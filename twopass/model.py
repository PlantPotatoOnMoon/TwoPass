"""Action-conditioned Wan2.2 TI2V-5B predictor used by TwoPass.

The pretrained backbone is extended with three zero-initialised pathways, so the
extended network reproduces the pretrained one before any training:

* an action embedder that adds a per-latent-frame increment to the six
  modulation vectors of every transformer block;
* a draft embedding, a second patch embedding that receives the clean-video
  draft (Self-Generated Draft Training);
* a late adapter, a residual bottleneck applied after the last transformer
  block in the refinement role only (Refinement-Only Late Adapter).
"""
import glob
import json
import os

import torch
import torch.nn as nn
from safetensors.torch import load_file

from wan.modules.model import WanModel, sinusoidal_embedding_1d

# Parameters that are not part of the pretrained backbone.
NEW_MODULES = ("action_embedder", "self_cond_embedding", "pass_adapter")
# The adapter is initialised from its own generator so that attaching it does
# not shift the global random stream.
ADAPTER_INIT_SEED = 20261009


def pixel_actions_to_latent(actions, latent_frames, vae_time_stride=4):
    """Aggregate per-frame actions [B, F, A] into latent groups [B, F_lat, A].

    Latent frame 0 is the observed frame and keeps its (zero) action; latent
    frame k >= 1 covers pixel frames 4k-3 .. 4k and receives their mean action.
    """
    batch, _, action_dim = actions.shape
    out = actions.new_zeros(batch, latent_frames, action_dim)
    out[:, 0] = actions[:, 0]
    for k in range(1, latent_frames):
        out[:, k] = actions[:, vae_time_stride * (k - 1) + 1:vae_time_stride * k + 1].mean(dim=1)
    return out


class ActionEmbedder(nn.Module):
    """Map latent-frame actions to increments of the six block modulation vectors."""

    def __init__(self, action_dim=7, dim=3072, hidden=1024):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(action_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
            nn.SiLU(),
        )
        self.proj = nn.Linear(dim, 6 * dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, latent_actions):
        """[B, F_lat, A] -> [B, F_lat, 6, dim]."""
        increments = self.proj(self.mlp(latent_actions))
        batch, latent_frames, _ = increments.shape
        return increments.view(batch, latent_frames, 6, self.dim)


class LateAdapter(nn.Module):
    """Tokenwise residual bottleneck with a zero-initialised up-projection."""

    def __init__(self, dim, rank=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim, elementwise_affine=False),
            nn.Linear(dim, rank),
            nn.GELU(),
            nn.Linear(rank, dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x.float())


class TwoPassWanModel(WanModel):
    """Wan backbone with action conditioning, a draft pathway and a late adapter."""

    def __init__(self, *args, action_dim=7, adapter_after=30, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_embedder = ActionEmbedder(action_dim=action_dim, dim=self.dim)
        # The draft has its own patch embedding instead of extra input channels
        # on `patch_embedding`, so the new pathway can use its own learning rate.
        self.self_cond_embedding = nn.Conv3d(
            self.in_dim, self.dim, kernel_size=self.patch_size, stride=self.patch_size)
        nn.init.zeros_(self.self_cond_embedding.weight)
        nn.init.zeros_(self.self_cond_embedding.bias)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(ADAPTER_INIT_SEED)
            self.pass_adapter = LateAdapter(self.dim)
        self.adapter_after = adapter_after
        # Diagnostics of the most recent forward pass.
        self.adapter_delta_rms = torch.zeros(())
        self.adapter_relative_rms = torch.zeros(())

    @classmethod
    def from_pretrained_wan(cls, wan_dir, action_dim=7, adapter_after=30):
        """Build the model and load the official Wan2.2-TI2V-5B transformer weights."""
        with open(os.path.join(wan_dir, "config.json")) as f:
            config = json.load(f)
        config.pop("_class_name", None)
        config.pop("_diffusers_version", None)
        model = cls(action_dim=action_dim, adapter_after=adapter_after, **config)
        shards = sorted(glob.glob(os.path.join(wan_dir, "diffusion_pytorch_model-*.safetensors")))
        state = {}
        for shard in shards:
            state.update(load_file(shard))
        missing, unexpected = model.load_state_dict(state, strict=False)
        missing_backbone = [key for key in missing if not key.startswith(NEW_MODULES)]
        assert not missing_backbone, f"missing backbone weights: {missing_backbone[:10]}"
        assert not unexpected, f"unexpected weights: {unexpected[:10]}"
        return model

    def forward(self, x, t, context, seq_len, actions, draft, refine=False):
        """Predict the flow velocity.

        Args:
            x: list of noisy latents, each [C, F_lat, H, W].
            t: model timesteps, [B].
            context: list of text embeddings, each [L, C_text].
            seq_len: maximum token sequence length.
            actions: scaled per-frame actions, [B, F, A].
            draft: list of clean-video drafts shaped like `x`; all-zero when no
                draft is available.
            refine: refinement role; enables the late adapter.

        Returns:
            list of velocity predictions, each [C, F_lat, H, W].
        """
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        x = [self.patch_embedding(u.unsqueeze(0)) + self.self_cond_embedding(v.unsqueeze(0))
             for u, v in zip(x, draft)]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
            for u in x
        ])

        t = t.expand(t.size(0), seq_len)
        with torch.amp.autocast('cuda', dtype=torch.float32):
            bt = t.size(0)
            t = t.flatten()
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t)
                .unflatten(0, (bt, seq_len)).float())
            e0 = self.time_projection(e).unflatten(2, (6, self.dim))

            # Broadcast each latent frame's action increment over its spatial tokens.
            add = e0.new_zeros(e0.shape)
            for b in range(e0.size(0)):
                latent_frames = int(grid_sizes[b, 0].item())
                spatial = int(grid_sizes[b, 1].item()) * int(grid_sizes[b, 2].item())
                latent_actions = pixel_actions_to_latent(actions[b:b + 1], latent_frames)
                increments = self.action_embedder(latent_actions)[0]
                add[b, :latent_frames * spatial] = increments.unsqueeze(1).expand(
                    latent_frames, spatial, 6, self.dim).reshape(latent_frames * spatial, 6, self.dim)
            e0 = e0 + add
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        context = self.text_embedding(
            torch.stack([
                torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        kwargs = dict(e=e0, seq_lens=seq_lens, grid_sizes=grid_sizes,
                      freqs=self.freqs, context=context, context_lens=None)
        self.adapter_delta_rms = x.new_zeros(())
        self.adapter_relative_rms = x.new_zeros(())
        for block_index, block in enumerate(self.blocks, 1):
            x = block(x, **kwargs)
            if refine and block_index == self.adapter_after:
                delta = self.pass_adapter(x)
                self.adapter_delta_rms = delta.detach().float().square().mean().sqrt()
                self.adapter_relative_rms = self.adapter_delta_rms / \
                    x.detach().float().square().mean().sqrt().clamp_min(1e-12)
                x = x + delta
        x = self.head(x, e)
        x = self.unpatchify(x, grid_sizes)
        return [u.float() for u in x]
