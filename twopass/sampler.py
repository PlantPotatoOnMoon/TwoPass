"""Draft-Reuse Iterative Refinement.

Each solver query returns one velocity prediction, which supplies both the
UniPC update and the clean-video draft for the next query. The first query
receives a zero draft with the late adapter disabled; later queries receive the
previous clean estimate and enable the adapter. A K-query sample therefore
uses exactly K network calls.
"""
import torch

from twopass.noise import build_scheduler


def sequence_length(n_frames, height, width):
    """Number of transformer tokens of a clip (4x temporal, 16x spatial VAE; 2x2 patches)."""
    return ((n_frames - 1) // 4 + 1) * (height // 16) * (width // 16) // 4


def clean_estimate(noisy, velocity, sigma):
    """Clean-video estimate [B, C, F, H, W]; the observed first latent frame is preserved."""
    result = (noisy - sigma * velocity).detach()
    result[:, :, 0] = noisy[:, :, 0]
    return result


@torch.no_grad()
def generate(model, vae, context, first_frame, actions, n_frames, height, width, device, nfe, shift):
    """Generate a clip [3, n_frames, H, W] in [-1, 1] from the first frame [3, H, W] and scaled actions."""
    z_first = vae.encode([first_frame.unsqueeze(1)])[0]
    channels, _, h, w = z_first.shape
    latent_frames = (n_frames - 1) // 4 + 1
    seq_len = sequence_length(n_frames, height, width)

    scheduler = build_scheduler(nfe, device, shift)
    latent = torch.randn(channels, latent_frames, h, w, device=device)
    latent[:, 0] = z_first[:, 0]
    draft = [torch.zeros_like(latent)]
    context = [c.float() for c in context]

    for query, t in enumerate(scheduler.timesteps):
        timestep = torch.tensor([t], device=device).float()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            velocity = model([latent], t=timestep, context=context, seq_len=seq_len,
                             actions=actions, draft=draft, refine=query > 0)[0]
        velocity = velocity.float()
        sigma = scheduler.sigmas[query].to(latent.device)
        draft = [clean_estimate(latent[None], velocity[None], sigma)[0]]
        latent = scheduler.step(velocity.unsqueeze(0), t, latent.unsqueeze(0), return_dict=False)[0].squeeze(0)
        latent[:, 0] = z_first[:, 0]

    return vae.decode([latent])[0]
