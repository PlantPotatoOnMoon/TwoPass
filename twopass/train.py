"""TwoPass training.

Every update draws one noise level from the solver-query grid. With probability
`draft_probability`, a no-gradient pass of the shared network first produces a
clean-video draft at the same noisy state; the supervised pass then receives
that draft and runs in the refinement role, with the late adapter enabled.
Otherwise the draft is zero and the adapter is disabled. The loss is the
standard flow-matching objective on the future latent frames.

Run one process per GPU (scripts/train.sh). Every process loads one window per
update, so the global batch size equals the number of GPUs.
"""
import argparse
import functools
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DistributedSampler

from wan.modules.model import WanAttentionBlock
from twopass.checkpoint import load_training_state, restore_rng, save_checkpoint
from twopass.data import CachedWindowDataset, collate
from twopass.model import NEW_MODULES, LateAdapter, TwoPassWanModel
from twopass.noise import QueryGrid
from twopass.resumable_loader import ResumableLoader
from twopass.sampler import sequence_length


def parse_args():
    parser = argparse.ArgumentParser(description='TwoPass training')
    parser.add_argument('--config', required=True, help='JSON file with the hyperparameters')
    parser.add_argument('--wan_dir', required=True, help='directory with the Wan2.2-TI2V-5B weights')
    parser.add_argument('--manifest', required=True, help='manifest of cached training windows')
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--resume_checkpoint', default='', help='stepXXXXXXX.pt to resume from')
    args = parser.parse_args()
    for key, value in json.loads(Path(args.config).read_text()).items():
        setattr(args, key, value)
    return args


def main():
    args = parse_args()
    dist.init_process_group('nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    resume = bool(args.resume_checkpoint)
    step = 0

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    model = TwoPassWanModel.from_pretrained_wan(
        args.wan_dir, action_dim=args.action_dim, adapter_after=args.adapter_after)
    if resume:
        model_path = Path(args.resume_checkpoint)
        state_dir = model_path.parent / 'checkpoints' / model_path.stem
        payload = torch.load(model_path, map_location='cpu', weights_only=False)
        step = int(payload['step'])
        model.load_state_dict(payload['model'], strict=True)
        del payload
    model.to(device)
    wrap_policy = functools.partial(
        transformer_auto_wrap_policy, transformer_layer_cls={WanAttentionBlock, LateAdapter})
    model = FSDP(model, auto_wrap_policy=wrap_policy,
                 mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                                                buffer_dtype=torch.float32),
                 sharding_strategy=ShardingStrategy.FULL_SHARD, device_id=local_rank,
                 use_orig_params=True, sync_module_states=True)

    # The pretrained backbone and the newly added modules use separate learning rates.
    named = list(model.named_parameters())
    backbone = [p for n, p in named if not any(key in n for key in NEW_MODULES)]
    new_modules = [p for n, p in named if any(key in n for key in NEW_MODULES)]
    learning_rates = [args.backbone_lr, args.new_module_lr]
    optimizer = torch.optim.AdamW(
        [dict(params=backbone, lr=learning_rates[0]), dict(params=new_modules, lr=learning_rates[1])],
        betas=tuple(args.adam_betas), weight_decay=args.weight_decay)

    # The draft decision uses its own generator, seeded identically on all ranks.
    draft_rng = random.Random()
    loader_state = None
    epoch = 0
    if resume:
        saved = load_training_state(state_dir / f'optimizer_rank{rank}.pt', optimizer, step, world_size, rank, args)
        restore_rng(saved, draft_rng, device)
        loader_state = saved['dataloader']
        epoch = int(saved['epoch'])
        del saved
    else:
        torch.manual_seed(args.seed + rank * 1000)
        torch.cuda.manual_seed(args.seed + rank * 1000)
        np.random.seed(args.seed + rank * 1000)
        random.seed(args.seed + rank * 1000)
        draft_rng.seed(args.seed + 9173)
    for group, lr in zip(optimizer.param_groups, learning_rates):
        group['lr'] = lr

    dataset = CachedWindowDataset(args.manifest)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True,
                                 seed=args.seed, drop_last=True)
    loader = ResumableLoader(dataset, sampler, collate, loader_state)
    # Starting the data workers must not advance the model-side random stream.
    loader.restoring = True

    grid = QueryGrid(device, args.shift)
    args.noise_grid = {str(nfe): sigmas.tolist() for nfe, sigmas in grid.sigmas.items()}
    args.model_timesteps = {str(nfe): timesteps.tolist() for nfe, timesteps in grid.timesteps.items()}
    if rank == 0:
        (out_dir / 'config.json').write_text(json.dumps(vars(args), indent=2))

    model.train()
    seq_len = sequence_length(args.n_frames, args.height, args.width)
    start_step = step
    started = time.time()
    tick = time.perf_counter()
    last_logged = step
    if rank == 0:
        print('READY', dict(step=step, target_step=args.target_step, world_size=world_size), flush=True)
    while step < args.target_step:
        sampler.set_epoch(epoch)
        for batch in loader:
            if step >= args.target_step:
                break
            x0 = batch['gt_latents'].to(device).float()
            boundary = batch['boundary_latent'].to(device).float()
            x0 = torch.cat((boundary, x0[:, :, 1:]), dim=2)
            actions = batch['actions'].to(device)
            context = [c.to(device).float() for c in batch['context']]

            eps = torch.randn_like(x0)
            uniform_draw = torch.rand(len(x0), device=device)
            use_draft = draft_rng.random() < args.draft_probability
            nfe, index = grid.select(uniform_draw[0])
            sigma = grid.sigmas[nfe][index].expand(len(x0))
            timestep = grid.timesteps[nfe][index].expand(len(x0))
            sig = sigma[:, None, None, None, None]
            xt = (1 - sig) * x0 + sig * eps
            xt[:, :, 0] = boundary[:, :, 0]
            target = eps - x0

            draft = [torch.zeros_like(v) for v in xt]
            if use_draft:
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    first = torch.stack(model(list(xt.unbind(0)), t=timestep, context=context, seq_len=seq_len,
                                              actions=actions, draft=draft, refine=False)).float()
                    clean = (xt - sig * first).detach()
                    clean[:, :, 0] = boundary[:, :, 0]
                    draft = list(clean.unbind(0))
                assert not clean.requires_grad

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                pred = torch.stack(model(list(xt.unbind(0)), t=timestep, context=context, seq_len=seq_len,
                                         actions=actions, draft=draft, refine=use_draft)).float()
            loss = (pred[:, :, 1:] - target[:, :, 1:]).square().mean()
            loss.backward()

            # The draft pathway and the adapter must learn exactly on draft updates.
            adapter_grad = torch.zeros((), device=device)
            draft_grad = torch.zeros((), device=device)
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if 'pass_adapter' in name:
                        adapter_grad += param.grad.detach().float().square().sum()
                    if 'self_cond_embedding.weight' in name:
                        draft_grad += param.grad.detach().float().square().sum()
            dist.all_reduce(adapter_grad)
            dist.all_reduce(draft_grad)
            if use_draft:
                assert draft_grad > 0, 'missing draft-pathway gradient'
                assert adapter_grad > 0, 'missing adapter gradient'
            else:
                assert adapter_grad == 0, 'adapter active without a draft'

            grad_norm = model.clip_grad_norm_(args.gradient_clip)
            assert torch.isfinite(loss) and torch.isfinite(grad_norm)
            optimizer.step()
            step += 1

            if step <= start_step + 8 or step % args.log_every == 0:
                mean_loss = loss.detach().clone()
                dist.all_reduce(mean_loss)
                now = time.perf_counter()
                row = dict(step=step, loss=float(mean_loss / world_size), loss_rank0=float(loss.detach()),
                           draft_used=use_draft, schedule_nfe=nfe, schedule_index=index,
                           sigma_rank0=float(sigma[0]), window_rank0=batch['id'][0],
                           gradient_norm=float(grad_norm), adapter_grad_norm=float(adapter_grad.sqrt()),
                           draft_grad_norm=float(draft_grad.sqrt()),
                           adapter_delta_rms=float(model.module.adapter_delta_rms),
                           adapter_relative_rms=float(model.module.adapter_relative_rms),
                           network_calls=1 + int(use_draft), data_cursor=loader.consumed, data_epoch=epoch,
                           seconds_per_update=(now - tick) / (step - last_logged),
                           learning_rates=[group['lr'] for group in optimizer.param_groups],
                           peak_memory_gib=torch.cuda.max_memory_allocated() / 2**30,
                           elapsed_s=time.time() - started)
                if rank == 0:
                    with (out_dir / 'history.jsonl').open('a') as f:
                        f.write(json.dumps(row) + '\n')
                    print(json.dumps(row), flush=True)
                tick = now
                last_logged = step
            if step % args.save_every == 0 and step < args.target_step:
                save_started = time.time()
                save_checkpoint(model, optimizer, args, step, epoch, loader, draft_rng)
                save_seconds = time.time() - save_started
                started += save_seconds
                tick += save_seconds
            if step >= args.target_step:
                break
        if step < args.target_step:
            epoch += 1

    loop_seconds = time.time() - started
    save_checkpoint(model, optimizer, args, step, epoch, loader, draft_rng)
    if rank == 0:
        (out_dir / 'training_end.json').write_text(
            json.dumps(dict(step=step, training_loop_seconds=loop_seconds)))
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
