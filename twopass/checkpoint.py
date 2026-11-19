"""Full training-state checkpoints: model, per-rank optimizer state, random states, data position."""
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullStateDictConfig, FullyShardedDataParallel as FSDP, StateDictType


def save_checkpoint(model, optimizer, args, step, epoch, loader, draft_rng):
    """Write `stepXXXXXXX.pt` (model) and `checkpoints/stepXXXXXXX/` (training state of every rank)."""
    out_dir = Path(args.out_dir)
    state_dir = out_dir / 'checkpoints' / f'step{step:07d}'
    state_dir.mkdir(parents=True, exist_ok=True)
    rank = dist.get_rank()
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT,
                              FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
        model_state = model.state_dict()
    if rank == 0:
        path = out_dir / f'step{step:07d}.pt'
        partial = path.with_suffix('.partial')
        torch.save(dict(model=model_state, step=step), partial)
        os.replace(partial, path)
    del model_state
    payload = dict(optimizer=optimizer.state_dict(), step=step, world_size=dist.get_world_size(), rank=rank,
                   config=vars(args), epoch=epoch, batch_index=loader.consumed - 1,
                   dataloader=loader.state_dict(), torch_rng=torch.get_rng_state(),
                   cuda_rng=torch.cuda.get_rng_state(), python_rng=random.getstate(),
                   numpy_rng=np.random.get_state(), sc_rng=draft_rng.getstate())
    path = state_dir / f'optimizer_rank{rank}.pt'
    partial = path.with_suffix('.partial')
    torch.save(payload, partial)
    os.replace(partial, path)
    del payload
    dist.barrier()
    if rank == 0:
        (state_dir / 'config.json').write_text(json.dumps(vars(args), indent=2))
        (state_dir / 'complete.json').write_text(json.dumps(dict(step=step, world_size=dist.get_world_size())))
    dist.barrier()


def load_training_state(path, optimizer, step, world_size, rank, args):
    """Load one rank's training state and restore its optimizer."""
    saved = torch.load(path, map_location='cpu', weights_only=False)
    assert saved['step'] == step, (saved['step'], step)
    assert saved['world_size'] == world_size and saved['rank'] == rank
    for key in ('dataset', 'seed'):
        assert saved['config'][key] == getattr(args, key), (key, saved['config'][key])
    state = saved.pop('optimizer')
    assert state['state'], 'Optimizer checkpoint has no Adam state'
    steps = [float(v['step']) for v in state['state'].values()]
    assert min(steps) > 0
    assert all('exp_avg' in v and 'exp_avg_sq' in v for v in state['state'].values())
    optimizer.load_state_dict(state)
    assert [float(v['step']) for v in optimizer.state.values()] == steps
    for param, moments in optimizer.state.items():
        assert moments['exp_avg'].shape == param.shape and moments['exp_avg_sq'].shape == param.shape
    return saved


def restore_rng(saved, draft_rng, device):
    """Restore the model-side random streams of one rank."""
    torch.set_rng_state(saved['torch_rng'])
    torch.cuda.set_rng_state(saved['cuda_rng'], device=device)
    random.setstate(saved['python_rng'])
    np.random.set_state(saved['numpy_rng'])
    draft_rng.setstate(saved['sc_rng'])
