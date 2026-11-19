"""Data loader whose position and worker random states can be saved and restored.

The saved worker states are those of consumed batches, not of prefetched ones.
Requires a stateless dataset and yields one sample per batch.
"""
import pickle
import random
from itertools import islice

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, get_worker_info


def rng_state():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state())


def set_rng(state):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])


class TrackedDataset(Dataset):
    def __init__(self, dataset, states=None, rotation=0):
        self.dataset = dataset
        self.states = states
        self.rotation = rotation
        self.initialized = False

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        worker = get_worker_info()
        wid = worker.id
        if not self.initialized:
            if self.states is not None:
                logical_id = (wid + self.rotation) % worker.num_workers
                set_rng(pickle.loads(self.states[logical_id]))
            self.initialized = True
        sample = self.dataset[index]
        return sample, wid, pickle.dumps(rng_state())


class TrackedCollate:
    def __init__(self, collate):
        self.collate = collate

    def __call__(self, items):
        assert len(items) == 1
        sample, wid, state = items[0]
        return self.collate([sample]), wid, state


class OffsetSampler:
    def __init__(self, sampler):
        self.sampler = sampler
        self.offset = 0

    def __iter__(self):
        return islice(iter(self.sampler), self.offset, None)

    def __len__(self):
        return len(self.sampler) - self.offset


class ResumableLoader:
    def __init__(self, dataset, sampler, collate_fn, state=None):
        self.sampler = OffsetSampler(sampler)
        self.workers = 4
        self.start_batch = state['consumed'] if state else 0
        self.consumed = self.start_batch
        self.states = {}
        rotation = self.start_batch % self.workers
        self.rotation = rotation if state else 0
        if state:
            assert state['num_workers'] == self.workers
            self.states = {i: state['worker_rng'][(i + rotation) % self.workers]
                           for i in range(self.workers)}
        self.dataset = TrackedDataset(dataset, state['worker_rng'] if state else None, rotation)
        self.loader = DataLoader(self.dataset, batch_size=1, sampler=self.sampler,
                                 num_workers=self.workers, collate_fn=TrackedCollate(collate_fn),
                                 pin_memory=True, drop_last=True, persistent_workers=True, timeout=180)
        self.restoring = state is not None

    def __iter__(self):
        self.sampler.offset = self.start_batch
        self.consumed = self.start_batch
        # Recreating workers must not consume the next model-side random draw.
        saved_rng = torch.get_rng_state() if self.restoring else None
        iterator = iter(self.loader)
        if saved_rng is not None:
            torch.set_rng_state(saved_rng)
        self.restoring = False
        for batch, wid, state in iterator:
            self.states[wid] = state
            self.consumed += 1
            yield batch
        self.start_batch = 0
        if self.rotation:
            canonical = {((i + self.rotation) % self.workers): s for i, s in self.states.items()}
            self.loader._iterator._shutdown_workers()
            self.dataset.states = canonical
            self.dataset.rotation = 0
            self.dataset.initialized = False
            self.loader._iterator = None
            self.states = canonical
            self.rotation = 0
            self.restoring = True

    def state_dict(self):
        assert len(self.states) == self.workers, 'Need at least one consumed batch per worker'
        return {'consumed': self.consumed, 'num_workers': self.workers,
                'worker_rng': {((i + self.rotation) % self.workers): s for i, s in self.states.items()},
                'sampler_epoch': self.sampler.sampler.epoch}
