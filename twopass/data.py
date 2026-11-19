"""Cached training windows: Wan-VAE latents, text context and actions."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# Per-dimension action scale: six pose deltas and the gripper state. Language
# Table's planar actions occupy the first two dimensions; the others are zero.
ACTION_SCALE = torch.tensor([20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 1.0], dtype=torch.float32)
CACHE_KEYS = ("gt_latents", "boundary_latent", "context", "actions")


class CachedWindowDataset(Dataset):
    """Windows listed in a manifest, each stored as one `.npz` file."""

    def __init__(self, manifest_path):
        self.rows = json.loads(Path(manifest_path).read_text())["rows"]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with np.load(row["cache_npz"], allow_pickle=False) as cache:
            sample = {key: torch.from_numpy(cache[key].copy()) for key in CACHE_KEYS}
        sample["actions"] = sample["actions"] * ACTION_SCALE
        sample["id"] = row["id"]
        return sample


def collate(samples):
    return {key: ([s[key] for s in samples] if key == "id" else torch.stack([s[key] for s in samples]))
            for key in samples[0]}
