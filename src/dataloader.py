# src/dataloader.py
"""
Dataset / DataLoader for prop-det event batches (.pt files).

Each .pt file holds an (N × 5) tensor:  x, y, t, polarity, label
* x, y   : pixel coords (int)
* t      : timestamp  (µs)
* polarity: 0 = OFF, 1 = ON
* label  : 0 = background, 1 = propeller   (not used here yet)

We voxelise into (z, y, x) with 100 µs time-slices:
    z = (t - t.min()) // 100
Features per voxel = [ON_count, OFF_count]  → 2 input channels
"""

from pathlib import Path
from typing import List, Dict, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class EventVoxelDataset(Dataset):
    def __init__(self, root: Union[str, Path], time_bin_us: int):
        root = Path(root)
        if not root.exists():
            raise FileNotFoundError(root)
        self.files = sorted(root.glob("*.pt"))
        if not self.files:
            raise RuntimeError(f"No .pt files in {root}")
        self.time_bin_us = time_bin_us

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict:
        """
        Return one sample with matching coords, feats, label per voxel.
        Features are simply [is_ON, is_OFF] from the *first* event that
        lands in each voxel slice.
        """
        ev = torch.load(self.files[idx])               # (N, 5)
        xs, ys, ts, ps, lbl = ev.T                     # unpack columns

        # temporal index (start at 0 for each batch)
        z = ((ts - ts.min()) // self.time_bin_us).int()

        # full per-event coordinate array
        coords_full = torch.stack((z, ys, xs), dim=1).cpu().numpy().astype(np.int32)

        # ------------------------------------------------------------------ #
        # create per-event two-channel features:  [ON, OFF]
        feats_full = np.stack((ps.numpy(), 1 - ps.numpy()), axis=1).astype(np.float32)
        # ------------------------------------------------------------------ #

        # build 1D voxel key and find first occurrence of each unique voxel
        key = coords_full[:, 0] * 1_000_000_000 + coords_full[:, 1] * 1_000_000 + coords_full[:, 2]
        _, inv = np.unique(key, return_inverse=True)
        keep_idx = np.unique(inv, return_index=True)[1]       # first hit of every voxel

        # voxel labels: positive if *any* event in that voxel is labelled 1
        # vox_target = np.zeros(len(keep_idx), dtype=np.uint8)
        # np.maximum.at(vox_target, inv[keep_idx], lbl.numpy())

        # slice arrays so coords, feats, target line up 1-to-1
        coords = coords_full[keep_idx]          # (M, 3)
        feats  = feats_full[keep_idx]           # (M, 2)
        target = lbl[keep_idx]

        return {
            "voxel_grids": [coords],
            "voxel_feats": [feats],
            "target":      [target],
        }



# -------------  simple collate & loader helpers ----------------
def collate_evt(batch: List[Dict]) -> List[Dict]:
    """Keep list-of-dicts format for SpConv wrapper."""
    return batch


def make_loaders(cfg) -> Tuple[DataLoader, DataLoader]:
    """
    Build train / val loaders from cfg.dataset.root,
    expecting subfolders  train/   and  val/
    """
    root = Path(cfg.dataset.labelled_root)
    bin_us = cfg.dataset.time_bin_us
    ds_train = EventVoxelDataset(root / "train", bin_us)
    ds_val   = EventVoxelDataset(root / "val",   bin_us)

    loader_train = DataLoader(
        ds_train,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        collate_fn=collate_evt,
    )
    loader_val = DataLoader(
        ds_val,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        collate_fn=collate_evt,
    )
    return loader_train, loader_val
