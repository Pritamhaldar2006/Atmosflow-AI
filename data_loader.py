"""
Dataset for training frame interpolation on geostationary satellite imagery.

Expected input layout (after running download_data.py or your own prep):

    data/frames/
        2026-07-01T00-00.npy
        2026-07-01T00-10.npy
        2026-07-01T00-20.npy
        ...

Each .npy is a single-channel float32 array (already calibrated to
brightness temperature for IR, or reflectance for VIS), full-disk or a
pre-cropped region of interest, normalized to roughly [0, 1] by
download_data.py.

The dataset builds (I0, I1, It) triplets from *consecutive* frames in
the sorted file list, where It is the true middle frame -- this is
the standard self-supervised setup for frame interpolation: no manual
labels needed, ground truth is just "the frame that actually happened
in between."

If your source cadence only gives you pairs (e.g. INSAT at 30 min with
no true 15-min ground truth available), use `SatelliteDataset` in
"pair mode" (skip=2 with return_t=True) purely for inference, and train
on a denser-cadence source (Himawari/GOES) where a true middle frame
exists.
"""

import glob
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset


class SatelliteTripletDataset(Dataset):
    def __init__(self, frames_dir, patch_size=256, skip=2, train=True,
                 augment=True, file_ext="npy"):
        """
        frames_dir : directory of time-sorted single-channel frames
        patch_size : random crop size for training, center crop for val
        skip       : distance (in frames) between I0 and I1;
                     skip=2 means I0=frame[i], target=frame[i+1], I1=frame[i+2]
        train      : if True, random crop + augment; else deterministic center crop
        augment    : random horizontal flip (safe for satellite imagery;
                     do NOT enable rotation, orientation is physically meaningful)
        """
        self.paths = sorted(glob.glob(os.path.join(frames_dir, f"*.{file_ext}")))
        if len(self.paths) < skip + 1:
            raise ValueError(
                f"Need at least {skip + 1} frames in {frames_dir}, found {len(self.paths)}"
            )
        self.patch_size = patch_size
        self.skip = skip
        self.train = train
        self.augment = augment
        self.n_triplets = len(self.paths) - skip

    def __len__(self):
        return self.n_triplets

    def _load(self, path):
        arr = np.load(path).astype(np.float32)
        arr = np.nan_to_num(arr, nan=0.0)
        if arr.ndim == 2:
            arr = arr[None, ...]  # add channel dim -> (1, H, W)
        return arr

    def _crop_coords(self, h, w):
        ps = self.patch_size
        if h < ps or w < ps:
            raise ValueError(f"Frame {h}x{w} smaller than patch_size {ps}")
        if self.train:
            top = random.randint(0, h - ps)
            left = random.randint(0, w - ps)
        else:
            top = (h - ps) // 2
            left = (w - ps) // 2
        return top, left

    def __getitem__(self, idx):
        i0_path = self.paths[idx]
        it_path = self.paths[idx + self.skip // 2]
        i1_path = self.paths[idx + self.skip]

        i0 = self._load(i0_path)
        it = self._load(it_path)
        i1 = self._load(i1_path)

        _, h, w = i0.shape
        top, left = self._crop_coords(h, w)
        ps = self.patch_size
        i0 = i0[:, top:top + ps, left:left + ps]
        it = it[:, top:top + ps, left:left + ps]
        i1 = i1[:, top:top + ps, left:left + ps]

        if self.train and self.augment and random.random() < 0.5:
            i0 = np.ascontiguousarray(i0[:, :, ::-1])
            it = np.ascontiguousarray(it[:, :, ::-1])
            i1 = np.ascontiguousarray(i1[:, :, ::-1])

        return {
            "I0": torch.from_numpy(i0),
            "I1": torch.from_numpy(i1),
            "It": torch.from_numpy(it),
            "t": 0.5,  # relative timestep of the target frame between I0 and I1
        }


def make_loaders(frames_dir, patch_size=256, batch_size=8, val_frac=0.1,
                  num_workers=2):
    from torch.utils.data import DataLoader, Subset

    full = SatelliteTripletDataset(frames_dir, patch_size=patch_size, train=True)
    n_val = max(1, int(len(full) * val_frac))
    n_train = len(full) - n_val

    train_ds = Subset(full, range(0, n_train))
    val_full = SatelliteTripletDataset(frames_dir, patch_size=patch_size, train=False,
                                        augment=False)
    val_ds = Subset(val_full, range(n_train, n_train + n_val))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, drop_last=True,
                               pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


if __name__ == "__main__":
    # smoke test with synthetic frames if no real data is present yet
    import tempfile

    tmp = tempfile.mkdtemp()
    for i in range(6):
        np.save(os.path.join(tmp, f"frame_{i:03d}.npy"),
                np.random.rand(300, 300).astype(np.float32))
    ds = SatelliteTripletDataset(tmp, patch_size=128)
    sample = ds[0]
    print({k: (v.shape if hasattr(v, "shape") else v) for k, v in sample.items()})
