"""
Run inference: given two real consecutive satellite frames, generate one
or more synthetic intermediate frames, effectively upsampling temporal
resolution (e.g. INSAT's 30-min cadence -> 10-min or better equivalent).

Usage:
    python inference.py \
        --checkpoint checkpoints/liteRIFE_epoch39.pt \
        --frame0 data/frames/2026-07-01T00-00.npy \
        --frame1 data/frames/2026-07-01T00-30.npy \
        --num-intermediate 2 \
        --out-dir results/

--num-intermediate 2 recursively interpolates to produce 2 evenly-spaced
frames between frame0 and frame1 (i.e. splits the 30-min gap into three
10-min steps): t=0.5 first, then t=0.25 and t=0.75 from the halves.
"""

import argparse
import os

import numpy as np
import torch

from model import LiteRIFE


def load_frame(path, crop_size=None):
    arr = np.load(path).astype(np.float32)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if crop_size is not None:
        _, h, w = arr.shape
        cy, cx = h // 2, w // 2
        cs = crop_size // 2
        arr = arr[:, cy-cs:cy+cs, cx-cs:cx+cs]
    return torch.from_numpy(arr).unsqueeze(0)  # (1, C, H, W)


@torch.no_grad()
def interpolate_recursive(model, I0, I1, num_intermediate, device):
    """Binary-recursive interpolation: for num_intermediate=1 -> just the
    midpoint; for num_intermediate=3 -> midpoint, then midpoints of each
    half, etc. Returns list of frames ordered in time between I0 and I1."""
    if num_intermediate == 0:
        return []
    mid, _, _ = model(I0.to(device), I1.to(device), t=0.5)
    mid = mid.cpu()
    if num_intermediate == 1:
        return [mid]
    n_left = num_intermediate // 2
    n_right = num_intermediate - 1 - n_left
    left = interpolate_recursive(model, I0, mid, n_left, device)
    right = interpolate_recursive(model, mid, I1, n_right, device)
    return left + [mid] + right


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LiteRIFE(in_channels=1).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    I0 = load_frame(args.frame0, args.crop_size)
    I1 = load_frame(args.frame1, args.crop_size)

    frames = interpolate_recursive(model, I0, I1, args.num_intermediate, device)

    os.makedirs(args.out_dir, exist_ok=True)
    base0 = os.path.splitext(os.path.basename(args.frame0))[0]
    base1 = os.path.splitext(os.path.basename(args.frame1))[0]
    for i, f in enumerate(frames):
        out_path = os.path.join(args.out_dir, f"{base0}_to_{base1}_interp{i}.npy")
        np.save(out_path, f.squeeze(0).numpy())
        print(f"saved {out_path}")

    print(f"generated {len(frames)} intermediate frame(s) between "
          f"{base0} and {base1}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--frame0", required=True)
    p.add_argument("--frame1", required=True)
    p.add_argument("--num-intermediate", type=int, default=1)
    p.add_argument("--out-dir", default="./results")
    p.add_argument("--crop-size", type=int, default=None,
                    help="Center crop size to avoid OOM on full frames")
    args = p.parse_args()
    main(args)
