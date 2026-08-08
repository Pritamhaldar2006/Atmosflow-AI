# Chrono cloud 
ChronoCloud is a deep learning system for temporal frame interpolation of geostationary satellite imagery. Using a RIFE-inspired architecture with learned optical flow, occlusion-aware fusion, and residual refinement, it predicts realistic intermediate frames between real satellite scans — effectively upsampling GOES/Himawari/INSAT's native 10–30 minute cadence toward near-real-time temporal resolution. This helps close the gap for time-sensitive monitoring tasks like cyclone tracking, wildfire spread, and flash-flood-triggering convection, where classical optical flow methods (Farneback, Lucas-Kanade) blur or fail on fast, non-rigid cloud motion.







# Satellite frame interpolation (LiteRIFE)

AI/ML optical-flow-based frame interpolation for geostationary satellite
imagery (INSAT-3D/3DR, Himawari-8/9, GOES-16/17/18), designed to fine-tune
on a single free-tier Colab T4 GPU.

## What this does

Given two consecutive real satellite frames, predicts the frames that
would have occurred in between -- effectively upsampling temporal
resolution (e.g. INSAT's 30-min cadence toward Himawari/GOES-like 10-min
or better), to support near-real-time monitoring of fires, cyclones,
thunderstorms, and floods.

Unlike classical optical flow (Farneback, Lucas-Kanade, TV-L1), the flow
estimator here is *learned* end-to-end jointly with a fusion mask and a
residual refinement network -- this is what fixes the blur/ghosting
classical methods produce on deforming, non-rigid cloud motion.

## Project Structure

| File/Folder | Purpose |
|---|---|
| `model.py` | LiteRIFE architecture: 3-scale flow estimation + warping + refinement U-Net |
| `data_loader.py` | Builds (I0, target, I1) training triplets from a folder of time-sorted frames |
| `download_data.py` | Pulls GOES-16/17/18 frames from AWS Open Data (free, no auth); converts locally-downloaded INSAT HDF5 (from MOSDAC) |
| `train.py` | Fine-tuning loop: mixed precision, gradient accumulation, Drive checkpointing |
| `inference.py` | Recursive multi-frame interpolation from a trained checkpoint |
| `visualize.py` | Generates static grids and GIF animations with AI timestamps |
| `requirements.txt` | Python dependencies |
| `data/` | Directory where downloaded/processed `.npy` frames are stored |
| `results/` | Directory where interpolated `.npy` frames and visualizations are saved |
| `checkpoints/` | Directory for saving trained model checkpoints |
| `myvenv/` | Local Python virtual environment |

## Colab setup

```
!git clone <your-repo-url> sat_interp
%cd sat_interp
!pip install -r requirements.txt --quiet

from google.colab import drive
drive.mount('/content/drive')
```

## 1. Get data

GOES (fastest way to get a real dataset, no credentials needed):

```
!python download_data.py --source goes \
    --bucket noaa-goes16 --product ABI-L2-CMIPF --band 13 \
    --start 2026-07-01T00:00 --num-frames 300 --interval-min 10 \
    --out-dir /content/data/frames
```

INSAT (after manually downloading L1B HDF5 files from MOSDAC into a
folder, since MOSDAC has no public unauthenticated bulk API):

```
!python download_data.py --source insat-local \
    --local-dir /content/raw_hdf5 --out-dir /content/data/frames
```

Band 13 (~10.3 micron, IR "clean window") works day and night. Add more
bands later by stacking channels if you want multi-spectral input.

## 2. Train

```
!python train.py \
    --frames-dir /content/data/frames \
    --checkpoint-dir /content/drive/MyDrive/sat_interp_ckpts \
    --epochs 40 --batch-size 8 --effective-batch 16 --patch-size 256
```

T4 (16 GB) notes:
- `--batch-size 8` at 256x256 with AMP fits comfortably; raise
  `--effective-batch` (gradient accumulation) instead of raw batch size
  if you want a larger effective batch without more memory.
- Checkpoints save to Google Drive every `--save-every` epochs (default
  5) so a Colab disconnect doesn't lose progress. Resume with:
  `--resume /content/drive/MyDrive/sat_interp_ckpts/liteRIFE_epoch19.pt`

## 3. Run inference

```
!python inference.py \
    --checkpoint /content/drive/MyDrive/sat_interp_ckpts/liteRIFE_epoch39.pt \
    --frame0 /content/data/frames/2026-07-01T00-00.npy \
    --frame1 /content/data/frames/2026-07-01T00-30.npy \
    --num-intermediate 2 \
    --out-dir /content/results
```

`--num-intermediate 2` splits a 30-min gap into three 10-min steps.

## 4. Visualization

To inspect the generated frames (which are saved as `.npy` raw arrays), use `visualize.py` to create a static side-by-side comparison and an animated GIF. This is great for hackathon demos!

```
!python visualize.py \
    --frames /content/data/frames/2026-07-01T00-00.npy \
             /content/results/2026-07-01T00-00_to_2026-07-01T00-30_interp0.npy \
             /content/results/2026-07-01T00-00_to_2026-07-01T00-30_interp1.npy \
             /content/data/frames/2026-07-01T00-30.npy \
    --out-prefix /content/results/demo \
    --cmap viridis \
    --ping-pong
```

## Evaluation beyond PSNR/SSIM

Pixel metrics don't tell you whether the model tracked the actual
weather system correctly. For a convincing capstone/research result,
also report:
- PSNR/SSIM against classical baselines (Farneback, TV-L1) on the same
  held-out triplets, to quantify the improvement.
- Domain-specific tracking error: e.g. cyclone eye-center displacement,
  or convective-cell centroid drift, between the true and interpolated
  middle frame.

## Extending

- Swap `LiteRIFE` for a pretrained checkpoint from the official
  `hzwer/Practical-RIFE` or `FLAVR` repos and fine-tune instead of
  training from scratch -- much faster convergence on a T4.
- Multi-band input: stack IR + WV + VIS channels, set
  `LiteRIFE(in_channels=N)` and adjust `data_loader.py` to load stacked
  arrays instead of single-channel `.npy` files.
