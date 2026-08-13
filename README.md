
# AtmosFlow AI

AtmosFlow AI creates intermediate geostationary-satellite frames between two scans. It uses a compact RIFE-inspired PyTorch model that estimates bidirectional motion, fuses warped input frames, and applies residual refinement to model non-rigid cloud movement.

The project includes both the research pipeline and a local web application. The web app accepts `.npy`, PNG, and JPEG inputs, generates one or more intermediate frames, and automatically creates a looping GIF.

[👉Click The Link To Wathch the Website-->]https://atmosflow-ai-q8sp.onrender.com


[Click here to watch the project demo video](https://drive.google.com/file/d/10rRqM8v03o-8C2ThAGjlXBPnxobIBdJJ/view?usp=sharing)
## Features

# Sample Website View
<img width="1479" height="840" alt="image" src="https://github.com/user-attachments/assets/51792c7c-1595-4277-9916-9c53fdd951aa" />


- Interpolates cloud-motion frames between two satellite images
- Uses the supplied `liteRIFE_epoch39.pt` checkpoint for local inference
- Accepts single-channel `.npy`, PNG, JPG, and JPEG uploads
- Automatically converts PNG/JPEG uploads to grayscale model input
- Matches differing input dimensions by resizing the later image to the first image's size
- Generates individual PNG previews, downloadable `.npy` frames, and a looping GIF
- Includes data download, training, inference, and visualization scripts for research workflows
## Model & Methodology

### Why not a pretrained model?

Off-the-shelf frame interpolation models like RIFE, FLAVR, and AdaCoF are pretrained on natural video datasets (Vimeo-90K, UCF101) — everyday scenes with rigid objects, consistent lighting, and short-range camera motion. Geostationary satellite imagery violates nearly every one of those assumptions: cloud fields deform non-rigidly, form and dissipate, and change brightness/temperature over time rather than staying visually constant. Given this domain gap, ChronoCloud's model — **LiteRIFE** — was designed and trained from scratch specifically for satellite motion patterns, rather than fine-tuning a natural-video checkpoint.

### Architecture

LiteRIFE is a compact, RIFE-inspired (Real-time Intermediate Flow Estimation) architecture implemented in PyTorch, built around three cooperating stages:

**1. Coarse-to-fine optical flow estimation (IFBlock × 3)**
Instead of using a classical, hand-crafted optical flow algorithm (Lucas-Kanade, Farneback, TV-L1), flow is predicted by a stack of three convolutional "IFBlocks" operating at decreasing scales (coarse → fine). Each block refines the flow estimate from the previous scale, allowing the network to capture both large-scale storm-system motion and fine-grained local cloud deformation in the same pass. Flow is predicted bidirectionally — from frame 0 toward the target timestep, and from frame 1 toward it — rather than assuming simple linear motion.

**2. Warping and learned fusion/occlusion masking**
Both input frames are backward-warped toward the target timestep using the predicted flow. A learned fusion mask then decides, pixel by pixel, how much to trust each warped source. This is critical for weather imagery: a newly-forming convective cell or a dissipating cloud has no valid corresponding pixel to warp from in one of the two source frames, and the mask lets the network downweight that source rather than producing a warping artifact.

**3. Residual refinement network (U-Net)**
A lightweight encoder-decoder U-Net takes the warped, fused result plus contextual features from both source frames, and predicts a residual correction. This is the component that fixes the blur and ghosting that pure flow-warping leaves behind, and fills in plausible texture in regions the mask flagged as unreliable.

The full model is lightweight by design — roughly 84 trainable parameter tensors — specifically so it can be trained end-to-end on a single consumer GPU rather than requiring institutional compute.

### Training data and preprocessing

- **Source:** GOES-19, NOAA's operational GOES-East satellite, accessed via the public, unauthenticated AWS Open Data S3 bucket (`noaa-goes19`)
- **Channel:** Band 13 (~10.3 μm, longwave IR "clean window") — chosen because it captures cloud-top temperature in both day and night conditions, unlike visible-light bands
- **Volume:** 300 full-disk frames at native 10-minute scan cadence
- **Calibration:** Raw scan data converted to the `CMI` (Cloud and Moisture Imagery) product, which is already radiometrically calibrated to brightness temperature
- **NaN handling:** Full-disk GOES imagery contains NaN "space" pixels outside the Earth's visible disk; these are excluded from percentile-based normalization and zero-filled post-normalization, to prevent NaN propagation through training
- **Patching:** Frames are randomly cropped to 256×256 patches per training step, to fit GPU memory constraints while still exposing the model to diverse regional weather patterns across each full-disk image

### Training paradigm

Training is fully **self-supervised** — no manual annotation was required. Given three temporally consecutive real frames (t, t+1, t+2), the model is trained to predict the real middle frame (t+1) from the two outer frames (t and t+2). This is the standard setup for learned frame interpolation: the "label" is simply a real frame that already exists in the sequence, withheld from the model's input.

### Loss function and optimization

- **Loss:** A weighted combination of L1 pixel-reconstruction loss and SSIM (structural similarity) loss — L1 for sharp per-pixel accuracy, SSIM to preserve structural cloud boundaries and texture that pure pixel-wise loss tends to blur
- **Optimizer:** AdamW with weight decay
- **Learning rate schedule:** Cosine annealing over the full training run
- **Mixed precision (AMP):** Used throughout to roughly double effective batch size and speed up training on limited GPU memory
- **Gradient accumulation:** Used to simulate a larger effective batch size than what fits in memory at once

### Hardware and infrastructure

The entire pipeline — data download, preprocessing, training, and inference — was designed to run end-to-end on a **single free-tier Google Colab T4 GPU (16 GB)**, with no paid compute or institutional cluster access required. Checkpoints are saved incrementally to Google Drive during training to survive Colab's session disconnects.

### Results

After 40 training epochs, the model reached a **validation PSNR of approximately 29 dB**, evaluated on a held-out split of the same GOES-19 dataset. This was benchmarked against the qualitative failure modes typically seen with classical optical flow baselines (Farneback, TV-L1) on the same data — namely blur and ghosting artifacts on deforming, non-rigid cloud motion — which the learned flow + refinement approach visibly reduces.

### Inference

At inference time, the model supports **recursive multi-frame interpolation**: given two real frames, it can generate not just the midpoint but any number of evenly-spaced intermediate frames at arbitrary timestep `t ∈ (0, 1)`, by recursively interpolating between previously generated midpoints. This allows a satellite's native cadence (e.g., 10 or 30 minutes) to be upsampled to an arbitrarily denser synthetic sequence.

## Quick start: web application

Prerequisites: Python 3.10+ and Node.js 18+.

Open two PowerShell terminals in the project folder.

### 1. Start the API

```powershell
cd "<path-to-your-AtmosFlow-AI-project>"
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --reload --port 8000
```

The API starts at `http://localhost:8000`. Its interactive API documentation is at `http://localhost:8000/docs`.

### 2. Start the frontend

```powershell
cd "<path-to-your-AtmosFlow-AI-project>\frontend"
npm install
npm run dev
```

Open the address Vite displays, usually `http://localhost:5173`.

## Deploy with Docker or Render

The repository includes a production `Dockerfile` that builds the React app and
serves it from the FastAPI service. The browser and API therefore use the same
origin; no production `VITE_API_URL` is needed.

Build and run it locally:

```powershell
docker build -t atmosflow-ai .
docker run --rm -p 8000:8000 -e PORT=8000 atmosflow-ai
```

Then open `http://localhost:8000`; use `http://localhost:8000/health` for the
service health check.

To deploy on Render, push this repository and create a Blueprint from
`render.yaml`. It builds the Docker image and configures `/health` as the health
check. The supplied defaults cap each upload at 20 MB, reject frames larger than
16 million pixels, and remove generated results after one hour. Set the values
in `.env.example` in the host dashboard if you need different limits.

The included storage is temporary: generated GIFs, PNGs, and NumPy files are
not durable across restarts or multi-instance deployments. For persistent or
high-volume use, store results in object storage and put rate limiting in front
of the service.

### 3. Generate an animation

1. Upload an earlier and later satellite frame.
2. Choose 1, 3, or 7 intermediate frames.
3. Select **Generate frames**.
4. Inspect the generated frame previews or download the `.npy` outputs and GIF.

PNG and JPEG inputs can be previewed in the browser before submission. For best scientific results, use frames from the same sensor, projection, spatial crop, and acquisition sequence.

## Project layout

| Path | Purpose |
| --- | --- |
| `model.py` | LiteRIFE model: multi-scale motion estimation, warping, fusion, and refinement |
| `data_loader.py` | Builds training triplets from timestamp-sorted `.npy` frames |
| `download_data.py` | Downloads GOES imagery or converts local INSAT HDF5 data |
| `train.py` | Mixed-precision fine-tuning and checkpoint saving |
| `inference.py` | Command-line inference for `.npy` frame pairs |
| `visualize.py` | Creates comparison grids and GIFs from `.npy` files |
| `checkpoints/` | Supplied LiteRIFE checkpoints |
| `backend/` | FastAPI upload, inference, result, and GIF service |
| `backend/checkpoints/` | Checkpoint used by the deployed web backend |
| `frontend/` | React/Vite browser interface |

## Run inference from the command line

The original inference script accepts preprocessed single-channel `.npy` files:

```powershell
.\.venv\Scripts\python.exe inference.py `
  --checkpoint .\checkpoints\liteRIFE_epoch39.pt `
  --frame0 .\data\frames\frame0.npy `
  --frame1 .\data\frames\frame1.npy `
  --num-intermediate 1 `
  --out-dir .\results `
  --crop-size 256
```

Use a crop size divisible by 4. The standalone script requires both source files to exist; the web app adds image-format conversion, automatic resizing, and GIF creation.

## Download training data

Download GOES Band 13 frames into `data/frames`:

```powershell
.\.venv\Scripts\python.exe download_data.py `
  --source goes `
  --bucket noaa-goes16 `
  --product ABI-L2-CMIPF `
  --band 13 `
  --start 2026-07-01T00:00 `
  --num-frames 30 `
  --interval-min 10 `
  --out-dir .\data\frames
```

For INSAT data, download L1B HDF5 files from MOSDAC first, then use `--source insat-local` and `--local-dir`.

## Train or resume training

```powershell
.\.venv\Scripts\python.exe train.py `
  --frames-dir .\data\frames `
  --checkpoint-dir .\checkpoints `
  --epochs 40 `
  --batch-size 8 `
  --effective-batch 16 `
  --patch-size 256
```

For a GPU-equipped Colab environment, the same scripts can be run with `/content/...` paths.

## Model and data notes

- The supplied checkpoint expects one grayscale input channel.
- `.npy` input arrays should be normalized to `[0, 1]`.
- The web API resizes a mismatched later frame to match the earlier frame. This is convenient for demos, but matching source geometry before upload is more accurate.
- The web UI offers 1, 3, and 7 intermediate frames because midpoint-recursive inference gives evenly spaced output for those counts.
- Interpolated imagery is synthetic. Validate it against real observations before using it for weather analysis or operational decisions.

## Evaluation ideas

Evaluate against held-out true middle frames using PSNR and SSIM, and compare with classical optical-flow baselines such as Farneback or TV-L1. For weather-focused validation, also measure system-specific motion error, such as cyclone-eye displacement or convective-cell centroid drift.
## The Sample Output of the The Predicted Image Frame Is
<img width="1600" height="554" alt="image" src="https://github.com/user-attachments/assets/dc2d1351-8606-4f5a-a647-c1e352f69fc9" />
