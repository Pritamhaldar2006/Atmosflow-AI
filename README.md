# AtmosFlow AI

AtmosFlow AI creates intermediate geostationary-satellite frames between two scans. It uses a compact RIFE-inspired PyTorch model that estimates bidirectional motion, fuses warped input frames, and applies residual refinement to model non-rigid cloud movement.

The project includes both the research pipeline and a local web application. The web app accepts `.npy`, PNG, and JPEG inputs, generates one or more intermediate frames, and automatically creates a looping GIF.

## Features

- Interpolates cloud-motion frames between two satellite images
- Uses the supplied `liteRIFE_epoch39.pt` checkpoint for local inference
- Accepts single-channel `.npy`, PNG, JPG, and JPEG uploads
- Automatically converts PNG/JPEG uploads to grayscale model input
- Matches differing input dimensions by resizing the later image to the first image's size
- Generates individual PNG previews, downloadable `.npy` frames, and a looping GIF
- Includes data download, training, inference, and visualization scripts for research workflows

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
