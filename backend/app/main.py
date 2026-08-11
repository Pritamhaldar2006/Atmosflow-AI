"""FastAPI application for local AtmosFlow AI inference."""

from __future__ import annotations

from contextlib import asynccontextmanager
import os
from pathlib import Path
import time
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import numpy as np

from .inference_service import InferenceService


BACKEND_ROOT = Path(__file__).resolve().parents[1]
UPLOAD_DIR = BACKEND_ROOT / "uploads"
RESULT_DIR = BACKEND_ROOT / "results"
FRONTEND_DIST_DIR = BACKEND_ROOT.parent / "frontend" / "dist"
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(20 * 1024 * 1024)))
MAX_INPUT_PIXELS = int(os.getenv("MAX_INPUT_PIXELS", str(16_000_000)))
RESULTS_TTL_SECONDS = int(os.getenv("RESULTS_TTL_SECONDS", "3600"))
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
    if origin.strip()
]
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)
service = InferenceService()


@asynccontextmanager
async def lifespan(_: FastAPI):
    UPLOAD_DIR.mkdir(exist_ok=True)
    RESULT_DIR.mkdir(exist_ok=True)
    service.initialize()
    yield


app = FastAPI(title="AtmosFlow AI API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/results", StaticFiles(directory=RESULT_DIR), name="results")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "device": service.device}


async def save_upload(upload: UploadFile, label: str) -> Path:
    allowed_extensions = {".npy", ".png", ".jpg", ".jpeg"}
    extension = Path(upload.filename or "").suffix.lower()
    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"{label} must be a .npy, .png, .jpg, or .jpeg file.",
        )
    destination = UPLOAD_DIR / f"{uuid4()}_{label}{extension}"
    # Read one byte past the limit, preventing a client from making this worker
    # retain arbitrarily large request bodies in memory.
    content = await upload.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Each upload must be smaller than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
        )
    destination.write_bytes(content)
    return destination


def cleanup_expired_results() -> None:
    """Bound temporary disk use on hosts with ephemeral application storage."""
    cutoff = time.time() - RESULTS_TTL_SECONDS
    for result in RESULT_DIR.iterdir():
        if result.is_file() and result.stat().st_mtime < cutoff:
            result.unlink(missing_ok=True)


@app.post("/interpolate")
async def interpolate(
    frame0: UploadFile = File(...),
    frame1: UploadFile = File(...),
    num_intermediate: int = 1,
) -> dict:
    """Generate 1, 3, or 7 evenly spaced midpoint-recursive frames."""
    cleanup_expired_results()
    first_path = await save_upload(frame0, "frame0")
    second_path = await save_upload(frame1, "frame1")
    try:
        first = service.load_frame(first_path)
        second = service.load_frame(second_path)
        resized_input = first.shape != second.shape
        frames = service.interpolate(first, second, num_intermediate)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        first_path.unlink(missing_ok=True)
        second_path.unlink(missing_ok=True)

    job_id = uuid4().hex
    results = []
    for index, frame in enumerate(frames, start=1):
        npy_name = f"{job_id}_frame_{index}.npy"
        png_name = f"{job_id}_frame_{index}.png"
        np.save(RESULT_DIR / npy_name, frame)
        service.save_preview(frame, RESULT_DIR / png_name)
        results.append({
            "index": index,
            "npy_url": f"/results/{npy_name}",
            "preview_url": f"/results/{png_name}",
        })
    gif_name = f"{job_id}_animation.gif"
    service.save_animation(frames, RESULT_DIR / gif_name)
    return {
        "job_id": job_id,
        "frames": results,
        "gif_url": f"/results/{gif_name}",
        "resized_input": resized_input,
    }


# In the production Docker image the compiled frontend is copied here, so the
# browser and API share one domain and no API URL needs to be embedded at build
# time.  The conditional keeps the local backend-only workflow working.
if FRONTEND_DIST_DIR.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST_DIR, html=True), name="frontend")
