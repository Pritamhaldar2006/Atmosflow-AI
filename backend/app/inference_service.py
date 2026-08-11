"""Safe adapter between the web API and AtmosFlow AI's existing model."""

from __future__ import annotations

from pathlib import Path
import os
import threading

import numpy as np
from PIL import Image, UnidentifiedImageError
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "backend" / "checkpoints" / "liteRIFE_epoch39.pt"
MAX_INPUT_PIXELS = int(os.getenv("MAX_INPUT_PIXELS", "16000000"))


class InferenceService:
    """Keeps the model loaded once and prepares uploaded NumPy frames."""

    def __init__(self, checkpoint: Path = DEFAULT_CHECKPOINT) -> None:
        self.checkpoint = checkpoint
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.lock = threading.Lock()

    def initialize(self) -> None:
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint}")

        # Import the existing model without changing the original project code.
        import sys
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        from model import LiteRIFE

        model = LiteRIFE(in_channels=1).to(self.device)
        checkpoint = torch.load(self.checkpoint, map_location=self.device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        self.model = model

    @staticmethod
    def load_frame(path: Path) -> np.ndarray:
        if path.suffix.lower() == ".npy":
            try:
                array = np.load(path, allow_pickle=False).astype(np.float32)
            except (OSError, ValueError) as exc:
                raise ValueError("The uploaded .npy file could not be read.") from exc
            if array.ndim == 2:
                array = array[None, ...]
        else:
            try:
                # RGB/RGBA images are converted to luminance because the
                # supplied LiteRIFE checkpoint accepts one input channel.
                with Image.open(path) as image:
                    width, height = image.size
                    if width * height > MAX_INPUT_PIXELS:
                        raise ValueError(
                            f"Each image must contain at most {MAX_INPUT_PIXELS:,} pixels."
                        )
                    array = np.asarray(image.convert("L"), dtype=np.float32)[None, ...] / 255.0
            except (OSError, UnidentifiedImageError) as exc:
                raise ValueError("The uploaded image could not be read.") from exc
        if array.ndim != 3 or array.shape[0] != 1:
            raise ValueError("Each upload must resolve to a single-channel 2D frame.")
        if array.shape[1] * array.shape[2] > MAX_INPUT_PIXELS:
            raise ValueError(f"Each frame must contain at most {MAX_INPUT_PIXELS:,} pixels.")
        if not np.isfinite(array).all():
            array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)
        return np.clip(array, 0.0, 1.0)

    @staticmethod
    def _pad_to_multiple_of_four(frame: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        _, height, width = frame.shape
        padded_height = (height + 3) // 4 * 4
        padded_width = (width + 3) // 4 * 4
        padding = ((0, 0), (0, padded_height - height), (0, padded_width - width))
        return np.pad(frame, padding, mode="edge"), (height, width)

    @staticmethod
    def match_dimensions(reference: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """Resize an uploaded later frame to the reference frame's grid."""
        if reference.shape == frame.shape:
            return frame
        _, height, width = reference.shape
        resized = F.interpolate(
            torch.from_numpy(frame).unsqueeze(0),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        return resized.squeeze(0).numpy()

    @torch.no_grad()
    def interpolate(self, frame0: np.ndarray, frame1: np.ndarray, count: int) -> list[np.ndarray]:
        if self.model is None:
            raise RuntimeError("Model is not initialized.")
        frame1 = self.match_dimensions(frame0, frame1)
        if count not in {1, 3, 7}:
            raise ValueError("Choose 1, 3, or 7 intermediate frames for evenly spaced output.")

        padded0, original_size = self._pad_to_multiple_of_four(frame0)
        padded1, _ = self._pad_to_multiple_of_four(frame1)
        tensor0 = torch.from_numpy(padded0).unsqueeze(0)
        tensor1 = torch.from_numpy(padded1).unsqueeze(0)

        def recursively_interpolate(left: torch.Tensor, right: torch.Tensor, n: int) -> list[torch.Tensor]:
            if n == 0:
                return []
            midpoint, _, _ = self.model(left.to(self.device), right.to(self.device), t=0.5)
            midpoint = midpoint.cpu()
            if n == 1:
                return [midpoint]
            left_count = n // 2
            right_count = n - 1 - left_count
            return (
                recursively_interpolate(left, midpoint, left_count)
                + [midpoint]
                + recursively_interpolate(midpoint, right, right_count)
            )

        with self.lock:
            outputs = recursively_interpolate(tensor0, tensor1, count)
        height, width = original_size
        return [output.squeeze(0).numpy()[:, :height, :width] for output in outputs]

    @staticmethod
    def save_preview(frame: np.ndarray, path: Path) -> None:
        image = (np.clip(frame.squeeze(), 0, 1) * 255).astype(np.uint8)
        Image.fromarray(image, mode="L").save(path)

    @staticmethod
    def save_animation(frames: list[np.ndarray], path: Path, duration_ms: int = 450) -> None:
        """Save generated frames as a looping, ping-pong GIF."""
        images = [
            Image.fromarray((np.clip(frame.squeeze(), 0, 1) * 255).astype(np.uint8), mode="L")
            for frame in frames
        ]
        if len(images) > 2:
            images.extend(images[-2:0:-1])
        images[0].save(
            path,
            format="GIF",
            save_all=True,
            append_images=images[1:],
            duration=duration_ms,
            loop=0,
            disposal=2,
        )
