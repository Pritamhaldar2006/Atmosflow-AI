# ChronoCloud local API

This is an additive wrapper around the existing ChronoCloud inference code. It does not change `model.py` or `inference.py`.

## Run

From the project root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --reload --port 8000
```

Visit `http://localhost:8000/docs` to test the API. Upload two matching `.npy`, PNG, or JPEG files. `.npy` arrays must be single-channel and normalized to `[0, 1]`; PNG/JPEG files are converted to grayscale and normalized automatically.

If the two uploaded frames have different dimensions, the later frame is resized to the earlier frame's dimensions before interpolation. For scientifically accurate satellite results, use two frames from the same sensor and spatial crop whenever possible.

The `/interpolate` endpoint accepts `num_intermediate` values 1, 3, and 7. Those are the counts that the model's midpoint-recursive inference can produce at evenly spaced timesteps.

Every successful response contains `gif_url`, a looping GIF generated from the intermediate frames, alongside the individual PNG previews and `.npy` files.
