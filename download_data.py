"""
Download and preprocess geostationary satellite frames into the
single-channel .npy format expected by data_loader.py.

GOES-16/17/18 (NOAA, AWS Open Data, public, no credentials needed):
    bucket: noaa-goes16 / noaa-goes17 / noaa-goes18
    product: ABI-L1b-RadF (full disk radiance) or ABI-L2-CMIPF (calibrated,
             easier -- brightness temp / reflectance already computed)

Himawari-8/9 (JAXA, AWS Open Data via `noaa-himawari8`):
    similar structure, HSD format -- needs `satpy` to decode cleanly.

INSAT-3D/3DR (ISRO, via MOSDAC):
    MOSDAC requires a free account and does not expose a public,
    unauthenticated bulk-download API like NOAA/JAXA do. For INSAT data,
    register at https://mosdac.gov.in, download L1B HDF5 files manually
    or via their order/subscription system, then point `--local-dir` at
    the folder of downloaded files and use `convert_local_hdf5()` below
    instead of the AWS fetch path.

This script targets a single IR channel (Band 13, ~10.3 micron, "clean
window" -- good for both day and night cloud-top temperature) by
default. Add VIS/other bands by repeating the fetch with a different
band number and stacking channels in data_loader.py if you want a
multi-spectral input.
"""

import argparse
import datetime as dt
import io
import os

import numpy as np


def fetch_goes_band(bucket, product, band, start_time, num_frames, interval_min,
                     out_dir):
    """Fetch `num_frames` consecutive GOES ABI frames for one band, decode
    radiance/BT, save as normalized float32 .npy patches.

    Requires: boto3, xarray, netCDF4 (or h5netcdf) -- install via:
        pip install boto3 xarray netCDF4 s3fs
    """
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
    import xarray as xr

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    os.makedirs(out_dir, exist_ok=True)

    t = start_time
    saved = 0
    while saved < num_frames:
        prefix = (f"{product}/{t.year}/{t.timetuple().tm_yday:03d}/"
                  f"{t.hour:02d}/")
        resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        candidates = [
            o["Key"] for o in resp.get("Contents", [])
            if f"C{band:02d}_" in o["Key"]
        ]
        valid_candidates = []
        for k in candidates:
            try:
                s_part = k.split("_s")[1][:13]
                k_time = dt.datetime.strptime(s_part, "%Y%j%H%M%S")
                if abs((k_time - t).total_seconds()) <= (interval_min * 60) / 2:
                    valid_candidates.append((k, k_time))
            except IndexError:
                continue

        if not valid_candidates:
            t += dt.timedelta(minutes=interval_min)
            continue

        valid_candidates.sort(key=lambda x: abs((x[1] - t).total_seconds()))
        key = valid_candidates[0][0]
        buf = io.BytesIO()
        s3.download_fileobj(bucket, key, buf)
        buf.seek(0)

        ds = xr.open_dataset(buf, engine="h5netcdf")
        # L1b uses 'Rad', L2-CMIP uses 'CMI' (brightness temp / reflectance)
        var_name = "Rad" if "Rad" in ds else "CMI"
        rad = ds[var_name].values.astype(np.float32)
        # normalize to roughly [0, 1] using robust percentiles, clip outliers
        lo, hi = np.nanpercentile(rad, [1, 99])
        norm = np.clip((rad - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        norm = np.nan_to_num(norm, nan=0.0)

        ts_label = t.strftime("%Y-%m-%dT%H-%M")
        np.save(os.path.join(out_dir, f"{ts_label}.npy"), norm)
        print(f"saved {ts_label}.npy  shape={norm.shape}")

        saved += 1
        t += dt.timedelta(minutes=interval_min)


def convert_local_hdf5(input_dir, out_dir, dataset_key="IMG_TIR1"):
    """Convert a folder of locally-downloaded INSAT-3D/3DR L1B HDF5 files
    (from MOSDAC) into the same normalized .npy format.

    dataset_key depends on the product -- INSAT-3D L1B files typically
    expose IMG_TIR1 / IMG_TIR2 (thermal IR), IMG_VIS, IMG_SWIR, IMG_WV
    as separate HDF5 datasets. Inspect with:
        h5dump -H your_file.h5 | less
    """
    import h5py

    os.makedirs(out_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(input_dir) if f.lower().endswith((".h5", ".hdf")))
    for fname in files:
        path = os.path.join(input_dir, fname)
        with h5py.File(path, "r") as f:
            arr = f[dataset_key][:].astype(np.float32)
        lo, hi = np.percentile(arr, [1, 99])
        norm = np.clip((arr - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        out_name = os.path.splitext(fname)[0] + ".npy"
        np.save(os.path.join(out_dir, out_name), norm)
        print(f"converted {fname} -> {out_name}  shape={norm.shape}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["goes", "insat-local"], required=True)
    p.add_argument("--bucket", default="noaa-goes16")
    p.add_argument("--product", default="ABI-L2-CMIPF")
    p.add_argument("--band", type=int, default=13)
    p.add_argument("--start", default="2025-01-01T00:00")
    p.add_argument("--num-frames", type=int, default=200)
    p.add_argument("--interval-min", type=int, default=10)
    p.add_argument("--local-dir", default="./raw_hdf5")
    p.add_argument("--out-dir", default="./data/frames")
    args = p.parse_args()

    if args.source == "goes":
        start = dt.datetime.strptime(args.start, "%Y-%m-%dT%H:%M")
        fetch_goes_band(args.bucket, args.product, args.band, start,
                         args.num_frames, args.interval_min, args.out_dir)
    else:
        convert_local_hdf5(args.local_dir, args.out_dir)
