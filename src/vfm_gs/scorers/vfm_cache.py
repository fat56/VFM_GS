import json
import re
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image


MANIFEST_NAME = "manifest.json"
SUPPORTED_STORAGE = ("npy_float32", "npy_float16", "npz_uint8")


def normalize_np(value, eps=1e-6):
    value = np.nan_to_num(value.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    value_min = float(value.min()) if value.size else 0.0
    value_max = float(value.max()) if value.size else 0.0
    denom = max(value_max - value_min, eps)
    return (value - value_min) / denom


def compute_edge_map_np(image):
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    luma = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    dx = np.zeros_like(luma, dtype=np.float32)
    dy = np.zeros_like(luma, dtype=np.float32)
    dx[:, 1:] = luma[:, 1:] - luma[:, :-1]
    dy[1:, :] = luma[1:, :] - luma[:-1, :]
    return normalize_np(np.sqrt(dx * dx + dy * dy + 1e-12))


def safe_cache_stem(image_name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", image_name)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_extension(storage):
    if storage in ("npy_float32", "npy_float16"):
        return ".npy"
    if storage == "npz_uint8":
        return ".npz"
    raise ValueError("Unsupported cache storage {!r}. Available: {}".format(storage, ", ".join(SUPPORTED_STORAGE)))


def save_feature(path, feature, storage):
    path = Path(path)
    feature = normalize_np(feature)
    if storage == "npy_float32":
        np.save(path, feature.astype(np.float32))
    elif storage == "npy_float16":
        np.save(path, feature.astype(np.float16))
    elif storage == "npz_uint8":
        quantized = np.clip(np.rint(feature * 255.0), 0, 255).astype(np.uint8)
        np.savez_compressed(path, feature=quantized)
    else:
        raise ValueError("Unsupported cache storage {!r}.".format(storage))


def load_feature(path, storage):
    path = Path(path)
    if storage in ("npy_float32", "npy_float16"):
        value = np.load(path)
    elif storage == "npz_uint8":
        with np.load(path) as archive:
            value = archive["feature"].astype(np.float32) / 255.0
    else:
        raise ValueError("Unsupported cache storage {!r}.".format(storage))
    return np.nan_to_num(value.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def write_manifest(cache_dir, manifest):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    with (cache_dir / MANIFEST_NAME).open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)


def read_manifest(cache_dir):
    manifest_path = Path(cache_dir) / MANIFEST_NAME
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    entries = manifest.get("entries", {})
    if not isinstance(entries, dict):
        raise ValueError("VFM cache manifest entries must be a mapping: {}".format(manifest_path))
    return manifest
