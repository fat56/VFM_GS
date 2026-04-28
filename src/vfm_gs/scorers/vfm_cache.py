import json
import re
from pathlib import Path

import numpy as np
from PIL import Image


MANIFEST_NAME = "manifest.json"


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
