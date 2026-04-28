import json
import re
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image


MANIFEST_NAME = "manifest.json"
SUPPORTED_STORAGE = ("npy_float32", "npy_float16", "npz_uint8")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


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


def save_feature(path, feature, storage, normalize=True):
    path = Path(path)
    if normalize:
        feature = normalize_np(feature)
    else:
        feature = np.nan_to_num(feature.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
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


def _iter_image_names(images_dir):
    images_dir = Path(images_dir)
    if not images_dir.exists():
        return set()
    return {
        path.stem
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix in IMAGE_SUFFIXES
    }


def validate_cache(cache_dir, backend=None, source_path=None, images=None, check_checksum=True, load_entries=True):
    cache_dir = Path(cache_dir)
    errors = []
    warnings = []
    try:
        manifest = read_manifest(cache_dir)
    except Exception as exc:
        return ["unable to read manifest in {}: {}".format(cache_dir, exc)], warnings, {"entries": {}}

    manifest_backend = manifest.get("backend")
    if backend and manifest_backend != backend:
        errors.append("backend mismatch: manifest={!r}, expected={!r}".format(manifest_backend, backend))

    manifest_storage = manifest.get("storage", "npy_float32")
    if manifest_storage not in SUPPORTED_STORAGE:
        errors.append("unsupported manifest storage: {!r}".format(manifest_storage))

    entries = manifest.get("entries", {})
    if not entries:
        errors.append("manifest has no entries")

    for image_name, entry in sorted(entries.items()):
        cache_file = entry.get("cache_file")
        if not cache_file:
            errors.append("{}: missing cache_file".format(image_name))
            continue

        storage = entry.get("storage", manifest_storage)
        if storage not in SUPPORTED_STORAGE:
            errors.append("{}: unsupported storage {!r}".format(image_name, storage))
            continue

        cache_path = cache_dir / cache_file
        if not cache_path.exists():
            errors.append("{}: missing cache file {}".format(image_name, cache_path))
            continue

        if check_checksum:
            expected_checksum = entry.get("checksum_sha256")
            if expected_checksum and sha256_file(cache_path) != expected_checksum:
                errors.append("{}: checksum mismatch".format(image_name))

        if load_entries:
            try:
                feature = load_feature(cache_path, storage)
            except Exception as exc:
                errors.append("{}: unable to load feature: {}".format(image_name, exc))
                continue

            expected_shape = entry.get("shape")
            if expected_shape and list(feature.shape) != list(expected_shape):
                errors.append("{}: shape mismatch loaded={} manifest={}".format(image_name, list(feature.shape), expected_shape))
            if feature.ndim not in (2, 3):
                errors.append("{}: expected 2D/3D feature map, got shape {}".format(image_name, list(feature.shape)))
            if feature.size == 0:
                errors.append("{}: empty feature map".format(image_name))

    if source_path and images:
        image_names = _iter_image_names(Path(source_path) / images)
        manifest_names = set(entries)
        missing = sorted(image_names - manifest_names)
        extra = sorted(manifest_names - image_names)
        if missing:
            errors.append("{} source images are missing from manifest, e.g. {}".format(len(missing), missing[:5]))
        if extra:
            warnings.append("{} manifest entries have no matching source image, e.g. {}".format(len(extra), extra[:5]))

    return errors, warnings, manifest
