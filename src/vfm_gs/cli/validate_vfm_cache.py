from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from vfm_gs.scorers.vfm_cache import SUPPORTED_STORAGE, load_feature, read_manifest, sha256_file


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def _iter_image_names(images_dir):
    images_dir = Path(images_dir)
    if not images_dir.exists():
        return set()
    return {
        path.stem
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix in IMAGE_SUFFIXES
    }


def validate_cache(cache_dir, backend=None, source_path=None, images=None, check_checksum=True):
    cache_dir = Path(cache_dir)
    manifest = read_manifest(cache_dir)
    errors = []
    warnings = []

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

        try:
            feature = load_feature(cache_path, storage)
        except Exception as exc:
            errors.append("{}: unable to load feature: {}".format(image_name, exc))
            continue

        expected_shape = entry.get("shape")
        if expected_shape and list(feature.shape) != list(expected_shape):
            errors.append("{}: shape mismatch loaded={} manifest={}".format(image_name, list(feature.shape), expected_shape))
        if feature.ndim != 2:
            errors.append("{}: expected 2D feature map, got shape {}".format(image_name, list(feature.shape)))
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


def main(argv=None):
    parser = ArgumentParser(description="Validate a VFM feature cache manifest and entries.")
    parser.add_argument("--cache_dir", "-c", required=True, type=str)
    parser.add_argument("--backend", default=None, type=str)
    parser.add_argument("--source_path", "-s", default=None, type=str)
    parser.add_argument("--images", "-i", default=None, type=str)
    parser.add_argument("--skip_checksum", action="store_true")
    args = parser.parse_args(argv)

    errors, warnings, manifest = validate_cache(
        args.cache_dir,
        backend=args.backend,
        source_path=args.source_path,
        images=args.images,
        check_checksum=not args.skip_checksum,
    )

    for warning in warnings:
        print("[WARN] {}".format(warning))
    if errors:
        for error in errors:
            print("[ERROR] {}".format(error))
        raise SystemExit(1)

    print(
        "Validated {} {} cache entries in {}".format(
            len(manifest.get("entries", {})),
            manifest.get("backend", "<unknown>"),
            Path(args.cache_dir).resolve(),
        )
    )


if __name__ == "__main__":
    main()
