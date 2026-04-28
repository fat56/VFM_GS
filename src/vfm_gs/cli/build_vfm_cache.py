from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from PIL import Image

from vfm_gs.scorers.vfm_cache import (
    SUPPORTED_STORAGE,
    cache_extension,
    compute_edge_map_np,
    safe_cache_stem,
    save_feature,
    sha256_file,
    write_manifest,
)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def _iter_images(images_dir):
    for path in sorted(Path(images_dir).iterdir()):
        if path.is_file() and path.suffix in IMAGE_SUFFIXES:
            yield path


def _resize_for_cache(image, max_width):
    if max_width is None or image.width <= max_width:
        return image
    height = max(1, round(image.height * (max_width / image.width)))
    return image.resize((max_width, height), Image.BILINEAR)


def build_edge_cache(source_path, images, output_dir, backend, max_width=None, storage="npy_float32"):
    source_path = Path(source_path)
    images_dir = source_path / images
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    entries = {}
    image_paths = list(_iter_images(images_dir))
    if not image_paths:
        raise FileNotFoundError("No images found in {}".format(images_dir))

    for image_path in image_paths:
        image_name = image_path.stem
        with Image.open(image_path) as image:
            source_width, source_height = image.size
            image = _resize_for_cache(image.convert("RGB"), max_width)
            edge_map = compute_edge_map_np(image)
            height, width = edge_map.shape

        cache_file = "{}{}".format(safe_cache_stem(image_name), cache_extension(storage))
        cache_path = output_dir / cache_file
        save_feature(cache_path, edge_map, storage)
        entries[image_name] = {
            "cache_file": cache_file,
            "checksum_sha256": sha256_file(cache_path),
            "image_file": image_path.name,
            "source_shape": [source_height, source_width],
            "shape": [height, width],
            "dtype": "float32" if storage != "npz_uint8" else "uint8",
            "normalization": "minmax_0_1",
            "storage": storage,
        }

    manifest = {
        "schema_version": 1,
        "backend": backend,
        "source_path": str(source_path.resolve()),
        "images": images,
        "entry_key": "image_name",
        "feature": "edge_magnitude",
        "max_width": max_width,
        "storage": storage,
        "entries": entries,
    }
    write_manifest(output_dir, manifest)
    return manifest


def main(argv=None):
    parser = ArgumentParser(description="Build cached VFM-style GT features.")
    parser.add_argument("--source_path", "-s", required=True, type=str)
    parser.add_argument("--images", "-i", default="images", type=str)
    parser.add_argument("--output_dir", "-o", required=True, type=str)
    parser.add_argument("--backend", default="cached_edge_l1", choices=["cached_edge_l1"])
    parser.add_argument("--max_width", default=None, type=int)
    parser.add_argument("--storage", default="npy_float32", choices=SUPPORTED_STORAGE)
    args = parser.parse_args(argv)

    manifest = build_edge_cache(args.source_path, args.images, args.output_dir, args.backend, args.max_width, args.storage)
    print(
        "Wrote {} {} cache entries to {}".format(
            len(manifest["entries"]), manifest["backend"], Path(args.output_dir).resolve()
        )
    )


if __name__ == "__main__":
    main()
