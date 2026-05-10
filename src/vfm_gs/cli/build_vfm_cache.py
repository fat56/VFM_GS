from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
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
from vfm_gs.scene.colmap_loader import (
    qvec2rotmat,
    read_extrinsics_binary,
    read_extrinsics_text,
    read_intrinsics_binary,
    read_intrinsics_text,
    read_next_bytes,
)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
COLMAP_DEPTH_BACKENDS = ("colmap_depth_edge_l1",)
DINO_BACKENDS = ("dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14")
DINO_DIMS = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
}
PATCH_SIZE = 14


def _iter_images(images_dir):
    for path in sorted(Path(images_dir).iterdir()):
        if path.is_file() and path.suffix in IMAGE_SUFFIXES:
            yield path


def _resize_for_cache(image, max_width):
    if max_width is None or image.width <= max_width:
        return image
    height = max(1, round(image.height * (max_width / image.width)))
    return image.resize((max_width, height), Image.BILINEAR)


def _read_points3d_xyz_by_id_binary(path):
    points = {}
    with open(path, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            point_data = read_next_bytes(fid, num_bytes=43, format_char_sequence="QdddBBBd")
            point_id = int(point_data[0])
            points[point_id] = np.array(point_data[1:4], dtype=np.float32)
            track_length = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            if track_length:
                read_next_bytes(fid, num_bytes=8 * track_length, format_char_sequence="ii" * track_length)
    return points


def _read_points3d_xyz_by_id_text(path):
    points = {}
    with open(path, "r", encoding="utf-8") as fid:
        for line in fid:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            elems = line.split()
            point_id = int(elems[0])
            points[point_id] = np.array(tuple(map(float, elems[1:4])), dtype=np.float32)
    return points


def _read_colmap_points3d_xyz_by_id(sparse_dir):
    sparse_dir = Path(sparse_dir)
    bin_path = sparse_dir / "points3D.bin"
    txt_path = sparse_dir / "points3D.txt"
    if bin_path.exists():
        return _read_points3d_xyz_by_id_binary(bin_path)
    if txt_path.exists():
        return _read_points3d_xyz_by_id_text(txt_path)
    raise FileNotFoundError("Unable to find COLMAP points3D.bin or points3D.txt in {}".format(sparse_dir))


def _read_colmap_cameras_and_images(sparse_dir):
    sparse_dir = Path(sparse_dir)
    try:
        images = read_extrinsics_binary(sparse_dir / "images.bin")
        cameras = read_intrinsics_binary(sparse_dir / "cameras.bin")
        return cameras, images
    except Exception:
        images = read_extrinsics_text(sparse_dir / "images.txt")
        cameras = read_intrinsics_text(sparse_dir / "cameras.txt")
        return cameras, images


def _dilate_sparse_values(value_map, valid_mask, radius):
    radius = max(0, int(radius))
    if radius <= 0:
        return value_map, valid_mask

    height, width = value_map.shape
    padded_values = np.pad(value_map, radius, mode="edge")
    padded_valid = np.pad(valid_mask.astype(np.float32), radius, mode="constant", constant_values=0.0)
    out_values = value_map.copy()
    out_valid = valid_mask.copy()
    out_dist = np.full((height, width), np.inf, dtype=np.float32)
    out_dist[valid_mask] = 0.0

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            distance = float(dx * dx + dy * dy)
            src_values = padded_values[radius + dy : radius + dy + height, radius + dx : radius + dx + width]
            src_valid = padded_valid[radius + dy : radius + dy + height, radius + dx : radius + dx + width] > 0
            update = src_valid & (distance < out_dist)
            out_values[update] = src_values[update]
            out_valid[update] = True
            out_dist[update] = distance
    return out_values, out_valid


def _box_blur_masked(value_map, valid_mask, radius, min_mask=1e-3):
    radius = max(0, int(radius))
    if radius <= 0:
        return value_map, valid_mask

    height, width = value_map.shape
    padded_value = np.pad(
        value_map * valid_mask.astype(np.float32),
        radius,
        mode="constant",
        constant_values=0.0,
    )
    padded_mask = np.pad(valid_mask.astype(np.float32), radius, mode="constant", constant_values=0.0)
    value_acc = np.zeros_like(value_map, dtype=np.float32)
    mask_acc = np.zeros_like(value_map, dtype=np.float32)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            value_acc += padded_value[dy : dy + height, dx : dx + width]
            mask_acc += padded_mask[dy : dy + height, dx : dx + width]
    blurred = np.zeros_like(value_map, dtype=np.float32)
    valid = mask_acc > float(min_mask)
    blurred[valid] = value_acc[valid] / np.maximum(mask_acc[valid], 1e-6)
    return blurred, valid


def _depth_edge_from_sparse_points(
    image_entry,
    camera_entry,
    points_by_id,
    source_size,
    cache_size,
    splat_radius,
    blur_radius,
):
    source_width, source_height = source_size
    cache_width, cache_height = cache_size
    scale_x = float(cache_width) / max(float(source_width), 1.0)
    scale_y = float(cache_height) / max(float(source_height), 1.0)

    depth = np.full((cache_height, cache_width), np.inf, dtype=np.float32)
    valid_ids = image_entry.point3D_ids >= 0
    if not np.any(valid_ids):
        return np.zeros((cache_height, cache_width), dtype=np.float32), 0

    rotation = qvec2rotmat(image_entry.qvec).astype(np.float32)
    translation = np.asarray(image_entry.tvec, dtype=np.float32)
    observed = 0
    for xy, point_id in zip(image_entry.xys[valid_ids], image_entry.point3D_ids[valid_ids]):
        point = points_by_id.get(int(point_id))
        if point is None:
            continue
        cam_point = rotation @ point + translation
        z = float(cam_point[2])
        if z <= 1e-6:
            continue
        x = int(round(float(xy[0]) * scale_x))
        y = int(round(float(xy[1]) * scale_y))
        if x < 0 or x >= cache_width or y < 0 or y >= cache_height:
            continue
        if z < depth[y, x]:
            depth[y, x] = z
        observed += 1

    valid_mask = np.isfinite(depth)
    if not np.any(valid_mask):
        return np.zeros((cache_height, cache_width), dtype=np.float32), observed

    inv_depth = np.zeros_like(depth, dtype=np.float32)
    inv_depth[valid_mask] = 1.0 / np.maximum(depth[valid_mask], 1e-6)
    valid_values = inv_depth[valid_mask]
    min_value = float(valid_values.min())
    max_value = float(valid_values.max())
    inv_depth[valid_mask] = (valid_values - min_value) / max(max_value - min_value, 1e-6)

    inv_depth, valid_mask = _dilate_sparse_values(inv_depth, valid_mask, splat_radius)
    inv_depth, valid_mask = _box_blur_masked(inv_depth, valid_mask, blur_radius)

    dx = np.zeros_like(inv_depth, dtype=np.float32)
    dy = np.zeros_like(inv_depth, dtype=np.float32)
    valid_x = valid_mask[:, 1:] & valid_mask[:, :-1]
    valid_y = valid_mask[1:, :] & valid_mask[:-1, :]
    dx[:, 1:][valid_x] = inv_depth[:, 1:][valid_x] - inv_depth[:, :-1][valid_x]
    dy[1:, :][valid_y] = inv_depth[1:, :][valid_y] - inv_depth[:-1, :][valid_y]
    depth_edge = np.sqrt(dx * dx + dy * dy + 1e-12)
    return depth_edge, observed


def _resize_to_patch_grid(image, max_width, patch_size=PATCH_SIZE):
    image = _resize_for_cache(image, max_width)
    width = max(patch_size, (image.width // patch_size) * patch_size)
    height = max(patch_size, (image.height // patch_size) * patch_size)
    if (width, height) != image.size:
        image = image.resize((width, height), Image.BILINEAR)
    return image


def _patch_torch112_scaled_dot_product_attention():
    try:
        import torch.nn.functional as torch_f
    except Exception:
        return

    if hasattr(torch_f, "scaled_dot_product_attention") or not hasattr(torch_f, "_scaled_dot_product_attention"):
        return

    def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False):
        if is_causal:
            raise RuntimeError("is_causal=True is not supported by the PyTorch 1.12 compatibility shim.")
        if query.dim() == 4:
            batch, heads, tokens, dim = query.shape
            query_3d = query.reshape(batch * heads, tokens, dim)
            key_3d = key.reshape(batch * heads, key.shape[-2], key.shape[-1])
            value_3d = value.reshape(batch * heads, value.shape[-2], value.shape[-1])
            output, _ = torch_f._scaled_dot_product_attention(query_3d, key_3d, value_3d, attn_mask, dropout_p)
            return output.reshape(batch, heads, tokens, dim)
        output, _ = torch_f._scaled_dot_product_attention(query, key, value, attn_mask, dropout_p)
        return output

    torch_f.scaled_dot_product_attention = scaled_dot_product_attention


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


def build_colmap_depth_edge_cache(
    source_path,
    images,
    output_dir,
    backend,
    max_width=None,
    storage="npy_float32",
    splat_radius=2,
    blur_radius=2,
):
    source_path = Path(source_path)
    images_dir = source_path / images
    sparse_dir = source_path / "sparse" / "0"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list(_iter_images(images_dir))
    if not image_paths:
        raise FileNotFoundError("No images found in {}".format(images_dir))

    cameras, colmap_images = _read_colmap_cameras_and_images(sparse_dir)
    points_by_id = _read_colmap_points3d_xyz_by_id(sparse_dir)
    colmap_by_name = {Path(entry.name).stem: entry for entry in colmap_images.values()}

    entries = {}
    for image_path in image_paths:
        image_name = image_path.stem
        if image_name not in colmap_by_name:
            raise KeyError("Image {!r} is missing from COLMAP sparse images in {}".format(image_name, sparse_dir))
        image_entry = colmap_by_name[image_name]
        camera_entry = cameras[image_entry.camera_id]
        with Image.open(image_path) as image:
            source_width, source_height = image.size
            resized = _resize_for_cache(image.convert("RGB"), max_width)
            cache_width, cache_height = resized.size

            depth_edge, observed_points = _depth_edge_from_sparse_points(
            image_entry,
            camera_entry,
            points_by_id,
            (int(camera_entry.width), int(camera_entry.height)),
            (cache_width, cache_height),
            splat_radius=splat_radius,
            blur_radius=blur_radius,
        )
        depth_edge = np.nan_to_num(depth_edge.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

        cache_file = "{}{}".format(safe_cache_stem(image_name), cache_extension(storage))
        cache_path = output_dir / cache_file
        save_feature(cache_path, depth_edge, storage, normalize=True)
        entries[image_name] = {
            "cache_file": cache_file,
            "checksum_sha256": sha256_file(cache_path),
            "image_file": image_path.name,
            "source_shape": [source_height, source_width],
            "shape": [cache_height, cache_width],
            "dtype": "float32" if storage != "npz_uint8" else "uint8",
            "normalization": "minmax_0_1",
            "storage": storage,
            "observed_sparse_points": int(observed_points),
        }

    manifest = {
        "schema_version": 1,
        "backend": backend,
        "source_path": str(source_path.resolve()),
        "images": images,
        "entry_key": "image_name",
        "feature": "colmap_depth_edge",
        "max_width": max_width,
        "storage": storage,
        "splat_radius": int(splat_radius),
        "blur_radius": int(blur_radius),
        "entries": entries,
    }
    write_manifest(output_dir, manifest)
    return manifest


def _image_to_dino_tensor(image, device):
    import torch

    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype).view(1, 3, 1, 1)
    return ((tensor - mean) / std).to(device=device)


def _normalize_torch01(value, eps=1e-6):
    import torch

    value = torch.nan_to_num(value.to(torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
    value_min = torch.min(value)
    value_max = torch.max(value)
    return (value - value_min) / torch.clamp(value_max - value_min, min=eps)


def _dinov2_token_edge_map_torch(token_map):
    import torch
    import torch.nn.functional as torch_f

    tokens = torch_f.normalize(token_map.to(torch.float32), dim=-1)
    dx = torch.zeros(tokens.shape[:2], dtype=tokens.dtype, device=tokens.device)
    dy = torch.zeros_like(dx)
    if tokens.shape[1] > 1:
        dx[:, 1:] = 1.0 - torch_f.cosine_similarity(tokens[:, 1:, :], tokens[:, :-1, :], dim=-1)
    if tokens.shape[0] > 1:
        dy[1:, :] = 1.0 - torch_f.cosine_similarity(tokens[1:, :, :], tokens[:-1, :, :], dim=-1)
    return _normalize_torch01(torch.sqrt(dx.square() + dy.square() + 1e-12))


def _load_dinov2_model(backend, dinov2_repo, device, pretrained=True):
    import torch

    _patch_torch112_scaled_dot_product_attention()
    try:
        if dinov2_repo:
            model = torch.hub.load(dinov2_repo, backend, source="local", pretrained=pretrained)
        else:
            model = torch.hub.load("facebookresearch/dinov2", backend, pretrained=pretrained, trust_repo=True)
    except Exception as exc:
        raise RuntimeError(
            "Unable to load DINOv2 backend {!r}. If torch.hub remote access is unavailable, "
            "clone https://github.com/facebookresearch/dinov2.git and pass --dinov2_repo.".format(backend)
        ) from exc
    return model.eval().to(device)


def build_dinov2_cache(
    source_path,
    images,
    output_dir,
    backend,
    max_width=None,
    storage="npy_float16",
    dinov2_repo=None,
    device="cuda",
    limit=None,
    pretrained=True,
    project_token_edge=False,
):
    import torch
    import torch.nn.functional as torch_f

    source_path = Path(source_path)
    images_dir = source_path / images
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list(_iter_images(images_dir))
    if limit is not None:
        image_paths = image_paths[:limit]
    if not image_paths:
        raise FileNotFoundError("No images found in {}".format(images_dir))

    model = _load_dinov2_model(backend, dinov2_repo, device, pretrained=pretrained)
    entries = {}

    for image_path in image_paths:
        image_name = image_path.stem
        with Image.open(image_path) as image:
            source_width, source_height = image.size
            image = _resize_to_patch_grid(image.convert("RGB"), max_width)
            width, height = image.size
            tensor = _image_to_dino_tensor(image, device)

        with torch.no_grad():
            features = model.forward_features(tensor)["x_norm_patchtokens"][0]
            features = torch_f.normalize(features, dim=-1)

        grid_h = height // PATCH_SIZE
        grid_w = width // PATCH_SIZE
        feature_map = features.reshape(grid_h, grid_w, DINO_DIMS[backend])
        if project_token_edge:
            feature_map = _dinov2_token_edge_map_torch(feature_map).detach().cpu().numpy()
            feature = "dinov2_token_edge"
            shape = [grid_h, grid_w]
            dtype = "uint8" if storage == "npz_uint8" else ("float16" if storage == "npy_float16" else "float32")
            normalization = "minmax_0_1"
        else:
            feature_map = feature_map.detach().cpu().numpy()
            feature = "dinov2_patchtokens"
            shape = [grid_h, grid_w, DINO_DIMS[backend]]
            dtype = "float16" if storage == "npy_float16" else "float32"
            normalization = "l2_channel"

        cache_file = "{}{}".format(safe_cache_stem(image_name), cache_extension(storage))
        cache_path = output_dir / cache_file
        save_feature(cache_path, feature_map, storage, normalize=False)
        entries[image_name] = {
            "cache_file": cache_file,
            "checksum_sha256": sha256_file(cache_path),
            "image_file": image_path.name,
            "source_shape": [source_height, source_width],
            "shape": shape,
            "dtype": dtype,
            "normalization": normalization,
            "storage": storage,
        }

    manifest = {
        "schema_version": 1,
        "backend": backend,
        "source_path": str(source_path.resolve()),
        "images": images,
        "entry_key": "image_name",
        "feature": feature,
        "project_token_edge": bool(project_token_edge),
        "patch_size": PATCH_SIZE,
        "max_width": max_width,
        "storage": storage,
        "pretrained": pretrained,
        "entries": entries,
    }
    write_manifest(output_dir, manifest)
    return manifest


def main(argv=None):
    parser = ArgumentParser(description="Build cached VFM-style GT features.")
    parser.add_argument("--source_path", "-s", required=True, type=str)
    parser.add_argument("--images", "-i", default="images", type=str)
    parser.add_argument("--output_dir", "-o", required=True, type=str)
    parser.add_argument("--backend", default="cached_edge_l1", choices=["cached_edge_l1", *COLMAP_DEPTH_BACKENDS, *DINO_BACKENDS])
    parser.add_argument("--max_width", default=None, type=int)
    parser.add_argument(
        "--storage",
        default=None,
        choices=SUPPORTED_STORAGE,
        help="Cache storage. Defaults to npy_float32 for cached_edge_l1 and npy_float16 for DINOv2.",
    )
    parser.add_argument("--dinov2_repo", default=None, type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--limit", default=None, type=int)
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--depth_splat_radius", default=2, type=int)
    parser.add_argument("--depth_blur_radius", default=2, type=int)
    parser.add_argument(
        "--project_token_edge",
        action="store_true",
        help="For DINOv2 backends, store the derived 2D token-edge map instead of full patch tokens.",
    )
    args = parser.parse_args(argv)

    if args.backend == "cached_edge_l1":
        storage = args.storage or "npy_float32"
        manifest = build_edge_cache(args.source_path, args.images, args.output_dir, args.backend, args.max_width, storage)
    elif args.backend in COLMAP_DEPTH_BACKENDS:
        storage = args.storage or "npy_float32"
        manifest = build_colmap_depth_edge_cache(
            args.source_path,
            args.images,
            args.output_dir,
            args.backend,
            args.max_width,
            storage,
            splat_radius=args.depth_splat_radius,
            blur_radius=args.depth_blur_radius,
        )
    else:
        storage = args.storage or "npy_float16"
        if storage == "npz_uint8" and not args.project_token_edge:
            raise ValueError("DINOv2 patch-token caches require floating-point storage.")
        manifest = build_dinov2_cache(
            args.source_path,
            args.images,
            args.output_dir,
            args.backend,
            max_width=args.max_width,
            storage=storage,
            dinov2_repo=args.dinov2_repo,
            device=args.device,
            limit=args.limit,
            pretrained=not args.no_pretrained,
            project_token_edge=args.project_token_edge,
        )
    print(
        "Wrote {} {} cache entries to {}".format(
            len(manifest["entries"]), manifest["backend"], Path(args.output_dir).resolve()
        )
    )


if __name__ == "__main__":
    main()
