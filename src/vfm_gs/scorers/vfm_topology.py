import math
import time

import torch
import torch.nn.functional as F

from vfm_gs.gaussian_renderer import render_fastgs
from vfm_gs.scorers.vfm_cache import load_feature, read_manifest, validate_cache
from vfm_gs.utils.fast_utils import compute_gaussian_score_fastgs, normalize01

from .registry import register_scorer


_CACHE_READERS = {}
_DINO_MODELS = {}
_DINO_CACHE_BACKENDS = ("dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14")
_DEPTH_ANYTHING_CACHE_BACKENDS = ("depth_anything", "depth_anything_v2")
_CACHED_BACKEND_MANIFESTS = {
    "cached_edge_l1": ("cached_edge_l1",),
    "colmap_depth_edge_l1": ("colmap_depth_edge_l1",),
    "colmap_depth_edge_prior": ("colmap_depth_edge_l1",),
    "depth_anything_depth_prior": _DEPTH_ANYTHING_CACHE_BACKENDS,
    "depth_anything_depth_edge_prior": _DEPTH_ANYTHING_CACHE_BACKENDS,
    "depth_anything_residual_orientation": _DEPTH_ANYTHING_CACHE_BACKENDS,
    "dinov2_token_edge_l1": _DINO_CACHE_BACKENDS,
    "dinov2_descriptor_cosine": _DINO_CACHE_BACKENDS,
    "dinov2_descriptor_cosine_l1": _DINO_CACHE_BACKENDS,
}
_DINO_PATCH_SIZE = 14
_PROFILE_STATE = {"calls": 0}


def _profile_enabled(args):
    return bool(getattr(args, "vfm_profile_scorer", False))


def _profile_interval(args):
    return max(1, int(getattr(args, "vfm_profile_interval", 1) or 1))


def _sync_if_needed(enabled):
    if enabled and torch.cuda.is_available():
        torch.cuda.synchronize()


def _elapsed_ms(start, enabled):
    if not enabled:
        return 0.0
    _sync_if_needed(True)
    return (time.perf_counter() - start) * 1000.0


def _is_vfm_active(args):
    iteration = int(getattr(args, "current_iteration", 0) or 0)
    if iteration <= 0:
        return True
    active_from = int(getattr(args, "vfm_active_from_iter", 0) or 0)
    active_until = int(getattr(args, "vfm_active_until_iter", 0) or 0)
    if active_from > 0 and iteration < active_from:
        return False
    if active_until > 0 and iteration > active_until:
        return False
    return True


def _vfm_densify_override_active(args):
    if not bool(getattr(args, "vfm_densify_override_enabled", False)):
        return False
    iteration = int(getattr(args, "current_iteration", 0) or 0)
    if iteration <= 0:
        return False
    from_iter = int(getattr(args, "vfm_densify_override_from_iter", 0) or 0)
    until_iter = int(getattr(args, "vfm_densify_override_until_iter", 0) or 0)
    if from_iter > 0 and iteration <= from_iter:
        return False
    if until_iter > 0 and iteration >= until_iter:
        return False
    return True


def _densify_prune_enabled(args):
    if _vfm_densify_override_active(args):
        return bool(getattr(args, "vfm_densify_override_prune_enabled", True))
    return bool(getattr(args, "densify_prune_enabled", True))


def _clear_rgb_rerank_reference(args):
    if hasattr(args, "vfm_rgb_broad_reference_score"):
        args.vfm_rgb_broad_reference_score = None


class VFMFeatureCache:
    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        self.manifest = read_manifest(cache_dir)
        self.entries = self.manifest["entries"]
        self.features = {}
        self.derived_features = {}

    def get_feature_map(self, image_name, device):
        if image_name not in self.entries:
            raise KeyError("Image {!r} is missing from VFM cache {}".format(image_name, self.cache_dir))
        if image_name not in self.features:
            entry = self.entries[image_name]
            storage = entry.get("storage", self.manifest.get("storage", "npy_float32"))
            path = "{}/{}".format(self.cache_dir.rstrip("/"), entry["cache_file"])
            self.features[image_name] = torch.from_numpy(load_feature(path, storage)).to(torch.float32)
        return self.features[image_name].to(device=device)

    def get_edge_map(self, image_name, device, size):
        edge_map = self.get_feature_map(image_name, device)
        if edge_map.ndim != 2:
            raise ValueError("Expected 2D edge cache for {!r}, got shape {}".format(image_name, list(edge_map.shape)))

        if tuple(edge_map.shape[-2:]) != tuple(size):
            edge_map = F.interpolate(
                edge_map.view(1, 1, *edge_map.shape[-2:]),
                size=size,
                mode="bilinear",
                align_corners=False,
            ).view(*size)
        return edge_map

    def get_dinov2_token_edge_map(self, image_name, device):
        key = ("dinov2_token_edge", image_name)
        if key not in self.derived_features:
            token_map = self.get_feature_map(image_name, "cpu")
            feature = self.manifest.get("feature")
            if feature == "dinov2_token_edge":
                if token_map.ndim != 2:
                    raise ValueError(
                        "Expected cached DINO token-edge map with shape [grid_h, grid_w], got {}".format(
                            list(token_map.shape)
                        )
                    )
                self.derived_features[key] = normalize01(token_map.to(torch.float32))
            else:
                self.derived_features[key] = _dinov2_token_edge_map(token_map)
        return self.derived_features[key].to(device=device)


def _get_cache(cache_dir):
    if not cache_dir:
        raise ValueError("vfm_cache_dir is required for cached VFM backends.")
    if cache_dir not in _CACHE_READERS:
        _CACHE_READERS[cache_dir] = VFMFeatureCache(cache_dir)
    return _CACHE_READERS[cache_dir]


def _render_with_sh0(viewpoint_cam, gaussians, pipe, bg, mult):
    active_sh_degree = gaussians.active_sh_degree
    try:
        gaussians.active_sh_degree = 0
        return render_fastgs(viewpoint_cam, gaussians, pipe, bg, mult)["render"]
    finally:
        gaussians.active_sh_degree = active_sh_degree


def _luma(image):
    weights = torch.tensor([0.299, 0.587, 0.114], dtype=image.dtype, device=image.device).view(3, 1, 1)
    return torch.sum(image[:3] * weights, dim=0, keepdim=True)


def _gradient_magnitude(luma_image):
    dx = torch.zeros_like(luma_image)
    dy = torch.zeros_like(luma_image)
    dx[:, :, 1:] = luma_image[:, :, 1:] - luma_image[:, :, :-1]
    dy[:, 1:, :] = luma_image[:, 1:, :] - luma_image[:, :-1, :]
    return torch.sqrt(dx.square() + dy.square() + 1e-12).squeeze(0)


def _dinov2_token_edge_map(token_map):
    if token_map.ndim != 3:
        raise ValueError(
            "Expected DINOv2 token map with shape [grid_h, grid_w, dim], got {}".format(list(token_map.shape))
        )
    tokens = F.normalize(token_map.to(torch.float32), dim=-1)
    dx = torch.zeros(tokens.shape[:2], dtype=tokens.dtype, device=tokens.device)
    dy = torch.zeros_like(dx)
    if tokens.shape[1] > 1:
        dx[:, 1:] = 1.0 - F.cosine_similarity(tokens[:, 1:, :], tokens[:, :-1, :], dim=-1)
    if tokens.shape[0] > 1:
        dy[1:, :] = 1.0 - F.cosine_similarity(tokens[1:, :, :], tokens[:-1, :, :], dim=-1)
    return normalize01(torch.sqrt(dx.square() + dy.square() + 1e-12))


def _get_dinov2_model(backend, repo, device):
    key = (backend, repo or "", device)
    if key not in _DINO_MODELS:
        from vfm_gs.cli.build_vfm_cache import _load_dinov2_model

        _DINO_MODELS[key] = _load_dinov2_model(backend, repo or None, device, pretrained=True)
    return _DINO_MODELS[key]


def _rendered_image_to_dino_tensor(rendered_image, grid_size):
    target_h = int(grid_size[0]) * _DINO_PATCH_SIZE
    target_w = int(grid_size[1]) * _DINO_PATCH_SIZE
    image = rendered_image[:3].detach().clamp(0.0, 1.0).unsqueeze(0)
    if tuple(image.shape[-2:]) != (target_h, target_w):
        image = F.interpolate(image, size=(target_h, target_w), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=image.dtype, device=image.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=image.dtype, device=image.device).view(1, 3, 1, 1)
    return (image - mean) / std


def _extract_rendered_dinov2_tokens(rendered_image, grid_size, backend, repo, device):
    model = _get_dinov2_model(backend, repo, device)
    tensor = _rendered_image_to_dino_tensor(rendered_image.to(device=device), grid_size)
    with torch.no_grad():
        tokens = model.forward_features(tensor)["x_norm_patchtokens"][0]
        tokens = F.normalize(tokens.to(torch.float32), dim=-1)
    return tokens.reshape(int(grid_size[0]), int(grid_size[1]), -1)


def _pool_to_patch_grid(value_map, grid_size):
    return F.adaptive_avg_pool2d(value_map.reshape(1, 1, *value_map.shape[-2:]), output_size=grid_size).view(*grid_size)


def _smooth_2d_map(value_map, kernel_size, arg_name):
    kernel_size = int(kernel_size)
    if kernel_size <= 1:
        return value_map
    if kernel_size % 2 == 0:
        raise ValueError("{} must be odd when it is greater than 1, got {}".format(arg_name, kernel_size))
    value = value_map.reshape(1, 1, *value_map.shape[-2:])
    value = F.pad(value, (kernel_size // 2,) * 4, mode="replicate")
    return F.avg_pool2d(value, kernel_size=kernel_size, stride=1).view(*value_map.shape[-2:])


def _mock_l1_error(rendered_image, gt_image):
    return torch.mean(torch.abs(rendered_image[:3] - gt_image[:3]), dim=0)


def _mock_edge_l1_error(rendered_image, gt_image):
    rgb_error = _mock_l1_error(rendered_image, gt_image)
    rendered_edges = _gradient_magnitude(_luma(rendered_image))
    gt_edges = _gradient_magnitude(_luma(gt_image))
    edge_error = torch.abs(rendered_edges - gt_edges)
    return 0.5 * rgb_error + 0.5 * edge_error


def _cached_edge_l1_error(rendered_image, viewpoint_cam, cache_dir):
    rendered_edges = normalize01(_gradient_magnitude(_luma(rendered_image)))
    gt_edges = _get_cache(cache_dir).get_edge_map(
        viewpoint_cam.image_name,
        rendered_image.device,
        rendered_edges.shape,
    )
    return torch.abs(rendered_edges - gt_edges)


def _colmap_depth_edge_l1_error(rendered_image, viewpoint_cam, cache_dir):
    rendered_edges = normalize01(_gradient_magnitude(_luma(rendered_image)))
    gt_depth_edges = _get_cache(cache_dir).get_edge_map(
        viewpoint_cam.image_name,
        rendered_image.device,
        rendered_edges.shape,
    )
    return torch.abs(rendered_edges - gt_depth_edges)


def _colmap_depth_edge_prior_error(rendered_image, viewpoint_cam, cache_dir):
    return _get_cache(cache_dir).get_edge_map(
        viewpoint_cam.image_name,
        rendered_image.device,
        rendered_image.shape[-2:],
    )


def _depth_anything_prior_error(rendered_image, viewpoint_cam, cache_dir):
    return _get_cache(cache_dir).get_edge_map(
        viewpoint_cam.image_name,
        rendered_image.device,
        rendered_image.shape[-2:],
    )


def _normalize_valid(value_map, valid_mask, eps=1e-6):
    out = torch.zeros_like(value_map, dtype=torch.float32)
    valid_values = value_map[valid_mask].to(torch.float32)
    if valid_values.numel() == 0:
        return out
    value_min = torch.min(valid_values)
    value_max = torch.max(valid_values)
    out[valid_mask] = (value_map[valid_mask].to(torch.float32) - value_min) / torch.clamp(
        value_max - value_min,
        min=eps,
    )
    return torch.clamp(out, 0.0, 1.0)


def _topk_mask_valid(value_map, topk_fraction, valid_mask=None):
    flat = value_map.detach().to(torch.float32).reshape(-1)
    if valid_mask is None:
        valid_flat = torch.ones_like(flat, dtype=torch.bool)
    else:
        valid_flat = valid_mask.reshape(-1)
    valid_indices = torch.nonzero(valid_flat, as_tuple=False).reshape(-1)
    mask = torch.zeros_like(flat, dtype=torch.bool)
    if valid_indices.numel() == 0:
        return mask.view_as(value_map)

    topk_fraction = min(max(float(topk_fraction), 0.0), 1.0)
    k = int(math.ceil(valid_indices.numel() * topk_fraction))
    if k <= 0:
        return mask.view_as(value_map)
    k = min(k, valid_indices.numel())
    valid_values = torch.nan_to_num(flat[valid_indices], nan=0.0, posinf=0.0, neginf=0.0)
    top_indices = torch.topk(valid_values, k=k, largest=True, sorted=False).indices
    mask[valid_indices[top_indices]] = True
    return mask.view_as(value_map)


def _mask_iou(left_mask, right_mask):
    union = torch.count_nonzero(left_mask | right_mask).item()
    if union == 0:
        return 0.0
    intersection = torch.count_nonzero(left_mask & right_mask).item()
    return float(intersection) / float(union)


def _proxy_center_zbuffer_depth(gaussians, viewpoint_cam, chunk_size, splat_radius):
    xyz = gaussians.get_xyz.detach()
    device = xyz.device
    height = int(viewpoint_cam.image_height)
    width = int(viewpoint_cam.image_width)
    depth_flat = torch.full((height * width,), float("inf"), dtype=torch.float32, device=device)
    chunk_size = max(1, int(chunk_size or 1))
    radius = max(0, int(splat_radius or 0))

    for start in range(0, int(xyz.shape[0]), chunk_size):
        points = xyz[start : start + chunk_size]
        ones = torch.ones((points.shape[0], 1), dtype=points.dtype, device=device)
        points_h = torch.cat([points, ones], dim=1)
        view_points = torch.matmul(points_h, viewpoint_cam.world_view_transform)
        clip = torch.matmul(points_h, viewpoint_cam.full_proj_transform)
        clip_w = clip[:, 3]
        ndc = clip[:, :3] / torch.clamp(clip_w[:, None], min=1e-7)
        depth = view_points[:, 2].to(torch.float32)
        valid = (
            torch.isfinite(depth)
            & torch.isfinite(ndc).all(dim=1)
            & (clip_w > 0.0)
            & (depth > float(getattr(viewpoint_cam, "znear", 0.01)))
            & (ndc[:, 0] >= -1.0)
            & (ndc[:, 0] <= 1.0)
            & (ndc[:, 1] >= -1.0)
            & (ndc[:, 1] <= 1.0)
        )
        if not torch.any(valid):
            continue

        x = torch.clamp(((ndc[valid, 0] + 1.0) * 0.5 * width).to(torch.long), 0, width - 1)
        y = torch.clamp(((1.0 - ndc[valid, 1]) * 0.5 * height).to(torch.long), 0, height - 1)
        valid_depth = depth[valid]
        for dy in range(-radius, radius + 1):
            yy = y + dy
            in_y = (yy >= 0) & (yy < height)
            if not torch.any(in_y):
                continue
            for dx in range(-radius, radius + 1):
                xx = x + dx
                in_bounds = in_y & (xx >= 0) & (xx < width)
                if not torch.any(in_bounds):
                    continue
                pixel_indices = yy[in_bounds] * width + xx[in_bounds]
                depth_flat.scatter_reduce_(
                    0,
                    pixel_indices,
                    valid_depth[in_bounds],
                    reduce="amin",
                    include_self=True,
                )

    valid_flat = torch.isfinite(depth_flat)
    return depth_flat.view(height, width), valid_flat.view(height, width)


def _depth_anything_residual_orientation_error(rendered_image, viewpoint_cam, cache_dir, gaussians, args):
    if gaussians is None:
        raise ValueError("depth_anything_residual_orientation requires gaussians for the center-depth proxy.")

    prior = normalize01(_depth_anything_prior_error(rendered_image, viewpoint_cam, cache_dir))
    depth, valid = _proxy_center_zbuffer_depth(
        gaussians,
        viewpoint_cam,
        getattr(args, "vfm_residual_proxy_chunk_size", 1_000_000),
        getattr(args, "vfm_residual_proxy_splat_radius", 1),
    )
    min_coverage = max(0.0, float(getattr(args, "vfm_residual_proxy_min_coverage", 0.05) or 0.0))
    valid_coverage = float(valid.to(torch.float32).mean().item()) if valid.numel() else 0.0
    if valid_coverage < min_coverage or torch.count_nonzero(valid).item() == 0:
        return prior

    depth_norm = _normalize_valid(depth, valid)
    inv_depth_norm = _normalize_valid(1.0 / torch.clamp(depth, min=1e-6), valid)
    residual_depth = torch.where(valid, torch.abs(depth_norm - prior), torch.zeros_like(prior))
    residual_inv = torch.where(valid, torch.abs(inv_depth_norm - prior), torch.zeros_like(prior))

    selector = str(getattr(args, "vfm_residual_orientation_selector", "edge_iou") or "edge_iou").lower()
    signals = {
        "prior": prior * valid.to(torch.float32),
        "residual_depth": residual_depth,
        "residual_inv": residual_inv,
    }
    if selector in signals:
        return signals[selector]
    if selector != "edge_iou":
        raise ValueError(
            "Unsupported vfm_residual_orientation_selector {!r}. Available: edge_iou, prior, residual_depth, residual_inv.".format(
                selector
            )
        )

    topk = float(
        getattr(
            args,
            "vfm_residual_orientation_topk",
            getattr(args, "vfm_metric_topk", 0.10),
        )
        or 0.0
    )
    gt_edges = normalize01(_gradient_magnitude(_luma(viewpoint_cam.original_image.to(rendered_image.device))))
    edge_mask = _topk_mask_valid(gt_edges, topk, valid)
    best_signal = "prior"
    best_score = -1.0
    for signal_name in ("prior", "residual_depth", "residual_inv"):
        signal_mask = _topk_mask_valid(signals[signal_name], topk, valid)
        score = _mask_iou(signal_mask, edge_mask)
        if score > best_score:
            best_signal = signal_name
            best_score = score
    return signals[best_signal]


def _dinov2_token_edge_l1_error(rendered_image, viewpoint_cam, cache_dir):
    gt_token_edges = _get_cache(cache_dir).get_dinov2_token_edge_map(viewpoint_cam.image_name, rendered_image.device)
    rendered_edges = normalize01(_gradient_magnitude(_luma(rendered_image)))
    rendered_patch_edges = normalize01(_pool_to_patch_grid(rendered_edges, gt_token_edges.shape[-2:]))
    patch_error = torch.abs(rendered_patch_edges - gt_token_edges)
    return F.interpolate(
        patch_error.view(1, 1, *patch_error.shape[-2:]),
        size=rendered_image.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).view(*rendered_image.shape[-2:])


def _dinov2_descriptor_cosine_error(
    rendered_image,
    viewpoint_cam,
    cache_dir,
    dinov2_repo,
    dinov2_device,
    descriptor_token_smooth_kernel=1,
):
    cache = _get_cache(cache_dir)
    gt_tokens = cache.get_feature_map(viewpoint_cam.image_name, rendered_image.device)
    if gt_tokens.ndim != 3:
        raise ValueError(
            "Expected DINOv2 descriptor cache with shape [grid_h, grid_w, dim], got {}".format(
                list(gt_tokens.shape)
            )
        )
    dinov2_backend = cache.manifest.get("backend")
    if dinov2_backend not in _DINO_CACHE_BACKENDS:
        raise ValueError("Expected DINOv2 cache backend, got {!r}".format(dinov2_backend))
    rendered_tokens = _extract_rendered_dinov2_tokens(
        rendered_image,
        gt_tokens.shape[:2],
        dinov2_backend,
        dinov2_repo,
        dinov2_device,
    ).to(device=rendered_image.device)
    gt_tokens = F.normalize(gt_tokens.to(torch.float32), dim=-1)
    patch_error = 0.5 * torch.clamp(
        1.0 - F.cosine_similarity(rendered_tokens, gt_tokens, dim=-1),
        min=0.0,
        max=2.0,
    )
    patch_error = _smooth_2d_map(
        patch_error,
        descriptor_token_smooth_kernel,
        "vfm_descriptor_token_smooth_kernel",
    )
    return F.interpolate(
        patch_error.view(1, 1, *patch_error.shape[-2:]),
        size=rendered_image.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).view(*rendered_image.shape[-2:])


def _compute_pixel_error_map(
    rendered_image,
    viewpoint_cam,
    backend,
    cache_dir,
    gaussians=None,
    args=None,
    dinov2_repo="",
    dinov2_device="cuda",
    descriptor_token_smooth_kernel=1,
):
    if backend in ("mock_l1", "photometric_l1"):
        gt_image = viewpoint_cam.original_image.to(rendered_image.device)
        return _mock_l1_error(rendered_image, gt_image)
    if backend in ("mock_edge_l1", "edge_l1"):
        gt_image = viewpoint_cam.original_image.to(rendered_image.device)
        return _mock_edge_l1_error(rendered_image, gt_image)
    if backend == "cached_edge_l1":
        return _cached_edge_l1_error(rendered_image, viewpoint_cam, cache_dir)
    if backend == "colmap_depth_edge_l1":
        return _colmap_depth_edge_l1_error(rendered_image, viewpoint_cam, cache_dir)
    if backend == "colmap_depth_edge_prior":
        return _colmap_depth_edge_prior_error(rendered_image, viewpoint_cam, cache_dir)
    if backend in ("depth_anything_depth_prior", "depth_anything_depth_edge_prior"):
        return _depth_anything_prior_error(rendered_image, viewpoint_cam, cache_dir)
    if backend == "depth_anything_residual_orientation":
        return _depth_anything_residual_orientation_error(rendered_image, viewpoint_cam, cache_dir, gaussians, args)
    if backend == "dinov2_token_edge_l1":
        return _dinov2_token_edge_l1_error(rendered_image, viewpoint_cam, cache_dir)
    if backend in ("dinov2_descriptor_cosine", "dinov2_descriptor_cosine_l1"):
        return _dinov2_descriptor_cosine_error(
            rendered_image,
            viewpoint_cam,
            cache_dir,
            dinov2_repo,
            dinov2_device,
            descriptor_token_smooth_kernel=descriptor_token_smooth_kernel,
        )
    raise ValueError(
        "Unsupported vfm_backend {!r}. Available backends: mock_l1, mock_edge_l1, cached_edge_l1, "
        "colmap_depth_edge_l1, colmap_depth_edge_prior, depth_anything_depth_prior, "
        "depth_anything_depth_edge_prior, depth_anything_residual_orientation, dinov2_token_edge_l1, "
        "dinov2_descriptor_cosine.".format(backend)
    )


def _topk_metric_map(normalized_error, topk_fraction):
    topk_fraction = min(max(float(topk_fraction), 0.0), 1.0)
    metric_flat = torch.zeros(normalized_error.numel(), dtype=torch.int, device=normalized_error.device)
    if topk_fraction <= 0.0 or normalized_error.numel() == 0:
        return metric_flat.view_as(normalized_error)

    flat_error = normalized_error.reshape(-1)
    positive_count = int(torch.count_nonzero(flat_error > 0).item())
    if positive_count == 0:
        return metric_flat.view_as(normalized_error)

    k = min(max(1, int(math.ceil(flat_error.numel() * topk_fraction))), positive_count)
    _, indices = torch.topk(flat_error, k=k, largest=True, sorted=False)
    metric_flat[indices] = 1
    return metric_flat.view_as(normalized_error)


def _topk_score_mask(score, topk_fraction):
    topk_fraction = min(max(float(topk_fraction), 0.0), 1.0)
    flat_score = torch.nan_to_num(score.detach().to(torch.float32).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    mask = torch.zeros(flat_score.shape[0], dtype=torch.bool, device=flat_score.device)
    if topk_fraction <= 0.0 or flat_score.numel() == 0:
        return mask.view_as(score)

    positive_mask = flat_score > 0
    positive_count = int(torch.count_nonzero(positive_mask).item())
    if positive_count == 0:
        return mask.view_as(score)

    k = min(max(1, int(math.ceil(flat_score.numel() * topk_fraction))), positive_count)
    candidate_score = flat_score.masked_fill(~positive_mask, float("-inf"))
    _, indices = torch.topk(candidate_score, k=k, largest=True, sorted=False)
    mask[indices] = True
    return mask.view_as(score)


def _percentile_metric_map(normalized_error, percentile):
    percentile = min(max(float(percentile), 0.0), 1.0)
    flat_error = normalized_error.reshape(-1)
    if flat_error.numel() == 0 or torch.max(flat_error) <= 0:
        return torch.zeros_like(normalized_error, dtype=torch.int)

    kth_index = min(max(1, int(math.ceil(flat_error.numel() * percentile))), flat_error.numel())
    threshold = torch.kthvalue(flat_error, kth_index).values
    return (normalized_error > threshold).int()


def _soft_topk_metric_map_layers(normalized_error, topk_fraction, levels):
    levels = max(1, int(levels))
    layers = []
    for level in range(1, levels + 1):
        layer_fraction = float(topk_fraction) * level / levels
        layers.append((_topk_metric_map(normalized_error, layer_fraction), 1.0 / levels))
    return layers


def _build_vfm_metric_map(pixel_error_map, args):
    return _build_vfm_metric_map_layers(pixel_error_map, args)[0][0]


def _build_vfm_metric_map_layers(pixel_error_map, args):
    normalized_error = normalize01(pixel_error_map.detach())
    mode = getattr(args, "vfm_metric_map_mode", "threshold").lower()
    if mode == "threshold":
        metric_map = (normalized_error > getattr(args, "vfm_loss_thresh", 0.5)).int()
        layers = [(metric_map, 1.0)]
    elif mode == "percentile":
        metric_map = _percentile_metric_map(
            normalized_error,
            getattr(args, "vfm_metric_percentile", 0.85),
        )
        layers = [(metric_map, 1.0)]
    elif mode == "topk":
        metric_map = _topk_metric_map(
            normalized_error,
            getattr(args, "vfm_metric_topk", 0.15),
        )
        layers = [(metric_map, 1.0)]
    elif mode == "soft_topk":
        layers = _soft_topk_metric_map_layers(
            normalized_error,
            getattr(args, "vfm_metric_topk", 0.15),
            getattr(args, "vfm_metric_soft_levels", 3),
        )
    else:
        raise ValueError(
            "Unsupported vfm_metric_map_mode {!r}. Available: threshold, percentile, topk, soft_topk.".format(mode)
        )
    return [(metric_map.reshape(-1).contiguous(), float(weight)) for metric_map, weight in layers]


def _support_metric_map(viewpoint_cam, device):
    return torch.ones(
        int(viewpoint_cam.image_height) * int(viewpoint_cam.image_width),
        dtype=torch.int,
        device=device,
    )


def _normalize_vfm_counts_by_support(vfm_counts, support_counts, args):
    normalizer = getattr(args, "vfm_importance_normalizer", "none").lower()
    if normalizer == "none":
        return vfm_counts
    if normalizer != "support_ratio":
        raise ValueError(
            "Unsupported vfm_importance_normalizer {!r}. Available: none, support_ratio.".format(normalizer)
        )

    min_count = max(float(getattr(args, "vfm_support_min_count", 1.0)), 1.0)
    power = max(float(getattr(args, "vfm_support_ratio_power", 1.0)), 0.0)
    support_counts = torch.clamp(support_counts.to(torch.float32), min=min_count)
    hit_ratio = torch.clamp(vfm_counts.to(torch.float32) / support_counts, min=0.0, max=1.0)
    if power != 1.0:
        hit_ratio = torch.pow(hit_ratio, power)
    return vfm_counts.to(torch.float32) * hit_ratio


def _rgb_prune_topk_candidate(rgb_pruning, args):
    scores = torch.nan_to_num(rgb_pruning.detach().to(torch.float32).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    total = scores.numel()
    candidate = torch.zeros_like(scores, dtype=torch.bool)
    if total == 0:
        return candidate.reshape_as(rgb_pruning).to(device=rgb_pruning.device)

    topk = float(getattr(args, "vfm_prune_protect_rgb_topk", 0.001) or 0.0)
    if topk <= 0.0:
        return candidate.reshape_as(rgb_pruning).to(device=rgb_pruning.device)
    if topk <= 1.0:
        count = int(math.ceil(total * topk))
    else:
        count = int(round(topk))
    count = max(1, min(total, count))

    if count >= total:
        candidate[:] = True
    else:
        selected = torch.topk(scores, count, largest=True).indices
        candidate[selected] = True
    return candidate.reshape_as(rgb_pruning).to(device=rgb_pruning.device)


def _topk_count(total, topk):
    if total <= 0 or topk <= 0.0:
        return 0
    if topk <= 1.0:
        count = int(math.ceil(total * topk))
    else:
        count = int(round(topk))
    return max(1, min(total, count))


def _rgb_prune_auto_topk_candidate(rgb_pruning, args):
    scores = torch.nan_to_num(rgb_pruning.detach().to(torch.float32).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    total = scores.numel()
    candidate = torch.zeros_like(scores, dtype=torch.bool)
    if total == 0:
        return candidate.reshape_as(rgb_pruning).to(device=rgb_pruning.device)

    min_topk = max(0.0, float(getattr(args, "vfm_prune_protect_rgb_auto_min_topk", 0.001) or 0.0))
    max_topk = max(min_topk, float(getattr(args, "vfm_prune_protect_rgb_auto_max_topk", 0.010) or 0.0))
    std_scale = max(0.0, float(getattr(args, "vfm_prune_protect_rgb_auto_std_scale", 2.0) or 0.0))
    min_count = _topk_count(total, min_topk)
    max_count = _topk_count(total, max_topk)
    if max_count <= 0:
        return candidate.reshape_as(rgb_pruning).to(device=rgb_pruning.device)

    threshold = scores.mean() + std_scale * scores.std(unbiased=False)
    selected = torch.nonzero(scores >= threshold, as_tuple=False).reshape(-1)
    count = int(selected.numel())
    count = max(min_count, min(max_count, count))
    if count >= total:
        candidate[:] = True
    elif count > 0:
        selected = torch.topk(scores, count, largest=True).indices
        candidate[selected] = True
    return candidate.reshape_as(rgb_pruning).to(device=rgb_pruning.device)


def _rgb_prune_candidate_mask(rgb_pruning, args, mode=None):
    selected_mode = str(mode or getattr(args, "vfm_prune_protect_mode", "vfm") or "vfm").lower()
    if selected_mode == "rgb_prune_candidate":
        min_score = float(getattr(args, "vfm_prune_protect_rgb_min_score", 0.9) or 0.0)
        return rgb_pruning.to(torch.float32) >= min_score
    if selected_mode == "rgb_prune_topk":
        return _rgb_prune_topk_candidate(rgb_pruning, args)
    if selected_mode == "rgb_prune_auto_topk":
        return _rgb_prune_auto_topk_candidate(rgb_pruning, args)
    return None


def _build_prune_protection(vfm_counts, rgb_pruning, args):
    weight = max(0.0, float(getattr(args, "vfm_prune_protect_weight", 0.0) or 0.0))
    if weight <= 0.0:
        return None, 0.0

    mode = str(getattr(args, "vfm_prune_protect_mode", "vfm") or "vfm").lower()
    min_count = max(0.0, float(getattr(args, "vfm_prune_protect_min_count", 1.0) or 0.0))
    power = max(0.0, float(getattr(args, "vfm_prune_protect_power", 1.0) or 0.0))

    protect = vfm_counts.to(torch.float32)
    if min_count > 0.0:
        protect = torch.where(protect >= min_count, protect, torch.zeros_like(protect))
    protect = normalize01(protect)
    if power != 1.0:
        protect = torch.pow(protect, power)

    if mode == "vfm":
        return protect, weight
    if mode == "rgb_aware":
        return protect * (1.0 - rgb_pruning.to(torch.float32)), weight
    if mode in ("rgb_prune_candidate", "rgb_prune_topk", "rgb_prune_auto_topk"):
        rgb_candidate = _rgb_prune_candidate_mask(rgb_pruning, args, mode)
        return protect * rgb_candidate.to(torch.float32), weight
    raise ValueError(
        "Unsupported vfm_prune_protect_mode {!r}. Available: vfm, rgb_aware, rgb_prune_candidate, rgb_prune_topk, rgb_prune_auto_topk.".format(
            mode
        )
    )


def _importance_budget_progress(gaussians, args):
    budget_count = int(getattr(args, "vfm_importance_budget_count", 0) or 0)
    if budget_count <= 0:
        return None

    start_ratio = float(getattr(args, "vfm_importance_budget_start_ratio", 0.9) or 0.0)
    start_ratio = max(0.0, min(start_ratio, 1.0))

    current_count = int(gaussians._xyz.shape[0])
    start_count = max(1, int(round(budget_count * start_ratio)))
    if current_count <= start_count:
        return 0.0
    if current_count >= budget_count:
        return 1.0

    span = max(1, budget_count - start_count)
    progress = float(current_count - start_count) / float(span)
    curve = str(getattr(args, "vfm_importance_budget_curve", "linear") or "linear").lower()
    if curve == "linear":
        pass
    elif curve == "quadratic":
        progress = progress * progress
    elif curve == "sqrt":
        progress = math.sqrt(progress)
    else:
        raise ValueError(
            "Unsupported vfm_importance_budget_curve {!r}. Available: linear, quadratic, sqrt.".format(curve)
        )
    return progress


def _budget_aware_importance_weight(base_weight, gaussians, args):
    progress = _importance_budget_progress(gaussians, args)
    if progress is None or progress <= 0.0:
        return base_weight

    min_weight = float(getattr(args, "vfm_importance_budget_min_weight", 0.0) or 0.0)
    min_weight = max(0.0, min(min_weight, base_weight))
    return base_weight - progress * (base_weight - min_weight)


def compute_gaussian_score_fastgs_with_vfm(camlist, gaussians, pipe, bg, args, DENSIFY=False):
    """Compute FastGS scores with an auxiliary VFM-style topology signal.

    The v1 scorer keeps the FastGS scoring contract intact: it turns a
    backend-specific pixel error map into a binary metric map, lets the
    rasterizer accumulate per-Gaussian hits, then fuses those scores with the
    original photometric FastGS scores.
    """

    profile_this = _profile_enabled(args)
    if profile_this:
        _PROFILE_STATE["calls"] += 1
    profile_call = int(_PROFILE_STATE["calls"]) if profile_this else 0
    profile_this = profile_this and (profile_call % _profile_interval(args) == 0)
    _sync_if_needed(profile_this)
    total_start = time.perf_counter()
    rgb_start = total_start
    backend = getattr(args, "vfm_backend", "mock_l1")
    vfm_weight = getattr(args, "vfm_weight", 0.25)
    vfm_importance_mode = getattr(args, "vfm_importance_mode", "max").lower()
    vfm_prune_protect_weight = max(0.0, float(getattr(args, "vfm_prune_protect_weight", 0.0) or 0.0))
    vfm_active = _is_vfm_active(args)
    skip_rgb_score = (
        vfm_active
        and DENSIFY
        and vfm_importance_mode == "vfm_only"
        and float(vfm_weight or 0.0) <= 0.0
        and vfm_prune_protect_weight <= 0.0
        and not _densify_prune_enabled(args)
    )
    if skip_rgb_score:
        rgb_importance = torch.zeros((gaussians._xyz.shape[0],), dtype=torch.float32, device=gaussians._xyz.device)
        rgb_pruning = torch.zeros_like(rgb_importance)
    else:
        rgb_importance, rgb_pruning = compute_gaussian_score_fastgs(
            camlist, gaussians, pipe, bg, args, DENSIFY=DENSIFY
        )
    rgb_ms = _elapsed_ms(rgb_start, profile_this)
    if not vfm_active:
        _clear_rgb_rerank_reference(args)
        if profile_this:
            total_ms = _elapsed_ms(total_start, profile_this)
            print(
                "[VFM PROFILE] call={} densify={} backend={} active=false gs={} total_ms={:.2f} rgb_ms={:.2f}".format(
                    profile_call,
                    bool(DENSIFY),
                    backend,
                    int(gaussians._xyz.shape[0]),
                    total_ms,
                    rgb_ms,
                )
            )
        return rgb_importance if DENSIFY else None, rgb_pruning

    vfm_counts_total = None
    vfm_support_total = None
    vfm_pruning_total = None
    vfm_importance_weight = max(0.0, getattr(args, "vfm_importance_weight", 1.0))
    if DENSIFY and vfm_importance_mode != "adaptive_weighted":
        vfm_importance_weight = _budget_aware_importance_weight(vfm_importance_weight, gaussians, args)
    vfm_importance_normalizer = getattr(args, "vfm_importance_normalizer", "none").lower()
    needs_vfm_counts = DENSIFY or vfm_prune_protect_weight > 0.0
    support_cap_mode = (
        DENSIFY
        and str(getattr(args, "densify_budget_candidate_cap_mode", "") or "").lower() == "screen_support"
    )
    use_albedo_sh0 = getattr(args, "vfm_use_albedo_sh0", True)
    cache_dir = getattr(args, "vfm_cache_dir", "")
    dinov2_repo = getattr(args, "vfm_dinov2_repo", "")
    dinov2_device = getattr(args, "vfm_dinov2_device", "cuda")
    descriptor_token_smooth_kernel = getattr(args, "vfm_descriptor_token_smooth_kernel", 1)
    render_ms = 0.0
    error_ms = 0.0
    metric_ms = 0.0
    count_ms = 0.0
    support_ms = 0.0
    combine_ms = 0.0
    view_count = 0
    layer_count = 0

    for viewpoint_cam in camlist:
        view_count += 1
        stage_start = time.perf_counter()
        if use_albedo_sh0:
            rendered_image = _render_with_sh0(viewpoint_cam, gaussians, pipe, bg, args.mult)
        else:
            rendered_image = render_fastgs(viewpoint_cam, gaussians, pipe, bg, args.mult)["render"]
        render_ms += _elapsed_ms(stage_start, profile_this)

        stage_start = time.perf_counter()
        pixel_error_map = _compute_pixel_error_map(
            rendered_image,
            viewpoint_cam,
            backend,
            cache_dir,
            gaussians=gaussians,
            args=args,
            dinov2_repo=dinov2_repo,
            dinov2_device=dinov2_device,
            descriptor_token_smooth_kernel=descriptor_token_smooth_kernel,
        )
        error_ms += _elapsed_ms(stage_start, profile_this)
        stage_start = time.perf_counter()
        metric_layers = _build_vfm_metric_map_layers(pixel_error_map, args)
        metric_ms += _elapsed_ms(stage_start, profile_this)
        counts = None
        for metric_map, layer_weight in metric_layers:
            layer_count += 1
            stage_start = time.perf_counter()
            render_pkg = render_fastgs(
                viewpoint_cam,
                gaussians,
                pipe,
                bg,
                args.mult,
                get_flag=True,
                metric_map=metric_map,
            )
            count_ms += _elapsed_ms(stage_start, profile_this)
            layer_counts = render_pkg["accum_metric_counts"]
            if layer_weight != 1.0:
                layer_counts = layer_counts.to(torch.float32) * layer_weight
            if counts is None:
                counts = layer_counts.clone()
            else:
                counts += layer_counts

        if needs_vfm_counts:
            if vfm_counts_total is None:
                vfm_counts_total = counts.clone()
            else:
                vfm_counts_total += counts

        if DENSIFY and (vfm_importance_normalizer == "support_ratio" or support_cap_mode):
            stage_start = time.perf_counter()
            support_pkg = render_fastgs(
                viewpoint_cam,
                gaussians,
                pipe,
                bg,
                args.mult,
                get_flag=True,
                metric_map=_support_metric_map(viewpoint_cam, rendered_image.device),
            )
            support_counts = support_pkg["accum_metric_counts"]
            if vfm_support_total is None:
                vfm_support_total = support_counts.clone()
            else:
                vfm_support_total += support_counts
            support_ms += _elapsed_ms(stage_start, profile_this)

        stage_start = time.perf_counter()
        weighted_counts = pixel_error_map.detach().mean() * counts.to(torch.float32)
        if vfm_pruning_total is None:
            vfm_pruning_total = weighted_counts.clone()
        else:
            vfm_pruning_total += weighted_counts
        combine_ms += _elapsed_ms(stage_start, profile_this)

    stage_start = time.perf_counter()
    vfm_pruning = normalize01(vfm_pruning_total)
    pruning_score = normalize01(rgb_pruning + vfm_weight * vfm_pruning)
    protection, protection_weight = _build_prune_protection(vfm_counts_total, rgb_pruning, args)
    if protection is not None:
        if not DENSIFY:
            protected = int((protection > 0).sum().item())
            rgb_candidate_mask = _rgb_prune_candidate_mask(rgb_pruning, args)
            rgb_candidates = int(rgb_candidate_mask.sum().item()) if rgb_candidate_mask is not None else 0
            print(
                "[VFM PRUNE PROTECT] iter={} mode={} weight={:.4f} protected={} rgb_candidates={} mean={:.6f} max={:.6f}".format(
                    int(getattr(args, "current_iteration", 0) or 0),
                    str(getattr(args, "vfm_prune_protect_mode", "vfm") or "vfm"),
                    float(protection_weight),
                    protected,
                    rgb_candidates,
                    float(protection.mean().item()) if protection.numel() else 0.0,
                    float(protection.max().item()) if protection.numel() else 0.0,
                )
            )
        pruning_score = normalize01(pruning_score - protection_weight * protection)

    if DENSIFY:
        if vfm_importance_normalizer != "none":
            if vfm_importance_normalizer == "support_ratio":
                vfm_counts_total = _normalize_vfm_counts_by_support(vfm_counts_total, vfm_support_total, args)
            else:
                raise ValueError(
                    "Unsupported vfm_importance_normalizer {!r}. Available: none, support_ratio.".format(
                        vfm_importance_normalizer
                    )
                )
        args.densify_budget_support_counts = vfm_support_total
        vfm_importance = torch.floor(vfm_counts_total.to(torch.float32) / len(camlist))
        if vfm_importance_mode == "rgb_only":
            importance_score = rgb_importance
        elif vfm_importance_mode == "vfm_only":
            importance_score = vfm_importance
        elif vfm_importance_mode == "weighted":
            blend = min(vfm_importance_weight, 1.0)
            importance_score = torch.floor(
                (1.0 - blend) * rgb_importance.to(torch.float32) + blend * vfm_importance.to(torch.float32)
            )
        elif vfm_importance_mode == "adaptive_weighted":
            progress = _importance_budget_progress(gaussians, args)
            if progress is None:
                progress = 1.0
            scaled_vfm_importance = vfm_importance
            if vfm_importance_weight != 1.0:
                scaled_vfm_importance = torch.floor(vfm_importance.to(torch.float32) * vfm_importance_weight)
            max_importance = torch.maximum(rgb_importance.to(torch.float32), scaled_vfm_importance.to(torch.float32))
            blend = min(vfm_importance_weight, 1.0)
            weighted_importance = torch.floor(
                (1.0 - blend) * rgb_importance.to(torch.float32) + blend * vfm_importance.to(torch.float32)
            )
            importance_score = torch.floor((1.0 - progress) * max_importance + progress * weighted_importance)
        elif vfm_importance_mode == "max":
            if vfm_importance_weight != 1.0:
                vfm_importance = torch.floor(vfm_importance.to(torch.float32) * vfm_importance_weight)
            importance_score = torch.maximum(rgb_importance.to(torch.float32), vfm_importance)
        elif vfm_importance_mode == "rgb_broad":
            rgb_score = rgb_importance.to(torch.float32)
            broad_mask = _topk_score_mask(rgb_score, getattr(args, "vfm_rgb_broad_topk", 0.50))
            importance_score = rgb_score.masked_fill(~broad_mask, 0.0)
        elif vfm_importance_mode == "rgb_rerank":
            rgb_score = rgb_importance.to(torch.float32)
            dino_score = normalize01(vfm_importance.to(torch.float32))
            if getattr(args, "vfm_rgb_broad_gate", True):
                broad_mask = _topk_score_mask(rgb_score, getattr(args, "vfm_rgb_broad_topk", 0.50))
            else:
                broad_mask = rgb_score > 0
            rerank_weight = max(0.0, float(getattr(args, "vfm_dino_rerank_lambda", 0.25) or 0.0))
            importance_score = rgb_score * (1.0 + rerank_weight * dino_score)
            importance_score = importance_score.masked_fill(~broad_mask, 0.0)
            if getattr(args, "vfm_rgb_rerank_final_topm", False):
                args.vfm_rgb_broad_reference_score = rgb_score.masked_fill(~broad_mask, 0.0).detach()
            else:
                _clear_rgb_rerank_reference(args)
        else:
            raise ValueError(
                "Unsupported vfm_importance_mode {!r}. Available: max, weighted, adaptive_weighted, "
                "rgb_only, rgb_broad, rgb_rerank.".format(
                    vfm_importance_mode
                )
            )
    else:
        importance_score = None
    combine_ms += _elapsed_ms(stage_start, profile_this)
    total_ms = _elapsed_ms(total_start, profile_this)

    if profile_this:
        print(
            "[VFM PROFILE] call={} densify={} backend={} views={} layers={} gs={} total_ms={:.2f} "
            "rgb_ms={:.2f} render_ms={:.2f} error_ms={:.2f} metric_ms={:.2f} count_ms={:.2f} "
            "support_ms={:.2f} combine_ms={:.2f}".format(
                profile_call,
                bool(DENSIFY),
                backend,
                view_count,
                layer_count,
                int(gaussians._xyz.shape[0]),
                total_ms,
                rgb_ms,
                render_ms,
                error_ms,
                metric_ms,
                count_ms,
                support_ms,
                combine_ms,
            )
        )

    return importance_score, pruning_score


def preflight_vfm_topology_scorer(dataset, args):
    backend = getattr(args, "vfm_backend", "mock_l1")
    expected_manifest_backends = _CACHED_BACKEND_MANIFESTS.get(backend)
    if expected_manifest_backends is None:
        return

    cache_dir = getattr(args, "vfm_cache_dir", "")
    source_path = getattr(dataset, "source_path", None)
    images = getattr(dataset, "images", None)
    errors, warnings, manifest = validate_cache(
        cache_dir,
        backend=None,
        source_path=source_path,
        images=images,
        check_checksum=True,
        load_entries=False,
    )
    manifest_backend = manifest.get("backend")
    if manifest_backend not in expected_manifest_backends:
        errors.append(
            "backend mismatch: manifest={!r}, expected one of {}".format(
                manifest_backend,
                ", ".join(repr(item) for item in expected_manifest_backends),
            )
        )
    if backend == "dinov2_token_edge_l1" and manifest.get("feature") not in ("dinov2_patchtokens", "dinov2_token_edge"):
        errors.append(
            "feature mismatch: manifest={!r}, expected 'dinov2_patchtokens' or 'dinov2_token_edge'".format(
                manifest.get("feature")
            )
        )
    if backend in ("dinov2_descriptor_cosine", "dinov2_descriptor_cosine_l1") and manifest.get("feature") != "dinov2_patchtokens":
        errors.append("feature mismatch: manifest={!r}, expected 'dinov2_patchtokens'".format(manifest.get("feature")))
    if backend in ("depth_anything_depth_prior", "depth_anything_residual_orientation") and manifest.get("feature") != "depth_anything_relative_depth":
        errors.append("feature mismatch: manifest={!r}, expected 'depth_anything_relative_depth'".format(manifest.get("feature")))
    if backend == "depth_anything_depth_edge_prior" and manifest.get("feature") != "depth_anything_depth_edge":
        errors.append("feature mismatch: manifest={!r}, expected 'depth_anything_depth_edge'".format(manifest.get("feature")))
    for warning in warnings:
        print("[VFM cache preflight warning] {}".format(warning))
    if errors:
        formatted = "\n".join("- {}".format(error) for error in errors[:10])
        raise ValueError("VFM cache preflight failed for {}:\n{}".format(cache_dir, formatted))
    print(
        "VFM cache preflight passed: {} {} entries at {}".format(
            len(manifest.get("entries", {})),
            manifest.get("backend", backend),
            cache_dir,
        )
    )


compute_gaussian_score_fastgs_with_vfm.preflight = preflight_vfm_topology_scorer
register_scorer("vfm_topology_scorer", compute_gaussian_score_fastgs_with_vfm)
