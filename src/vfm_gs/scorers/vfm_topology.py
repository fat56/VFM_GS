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
_CACHED_BACKEND_MANIFESTS = {
    "cached_edge_l1": ("cached_edge_l1",),
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
        "dinov2_token_edge_l1, dinov2_descriptor_cosine.".format(backend)
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
    raise ValueError(
        "Unsupported vfm_prune_protect_mode {!r}. Available: vfm, rgb_aware.".format(mode)
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
    rgb_importance, rgb_pruning = compute_gaussian_score_fastgs(
        camlist, gaussians, pipe, bg, args, DENSIFY=DENSIFY
    )
    rgb_ms = _elapsed_ms(rgb_start, profile_this)
    backend = getattr(args, "vfm_backend", "mock_l1")
    if not _is_vfm_active(args):
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
    vfm_weight = getattr(args, "vfm_weight", 0.25)
    vfm_importance_mode = getattr(args, "vfm_importance_mode", "max").lower()
    vfm_importance_weight = max(0.0, getattr(args, "vfm_importance_weight", 1.0))
    if DENSIFY and vfm_importance_mode != "adaptive_weighted":
        vfm_importance_weight = _budget_aware_importance_weight(vfm_importance_weight, gaussians, args)
    vfm_importance_normalizer = getattr(args, "vfm_importance_normalizer", "none").lower()
    vfm_prune_protect_weight = max(0.0, float(getattr(args, "vfm_prune_protect_weight", 0.0) or 0.0))
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
        else:
            raise ValueError(
                "Unsupported vfm_importance_mode {!r}. Available: max, weighted, adaptive_weighted, rgb_only.".format(
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
