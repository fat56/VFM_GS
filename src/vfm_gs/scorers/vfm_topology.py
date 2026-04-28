import torch
import torch.nn.functional as F

from vfm_gs.gaussian_renderer import render_fastgs
from vfm_gs.scorers.vfm_cache import load_feature, read_manifest, validate_cache
from vfm_gs.utils.fast_utils import compute_gaussian_score_fastgs, normalize01

from .registry import register_scorer


_CACHE_READERS = {}
_DINO_CACHE_BACKENDS = ("dinov2_vits14", "dinov2_vitb14")
_CACHED_BACKEND_MANIFESTS = {
    "cached_edge_l1": ("cached_edge_l1",),
    "dinov2_token_edge_l1": _DINO_CACHE_BACKENDS,
}


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


def _pool_to_patch_grid(value_map, grid_size):
    return F.adaptive_avg_pool2d(value_map.reshape(1, 1, *value_map.shape[-2:]), output_size=grid_size).view(*grid_size)


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


def _compute_pixel_error_map(rendered_image, viewpoint_cam, backend, cache_dir):
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
    raise ValueError(
        "Unsupported vfm_backend {!r}. Available backends: mock_l1, mock_edge_l1, cached_edge_l1, "
        "dinov2_token_edge_l1.".format(backend)
    )


def compute_gaussian_score_fastgs_with_vfm(camlist, gaussians, pipe, bg, args, DENSIFY=False):
    """Compute FastGS scores with an auxiliary VFM-style topology signal.

    The v1 scorer keeps the FastGS scoring contract intact: it turns a
    backend-specific pixel error map into a binary metric map, lets the
    rasterizer accumulate per-Gaussian hits, then fuses those scores with the
    original photometric FastGS scores.
    """

    rgb_importance, rgb_pruning = compute_gaussian_score_fastgs(
        camlist, gaussians, pipe, bg, args, DENSIFY=DENSIFY
    )

    vfm_counts_total = None
    vfm_pruning_total = None
    backend = getattr(args, "vfm_backend", "mock_l1")
    loss_thresh = getattr(args, "vfm_loss_thresh", 0.5)
    vfm_weight = getattr(args, "vfm_weight", 0.25)
    use_albedo_sh0 = getattr(args, "vfm_use_albedo_sh0", True)
    cache_dir = getattr(args, "vfm_cache_dir", "")

    for viewpoint_cam in camlist:
        if use_albedo_sh0:
            rendered_image = _render_with_sh0(viewpoint_cam, gaussians, pipe, bg, args.mult)
        else:
            rendered_image = render_fastgs(viewpoint_cam, gaussians, pipe, bg, args.mult)["render"]

        pixel_error_map = _compute_pixel_error_map(rendered_image, viewpoint_cam, backend, cache_dir)
        metric_map = (normalize01(pixel_error_map) > loss_thresh).int().reshape(-1).contiguous()
        render_pkg = render_fastgs(
            viewpoint_cam,
            gaussians,
            pipe,
            bg,
            args.mult,
            get_flag=True,
            metric_map=metric_map,
        )
        counts = render_pkg["accum_metric_counts"]

        if DENSIFY:
            if vfm_counts_total is None:
                vfm_counts_total = counts.clone()
            else:
                vfm_counts_total += counts

        weighted_counts = pixel_error_map.detach().mean() * counts.to(torch.float32)
        if vfm_pruning_total is None:
            vfm_pruning_total = weighted_counts.clone()
        else:
            vfm_pruning_total += weighted_counts

    vfm_pruning = normalize01(vfm_pruning_total)
    pruning_score = normalize01(rgb_pruning + vfm_weight * vfm_pruning)

    if DENSIFY:
        vfm_importance = torch.div(vfm_counts_total, len(camlist), rounding_mode="floor")
        importance_score = torch.maximum(rgb_importance, vfm_importance)
    else:
        importance_score = None

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
    if backend == "dinov2_token_edge_l1" and manifest.get("feature") != "dinov2_patchtokens":
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
