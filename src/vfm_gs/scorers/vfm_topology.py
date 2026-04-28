import torch
import torch.nn.functional as F

from vfm_gs.gaussian_renderer import render_fastgs
from vfm_gs.scorers.vfm_cache import load_feature, read_manifest
from vfm_gs.utils.fast_utils import compute_gaussian_score_fastgs, normalize01

from .registry import register_scorer


_CACHE_READERS = {}


class VFMFeatureCache:
    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        self.manifest = read_manifest(cache_dir)
        self.entries = self.manifest["entries"]
        self.features = {}

    def get_edge_map(self, image_name, device, size):
        if image_name not in self.entries:
            raise KeyError("Image {!r} is missing from VFM cache {}".format(image_name, self.cache_dir))
        if image_name not in self.features:
            entry = self.entries[image_name]
            storage = entry.get("storage", self.manifest.get("storage", "npy_float32"))
            path = "{}/{}".format(self.cache_dir.rstrip("/"), entry["cache_file"])
            edge_map = torch.from_numpy(load_feature(path, storage)).to(torch.float32)
            self.features[image_name] = edge_map

        edge_map = self.features[image_name].to(device=device)
        if tuple(edge_map.shape[-2:]) != tuple(size):
            edge_map = F.interpolate(
                edge_map.view(1, 1, *edge_map.shape[-2:]),
                size=size,
                mode="bilinear",
                align_corners=False,
            ).view(*size)
        return edge_map


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


def _compute_pixel_error_map(rendered_image, viewpoint_cam, backend, cache_dir):
    gt_image = viewpoint_cam.original_image.to(rendered_image.device)
    if backend in ("mock_l1", "photometric_l1"):
        return _mock_l1_error(rendered_image, gt_image)
    if backend in ("mock_edge_l1", "edge_l1"):
        return _mock_edge_l1_error(rendered_image, gt_image)
    if backend == "cached_edge_l1":
        return _cached_edge_l1_error(rendered_image, viewpoint_cam, cache_dir)
    raise ValueError(
        "Unsupported vfm_backend {!r}. Available backends: mock_l1, mock_edge_l1, cached_edge_l1.".format(backend)
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


register_scorer("vfm_topology_scorer", compute_gaussian_score_fastgs_with_vfm)
