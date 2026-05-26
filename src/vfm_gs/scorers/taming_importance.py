import math

import torch

from vfm_gs.gaussian_renderer import render_fastgs
from vfm_gs.utils.fast_utils import compute_photometric_loss, get_loss, normalize, normalize01

from .registry import register_scorer


_DEFAULT_COEFFICIENTS = {
    "view": 50.0,
    "mse": 50.0,
    "edge": 50.0,
    "grad": 25.0,
    "opacity": 100.0,
    "depth": 5.0,
    "loss": 10.0,
    "radii": 10.0,
    "scale": 25.0,
}


def _coeff(args, name):
    return float(getattr(args, "taming_{}_importance".format(name), _DEFAULT_COEFFICIENTS[name]))


def _gt_edge_map(image):
    rgb = image.detach().to(torch.float32)
    luma = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    dx = torch.zeros_like(luma)
    dy = torch.zeros_like(luma)
    dx[:, 1:] = luma[:, 1:] - luma[:, :-1]
    dy[1:, :] = luma[1:, :] - luma[:-1, :]
    return normalize01(torch.sqrt(dx * dx + dy * dy + 1e-12))


def _taming_loss_map(render_image, gt_image, args):
    l1_norm = get_loss(render_image, gt_image)
    edge_norm = _gt_edge_map(gt_image)
    return normalize01(_coeff(args, "mse") * l1_norm + _coeff(args, "edge") * edge_norm)


def _metric_map_from_loss(loss_map, args):
    flat = loss_map.reshape(-1).detach().to(torch.float32)
    topk_fraction = float(getattr(args, "taming_metric_topk", 0.0) or 0.0)
    metric = torch.zeros_like(flat, dtype=torch.int32)
    if 0.0 < topk_fraction < 1.0:
        k = max(1, int(math.ceil(flat.numel() * topk_fraction)))
        k = min(k, flat.numel())
        selected = torch.topk(flat, k=k, largest=True, sorted=False).indices
        metric[selected] = 1
        return metric.contiguous()

    threshold = float(getattr(args, "taming_metric_loss_thresh", getattr(args, "loss_thresh", 0.1)) or 0.1)
    return (flat > threshold).to(torch.int32).contiguous()


def _camera_depths(gaussians, viewpoint_cam):
    xyz = gaussians.get_xyz.detach()
    ones = torch.ones((xyz.shape[0], 1), dtype=xyz.dtype, device=xyz.device)
    points_h = torch.cat([xyz, ones], dim=1)
    view_points = torch.matmul(points_h, viewpoint_cam.world_view_transform)
    return torch.nan_to_num(view_points[:, 2].to(torch.float32), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)


def _base_primitive_scores(gaussians):
    grads = gaussians.xyz_gradient_accum / gaussians.denom
    grads[grads.isnan()] = 0.0
    grad_score = torch.norm(grads.detach(), dim=-1)
    opacity_score = gaussians.get_opacity.detach().squeeze()
    scale_score = torch.prod(gaussians.get_scaling.detach(), dim=1)
    return grad_score, opacity_score, scale_score


def _normalize_output_score(raw_score, args):
    median = float(getattr(args, "taming_importance_output_median", 5.0) or 0.0)
    if median <= 0.0:
        return torch.nan_to_num(raw_score.detach().to(torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return normalize(median, raw_score.detach())


def compute_gaussian_score_taming_importance_fastgs_prune(camlist, gaussians, pipe, bg, args, DENSIFY=False):
    """Use Taming-3DGS-style densify importance while keeping FastGS pruning score.

    The current FastGS rasterizer exposes binary per-Gaussian metric counts, not
    Taming-3DGS' full weighted accumulators. This scorer therefore swaps the
    densification importance to the Taming primitive/loss score terms available
    in this pipeline, then rescales them to the FastGS threshold contract.
    """

    full_metric_score = None
    raw_importance = None
    grad_score, opacity_score, scale_score = _base_primitive_scores(gaussians)

    for viewpoint_cam in camlist:
        render_pkg = render_fastgs(viewpoint_cam, gaussians, pipe, bg, args.mult)
        render_image = render_pkg["render"]
        gt_image = viewpoint_cam.original_image.cuda()
        photometric_loss = compute_photometric_loss(viewpoint_cam, render_image).detach()

        fastgs_metric_map = (get_loss(render_image, gt_image) > args.loss_thresh).int().reshape(-1).contiguous()
        fastgs_pkg = render_fastgs(
            viewpoint_cam,
            gaussians,
            pipe,
            bg,
            args.mult,
            get_flag=True,
            metric_map=fastgs_metric_map,
        )
        fastgs_counts = fastgs_pkg["accum_metric_counts"].to(torch.float32)
        fastgs_view_score = photometric_loss * fastgs_counts
        full_metric_score = fastgs_view_score if full_metric_score is None else full_metric_score + fastgs_view_score

        if not DENSIFY:
            continue

        taming_metric_map = _metric_map_from_loss(_taming_loss_map(render_image, gt_image, args), args)
        taming_pkg = render_fastgs(
            viewpoint_cam,
            gaussians,
            pipe,
            bg,
            args.mult,
            get_flag=True,
            metric_map=taming_metric_map,
        )
        loss_counts = taming_pkg["accum_metric_counts"].to(torch.float32)
        radii_score = render_pkg["radii"].detach().to(torch.float32)
        depth_score = _camera_depths(gaussians, viewpoint_cam)

        primitive_importance = (
            normalize(_coeff(args, "grad"), grad_score)
            + normalize(_coeff(args, "opacity"), opacity_score)
            + normalize(_coeff(args, "depth"), depth_score)
            + normalize(_coeff(args, "radii"), radii_score)
            + normalize(_coeff(args, "scale"), scale_score)
        )
        pixel_importance = normalize(_coeff(args, "loss"), loss_counts)
        view_importance = _coeff(args, "view") * photometric_loss * (primitive_importance + pixel_importance)
        visible = render_pkg["radii"] > 0
        view_importance = view_importance.masked_fill(~visible, 0.0)
        raw_importance = view_importance if raw_importance is None else raw_importance + view_importance

    pruning_score = normalize01(full_metric_score)
    importance_score = _normalize_output_score(raw_importance, args) if DENSIFY else None
    return importance_score, pruning_score


register_scorer("taming_importance_fastgs_prune", compute_gaussian_score_taming_importance_fastgs_prune)
