from __future__ import annotations

#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os, random, time
from random import randint
import sys
import uuid
from argparse import ArgumentParser, Namespace
from vfm_gs.config.legacy_args import ModelParams, PipelineParams, OptimizationParams
from vfm_gs.config.loader import DEFAULT_VARIANT, apply_argparse_defaults, load_train_config

TENSORBOARD_FOUND = False
RUNTIME_IMPORT_ERROR = None

try:
    import torch
    import numpy as np
    from vfm_gs.lpips_pytorch import lpips
    from vfm_gs.utils.loss_utils import l1_loss, l2_loss
    from fused_ssim import fused_ssim as fast_ssim
    from vfm_gs.gaussian_renderer import render_fastgs, network_gui_ws
    from vfm_gs.scene import Scene, GaussianModel
    from vfm_gs.utils.general_utils import safe_state
    from tqdm import tqdm
    from vfm_gs.utils.image_utils import psnr
except ModuleNotFoundError as exc:
    RUNTIME_IMPORT_ERROR = exc


def require_runtime_imports():
    if RUNTIME_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Training requires the FastGS runtime dependencies. "
            "Create the conda environment from environment.yml before running training."
        ) from RUNTIME_IMPORT_ERROR

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except (ImportError, NameError):
    TENSORBOARD_FOUND = False


def _target_gaussian_count(opt):
    return int(getattr(opt, "target_gaussian_count", 0) or 0)


def _prune_min_gaussian_count(opt):
    explicit_floor = int(getattr(opt, "prune_min_gaussian_count", 0) or 0)
    if explicit_floor > 0:
        return explicit_floor
    target_count = _target_gaussian_count(opt)
    target_ratio = float(getattr(opt, "prune_min_gaussian_target_ratio", 0.0) or 0.0)
    if target_count <= 0 or target_ratio <= 0.0:
        return 0
    return max(0, int(round(target_count * target_ratio)))


def _target_gaussian_prune_order(opt):
    order = str(getattr(opt, "target_gaussian_prune_order", "lowest_score") or "lowest_score").lower()
    if order not in ("lowest_score", "highest_score", "lowest_opacity"):
        raise ValueError(
            "Unsupported target_gaussian_prune_order {!r}. Available: lowest_score, highest_score, lowest_opacity.".format(
                order
            )
        )
    return order


def _staged_target_gaussian_count(opt):
    target_count = _target_gaussian_count(opt)
    if target_count <= 0:
        return 0
    margin = max(1.0, float(getattr(opt, "target_gaussian_stage_margin", 1.2) or 1.0))
    return max(target_count, int(round(target_count * margin)))


def _should_run_staged_target_prune(opt, iteration, staged_target_count):
    if not bool(getattr(opt, "target_gaussian_staged", False)):
        return False
    if staged_target_count <= 0:
        return False
    stage_start = int(getattr(opt, "target_gaussian_stage_start", 0) or 0)
    stage_interval = max(1, int(getattr(opt, "target_gaussian_stage_interval", 500) or 1))
    return iteration >= stage_start and iteration % stage_interval == 0


def _prune_to_target_budget(scene, gaussians, gaussian_scorer, pipe, bg, opt, target_count, label, iteration=None):
    current_count = gaussians._xyz.shape[0]
    if target_count <= 0 or current_count <= target_count:
        return 0, 0.0

    budget_start = time.time()
    my_viewpoint_stack = scene.getTrainCameras().copy()
    from vfm_gs.utils.fast_utils import sampling_cameras

    camlist = sampling_cameras(my_viewpoint_stack)
    opt.current_iteration = iteration if iteration is not None else int(getattr(opt, "iterations", 0) or 0)
    _, pruning_score = gaussian_scorer(camlist, gaussians, pipe, bg, opt)
    prune_order = _target_gaussian_prune_order(opt)
    pruned_count = gaussians.prune_to_target_count(
        target_count,
        pruning_score=pruning_score,
        order=prune_order,
    )
    torch.cuda.synchronize()
    elapsed = time.time() - budget_start
    prefix = "[ITER {}] ".format(iteration) if iteration is not None else ""
    print(
        "{}{} Gaussian prune: {} -> {} (removed {}, target {}, order {})".format(
            prefix,
            label,
            current_count,
            gaussians._xyz.shape[0],
            pruned_count,
            target_count,
            prune_order,
        )
    )
    return pruned_count, elapsed


def _post_prune_finetune_iterations(opt):
    return max(0, int(getattr(opt, "post_prune_finetune_iterations", 0) or 0))


def _post_prune_finetune_interval(opt, attr_name):
    value = int(getattr(opt, attr_name, 0) or 0)
    return max(1, value) if value > 0 else None


def _post_prune_finetune_lr_step(opt, iteration, local_iteration):
    lr_mode = str(getattr(opt, "post_prune_finetune_lr_mode", "continue") or "continue").lower()
    if lr_mode == "continue":
        return iteration
    if lr_mode in ("local", "restart"):
        return local_iteration
    raise ValueError(
        "Unsupported post_prune_finetune_lr_mode {!r}. Available: continue, local, restart.".format(lr_mode)
    )


def _should_run_post_prune_finetune(opt, target_pruned_count, staged_pruned_count):
    trigger = str(getattr(opt, "post_prune_finetune_trigger", "final_prune") or "final_prune").lower()
    final_pruned = target_pruned_count > 0
    staged_pruned = staged_pruned_count > 0
    if trigger == "final_prune":
        return final_pruned
    if trigger == "staged_prune":
        return staged_pruned
    if trigger == "any_prune":
        return final_pruned or staged_pruned
    if trigger == "always":
        return True
    raise ValueError(
        "Unsupported post_prune_finetune_trigger {!r}. Available: final_prune, staged_prune, any_prune, always.".format(
            trigger
        )
    )


def _zero_optimizer_gradients(gaussians, opt):
    if opt.optimizer_type == "default":
        gaussians.optimizer.zero_grad(set_to_none=True)
        gaussians.shoptimizer.zero_grad(set_to_none=True)
    elif opt.optimizer_type == "sparse_adam":
        gaussians.optimizer.zero_grad(set_to_none=True)


def _run_post_prune_finetune(scene, gaussians, pipe, bg, opt, start_iteration, finetune_iterations):
    if finetune_iterations <= 0:
        return 0.0, start_iteration

    _zero_optimizer_gradients(gaussians, opt)

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    optim_start = torch.cuda.Event(enable_timing=True)
    optim_end = torch.cuda.Event(enable_timing=True)
    total_time = 0.0
    ema_loss_for_log = 0.0
    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    progress_bar = tqdm(range(finetune_iterations), desc="Post-prune fine-tune")
    last_progress_update = 0
    step_interval = _post_prune_finetune_interval(opt, "post_prune_finetune_step_interval")
    sh_step_interval = _post_prune_finetune_interval(opt, "post_prune_finetune_sh_step_interval")
    lr_scale = max(0.0, float(getattr(opt, "post_prune_finetune_lr_scale", 1.0) or 0.0))

    for local_iteration in range(1, finetune_iterations + 1):
        iteration = start_iteration + local_iteration
        iter_start.record()

        lr_step = _post_prune_finetune_lr_step(opt, iteration, local_iteration)
        gaussians.update_learning_rate(iteration, lr_step=lr_step, lr_scale=lr_scale)
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        _ = viewpoint_indices.pop(rand_idx)

        render_pkg = render_fastgs(viewpoint_cam, gaussians, pipe, bg, opt.mult)
        image = render_pkg["render"]
        radii = render_pkg["radii"]

        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        Ll2 = l2_loss(image, gt_image)
        ssim_value = fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        l2_blend = max(0.0, min(1.0, float(getattr(opt, "lambda_l2", 0.0) or 0.0)))
        pixel_loss = (1.0 - l2_blend) * Ll1 + l2_blend * Ll2
        loss = (1.0 - opt.lambda_dssim) * pixel_loss + opt.lambda_dssim * (1.0 - ssim_value)
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if local_iteration % 10 == 0 or local_iteration == finetune_iterations:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(local_iteration - last_progress_update)
                last_progress_update = local_iteration
            if local_iteration == finetune_iterations:
                progress_bar.close()

            iter_time = iter_start.elapsed_time(iter_end)
            optim_start.record()
            if opt.optimizer_type == "default":
                gaussians.optimizer_step(
                    iteration,
                    step_interval=step_interval,
                    sh_step_interval=sh_step_interval,
                )
            elif opt.optimizer_type == "sparse_adam":
                visible = radii > 0
                gaussians.optimizer.step(visible, radii.shape[0])
                gaussians.optimizer.zero_grad(set_to_none=True)
            optim_end.record()
            torch.cuda.synchronize()
            optim_time = optim_start.elapsed_time(optim_end)
            total_time += (iter_time + optim_time) / 1e3

    save_iteration = start_iteration + finetune_iterations
    print("\n[ITER {}] Saving post-prune fine-tuned Gaussians".format(save_iteration))
    scene.save(save_iteration)
    return total_time, save_iteration


def training(
    dataset,
    opt,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
    websockets,
    start_pointcloud_iteration=0,
):
    require_runtime_imports()
    from vfm_gs.scorers import get_scorer
    from vfm_gs.utils.fast_utils import sampling_cameras

    first_iter = 0
    start_pointcloud_iteration = int(start_pointcloud_iteration or 0)
    scorer_name = getattr(opt, "scorer", "fastgs_photometric")
    if getattr(opt, "vfm_enable", False):
        scorer_name = "vfm_topology_scorer"
    gaussian_scorer = get_scorer(scorer_name)
    scorer_preflight = getattr(gaussian_scorer, "preflight", None)
    if scorer_preflight is not None:
        scorer_preflight(dataset, opt)
    print("Using Gaussian scorer: {}".format(scorer_name))

    target_gaussian_count = _target_gaussian_count(opt)
    effective_prune_min_gaussian_count = _prune_min_gaussian_count(opt)
    opt.prune_min_gaussian_count = effective_prune_min_gaussian_count
    if effective_prune_min_gaussian_count > 0:
        print(
            "Effective prune_min_gaussian_count: {} (target {}, ratio {})".format(
                effective_prune_min_gaussian_count,
                target_gaussian_count,
                getattr(opt, "prune_min_gaussian_target_ratio", 0.0),
            )
        )

    tb_writer = prepare_output_and_logger(dataset, opt, pipe)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians, load_iteration=start_pointcloud_iteration if start_pointcloud_iteration > 0 else None)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    elif start_pointcloud_iteration > 0:
        first_iter = start_pointcloud_iteration

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))

    # record time
    optim_start = torch.cuda.Event(enable_timing=True)
    optim_end = torch.cuda.Event(enable_timing=True)
    total_time = 0.0

    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    bg = torch.rand((3), device="cuda") if opt.random_background else background
    target_gaussian_staged = bool(getattr(opt, "target_gaussian_staged", False))
    staged_target_gaussian_count = _staged_target_gaussian_count(opt)
    target_gaussian_stage_after_densify = bool(getattr(opt, "target_gaussian_stage_after_densify", False))
    staged_pruned_count = 0

    for iteration in range(first_iter, opt.iterations + 1):

        if websockets:
            if network_gui_ws.curr_id >= 0 and network_gui_ws.curr_id < len(scene.getTrainCameras()):
                cam = scene.getTrainCameras()[network_gui_ws.curr_id]
                net_image = render_fastgs(cam, gaussians, pipe, background, opt.mult, 1.0)["render"]
                network_gui_ws.latest_width = cam.image_width
                network_gui_ws.latest_height = cam.image_height
                network_gui_ws.latest_result = net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())

        iter_start.record()
        
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        _ = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        render_pkg = render_fastgs(viewpoint_cam, gaussians, pipe, bg, opt.mult)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        Ll2 = l2_loss(image, gt_image)
        ssim_value = fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        l2_blend = max(0.0, min(1.0, float(getattr(opt, "lambda_l2", 0.0) or 0.0)))
        pixel_loss = (1.0 - l2_blend) * Ll1 + l2_blend * Ll2
        loss = (1.0 - opt.lambda_dssim) * pixel_loss + opt.lambda_dssim * (1.0 - ssim_value)
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            iter_time = iter_start.elapsed_time(iter_end)
            # Log and save
            # training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_time, testing_iterations, scene, render_fastgs, (pipe, background, opt.mult))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            
            optim_start.record()
            
            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    my_viewpoint_stack = scene.getTrainCameras().copy()
                    camlist = sampling_cameras(my_viewpoint_stack)

                    # The multiview consistent densification of fastgs
                    opt.current_iteration = iteration
                    importance_score, pruning_score = gaussian_scorer(camlist, gaussians, pipe, bg, opt, DENSIFY=True)
                    gaussians.densify_and_prune_fastgs(max_screen_size = size_threshold, 
                                                min_opacity = 0.005, 
                                                extent = scene.cameras_extent, 
                                                radii=radii,
                                                args = opt,
                                                importance_score = importance_score,
                                                pruning_score = pruning_score)
                    if _should_run_staged_target_prune(opt, iteration, staged_target_gaussian_count):
                        pruned_count, budget_time = _prune_to_target_budget(
                            scene,
                            gaussians,
                            gaussian_scorer,
                            pipe,
                            bg,
                            opt,
                            staged_target_gaussian_count,
                            "Staged target",
                            iteration=iteration,
                        )
                        staged_pruned_count += pruned_count
                        total_time += budget_time

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # The multiview consistent pruning of fastgs. We do it every 3k iterations after 15k
            # In this stage, the model converge basically. So we can prune more aggressively without degrading rendering quality.
            # You can check the rendering results of 20K iterations in arxiv version (https://arxiv.org/abs/2511.04283), the rendering quality is already very good.
            if (
                bool(getattr(opt, "final_prune_enabled", True))
                and iteration % 3000 == 0
                and iteration > 15_000
                and iteration < 30_000
            ):
                my_viewpoint_stack = scene.getTrainCameras().copy()
                camlist = sampling_cameras(my_viewpoint_stack)

                opt.current_iteration = iteration
                _, pruning_score = gaussian_scorer(camlist, gaussians, pipe, bg, opt)
                gaussians.final_prune_fastgs(
                    min_opacity=0.1,
                    pruning_score=pruning_score,
                    min_gaussian_count=getattr(opt, "prune_min_gaussian_count", 0),
                    remove_fraction=getattr(opt, "final_prune_remove_fraction", 1.0),
                )

            if (
                target_gaussian_staged
                and target_gaussian_stage_after_densify
                and iteration >= opt.densify_until_iter
                and _should_run_staged_target_prune(opt, iteration, staged_target_gaussian_count)
            ):
                pruned_count, budget_time = _prune_to_target_budget(
                    scene,
                    gaussians,
                    gaussian_scorer,
                    pipe,
                    bg,
                    opt,
                    staged_target_gaussian_count,
                    "Post-densify staged target",
                    iteration=iteration,
                )
                staged_pruned_count += pruned_count
                total_time += budget_time
        
            # Optimization step
            if iteration < opt.iterations:
                if opt.optimizer_type == "default":
                    gaussians.optimizer_step(iteration)
                elif opt.optimizer_type == "sparse_adam":
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)

            # record time
            optim_end.record()
            torch.cuda.synchronize()
            optim_time = optim_start.elapsed_time(optim_end)
            total_time += (iter_time + optim_time) / 1e3

    target_pruned_count = 0
    if target_gaussian_count > 0:
        current_count = gaussians._xyz.shape[0]
        if current_count > target_gaussian_count:
            target_pruned_count, budget_time = _prune_to_target_budget(
                scene,
                gaussians,
                gaussian_scorer,
                pipe,
                bg,
                opt,
                target_gaussian_count,
                "Target",
            )
            total_time += budget_time
            if opt.iterations in saving_iterations:
                print("\n[ITER {}] Saving target-pruned Gaussians".format(opt.iterations))
                scene.save(opt.iterations)
        else:
            print("Target Gaussian prune skipped: {} <= {}".format(current_count, target_gaussian_count))

    post_prune_finetune_iterations = _post_prune_finetune_iterations(opt)
    if post_prune_finetune_iterations > 0:
        if _should_run_post_prune_finetune(opt, target_pruned_count, staged_pruned_count):
            finetune_time, finetune_save_iteration = _run_post_prune_finetune(
                scene,
                gaussians,
                pipe,
                bg,
                opt,
                opt.iterations,
                post_prune_finetune_iterations,
            )
            total_time += finetune_time
            print(
                "Post-prune fine-tune complete: {} iterations, saved iteration {}".format(
                    post_prune_finetune_iterations,
                    finetune_save_iteration,
                )
            )
        else:
            print("Post-prune fine-tune skipped: configured prune trigger did not fire")

    print(f"Gaussian number: {gaussians._xyz.shape[0]}")
    print(f"Training time: {total_time}")
    
def _namespace_from_groups(*groups):
    values = {}
    for group in groups:
        if group is not None:
            values.update(vars(group))
    return Namespace(**values)


def prepare_output_and_logger(args, opt=None, pipe=None):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str)
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(_namespace_from_groups(args, opt, pipe)))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test, ssim_test, lpips_test = 0.0, 0.0, 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0)).mean().double()
                    lpips_test += lpips(image, gt_image, net_type='vgg').mean().double()
                psnr_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    pre_parser = ArgumentParser(add_help=False)
    pre_parser.add_argument("--variant", default=DEFAULT_VARIANT)
    pre_parser.add_argument("--config", default=None)
    pre_args, _ = pre_parser.parse_known_args(argv)

    if any(arg in ("-h", "--help") for arg in argv):
        train_config = {"name": pre_args.variant, "scorer": "fastgs_photometric", "training_args": {}}
    else:
        train_config = load_train_config(pre_args.variant, pre_args.config)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--variant", default=train_config.get("name", pre_args.variant))
    parser.add_argument("--config", default=pre_args.config)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--start_pointcloud_iteration", type=int, default=0)
    parser.add_argument("--websockets", action='store_true', default=False)
    parser.add_argument("--benchmark_dir", type=str, default=None)
    apply_argparse_defaults(parser, train_config)
    args = parser.parse_args(argv)
    require_runtime_imports()
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    if(args.websockets):
        network_gui_ws.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    training(
        lp.extract(args), 
        op.extract(args), 
        pp.extract(args), 
        args.test_iterations, 
        args.save_iterations, 
        args.checkpoint_iterations, 
        args.start_checkpoint, 
        args.debug_from, 
        args.websockets,
        args.start_pointcloud_iteration,
    )

    # All done
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
