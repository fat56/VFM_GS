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

import math

import torch
import numpy as np
from vfm_gs.utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation, identity_gate
from torch import nn
import os
from vfm_gs.utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from vfm_gs.utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from vfm_gs.utils.graphics_utils import BasicPointCloud
from vfm_gs.utils.general_utils import strip_symmetric, build_scaling_rotation

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def modify_functions(self):
        old_opacities = self.get_opacity.clone()
        self.opacity_activation = torch.abs
        self.inverse_opacity_activation = identity_gate
        self._opacity = self.opacity_activation(old_opacities)

    def __init__(self, sh_degree, optimizer_type="default"):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.xyz_gradient_accum_abs = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.shoptimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self, optimizer_type):
        if optimizer_type == "default":
            return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.xyz_gradient_accum_abs,
            self.denom,
            self.optimizer.state_dict(),
            self.shoptimizer.state_dict(),
            self.spatial_lr_scale,
        )
        else:
            return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.xyz_gradient_accum_abs,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum,
        xyz_gradient_accum_abs, 
        denom,
        opt_dict, 
        shopt_dict,
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.xyz_gradient_accum_abs = xyz_gradient_accum_abs
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
        self.shoptimizer.load_state_dict(shopt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.lowfeature_lr, "name": "f_dc"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]
        sh_l = [{'params': [self._features_rest], 'lr': training_args.highfeature_lr / 20.0, "name": "f_rest"}]

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
            self.shoptimizer = torch.optim.Adam(sh_l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            self.optimizer = SparseGaussianAdam(l + sh_l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration, lr_step=None, lr_scale=1.0):
        ''' Learning rate scheduling per step '''
        scheduler_step = iteration if lr_step is None else lr_step
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(scheduler_step) * lr_scale
                param_group['lr'] = lr
                return lr

    def optimizer_step(self, iteration, step_interval=None, sh_step_interval=None):
        ''' An optimization schdeuler. The goal is similar to the sparse Adam of taming 3dgs.'''
        if step_interval is not None or sh_step_interval is not None:
            default_step, default_sh_step = self._optimizer_step_flags(iteration)
            do_step = default_step
            do_sh_step = default_sh_step
            if step_interval is not None:
                step_interval = max(1, int(step_interval))
                do_step = iteration % step_interval == 0
            if sh_step_interval is not None:
                sh_step_interval = max(1, int(sh_step_interval))
                do_sh_step = iteration % sh_step_interval == 0
            if do_step:
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none = True)
            if do_sh_step:
                self.shoptimizer.step()
                self.shoptimizer.zero_grad(set_to_none = True)
            return

        if iteration <= 15000:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none = True)
            if iteration % 16 == 0:
                self.shoptimizer.step()
                self.shoptimizer.zero_grad(set_to_none = True)
        elif iteration <= 20000:
            if iteration % 32 ==0:
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none = True)
                self.shoptimizer.step()
                self.shoptimizer.zero_grad(set_to_none = True)
        else:
            if iteration % 64 ==0:
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none = True)
                self.shoptimizer.step()
                self.shoptimizer.zero_grad(set_to_none = True)

    def _optimizer_step_flags(self, iteration):
        if iteration <= 15000:
            return True, iteration % 16 == 0
        if iteration <= 20000:
            should_step = iteration % 32 == 0
            return should_step, should_step
        should_step = iteration % 64 == 0
        return should_step, should_step

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        optimizers = [self.optimizer]
        if self.shoptimizer: optimizers.append(self.shoptimizer)

        for opt in optimizers:
            for group in opt.param_groups:
                stored_state = opt.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                    del opt.state[group['params'][0]]
                    group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                    opt.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.xyz_gradient_accum_abs = self.xyz_gradient_accum_abs[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        if self.tmp_radii is not None:
            self.tmp_radii = self.tmp_radii[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        optimizers = [self.optimizer]
        if self.shoptimizer: optimizers.append(self.shoptimizer)

        for opt in optimizers:
            for group in opt.param_groups:
                assert len(group["params"]) == 1
                extension_tensor = tensors_dict[group["name"]]
                stored_state = opt.state.get(group['params'][0], None)
                if stored_state is not None:

                    stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                    stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                    del opt.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    opt.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")  # abs
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split_fastgs(self, metric_mask, filter, N=2):
        n_init_points = self.get_xyz.shape[0]

        selected_pts_mask = torch.zeros((n_init_points), dtype=bool, device="cuda")
        mask = torch.logical_and(metric_mask, filter)
        selected_pts_mask[:mask.shape[0]] = mask

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone_fastgs(self, metric_mask, filter):
        selected_pts_mask = torch.logical_and(metric_mask, filter)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_tmp_radii = self.tmp_radii[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii)

    def _densify_metric_threshold(self, args):
        base_threshold = float(getattr(args, "densify_metric_thresh", 5.0) or 5.0)
        budget_count = int(getattr(args, "densify_budget_count", 0) or 0)
        if budget_count <= 0:
            return base_threshold

        current_count = int(self.get_xyz.shape[0])
        start_ratio = float(getattr(args, "densify_budget_start_ratio", 0.9) or 0.9)
        start_ratio = min(max(start_ratio, 0.0), 0.999)
        start_count = int(budget_count * start_ratio)
        if current_count <= start_count:
            return base_threshold

        max_threshold = float(getattr(args, "densify_budget_max_metric_thresh", base_threshold) or base_threshold)
        max_threshold = max(base_threshold, max_threshold)
        if budget_count <= start_count:
            return max_threshold

        progress = (current_count - start_count) / float(budget_count - start_count)
        progress = min(max(progress, 0.0), 1.0)
        curve = str(getattr(args, "densify_budget_curve", "linear") or "linear").lower()
        if curve == "linear":
            pass
        elif curve == "quadratic":
            progress = progress * progress
        elif curve == "sqrt":
            progress = math.sqrt(progress)
        else:
            raise ValueError(
                "Unsupported densify_budget_curve {!r}. Available: linear, quadratic, sqrt.".format(curve)
            )
        return base_threshold + progress * (max_threshold - base_threshold)

    def _cap_densify_candidates(self, args, importance_score, metric_mask, all_clones, all_splits):
        if not getattr(args, "densify_budget_candidate_cap", False):
            return metric_mask

        budget_count = int(getattr(args, "densify_budget_count", 0) or 0)
        if budget_count <= 0:
            return metric_mask

        remaining = budget_count - int(self.get_xyz.shape[0])
        if remaining <= 0:
            return torch.zeros_like(metric_mask, dtype=torch.bool)

        candidates = torch.logical_and(metric_mask, torch.logical_or(all_clones, all_splits))
        candidate_count = int(candidates.sum().item())
        if candidate_count <= remaining:
            return metric_mask

        cap_mode = str(getattr(args, "densify_budget_candidate_cap_mode", "global") or "global")
        candidate_indices = candidates.nonzero(as_tuple=False).squeeze(1)
        candidate_scores = importance_score[candidate_indices]
        if cap_mode == "branch_type":
            clone_indices = torch.logical_and(metric_mask, all_clones).nonzero(as_tuple=False).squeeze(1)
            split_indices = torch.logical_and(metric_mask, all_splits).nonzero(as_tuple=False).squeeze(1)
            keep_mask = self._cap_densify_candidates_branch_type(
                metric_mask,
                clone_indices,
                importance_score[clone_indices],
                split_indices,
                importance_score[split_indices],
                remaining,
            )
            return torch.logical_and(metric_mask, keep_mask)

        if cap_mode == "screen_support":
            support_counts = getattr(args, "densify_budget_support_counts", None)
            if support_counts is not None:
                keep_mask = self._cap_densify_candidates_screen_support(
                    args,
                    metric_mask,
                    candidate_indices,
                    candidate_scores,
                    support_counts[candidate_indices],
                    remaining,
                )
                return torch.logical_and(metric_mask, keep_mask)

        if cap_mode == "spatial_xyz":
            keep_mask = self._cap_densify_candidates_spatial_xyz(
                args,
                metric_mask,
                candidate_indices,
                candidate_scores,
                remaining,
            )
            return torch.logical_and(metric_mask, keep_mask)

        topk_indices = torch.topk(candidate_scores, k=remaining, largest=True, sorted=False).indices
        keep_mask = torch.zeros_like(metric_mask, dtype=torch.bool)
        keep_mask[candidate_indices[topk_indices]] = True
        return torch.logical_and(metric_mask, keep_mask)

    def _rgb_rerank_final_topm_mask(self, args, importance_score, metric_threshold, all_clones, all_splits):
        reference_score = getattr(args, "vfm_rgb_broad_reference_score", None)
        if reference_score is None:
            return None
        if reference_score.shape[0] != importance_score.shape[0]:
            return None

        densifiable = torch.logical_or(all_clones, all_splits)
        reference_candidates = torch.logical_and(reference_score > metric_threshold, densifiable)
        quota = int(reference_candidates.sum().item())
        keep_mask = torch.zeros_like(importance_score, dtype=torch.bool)
        if quota <= 0:
            return keep_mask

        candidates = torch.logical_and(importance_score > 0.0, densifiable)
        candidate_indices = candidates.nonzero(as_tuple=False).squeeze(1)
        candidate_count = int(candidate_indices.numel())
        if candidate_count <= 0:
            return keep_mask
        if candidate_count <= quota:
            keep_mask[candidate_indices] = True
            return keep_mask

        candidate_scores = importance_score[candidate_indices]
        selected = torch.topk(candidate_scores, k=quota, largest=True, sorted=False).indices
        keep_mask[candidate_indices[selected]] = True
        return keep_mask

    def _cap_densify_candidates_branch_type(self, metric_mask, clone_indices, clone_scores, split_indices, split_scores, remaining):
        keep_mask = torch.zeros_like(metric_mask, dtype=torch.bool)
        clone_count = int(clone_indices.numel())
        split_count = int(split_indices.numel())
        total_count = clone_count + split_count
        if total_count <= 0:
            return keep_mask

        clone_quota = int(float(remaining) * float(clone_count) / float(total_count))
        split_quota = int(float(remaining) * float(split_count) / float(total_count))
        allocated = clone_quota + split_quota

        remainders = [
            ("clone", float(remaining) * float(clone_count) / float(total_count) - float(clone_quota)),
            ("split", float(remaining) * float(split_count) / float(total_count) - float(split_quota)),
        ]
        for branch, _ in sorted(remainders, key=lambda item: item[1], reverse=True):
            if allocated >= remaining:
                break
            if branch == "clone" and clone_quota < clone_count:
                clone_quota += 1
                allocated += 1
            elif branch == "split" and split_quota < split_count:
                split_quota += 1
                allocated += 1

        if allocated < remaining:
            if clone_count - clone_quota >= split_count - split_quota:
                first, second = "clone", "split"
            else:
                first, second = "split", "clone"
            for branch in (first, second):
                if allocated >= remaining:
                    break
                if branch == "clone":
                    add = min(remaining - allocated, clone_count - clone_quota)
                    clone_quota += add
                    allocated += add
                else:
                    add = min(remaining - allocated, split_count - split_quota)
                    split_quota += add
                    allocated += add

        if clone_quota > 0:
            selected = torch.topk(clone_scores, k=min(clone_quota, clone_count), largest=True, sorted=False).indices
            keep_mask[clone_indices[selected]] = True
        if split_quota > 0:
            selected = torch.topk(split_scores, k=min(split_quota, split_count), largest=True, sorted=False).indices
            keep_mask[split_indices[selected]] = True

        return keep_mask

    def _cap_densify_candidates_screen_support(
        self,
        args,
        metric_mask,
        candidate_indices,
        candidate_scores,
        candidate_support,
        remaining,
    ):
        keep_mask = torch.zeros_like(metric_mask, dtype=torch.bool)
        support = candidate_support.detach().to(torch.float32)
        support_positive = support[support > 0]
        if support_positive.numel() == 0:
            topk_indices = torch.topk(candidate_scores, k=remaining, largest=True, sorted=False).indices
            keep_mask[candidate_indices[topk_indices]] = True
            return keep_mask

        bins = int(getattr(args, "densify_budget_support_bins", 4) or 4)
        bins = max(1, min(bins, 16))
        quantiles = torch.linspace(0.0, 1.0, bins + 1, device=support.device)
        edges = torch.quantile(support_positive, quantiles)
        bucket_ids = torch.bucketize(support, edges[1:-1], right=False)

        _, inverse_bins, bin_counts = torch.unique(
            bucket_ids,
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )
        raw_quotas = bin_counts.to(torch.float32) * (float(remaining) / float(candidate_indices.numel()))
        quotas = torch.floor(raw_quotas).to(torch.long)
        quotas = torch.minimum(quotas, bin_counts)

        allocated = int(quotas.sum().item())
        if allocated < remaining:
            fractional = raw_quotas - torch.floor(raw_quotas)
            grow_order = torch.argsort(fractional, descending=True)
            for bin_pos in grow_order.tolist():
                if allocated >= remaining:
                    break
                if quotas[bin_pos] >= bin_counts[bin_pos]:
                    continue
                quotas[bin_pos] += 1
                allocated += 1

        if allocated < remaining:
            grow_order = torch.argsort(bin_counts - quotas, descending=True)
            for bin_pos in grow_order.tolist():
                if allocated >= remaining:
                    break
                if quotas[bin_pos] >= bin_counts[bin_pos]:
                    continue
                quotas[bin_pos] += 1
                allocated += 1

        for local_bin, quota in enumerate(quotas.tolist()):
            if quota <= 0:
                continue
            bin_mask = inverse_bins == local_bin
            bin_candidate_indices = candidate_indices[bin_mask]
            quota = min(quota, bin_candidate_indices.numel())
            bin_scores = candidate_scores[bin_mask]
            selected = torch.topk(bin_scores, k=quota, largest=True, sorted=False).indices
            keep_mask[bin_candidate_indices[selected]] = True

        remaining_capacity = remaining - int(keep_mask[candidate_indices].sum().item())
        if remaining_capacity > 0:
            unused_candidate_flags = torch.logical_not(keep_mask[candidate_indices])
            unused_indices = candidate_indices[unused_candidate_flags]
            unused_scores = candidate_scores[unused_candidate_flags]
            quota = min(remaining_capacity, unused_indices.numel())
            if quota > 0:
                selected = torch.topk(unused_scores, k=quota, largest=True, sorted=False).indices
                keep_mask[unused_indices[selected]] = True

        return keep_mask

    def _cap_densify_candidates_spatial_xyz(self, args, metric_mask, candidate_indices, candidate_scores, remaining):
        keep_mask = torch.zeros_like(metric_mask, dtype=torch.bool)
        bins_per_axis = int(getattr(args, "densify_budget_spatial_bins", 8) or 8)
        bins_per_axis = max(1, min(bins_per_axis, 64))

        xyz = self.get_xyz.detach()
        candidate_xyz = xyz[candidate_indices]
        xyz_min = xyz.min(dim=0).values
        xyz_extent = (xyz.max(dim=0).values - xyz_min).clamp_min(1e-6)
        bin_xyz = torch.floor((candidate_xyz - xyz_min) / xyz_extent * bins_per_axis)
        bin_xyz = torch.clamp(bin_xyz, min=0, max=bins_per_axis - 1).to(torch.long)
        bin_ids = bin_xyz[:, 0] * bins_per_axis * bins_per_axis + bin_xyz[:, 1] * bins_per_axis + bin_xyz[:, 2]
        _, inverse_bins, bin_counts = torch.unique(
            bin_ids,
            sorted=False,
            return_inverse=True,
            return_counts=True,
        )

        raw_quotas = bin_counts.to(torch.float32) * (float(remaining) / float(candidate_indices.numel()))
        quotas = torch.floor(raw_quotas).to(torch.long)
        quotas = torch.minimum(quotas, bin_counts)

        allocated = int(quotas.sum().item())
        if allocated < remaining:
            fractional = raw_quotas - torch.floor(raw_quotas)
            grow_order = torch.argsort(fractional, descending=True)
            for bin_pos in grow_order.tolist():
                if allocated >= remaining:
                    break
                if quotas[bin_pos] >= bin_counts[bin_pos]:
                    continue
                quotas[bin_pos] += 1
                allocated += 1

        if allocated < remaining:
            grow_order = torch.argsort(bin_counts - quotas, descending=True)
            for bin_pos in grow_order.tolist():
                if allocated >= remaining:
                    break
                if quotas[bin_pos] >= bin_counts[bin_pos]:
                    continue
                quotas[bin_pos] += 1
                allocated += 1

        for local_bin, quota in enumerate(quotas.tolist()):
            if quota <= 0:
                continue
            bin_mask = inverse_bins == local_bin
            bin_candidate_indices = candidate_indices[bin_mask]
            quota = min(quota, bin_candidate_indices.numel())
            bin_scores = candidate_scores[bin_mask]
            selected = torch.topk(bin_scores, k=quota, largest=True, sorted=False).indices
            keep_mask[bin_candidate_indices[selected]] = True

        remaining_capacity = remaining - int(keep_mask[candidate_indices].sum().item())
        if remaining_capacity > 0:
            unused_candidate_flags = torch.logical_not(keep_mask[candidate_indices])
            unused_indices = candidate_indices[unused_candidate_flags]
            unused_scores = candidate_scores[unused_candidate_flags]
            quota = min(remaining_capacity, unused_indices.numel())
            if quota > 0:
                selected = torch.topk(unused_scores, k=quota, largest=True, sorted=False).indices
                keep_mask[unused_indices[selected]] = True

        return keep_mask

    def densify_and_prune_fastgs(self, max_screen_size, min_opacity, extent, radii, args, importance_score = None, pruning_score = None):
        
        ''' 
            Densification and Pruning based on FastGS criteria:
            1.  The gaussians candidate for densification are selected based on the gradient of their position first.
            2.  Then, based on their average metric score (computed over multiple sampled views), they are either densified (cloned) or split.
                This is our main contribution compared to the vanilla 3DGS.
            3.  Finally, gaussians with low opacity or very large size are pruned.
        '''
        grad_vars = self.xyz_gradient_accum / self.denom
        grad_vars[grad_vars.isnan()] = 0.0
        self.tmp_radii = radii

        grads_abs = self.xyz_gradient_accum_abs / self.denom
        grads_abs[grads_abs.isnan()] = 0.0

        grad_qualifiers = torch.where(torch.norm(grad_vars, dim=-1) >= args.grad_thresh, True, False)
        grad_qualifiers_abs = torch.where(torch.norm(grads_abs, dim=-1) >= args.grad_abs_thresh, True, False)
        clone_qualifiers = torch.max(self.get_scaling, dim=1).values <= args.dense*extent
        split_qualifiers = torch.max(self.get_scaling, dim=1).values > args.dense*extent

        all_clones = torch.logical_and(clone_qualifiers, grad_qualifiers)
        all_splits = torch.logical_and(split_qualifiers, grad_qualifiers_abs)

        # This is our multi-view consisent metric for densification
        # We use this metric to further filter the candidates for densification, which is similar to taming 3dgs.
        metric_threshold = self._densify_metric_threshold(args)
        metric_mask = importance_score > metric_threshold
        if getattr(args, "vfm_rgb_rerank_final_topm", False):
            final_topm_mask = self._rgb_rerank_final_topm_mask(
                args,
                importance_score,
                metric_threshold,
                all_clones,
                all_splits,
            )
            if final_topm_mask is not None:
                metric_mask = final_topm_mask
        metric_mask = self._cap_densify_candidates(args, importance_score, metric_mask, all_clones, all_splits)

        self.densify_and_clone_fastgs(metric_mask, all_clones)
        self.densify_and_split_fastgs(metric_mask, all_splits)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        scores = 1 - pruning_score 
        to_remove = torch.sum(prune_mask)
        remove_budget = int(0.5 * to_remove)

        # Keep diagnostic capacity floors from pruning below a known baseline.
        min_gaussian_count = max(0, int(getattr(args, "prune_min_gaussian_count", 0) or 0))
        if min_gaussian_count > 0:
            current_count = self.get_xyz.shape[0]
            protected_budget = max(0, current_count - min_gaussian_count)
            remove_budget = min(remove_budget, protected_budget)

        if remove_budget:
            n_init_points = self.get_xyz.shape[0]
            padded_importance = torch.zeros((n_init_points), dtype=torch.float32)
            padded_importance[:scores.shape[0]] = 1 / (1e-6 + scores.squeeze())
            selected_pts_mask = torch.zeros_like(padded_importance, dtype=bool, device="cuda")
            sampled_indices = torch.multinomial(padded_importance, remove_budget, replacement=False)
            selected_pts_mask[sampled_indices] = True
            final_prune = torch.logical_and(prune_mask, selected_pts_mask)
            self.prune_points(final_prune)
        
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.8))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]
        tmp_radii = self.tmp_radii
        self.tmp_radii = None

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.xyz_gradient_accum_abs[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, 2:], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def final_prune_fastgs(self, min_opacity, pruning_score = None, min_gaussian_count=0):
        """Final-stage pruning: remove Gaussians based on opacity and multi-view consistency.
        In the final stage we remove Gaussians that have low opacity or that are flagged by
        our multi-view reconstruction consistency metric (provided as `pruning_score`)."""
        prune_mask = (self.get_opacity < min_opacity).squeeze() 
        scores_mask = pruning_score > 0.9
        final_prune = torch.logical_or(prune_mask, scores_mask)
        min_gaussian_count = max(0, int(min_gaussian_count or 0))
        if min_gaussian_count > 0:
            current_count = self.get_xyz.shape[0]
            remove_count = int(final_prune.sum().item())
            max_remove = max(0, current_count - min_gaussian_count)
            if remove_count > max_remove:
                if max_remove <= 0:
                    final_prune = torch.zeros_like(final_prune)
                else:
                    scores = pruning_score.detach().squeeze().to(dtype=torch.float32, device=self.get_xyz.device)
                    if scores.numel() < current_count:
                        padded_scores = torch.zeros((current_count), dtype=torch.float32, device=self.get_xyz.device)
                        padded_scores[: scores.numel()] = scores
                        scores = padded_scores
                    elif scores.numel() > current_count:
                        scores = scores[:current_count]
                    scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
                    candidate_scores = scores.masked_fill(~final_prune, float("-inf"))
                    selected = torch.topk(candidate_scores, max_remove, largest=True).indices
                    protected_prune = torch.zeros_like(final_prune)
                    protected_prune[selected] = True
                    final_prune = protected_prune
        self.prune_points(final_prune)

    def prune_to_target_count(self, target_count, pruning_score=None, order="lowest_score"):
        target_count = int(target_count)
        current_count = self.get_xyz.shape[0]
        if target_count <= 0 or current_count <= target_count:
            return 0

        order = str(order or "lowest_score").lower()
        if order not in ("lowest_score", "highest_score", "lowest_opacity"):
            raise ValueError(
                "Unsupported target Gaussian prune order {!r}. Available: lowest_score, highest_score, lowest_opacity.".format(
                    order
                )
            )

        remove_count = current_count - target_count
        device = self.get_xyz.device
        if order == "lowest_opacity" or pruning_score is None:
            scores = self.get_opacity.detach().squeeze().to(dtype=torch.float32, device=device)
        else:
            scores = pruning_score.detach().squeeze().to(dtype=torch.float32, device=device)
            if scores.numel() < current_count:
                padded_scores = torch.zeros((current_count), dtype=torch.float32, device=device)
                padded_scores[: scores.numel()] = scores
                scores = padded_scores
            elif scores.numel() > current_count:
                scores = scores[:current_count]
            scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

        prune_mask = torch.zeros((current_count), dtype=bool, device=device)
        prune_indices = torch.topk(scores, remove_count, largest=(order == "highest_score")).indices
        prune_mask[prune_indices] = True
        self.prune_points(prune_mask)
        return remove_count
