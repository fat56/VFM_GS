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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.separate_sh = True
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.antialiasing = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025 
        self.shfeature_lr = 0.005 
        self.opacity_lr = 0.025 
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.001
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        
        # fastgs parameters
        self.loss_thresh = 0.1
        self.grad_abs_thresh = 0.0012  
        self.highfeature_lr = 0.005
        self.lowfeature_lr = 0.0025
        self.grad_thresh = 0.0002
        self.dense = 0.001
        self.mult = 0.5      # multiplier for the compact box to control the tile number of each splat

        self.random_background = False
        self.optimizer_type = "default"
        self.scorer = "fastgs_photometric"
        self.vfm_enable = False
        self.vfm_backend = "mock_l1"
        self.vfm_cache_dir = ""
        self.vfm_loss_thresh = 0.5
        self.vfm_metric_map_mode = "threshold"
        self.vfm_metric_percentile = 0.85
        self.vfm_metric_topk = 0.15
        self.vfm_metric_soft_levels = 3
        self.vfm_descriptor_token_smooth_kernel = 1
        self.vfm_weight = 0.25
        self.vfm_importance_weight = 1.0
        self.vfm_importance_mode = "max"
        self.vfm_importance_normalizer = "none"
        self.vfm_support_ratio_power = 1.0
        self.vfm_support_min_count = 1.0
        self.vfm_prune_protect_weight = 0.0
        self.vfm_prune_protect_mode = "vfm"
        self.vfm_prune_protect_min_count = 1.0
        self.vfm_prune_protect_power = 1.0
        self.vfm_use_albedo_sh0 = True
        self.vfm_dinov2_repo = ""
        self.vfm_dinov2_device = "cuda"
        self.target_gaussian_count = 0
        self.target_gaussian_staged = False
        self.target_gaussian_stage_margin = 1.2
        self.target_gaussian_stage_start = 0
        self.target_gaussian_stage_interval = 500
        self.prune_min_gaussian_count = 0
        self.post_prune_finetune_iterations = 0
        self.post_prune_finetune_step_interval = 0
        self.post_prune_finetune_sh_step_interval = 0
        self.post_prune_finetune_lr_mode = "continue"
        self.post_prune_finetune_lr_scale = 1.0
        self.post_prune_finetune_trigger = "final_prune"
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
