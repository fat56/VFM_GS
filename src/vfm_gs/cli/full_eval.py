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

import os
import time
from argparse import ArgumentParser


mipnerf360_outdoor_scenes = ["bicycle", "flowers", "garden", "stump", "treehill"]
mipnerf360_indoor_scenes = ["room", "counter", "kitchen", "bonsai"]
tanks_and_temples_scenes = ["truck", "train"]
deep_blending_scenes = ["drjohnson", "playroom"]


def all_scenes():
    scenes = []
    scenes.extend(mipnerf360_outdoor_scenes)
    scenes.extend(mipnerf360_indoor_scenes)
    scenes.extend(tanks_and_temples_scenes)
    scenes.extend(deep_blending_scenes)
    return scenes


def run_cmd(command, args):
    print(command)
    if not args.dry_run:
        os.system(command)


def train_scene_commands(args, mode_suffix, variant, python_bin):
    common_args = " --quiet --eval --test_iterations -1 "
    common_args += " --optimizer_type {}".format(args.optimizer_type)
    common_args += " --variant {}".format(variant)

    for scene in mipnerf360_outdoor_scenes:
        source = args.mipnerf360 + "/" + scene
        output = args.output_path + "/" + "{}_{}".format(scene, mode_suffix)
        yield "{} -m vfm_gs.cli.train -s {} -i images_4 -m {} {}".format(python_bin, source, output, common_args)

    for scene in mipnerf360_indoor_scenes:
        source = args.mipnerf360 + "/" + scene
        output = args.output_path + "/" + "{}_{}".format(scene, mode_suffix)
        yield "{} -m vfm_gs.cli.train -s {} -i images_2 -m {} {}".format(python_bin, source, output, common_args)

    for scene in tanks_and_temples_scenes:
        source = args.tanksandtemples + "/" + scene
        output = args.output_path + "/" + "{}_{}".format(scene, mode_suffix)
        yield "{} -m vfm_gs.cli.train -s {} -m {} {}".format(python_bin, source, output, common_args)

    for scene in deep_blending_scenes:
        source = args.deepblending + "/" + scene
        output = args.output_path + "/" + "{}_{}".format(scene, mode_suffix)
        yield "{} -m vfm_gs.cli.train -s {} -m {} {}".format(python_bin, source, output, common_args)


def main(argv=None):
    parser = ArgumentParser(description="Full evaluation script parameters")
    parser.add_argument("--skip_training", action="store_true")
    parser.add_argument("--skip_rendering", action="store_true")
    parser.add_argument("--skip_metrics", action="store_true")
    parser.add_argument("--output_path", default="./eval")
    parser.add_argument("--mode", type=str, default="big", choices=["budget", "big"])
    parser.add_argument("--optimizer_type", type=str, default="default")
    parser.add_argument("--sh_lower", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--mipnerf360", "-m360", type=str)
    parser.add_argument("--tanksandtemples", "-tat", type=str)
    parser.add_argument("--deepblending", "-db", type=str)
    args = parser.parse_args(argv)

    if not args.skip_training or not args.skip_rendering:
        missing = [
            name
            for name in ("mipnerf360", "tanksandtemples", "deepblending")
            if getattr(args, name) is None
        ]
        if missing:
            parser.error("Missing dataset roots: {}".format(", ".join(missing)))

    variant = "fastgs_big" if args.mode == "big" else "fastgs_baseline"
    mode_suffix = "big" if args.mode == "big" else "budget"
    python_bin = os.environ.get("PYTHON", "python3")

    timings = {}
    if not args.skip_training:
        os.makedirs(args.output_path, exist_ok=True)
        start_time = time.time()
        for command in train_scene_commands(args, mode_suffix, variant, python_bin):
            run_cmd(command, args)
        timings["training"] = (time.time() - start_time) / 60.0

    if timings and not args.dry_run:
        with open(os.path.join(args.output_path, "timing.txt"), "w", encoding="utf-8") as file:
            for name, minutes in sorted(timings.items()):
                file.write("{}: {} minutes\n".format(name, minutes))

    if not args.skip_rendering:
        for scene in all_scenes():
            output_path = args.output_path + "/" + scene + "_" + mode_suffix
            run_cmd("{} -m vfm_gs.cli.render -m {}".format(python_bin, output_path), args)

    if not args.skip_metrics:
        for scene in all_scenes():
            output_path = args.output_path + "/" + scene + "_" + mode_suffix
            run_cmd("{} -m vfm_gs.cli.metrics -m {}".format(python_bin, output_path), args)


if __name__ == "__main__":
    main()
