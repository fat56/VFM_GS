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

import logging
import os
import shutil
from argparse import ArgumentParser


def run_command(command):
    exit_code = os.system(command)
    if exit_code != 0:
        logging.error("Command failed with code %s: %s", exit_code, command)
        raise SystemExit(exit_code)


def main(argv=None):
    parser = ArgumentParser("Colmap converter")
    parser.add_argument("--no_gpu", action="store_true")
    parser.add_argument("--skip_matching", action="store_true")
    parser.add_argument("--source_path", "-s", required=True, type=str)
    parser.add_argument("--camera", default="OPENCV", type=str)
    parser.add_argument("--colmap_executable", default="", type=str)
    parser.add_argument("--resize", action="store_true")
    parser.add_argument("--magick_executable", default="", type=str)
    args = parser.parse_args(argv)

    colmap_command = '"{}"'.format(args.colmap_executable) if args.colmap_executable else "colmap"
    magick_command = '"{}"'.format(args.magick_executable) if args.magick_executable else "magick"
    use_gpu = 0 if args.no_gpu else 1

    if not args.skip_matching:
        os.makedirs(args.source_path + "/distorted/sparse", exist_ok=True)

        feature_extraction = (
            colmap_command
            + " feature_extractor "
            + "--database_path "
            + args.source_path
            + "/distorted/database.db "
            + "--image_path "
            + args.source_path
            + "/input "
            + "--ImageReader.single_camera 1 "
            + "--ImageReader.camera_model "
            + args.camera
            + " "
            + "--SiftExtraction.use_gpu "
            + str(use_gpu)
        )
        run_command(feature_extraction)

        feature_matching = (
            colmap_command
            + " exhaustive_matcher "
            + "--database_path "
            + args.source_path
            + "/distorted/database.db "
            + "--SiftMatching.use_gpu "
            + str(use_gpu)
        )
        run_command(feature_matching)

        mapper = (
            colmap_command
            + " mapper "
            + "--database_path "
            + args.source_path
            + "/distorted/database.db "
            + "--image_path "
            + args.source_path
            + "/input "
            + "--output_path "
            + args.source_path
            + "/distorted/sparse "
            + "--Mapper.ba_global_function_tolerance=0.000001"
        )
        run_command(mapper)

    undistort = (
        colmap_command
        + " image_undistorter "
        + "--image_path "
        + args.source_path
        + "/input "
        + "--input_path "
        + args.source_path
        + "/distorted/sparse/0 "
        + "--output_path "
        + args.source_path
        + " "
        + "--output_type COLMAP"
    )
    run_command(undistort)

    files = os.listdir(args.source_path + "/sparse")
    os.makedirs(args.source_path + "/sparse/0", exist_ok=True)
    for file_name in files:
        if file_name == "0":
            continue
        source_file = os.path.join(args.source_path, "sparse", file_name)
        destination_file = os.path.join(args.source_path, "sparse", "0", file_name)
        shutil.move(source_file, destination_file)

    if args.resize:
        print("Copying and resizing...")
        for suffix, ratio in (("images_2", "50%"), ("images_4", "25%"), ("images_8", "12.5%")):
            os.makedirs(os.path.join(args.source_path, suffix), exist_ok=True)
            for file_name in os.listdir(args.source_path + "/images"):
                source_file = os.path.join(args.source_path, "images", file_name)
                destination_file = os.path.join(args.source_path, suffix, file_name)
                shutil.copy2(source_file, destination_file)
                run_command(magick_command + " mogrify -resize " + ratio + " " + destination_file)

    print("Done.")


if __name__ == "__main__":
    main()
