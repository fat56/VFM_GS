from __future__ import annotations

import importlib.metadata
import json
import math
import platform
from argparse import ArgumentParser


DINOV2_VARIANTS = {
    "dinov2_vits14": {"params_m": 21, "dim": 384},
    "dinov2_vitb14": {"params_m": 86, "dim": 768},
    "dinov2_vitl14": {"params_m": 300, "dim": 1024},
    "dinov2_vitg14": {"params_m": 1100, "dim": 1536},
}


OPTIONAL_PACKAGES = {
    "transformers": "transformers",
    "timm": "timm",
    "xformers": "xformers",
    "opencv-python": "cv2",
}


def _package_version(package_name):
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def probe_environment():
    result = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": None,
        "cuda_available": False,
        "cuda": None,
        "gpu_name": None,
        "gpu_total_gb": None,
    }
    try:
        import torch
    except Exception as exc:
        result["torch_error"] = str(exc)
        return result

    result["torch"] = getattr(torch, "__version__", None)
    result["cuda_available"] = bool(torch.cuda.is_available())
    result["cuda"] = getattr(torch.version, "cuda", None)
    if result["cuda_available"]:
        result["gpu_name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        result["gpu_total_gb"] = round(props.total_memory / 1024**3, 2)
    return result


def probe_packages():
    packages = {}
    for package_name, import_name in OPTIONAL_PACKAGES.items():
        version = _package_version(package_name)
        packages[package_name] = {
            "installed": version is not None,
            "version": version,
            "import": import_name,
        }
    return packages


def estimate_dinov2(width, height, num_images, patch_size=14):
    grid_w = math.ceil(width / patch_size)
    grid_h = math.ceil(height / patch_size)
    tokens = grid_w * grid_h
    estimates = {}
    for name, spec in DINOV2_VARIANTS.items():
        dim = spec["dim"]
        feature_mb_float32 = tokens * dim * 4 / 1024**2
        feature_gb_float32_dataset = feature_mb_float32 * num_images / 1024
        estimates[name] = {
            "params_m": spec["params_m"],
            "dim": dim,
            "tokens_per_image": tokens,
            "feature_mb_per_image_float32": round(feature_mb_float32, 2),
            "feature_gb_dataset_float32": round(feature_gb_float32_dataset, 2),
        }
    return {
        "width": width,
        "height": height,
        "num_images": num_images,
        "patch_size": patch_size,
        "grid": [grid_h, grid_w],
        "variants": estimates,
    }


def make_recommendation(environment, packages, estimates):
    torch_version = environment.get("torch") or ""
    gpu_total_gb = environment.get("gpu_total_gb") or 0
    recommendations = []

    if torch_version.startswith("1.12"):
        recommendations.append(
            "Prefer torch.hub DINOv2 integration over current Transformers, because this environment uses PyTorch 1.12."
        )
    if not packages["transformers"]["installed"]:
        recommendations.append("Do not add Transformers as the first path until its torch requirement is checked in an isolated env.")
    if gpu_total_gb and gpu_total_gb >= 20:
        recommendations.append("ViT-S/14 or ViT-B/14 feature extraction at cache-building time is plausible on this GPU.")
    recommendations.append("Start with max_width around 518-640 and cache projected/normalized features, not raw full-resolution tokens.")
    recommendations.append("Keep cached_edge_l1 as the deterministic smoke fallback for CI and environment checks.")
    return recommendations


def main(argv=None):
    parser = ArgumentParser(description="Probe VFM backend dependency and cache-size feasibility.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=426)
    parser.add_argument("--num_images", type=int, default=194)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    environment = probe_environment()
    packages = probe_packages()
    estimates = estimate_dinov2(args.width, args.height, args.num_images)
    result = {
        "environment": environment,
        "packages": packages,
        "dinov2_estimates": estimates,
        "recommendations": make_recommendation(environment, packages, estimates),
    }

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    print("Python: {}".format(environment["python"]))
    print("Torch: {}".format(environment.get("torch")))
    print("CUDA: {} ({})".format(environment.get("cuda"), environment.get("gpu_name")))
    print("GPU total: {} GB".format(environment.get("gpu_total_gb")))
    for package_name, package in packages.items():
        status = package["version"] if package["installed"] else "missing"
        print("{}: {}".format(package_name, status))
    print("DINOv2 estimate for {}x{} / {} images:".format(args.width, args.height, args.num_images))
    for name, estimate in estimates["variants"].items():
        print(
            "  {}: {} tokens, {} MB/image fp32, {} GB dataset fp32".format(
                name,
                estimate["tokens_per_image"],
                estimate["feature_mb_per_image_float32"],
                estimate["feature_gb_dataset_float32"],
            )
        )
    print("Recommendations:")
    for recommendation in result["recommendations"]:
        print("- {}".format(recommendation))


if __name__ == "__main__":
    main()
