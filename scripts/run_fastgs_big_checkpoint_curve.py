#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_0001_fastgs_big_eval import (
    DEFAULT_METHOD,
    DEFAULT_RUN_NAME,
    DEFAULT_VFM_CACHE_BACKEND,
    DEFAULT_VFM_CACHE_DEVICE,
    DEFAULT_VFM_CACHE_FEATURE,
    DEFAULT_VFM_CACHE_MAX_WIDTH,
    DEFAULT_VFM_CACHE_STORAGE,
    build_vfm_cache,
    load_metrics_map,
    point_count_at_iteration,
    render_iteration,
    run_metrics,
    scene_overrides,
    train_baseline,
)


PLOT_AVAILABLE = True


def checkpoint_iterations(total_iterations: int, interval: int) -> list[int]:
    interval = max(1, int(interval))
    total_iterations = int(total_iterations)
    values = list(range(interval, total_iterations + 1, interval))
    if not values or values[-1] != total_iterations:
        values.append(total_iterations)
    return sorted(set(values))


def read_checkpoint_rows(
    dataset: str,
    scene: str,
    method: str,
    run_dir: Path,
    iterations: list[int],
) -> list[dict[str, object]]:
    metrics_map = load_metrics_map(run_dir)
    rows: list[dict[str, object]] = []
    for iteration in iterations:
        method_key = "ours_{}".format(iteration)
        metrics = metrics_map.get(method_key)
        if metrics is None:
            raise RuntimeError("Missing metrics for {} in {}".format(method_key, run_dir / "results.json"))
        rows.append(
            {
                "dataset": dataset,
                "scene": scene,
                "method": method,
                "iteration": iteration,
                "psnr": metrics.get("PSNR"),
                "ssim": metrics.get("SSIM"),
                "lpips": metrics.get("LPIPS"),
                "gs_num": point_count_at_iteration(run_dir, iteration),
                "run_dir": str(run_dir),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def svg_escape(value: object) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def fmt_metric(value: float) -> str:
    return "{:.4f}".format(value)


def fmt_gs_million(value: float) -> str:
    return "{:.2f}".format(value)


def _ticks(min_value: float, max_value: float, count: int = 5) -> list[float]:
    if count <= 1 or not math.isfinite(min_value) or not math.isfinite(max_value):
        return [min_value]
    if abs(max_value - min_value) < 1e-12:
        return [min_value for _ in range(count)]
    step = (max_value - min_value) / float(count - 1)
    return [min_value + step * idx for idx in range(count)]


def _panel_svg(
    panel_title: str,
    x_values: list[int],
    y_values: list[float],
    *,
    color: str,
    y_label: str,
    y_formatter,
    width: int,
    height: int,
    left: int,
    top: int,
    panel_width: int,
    panel_height: int,
) -> list[str]:
    margin_left = 58
    margin_right = 16
    margin_top = 28
    margin_bottom = 34
    plot_x = left + margin_left
    plot_y = top + margin_top
    plot_w = panel_width - margin_left - margin_right
    plot_h = panel_height - margin_top - margin_bottom
    x_min = min(x_values)
    x_max = max(x_values)
    y_min = min(y_values)
    y_max = max(y_values)
    if abs(y_max - y_min) < 1e-12:
        y_min -= 1.0
        y_max += 1.0
    else:
        pad = (y_max - y_min) * 0.08
        y_min -= pad
        y_max += pad

    def x_to_px(value: float) -> float:
        if x_max == x_min:
            return plot_x + plot_w / 2.0
        return plot_x + (value - x_min) / (x_max - x_min) * plot_w

    def y_to_px(value: float) -> float:
        return plot_y + (1.0 - (value - y_min) / (y_max - y_min)) * plot_h

    lines: list[str] = []
    lines.append(
        '<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="white" stroke="#d0d7de" />'.format(
            x=left, y=top, w=panel_width, h=panel_height
        )
    )
    lines.append(
        '<text x="{x}" y="{y}" font-size="14" font-weight="600" fill="#111827">{text}</text>'.format(
            x=left + 10, y=top + 18, text=svg_escape(panel_title)
        )
    )
    lines.append(
        '<text x="{x}" y="{y}" font-size="11" fill="#374151">{text}</text>'.format(
            x=left + panel_width - 10, y=top + panel_height - 8, text=svg_escape(y_label)
        )
    )

    for tick in _ticks(y_min, y_max, 5):
        y = y_to_px(tick)
        lines.append('<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="#e5e7eb" stroke-width="1" />'.format(x1=plot_x, x2=plot_x + plot_w, y=y))
        lines.append(
            '<text x="{x}" y="{y}" font-size="10" text-anchor="end" dominant-baseline="middle" fill="#6b7280">{text}</text>'.format(
                x=plot_x - 6,
                y=y,
                text=svg_escape(y_formatter(tick)),
            )
        )

    x_ticks = _ticks(float(x_min), float(x_max), min(5, len(x_values)))
    for tick in x_ticks:
        x = x_to_px(tick)
        lines.append('<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2}" stroke="#e5e7eb" stroke-width="1" />'.format(x=x, y1=plot_y, y2=plot_y + plot_h))
        lines.append(
            '<text x="{x}" y="{y}" font-size="10" text-anchor="middle" fill="#6b7280">{text}</text>'.format(
                x=x,
                y=plot_y + plot_h + 14,
                text=svg_escape(int(round(tick))),
            )
        )

    points = " ".join("{:.1f},{:.1f}".format(x_to_px(xv), y_to_px(yv)) for xv, yv in zip(x_values, y_values))
    lines.append('<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{points}" />'.format(color=color, points=points))
    for xv, yv in zip(x_values, y_values):
        lines.append(
            '<circle cx="{x}" cy="{y}" r="3.5" fill="{color}" stroke="white" stroke-width="1" />'.format(
                x="{:.1f}".format(x_to_px(xv)),
                y="{:.1f}".format(y_to_px(yv)),
                color=color,
            )
        )
    return lines


def write_svg_figure(
    title: str,
    panels: list[dict[str, object]],
    output_path: Path,
) -> None:
    width = 1100
    panel_width = 1060
    panel_height = 215
    gap = 20
    height = 60 + len(panels) * panel_height + max(0, len(panels) - 1) * gap + 20
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'.format(
            width=width, height=height
        ),
        '<rect width="100%" height="100%" fill="#f9fafb" />',
        '<text x="{x}" y="32" font-size="20" font-weight="700" fill="#111827">{text}</text>'.format(
            x=20, text=svg_escape(title)
        ),
    ]
    top = 48
    for panel in panels:
        lines.extend(
            _panel_svg(
                str(panel["title"]),
                list(panel["x_values"]),
                list(panel["y_values"]),
                color=str(panel["color"]),
                y_label=str(panel["y_label"]),
                y_formatter=panel["y_formatter"],
                width=width,
                height=height,
                left=20,
                top=top,
                panel_width=panel_width,
                panel_height=panel_height,
            )
        )
        top += panel_height + gap
    lines.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def aggregate_by_iteration(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["iteration"])].append(row)
    output: list[dict[str, object]] = []
    for iteration in sorted(grouped):
        members = grouped[iteration]
        output.append(
            {
                "iteration": iteration,
                "psnr": mean([float(row["psnr"]) for row in members if row["psnr"] is not None]),
                "ssim": mean([float(row["ssim"]) for row in members if row["ssim"] is not None]),
                "lpips": mean([float(row["lpips"]) for row in members if row["lpips"] is not None]),
                "gs_num": mean([float(row["gs_num"]) for row in members if row["gs_num"] is not None]),
                "scene_count": len(members),
            }
        )
    return output


def plot_scene_curve(scene: str, rows: list[dict[str, object]], output_path: Path) -> None:
    iterations = [int(row["iteration"]) for row in rows]
    psnr = [float(row["psnr"]) for row in rows]
    ssim = [float(row["ssim"]) for row in rows]
    lpips = [float(row["lpips"]) for row in rows]
    gs_num = [float(row["gs_num"]) / 1_000_000.0 if row["gs_num"] is not None else float("nan") for row in rows]
    write_svg_figure(
        scene,
        [
            {"title": "PSNR", "x_values": iterations, "y_values": psnr, "color": "#2563eb", "y_label": "PSNR", "y_formatter": fmt_metric},
            {"title": "SSIM", "x_values": iterations, "y_values": ssim, "color": "#16a34a", "y_label": "SSIM", "y_formatter": fmt_metric},
            {"title": "LPIPS", "x_values": iterations, "y_values": lpips, "color": "#d97706", "y_label": "LPIPS", "y_formatter": fmt_metric},
            {"title": "GS num (M)", "x_values": iterations, "y_values": gs_num, "color": "#dc2626", "y_label": "GS num (M)", "y_formatter": fmt_gs_million},
        ],
        output_path,
    )


def write_check_md(
    output_md: Path,
    scene_rows: dict[str, list[dict[str, object]]],
    average_rows: list[dict[str, object]],
    plot_dir: Path,
    title: str,
) -> None:
    lines: list[str] = []
    lines.append("# {}".format(title))
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append("| iteration | avg PSNR | avg SSIM | avg LPIPS | avg GS |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in average_rows:
        lines.append(
            "| {iteration} | {psnr:.4f} | {ssim:.4f} | {lpips:.4f} | {gs_num:.0f} |".format(
                iteration=row["iteration"],
                psnr=float(row["psnr"]) if row["psnr"] is not None else float("nan"),
                ssim=float(row["ssim"]) if row["ssim"] is not None else float("nan"),
                lpips=float(row["lpips"]) if row["lpips"] is not None else float("nan"),
                gs_num=float(row["gs_num"]) if row["gs_num"] is not None else float("nan"),
            )
        )
    lines.append("")
    lines.append("![](./{}/average.svg)".format(plot_dir.name))
    lines.append("")
    for scene in sorted(scene_rows):
        lines.append("## {}".format(scene))
        lines.append("")
        lines.append("| iteration | PSNR | SSIM | LPIPS | GS |")
        lines.append("|---|---:|---:|---:|---:|")
        for row in scene_rows[scene]:
            lines.append(
                "| {iteration} | {psnr:.4f} | {ssim:.4f} | {lpips:.4f} | {gs_num:.0f} |".format(
                    iteration=row["iteration"],
                    psnr=float(row["psnr"]) if row["psnr"] is not None else float("nan"),
                    ssim=float(row["ssim"]) if row["ssim"] is not None else float("nan"),
                    lpips=float(row["lpips"]) if row["lpips"] is not None else float("nan"),
                    gs_num=float(row["gs_num"]) if row["gs_num"] is not None else float("nan"),
                )
            )
        lines.append("")
        lines.append("![](./{}/{})".format(plot_dir.name, "{}.svg".format(scene)))
        lines.append("")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FastGS checkpoint curves with periodic render/metrics.")
    parser.add_argument("--dataset-name", required=True, choices=["mipnerf360", "db", "tandt"])
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--train-images", default="images")
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--checkpoint-interval", type=int, default=2000)
    parser.add_argument("--resolution", type=int, default=-1)
    parser.add_argument("--variant", default="fastgs_big")
    parser.add_argument("--densification-interval", type=int, default=100)
    parser.add_argument("--method-name", default=DEFAULT_METHOD)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--config", default=None)
    parser.add_argument("--vfm-cache-template", default=None)
    parser.add_argument("--vfm-cache-backend", default=DEFAULT_VFM_CACHE_BACKEND)
    parser.add_argument("--vfm-cache-feature", default=DEFAULT_VFM_CACHE_FEATURE)
    parser.add_argument("--vfm-cache-storage", default=DEFAULT_VFM_CACHE_STORAGE)
    parser.add_argument("--vfm-cache-max-width", type=int, default=DEFAULT_VFM_CACHE_MAX_WIDTH)
    parser.add_argument("--vfm-cache-device", default=DEFAULT_VFM_CACHE_DEVICE)
    parser.add_argument("--no-scene-overrides", action="store_true")
    parser.add_argument("--check-md", type=Path, default=None)
    parser.add_argument("--check-plot-dir", type=Path, default=None)
    parser.add_argument("--check-csv", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = Path.cwd()
    use_overrides = not args.no_scene_overrides
    checkpoint_iters = checkpoint_iterations(args.iterations, args.checkpoint_interval)

    all_rows: list[dict[str, object]] = []
    scene_rows: dict[str, list[dict[str, object]]] = {}

    for scene in args.scenes:
        scene_path = args.dataset_root / scene
        if not scene_path.exists():
            raise FileNotFoundError(scene_path)
        overrides = scene_overrides(args.dataset_name, scene, use_overrides)
        run_dir = args.output_root / scene / args.run_name
        log_dir = args.output_root / scene / "logs" / args.run_name
        cache_dir = None
        if args.vfm_cache_template:
            cache_dir = Path(args.vfm_cache_template.format(scene=scene))
            build_vfm_cache(scene_path, scene, args, repo, log_dir, cache_dir)

        print("[{}] {} checkpoint curve".format(scene, args.method_name), flush=True)
        train_baseline(
            scene_path,
            scene,
            args,
            repo,
            run_dir,
            log_dir,
            overrides,
            cache_dir=cache_dir,
            save_iterations=checkpoint_iters,
        )
        for iteration in checkpoint_iters:
            render_iteration(run_dir, log_dir, repo, overrides, iteration)
        run_metrics(run_dir, log_dir, repo)

        rows = read_checkpoint_rows(args.dataset_name, scene, args.method_name, run_dir, checkpoint_iters)
        scene_rows[scene] = rows
        all_rows.extend(rows)

    average_rows = aggregate_by_iteration(all_rows)

    check_dir = args.output_root
    plot_dir = args.check_plot_dir or (check_dir / "plots")
    if PLOT_AVAILABLE:
        for scene, rows in scene_rows.items():
            plot_scene_curve(scene, rows, plot_dir / "{}.png".format(scene))
        plot_scene_curve("average", average_rows, plot_dir / "average.png")

    if args.check_csv is not None:
        write_csv(args.check_csv, all_rows)
    else:
        write_csv(check_dir / "check.csv", all_rows)

    if args.check_md is not None:
        md_path = args.check_md
    else:
        md_path = check_dir / "check.md"
    write_check_md(md_path, scene_rows, average_rows, plot_dir, "{} checkpoint curve".format(args.method_name))

    (check_dir / "check.json").write_text(
        json.dumps({"rows": all_rows, "average_rows": average_rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("Wrote {}".format(md_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
