#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_fastgs_big_checkpoint_curve import fmt_gs_million, fmt_metric, write_svg_figure


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def load_check(dataset_root: Path) -> dict:
    check_path = dataset_root / "check.json"
    if not check_path.exists():
        raise FileNotFoundError(check_path)
    return json.loads(check_path.read_text(encoding="utf-8"))


def group_by_scene(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["scene"])].append(row)
    for scene in grouped:
        grouped[scene] = sorted(grouped[scene], key=lambda row: int(row["iteration"]))
    return dict(sorted(grouped.items()))


def aggregate_by_iteration(rows: list[dict]) -> list[dict]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[int(row["iteration"])].append(row)
    output = []
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


def write_csv(path: Path, rows: list[dict]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_check_md(
    output_md: Path,
    output_root: Path,
    dataset_payloads: list[tuple[Path, dict]],
    overall_rows: list[dict],
) -> None:
    lines: list[str] = []
    lines.append("# FastGS Big Check")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append("| iteration | avg PSNR | avg SSIM | avg LPIPS | avg GS |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in aggregate_by_iteration(overall_rows):
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
    lines.append("![](./plots/overall.svg)")
    lines.append("")

    for dataset_root, payload in dataset_payloads:
        dataset_name = dataset_root.name
        rows = sorted(payload["rows"], key=lambda row: (str(row["scene"]), int(row["iteration"])))
        dataset_avg = payload["average_rows"]
        scene_rows = group_by_scene(rows)
        lines.append("## {}".format(dataset_name))
        lines.append("")
        lines.append("| iteration | avg PSNR | avg SSIM | avg LPIPS | avg GS |")
        lines.append("|---|---:|---:|---:|---:|")
        for row in dataset_avg:
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
        lines.append("![](./{}/plots/average.svg)".format(dataset_name))
        lines.append("")
        for scene, scene_rows_for_plot in scene_rows.items():
            lines.append("### {}".format(scene))
            lines.append("")
            lines.append("| iteration | PSNR | SSIM | LPIPS | GS |")
            lines.append("|---|---:|---:|---:|---:|")
            for row in scene_rows_for_plot:
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
            lines.append("![](./{}/plots/{}.svg)".format(dataset_name, scene))
            lines.append("")

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge checkpoint check outputs into one markdown summary.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-roots", nargs="+", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = args.output_root
    dataset_payloads: list[tuple[Path, dict]] = []
    all_rows: list[dict] = []
    for dataset_root in args.dataset_roots:
        payload = load_check(dataset_root)
        dataset_payloads.append((dataset_root, payload))
        all_rows.extend(payload["rows"])

    overall_rows = aggregate_by_iteration(all_rows)
    if overall_rows:
        write_svg_figure(
            "overall",
            [
                {"title": "PSNR", "x_values": [int(row["iteration"]) for row in overall_rows], "y_values": [float(row["psnr"]) for row in overall_rows], "color": "#2563eb", "y_label": "PSNR", "y_formatter": fmt_metric},
                {"title": "SSIM", "x_values": [int(row["iteration"]) for row in overall_rows], "y_values": [float(row["ssim"]) for row in overall_rows], "color": "#16a34a", "y_label": "SSIM", "y_formatter": fmt_metric},
                {"title": "LPIPS", "x_values": [int(row["iteration"]) for row in overall_rows], "y_values": [float(row["lpips"]) for row in overall_rows], "color": "#d97706", "y_label": "LPIPS", "y_formatter": fmt_metric},
                {"title": "GS num (M)", "x_values": [int(row["iteration"]) for row in overall_rows], "y_values": [float(row["gs_num"]) / 1_000_000.0 for row in overall_rows], "color": "#dc2626", "y_label": "GS num (M)", "y_formatter": fmt_gs_million},
            ],
            output_root / "plots" / "overall.svg",
        )

    output_md = args.output_md or (output_root / "check.md")
    output_csv = args.output_csv or (output_root / "check.csv")
    output_json = args.output_json or (output_root / "check.json")

    write_check_md(output_md, output_root, dataset_payloads, all_rows)
    write_csv(output_csv, all_rows)
    output_json.write_text(
        json.dumps({"rows": all_rows, "overall_rows": overall_rows, "datasets": [str(path) for path, _ in dataset_payloads]}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("Wrote {}".format(output_md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
