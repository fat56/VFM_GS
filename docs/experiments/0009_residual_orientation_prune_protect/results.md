# 0009 Residual Orientation Prune-Protect 结果

## 当前状态

实现、配置、smoke 与 Round 1 三场景 pilot 已完成。Round 1 主口径相对 Phase0 FastGS big baseline 略正但很薄；按 proposal 继续扩到 MipNeRF360 全 9 场景，先补跑剩余 6 个场景。

## Smoke

已完成：

- `py_compile` 通过：`src/vfm_gs/scorers/vfm_topology.py`、`src/vfm_gs/config/legacy_args.py`。
- `treehill -r 8` 700-step 训练通过，并在第 600 step 实际触发 `depth_anything_residual_orientation` scorer。
- Render / metrics 通过：700-step `treehill` 为 PSNR 20.7703 / SSIM 0.5537 / LPIPS 0.5305。
- 直接 scorer smoke 覆盖 late prune-protect 分支：`[VFM PRUNE PROTECT] iter=24000 mode=rgb_prune_auto_topk weight=0.2500 protected=239 rgb_candidates=271 mean=0.000138 max=1.000000`。

Smoke 输出：

- `output/0009/residual_orientation_protect_smoke/treehill/smoke_700_r8`

## Round 1 Pilot

配置：`configs/experiments/0009_residual_orientation_protect_start24000_auto_topk005.yaml`

输出：

- `output/0009/residual_orientation_protect_pilot/mip_g0/summary.csv`
- `output/0009/residual_orientation_protect_pilot/mip_g1/summary.csv`
- `output/0009/residual_orientation_protect_pilot/comparison/comparison.csv`
- `output/0009/residual_orientation_protect_pilot/comparison/summary.json`

主对照使用 Phase0 FastGS big baseline：`output/0002/phase0_5090_fastgs_big_baseline_fix1/mipnerf360_combined/summary.csv`。`curve ΔPSNR` 额外给出相对 0004 使用的 checkpoint-curve 30k baseline：`output/0002/fastgs_big_baseline_checkpoint_curve/mipnerf360/check.csv`。LPIPS 越低越好，因此正的 `ΔLPIPS` 是变差。

| scene | phase0 PSNR | 0009 PSNR | ΔPSNR | ΔSSIM | ΔLPIPS | ΔGS | curve ΔPSNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| room | 32.2136 | 32.2382 | +0.0246 | +0.00012 | +0.00020 | -2,722 | -0.0625 |
| treehill | 22.8275 | 22.8685 | +0.0410 | -0.00003 | -0.00014 | +467 | +0.0059 |
| stump | 27.1784 | 27.1408 | -0.0376 | -0.00108 | +0.00127 | -9,189 | +0.0026 |
| avg | 27.4065 | 27.4158 | +0.0093 | -0.00033 | +0.00045 | -3,815 | -0.0180 |

和 0004 static depth prior 的同口径曲线对照：

| scene | 0009 curve ΔPSNR | 0004 start24000 ΔPSNR | 0004 start24000 topk0.5 ΔPSNR | 观察 |
|---|---:|---:|---:|---|
| room | -0.0625 | -0.0498 | -0.1161 | residual orientation 没有修复 `room`，但比更窄 static topk0.5 少伤一些 |
| treehill | +0.0059 | n/a | n/a | 薄正向，符合 0008 离线 proxy 对 `treehill` 的预期 |
| stump | +0.0026 | +0.0357 | n/a | 没有保住 0004 static prior 在 `stump` 的 PSNR 收益 |

保护规模日志：

| scene | iter 24000 protected / rgb candidates | iter 27000 protected / rgb candidates |
|---|---:|---:|
| room | 2,628 / 2,889 | 2,446 / 2,864 |
| treehill | 3,777 / 5,110 | 3,714 / 5,075 |
| stump | 3,790 / 5,334 | 3,915 / 5,302 |

## 决策

Round 1 不是强正例：Phase0 主口径只有 +0.0093 PSNR，SSIM/LPIPS 不同步；相对 checkpoint-curve baseline 反而是 -0.0180 PSNR。它更像“容量受控的薄信号”，不是可默认化结果。

但三场景没有出现 0003 那种明显崩坏，Gaussian 数也没有膨胀；按 proposal 的最低门槛，继续补跑剩余 6 个 MipNeRF360 场景。Round 2 若全 9 场景仍不能同时优于 Phase0 baseline 和 0002 `depth_auto_topk`，则停止训练接入，把 residual orientation 留作离线诊断/selector 特征。
