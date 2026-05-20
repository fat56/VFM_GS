# 0009 Residual Orientation Prune-Protect 结果

## 当前状态

实现、配置、smoke、Round 1 三场景 pilot 与 Round 2 MipNeRF360 全 9 场景扩展均已完成。最终结论：residual orientation late prune-protect 不是新的默认训练接入；它相对 Phase0 baseline 只有薄 PSNR 正向，且没有超过 0002 `depth_auto_topk`。

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

## Round 1 决策

Round 1 不是强正例：Phase0 主口径只有 +0.0093 PSNR，SSIM/LPIPS 不同步；相对 checkpoint-curve baseline 反而是 -0.0180 PSNR。它更像“容量受控的薄信号”，不是可默认化结果。

但三场景没有出现 0003 那种明显崩坏，Gaussian 数也没有膨胀；按 proposal 的最低门槛，继续补跑剩余 6 个 MipNeRF360 场景。Round 2 若全 9 场景仍不能同时优于 Phase0 baseline 和 0002 `depth_auto_topk`，则停止训练接入，把 residual orientation 留作离线诊断/selector 特征。

## Round 2 Full MipNeRF360

Round 2 只补跑 Round 1 未覆盖的 6 个场景：`bicycle/flowers/garden/bonsai/counter/kitchen`。最终全 9 场景由 pilot 三场景和补跑六场景合并得到。

输出：

- `output/0009/residual_orientation_protect_full_missing/mip_g0/summary.csv`
- `output/0009/residual_orientation_protect_full_missing/mip_g1/summary.csv`
- `output/0009/residual_orientation_protect_full/comparison/comparison.csv`
- `output/0009/residual_orientation_protect_full/comparison/summary.json`

对照说明：

- `phase0`：`output/0002/phase0_5090_fastgs_big_baseline_fix1/mipnerf360_combined/summary.csv`
- `curve`：`output/0002/fastgs_big_baseline_checkpoint_curve/mipnerf360/check.csv` 的 30k baseline
- `auto`：0002 `depth_anything_depth_prior_prune_protect_auto_topk_full`

| scene | 0009 PSNR | ΔPSNR phase0 | ΔPSNR curve | ΔPSNR auto | ΔSSIM phase0 | ΔLPIPS phase0 | ΔGS phase0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| bicycle | 25.2485 | -0.0084 | -0.0162 | -0.0056 | +0.00009 | -0.00025 | -2,640 |
| flowers | 21.6278 | -0.0006 | +0.0065 | -0.0268 | -0.00008 | +0.00134 | +2,989 |
| garden | 27.6532 | +0.0186 | +0.0409 | +0.0005 | +0.00021 | -0.00002 | -9,207 |
| stump | 27.1408 | -0.0376 | +0.0026 | -0.0030 | -0.00108 | +0.00127 | -9,189 |
| treehill | 22.8685 | +0.0410 | +0.0059 | +0.0496 | -0.00003 | -0.00014 | +467 |
| room | 32.2382 | +0.0246 | -0.0625 | +0.0937 | +0.00012 | +0.00020 | -2,722 |
| counter | 29.5940 | +0.0680 | +0.0528 | -0.0013 | +0.00013 | -0.00021 | +957 |
| kitchen | 32.3070 | +0.0260 | -0.1113 | -0.1339 | +0.00016 | -0.00058 | +909 |
| bonsai | 33.0176 | -0.0670 | +0.0560 | -0.0283 | -0.00041 | +0.00009 | +3,580 |
| avg | 27.9662 | +0.0072 | -0.0028 | -0.0061 | -0.00010 | +0.00019 | -1,651 |

保护规模：

| scene | iter 24000 protected / rgb candidates | iter 27000 protected / rgb candidates |
|---|---:|---:|
| bicycle | 5,998 / 7,856 | 6,484 / 7,817 |
| flowers | 3,905 / 4,479 | 4,062 / 4,309 |
| garden | 8,970 / 13,230 | 8,730 / 13,172 |
| stump | 3,790 / 5,334 | 3,915 / 5,302 |
| treehill | 3,777 / 5,110 | 3,714 / 5,075 |
| room | 2,628 / 2,889 | 2,446 / 2,864 |
| counter | 2,151 / 2,382 | 1,927 / 2,369 |
| kitchen | 4,280 / 5,984 | 4,608 / 5,933 |
| bonsai | 2,364 / 4,336 | 3,030 / 4,281 |

Round 2 结论：0009 相对 Phase0 baseline 是 +0.0072 PSNR、-0.00010 SSIM、LPIPS +0.00019、GS -1,651；相对 checkpoint-curve baseline 是 -0.0028 PSNR；相对 0002 `depth_auto_topk` 是 -0.0061 PSNR、+0.00003 SSIM、LPIPS -0.00030、GS -1,958。它是一个很薄的 tradeoff，不是压过已有 auto-topk 的新默认。

正向集中在 `counter/treehill/room/garden`；负向集中在 `bonsai/stump`，且 `kitchen` 虽然相对 Phase0 略正，但明显低于 0002 auto-topk 和 curve baseline。0008 的离线 orientation proxy 上限没有稳定传导成训练收益。

## 最终决策

停止 0009 训练接入主线，不继续扫固定 top-k / weight / start iter。`depth_anything_residual_orientation` 可以保留为离线诊断或未来 selector 的候选特征，但不能作为 FastGS 默认 prune-protect 策略。
