# 0004 Late Scene-Adaptive Auxiliary 结果

## 当前状态

已完成前置 baseline curve 诊断和第一轮 6 场景 Depth Anything prune-protect auto-topk pilot。第一轮不是默认正解：它在 `stump/counter/bicycle` 上接近中性或小正向，但 `room` 明显负向，6 场景平均也低于 baseline。

## 前置 Baseline Curve

来源：`output/0002/fastgs_big_baseline_checkpoint_curve/mipnerf360/check.md`

| 场景集合 | 16k -> 30k PSNR | 20k -> 30k PSNR | 24k -> 30k PSNR | 30k GS |
|---|---:|---:|---:|---:|
| MipNeRF360 全 9 场景平均 | +0.3316 | +0.1525 | +0.0630 | 1,161,267 |

逐场景 24k -> 30k：

| 场景 | 24k -> 30k PSNR | 30k PSNR | 30k SSIM | 30k LPIPS | 30k GS |
|---|---:|---:|---:|---:|---:|
| bicycle | +0.0268 | 25.2646 | 0.7556 | 0.2447 | 1,558,080 |
| bonsai | +0.2177 | 32.9616 | 0.9531 | 0.1601 | 844,773 |
| counter | +0.1049 | 29.5411 | 0.9179 | 0.1765 | 471,258 |
| flowers | +0.0099 | 21.6213 | 0.6022 | 0.3407 | 1,134,832 |
| garden | +0.0201 | 27.6123 | 0.8643 | 0.1099 | 2,631,613 |
| kitchen | +0.0952 | 32.4183 | 0.9394 | 0.1043 | 1,179,861 |
| room | +0.1058 | 32.3007 | 0.9307 | 0.1881 | 570,190 |
| stump | -0.0194 | 27.1382 | 0.7863 | 0.2398 | 1,052,292 |
| treehill | +0.0055 | 22.8626 | 0.6323 | 0.3769 | 1,008,508 |

结论：0004 pilot 应优先关注后期窗口差异。`room` 这种室内场景 24k 后仍有明显 PSNR 空间；`stump` 这种场景 PSNR 在 22k 达峰但 LPIPS 仍改善；`bicycle` 后期收益薄但稳定。只看 30k 平均容易掩盖 intervention timing 的场景差异。

## 预期观察

- 先看小 pilot 是否能比 0002 / 0003 更稳地控制 Gaussian 数量。
- 再看场景间差异是否真的能被 policy 吸收，而不是只换一种失败方式。

## Round 1：Depth Anything prune-protect auto-topk，start15001

配置：`configs/experiments/0004_late_scene_adaptive_auxiliary.yaml`

输出：

- `output/0004/late_scene_adaptive_auxiliary/depth_prune_auto_topk_pilot_gpu1/check.md`
- `output/0004/late_scene_adaptive_auxiliary/depth_prune_auto_topk_indoor_gpu0/check.md`
- `output/0004/late_scene_adaptive_auxiliary/depth_prune_auto_topk_comparison.csv`

这个配置保持 `vfm_weight=0.0` 和 `vfm_importance_mode=rgb_only`，不改变 densification；Depth Anything 只在 RGB pruning candidates 内做 protect。由于 FastGS pruning 在 18k/21k/24k/27k 触发，`vfm_active_from_iter=15001` 实际上会覆盖四次晚期 pruning。

30k 相对 baseline：

| 场景 | PSNR | ΔPSNR | ΔSSIM | ΔLPIPS | ΔGS |
|---|---:|---:|---:|---:|---:|
| bicycle | 25.2714 | +0.0067 | +0.00032 | -0.00028 | +1,966 |
| stump | 27.1635 | +0.0252 | -0.00052 | +0.00094 | +1,042 |
| room | 32.0788 | -0.2219 | -0.00062 | +0.00004 | -1,622 |
| bonsai | 33.0319 | +0.0703 | -0.00001 | -0.00014 | +2,868 |
| counter | 29.5863 | +0.0452 | -0.00003 | +0.00022 | -323 |
| kitchen | 32.3477 | -0.0706 | -0.00000 | +0.00002 | -758 |

分组均值：

| 分组 | ΔPSNR | ΔSSIM | ΔLPIPS | ΔGS |
|---|---:|---:|---:|---:|
| pilot: bicycle/stump/room | -0.0633 | -0.00027 | +0.00023 | +462 |
| indoor: bonsai/counter/kitchen | +0.0150 | -0.00001 | +0.00004 | +596 |
| all 6 | -0.0242 | -0.00014 | +0.00014 | +529 |

窗口增量差值：下面数值是“实验 late gain - baseline late gain”。它比单点 30k 更能判断 prior 是否真的改善了后期阶段。

| 场景 | 16k -> 30k | 20k -> 30k | 24k -> 30k | 观察 |
|---|---:|---:|---:|---|
| bicycle | -0.0001 | -0.0035 | -0.0006 | 基本中性 |
| stump | +0.0156 | +0.0145 | +0.0111 | PSNR 后期回落被减轻，但 LPIPS 变差 |
| room | -0.2403 | -0.1696 | -0.1589 | 明显伤害自然后期增益 |
| bonsai | -0.1168 | -0.0625 | -0.0517 | 30k 单点正向主要来自早期曲线偏移，后期窗口反而变差 |
| counter | +0.0164 | +0.0015 | +0.0058 | 小正向 |
| kitchen | -0.0039 | +0.0199 | +0.0124 | 后期略正，但 30k 单点仍负 |

结论：start15001 的 auto-topk protect 仍太早、太粗。它没有改变 Gaussian 数量到失控，但也没有稳定提升质量；`room` 的强负例说明 18k/21k 介入可能干扰了仍在自然恢复的室内场景。

## Round 2：Depth Anything prune-protect auto-topk，start24000

配置：`configs/experiments/0004_late_scene_adaptive_auxiliary_start24000.yaml`

目的：只改变介入时机，跳过 18k/21k，只在 24k/27k 做 protect。若 `room` 退化显著收窄，同时 `stump/counter` 的小正向还能保留，说明 0004 的关键变量确实是 late timing；若仍不稳，则 prune-protect auto-topk 应收束为近中性诊断工具。

## 记录表

| 场景 / 数据集 | 配置 | PSNR | SSIM | LPIPS | Gaussian 数量 | 备注 |
|---|---|---:|---:|---:|---:|---|
| MipNeRF360 6 scenes | `0004_late_scene_adaptive_auxiliary` | mixed | mixed | mixed | +529 avg | 6 场景平均负，见 Round 1 |
| MipNeRF360 6 scenes | `0004_late_scene_adaptive_auxiliary_start24000` | TBD | TBD | TBD | TBD | 下一轮 |
