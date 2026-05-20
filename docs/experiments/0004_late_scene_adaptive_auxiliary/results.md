# 0004 Late Scene-Adaptive Auxiliary 结果

## 当前状态

已完成前置 baseline curve 诊断、第一轮 `start15001` pilot、第二轮 `start24000` timing 复验，以及第三轮 final-only 低成本小扫。`start24000` 明显修复了 `room` 的 late-window 伤害，但后续降低 protect weight 或收窄 auto-topk 都没有形成稳定正向，Depth Anything prune-protect auto-topk 不能作为默认在线辅助。

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

输出：

- `output/0004/late_scene_adaptive_auxiliary/depth_prune_auto_topk_start24000_pilot_gpu1/check.md`
- `output/0004/late_scene_adaptive_auxiliary/depth_prune_auto_topk_start24000_indoor_gpu0/check.md`
- `output/0004/late_scene_adaptive_auxiliary/depth_prune_auto_topk_start24000_comparison.csv`

目的：只改变介入时机，跳过 18k/21k，只在 24k/27k 做 protect。结果说明 timing 是真实变量：`room` 的 24k -> 30k late gain 从 Round 1 的 -0.1589 差值恢复到 -0.0004，30k PSNR 负向也从 -0.2219 收窄到 -0.0498。但这个修复没有让整体变成正解，6 场景平均仍为 -0.0340 PSNR。

30k 相对 baseline：

| 场景 | PSNR | ΔPSNR | ΔSSIM | ΔLPIPS | ΔGS |
|---|---:|---:|---:|---:|---:|
| bicycle | 25.2544 | -0.0103 | -0.00005 | -0.00011 | +5,458 |
| stump | 27.1739 | +0.0357 | -0.00040 | +0.00041 | -259 |
| room | 32.2509 | -0.0498 | +0.00011 | -0.00009 | +2,042 |
| bonsai | 32.8333 | -0.1283 | -0.00148 | -0.00006 | +2,972 |
| counter | 29.5787 | +0.0375 | +0.00031 | -0.00041 | -269 |
| kitchen | 32.3296 | -0.0887 | -0.00023 | +0.00032 | +101 |

分组均值：

| 分组 | ΔPSNR | ΔSSIM | ΔLPIPS | ΔGS |
|---|---:|---:|---:|---:|
| pilot: bicycle/stump/room | -0.0081 | -0.00011 | +0.00007 | +2,414 |
| indoor: bonsai/counter/kitchen | -0.0598 | -0.00047 | -0.00005 | +935 |
| all 6 | -0.0340 | -0.00029 | +0.00001 | +1,674 |

窗口增量差值：

| 场景 | 16k -> 30k | 20k -> 30k | 24k -> 30k | 观察 |
|---|---:|---:|---:|---|
| bicycle | -0.0013 | -0.0003 | +0.0019 | 中性 |
| stump | +0.0346 | +0.0242 | +0.0239 | 继续减轻 PSNR 后期回落，但 LPIPS 仍略差 |
| room | -0.0687 | -0.0037 | -0.0004 | 24k 后窗口基本恢复，说明跳过 18k/21k 有效 |
| bonsai | +0.0026 | -0.0368 | -0.0385 | 新主要负例，30k 单点和后期窗口都不稳 |
| counter | +0.0310 | +0.0212 | +0.0116 | 稳定小正向 |
| kitchen | -0.0314 | -0.0150 | +0.0070 | 24k 后略正，但 30k 单点负 |

结论：`start24000` 验证了 late timing 的重要性，但没有验证出可默认化的在线辅助器。下一步若继续 0004，应只做 final-only 的低成本小扫：例如更低 protect weight 或更窄 auto-topk 上限；若仍不能让 `bonsai/kitchen` 回到 baseline 附近，就把 Depth Anything prune-protect 收束为诊断工具。

## Round 3：final-only 低成本小扫

目的：承接 Round 2 的结论，只做最终 30k render / metrics，不再每 2k checkpoint 评测。两个分支分别测试：

- `configs/experiments/0004_late_scene_adaptive_auxiliary_start24000_weight015.yaml`：保持 auto-topk 1.0%，把 protect weight 从 0.25 降到 0.15。
- `configs/experiments/0004_late_scene_adaptive_auxiliary_start24000_auto_topk005.yaml`：保持 protect weight 0.25，把 auto-topk 上限从 1.0% 收窄到 0.5%。

输出：

- `output/0004/late_scene_adaptive_auxiliary/final_only_start24000_weight015_gpu0/summary.csv`
- `output/0004/late_scene_adaptive_auxiliary/final_only_start24000_auto_topk005_gpu1/summary.csv`

### start24000 + weight0.15

30k 相对 baseline：

| 场景 | PSNR | ΔPSNR | ΔSSIM | ΔLPIPS | ΔGS | 相对 Round 2 ΔPSNR |
|---|---:|---:|---:|---:|---:|---:|
| room | 32.2429 | -0.0578 | +0.00004 | -0.00036 | +2,058 | -0.0081 |
| bonsai | 32.8156 | -0.1460 | -0.00139 | +0.00006 | +3,625 | -0.0177 |
| counter | 29.5894 | +0.0482 | -0.00015 | +0.00015 | -227 | +0.0107 |
| kitchen | 32.2064 | -0.2118 | -0.00058 | +0.00113 | -790 | -0.1231 |

四场景均值：ΔPSNR -0.0919，ΔSSIM -0.00052，ΔLPIPS +0.00025，ΔGS +1,166。相对 Round 2 四场景均值又低 -0.0346 PSNR。降低 weight 没有把 `bonsai/kitchen` 拉回 baseline，反而明显伤害 `kitchen`，因此单纯降低保护强度不是解法。

### start24000 + auto-topk0.5%

30k 相对 baseline：

| 场景 | PSNR | ΔPSNR | ΔSSIM | ΔLPIPS | ΔGS | 相对 Round 2 ΔPSNR |
|---|---:|---:|---:|---:|---:|---:|
| room | 32.1846 | -0.1161 | -0.00005 | +0.00027 | +4,025 | -0.0663 |
| bonsai | 32.8962 | -0.0654 | -0.00197 | -0.00020 | +2,383 | +0.0630 |
| counter | 29.5493 | +0.0081 | -0.00003 | +0.00019 | -263 | -0.0294 |
| kitchen | 32.4513 | +0.0331 | +0.00007 | -0.00008 | +798 | +0.1218 |

四场景均值：ΔPSNR -0.0351，ΔSSIM -0.00050，ΔLPIPS +0.00004，ΔGS +1,736。相对 Round 2 四场景均值高 +0.0222 PSNR，但仍低于 baseline。它能明显修复 `kitchen` 并减轻 `bonsai` PSNR 负向，但把 `room` 从 -0.0498 拉低到 -0.1161，且 `counter` 的小正向基本被吃掉。

结论：Round 3 说明“更窄 proposal”比“更低 weight”更有希望，但仍不是稳定策略。Depth prior 在 `room/bonsai/kitchen/counter` 之间的最优 proposal 宽度不一致，固定 topk 或固定 weight 不能成为默认在线辅助。后续若继续 0004，应转向 validation-driven selector / scene-conditioned policy，而不是继续手工扫固定阈值。

## 记录表

| 场景 / 数据集 | 配置 | PSNR | SSIM | LPIPS | Gaussian 数量 | 备注 |
|---|---|---:|---:|---:|---:|---|
| MipNeRF360 6 scenes | `0004_late_scene_adaptive_auxiliary` | mixed | mixed | mixed | +529 avg | 6 场景平均负，见 Round 1 |
| MipNeRF360 6 scenes | `0004_late_scene_adaptive_auxiliary_start24000` | mixed | mixed | mixed | +1,674 avg | room 修复明显，但均值仍负 |
| MipNeRF360 4 scenes | `start24000_weight015` | mixed | mixed | mixed | +1,166 avg | 降权重更差，kitchen 明显负 |
| MipNeRF360 4 scenes | `start24000_auto_topk005` | mixed | mixed | mixed | +1,736 avg | 修复 kitchen，但 room/counter 变差 |
