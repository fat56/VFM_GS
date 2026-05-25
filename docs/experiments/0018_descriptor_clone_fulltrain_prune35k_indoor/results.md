# 0018 结果

状态：已完成。

## 完整性

tmux 双卡任务已结束，合并结果完整。

```text
summary.csv: 20 行 = 4 个场景 x 5 个评测点
comparison_20k_to_eval.csv: 12 行 = 4 个场景 x 3 个 prune-tail 增量评测点
comparison_vs_baseline30.csv: 8 行 = 4 个场景 x 0018 的 30K/35K 对标准 baseline 30K
aggregate_20k_to_eval.csv: 3 行
aggregate_vs_baseline30.csv: 2 行
```

主要输出：

```text
output/0018/descriptor_clone_fulltrain_prune35k_indoor/mipnerf360_indoor_combined/summary.csv
output/0018/descriptor_clone_fulltrain_prune35k_indoor/mipnerf360_indoor_combined/comparison_20k_to_eval.csv
output/0018/descriptor_clone_fulltrain_prune35k_indoor/mipnerf360_indoor_combined/aggregate_20k_to_eval.csv
output/0018/descriptor_clone_fulltrain_prune35k_indoor/mipnerf360_indoor_combined/comparison_vs_baseline30.csv
output/0018/descriptor_clone_fulltrain_prune35k_indoor/mipnerf360_indoor_combined/aggregate_vs_baseline30.csv
```

## 汇总

相对 20K descriptor-clone 结束点：

| 评测点 | dPSNR | dSSIM | dLPIPS | dGS |
|---:|---:|---:|---:|---:|
| 25K | +0.2700 | +0.00172 | -0.00237 | -188,652 |
| 30K | +0.4000 | +0.00235 | -0.00338 | -193,566 |
| 35K | +0.4627 | +0.00254 | -0.00378 | -201,264 |

相对标准 FastGS 30K baseline：

| 评测点 | dPSNR | dSSIM | dLPIPS | dGS | 胜出数 |
|---:|---:|---:|---:|---:|---|
| 30K | -0.0054 | -0.00006 | -0.00086 | +1,868 | PSNR 2/4，SSIM 2/4，LPIPS 4/4 |
| 35K | +0.0574 | +0.00013 | -0.00126 | -5,830 | PSNR 3/4，SSIM 3/4，LPIPS 4/4，点数 4/4 更少 |

15K -> 20K descriptor 窗口：

| 区间 | dPSNR | dSSIM | dLPIPS | dGS |
|---|---:|---:|---:|---:|
| 15K -> 20K | +0.6698 | +0.00579 | -0.00985 | +576 |

这一段不能直接解读为 descriptor-only 证据：与已有 FastGS checkpoint-curve 的 20K 相比，0018 的 20K 平均 PSNR 仍低 `-0.1069`，并且多 `+171,637` 个 Gaussians。真正把这条轨迹转成有效最终模型的是后续 prune tail。

## 分场景

35K 相对标准 FastGS 30K：

| 场景 | dPSNR | dSSIM | dLPIPS | dGS |
|---|---:|---:|---:|---:|
| room | -0.0058 | +0.00050 | -0.00161 | -2,877 |
| counter | +0.1129 | +0.00081 | -0.00199 | -1,787 |
| kitchen | +0.0111 | +0.00021 | -0.00061 | -6,749 |
| bonsai | +0.1113 | -0.00101 | -0.00084 | -11,907 |

20K -> 35K prune tail：

| 场景 | dPSNR | dSSIM | dLPIPS | dGS |
|---|---:|---:|---:|---:|
| room | +0.4320 | +0.00214 | -0.00350 | -128,085 |
| counter | +0.3480 | +0.00335 | -0.00506 | -85,592 |
| kitchen | +0.4339 | +0.00258 | -0.00374 | -344,017 |
| bonsai | +0.6371 | +0.00209 | -0.00281 | -247,362 |

## 解读

0018 对四个室内场景是正向结果，但因果解释不是“descriptor clone 单独赢”。更干净的读法是：

- 从头完整训练时，15K-20K 的 descriptor clone 加 shifted dense optimizer，本身没有超过标准 FastGS 20K。
- 20K-35K 的 shifted FastGS prune tail 很有效：平均删除约 201K 个 Gaussians，并且 PSNR / SSIM / LPIPS 都持续改善。
- 到 30K 时，0018 与 FastGS 30K 基本打平：PSNR 略低，但 LPIPS 更好。
- 到 35K 时，0018 平均超过 FastGS 30K，并且四个室内场景的 Gaussian 数量都更少。

决策：这条分支在室内场景上值得继续，但必须结合下面的补充对照理解；不能把 35K 收益归因到 descriptor clone 单独胜出。

## 补充对照

状态：已完成。

追加两个 0018 内部对照，不另开 0019：

- `rgb_fastgs_extra_fulltrain`：15K-20K 使用 RGB/FastGS extra-densify，其他 optimizer / final-prune schedule 与主实验一致，用来判断 35K 收益是否来自更长、更后移的 FastGS schedule。
- `desc16k21k_prune35k`：复用 0017 descriptor clone-only 的 21K PLY，从 21K 接 FastGS final-prune tail 到 35K，用来观察 PLY 续跑与从头完整训练的差异。该对照从 PLY 加载并重新初始化 optimizer，不是严格的 optimizer-state 继承实验。

完整性：

```text
summary.csv: 36 行 = RGB/FastGS fulltrain 20 行 + 0017-21K continuation 16 行
comparison_start_to_eval.csv: 24 行
comparison_vs_baseline30.csv: 16 行
aggregate_start_to_eval.csv: 6 行
aggregate_vs_baseline30.csv: 4 行
```

相对各自起点的 prune tail：

| 对照 | 起点 | 终点 | dPSNR | dSSIM | dLPIPS | dGS |
|---|---:|---:|---:|---:|---:|---:|
| RGB/FastGS extra fulltrain | 20K | 35K | +0.9076 | +0.00592 | -0.00908 | -148,432 |
| 0017-21K PLY continuation | 21K | 35K | +0.2365 | +0.00094 | -0.00088 | -183,409 |
| 主实验 descriptor fulltrain | 20K | 35K | +0.4627 | +0.00254 | -0.00378 | -201,264 |

相对标准 FastGS 30K baseline：

| 对照 | 终点 | dPSNR | dSSIM | dLPIPS | dGS | 胜出数 |
|---|---:|---:|---:|---:|---:|---|
| RGB/FastGS extra fulltrain | 35K | +0.0720 | +0.00090 | -0.00176 | +20,678 | PSNR 3/4，SSIM 4/4，LPIPS 4/4，点数 0/4 更少 |
| 0017-21K PLY continuation | 35K | -0.1118 | -0.00089 | +0.00209 | +7,003 | PSNR 0/4，SSIM 0/4，LPIPS 0/4，点数 0/4 更少 |
| 主实验 descriptor fulltrain | 35K | +0.0574 | +0.00013 | -0.00126 | -5,830 | PSNR 3/4，SSIM 3/4，LPIPS 4/4，点数 4/4 更少 |

与主实验 descriptor fulltrain 35K 直接相比：

| 对照 | dPSNR | dSSIM | dLPIPS | dGS | 说明 |
|---|---:|---:|---:|---:|---|
| RGB/FastGS extra fulltrain 35K | +0.0146 | +0.00077 | -0.00050 | +26,508 | 质量略高，但点数更多；PSNR 2/4 胜，LPIPS 4/4 胜，点数 0/4 更少 |
| 0017-21K PLY continuation 35K | -0.1692 | -0.00102 | +0.00336 | +12,833 | 全面低于从头主实验；PSNR 0/4 胜，LPIPS 0/4 胜 |

补充解读：

- RGB/FastGS extra-densify 的 20K 起点明显弱于 descriptor 20K：平均 `-0.4302 PSNR`、`-0.00261 SSIM`、`LPIPS +0.00480`，但它在 20K-35K tail 中获得更大的增益，最终 35K 质量略高于主实验。
- 因此，0018 最终质量收益主要不能被解释为 descriptor guidance 单独贡献；更强的解释是 **更长、更后移的 optimizer / final-prune schedule** 是关键变量。
- descriptor fulltrain 的优势主要体现在容量权衡：35K 相对 baseline30 少 5,830 点，而 RGB extra 35K 多 20,678 点；RGB 质量略高但不省点。
- 0017-21K PLY continuation 明显弱于从头完整训练，说明训练路径很重要。由于该对照从 PLY 加载并重置 optimizer，只能说明 “0017 PLY 续跑路径” 不如 0018 从头路径，不能当作严格的 optimizer-state 继承结论。
