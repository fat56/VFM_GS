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

决策：这条分支在室内场景上值得继续。下一步必须做同样 shifted optimizer / prune schedule 下的无 descriptor 对照，或者 RGB/FastGS extra-densify 对照，用来分离 35K 收益到底来自 descriptor guidance，还是来自更长、更后移的训练 schedule。
