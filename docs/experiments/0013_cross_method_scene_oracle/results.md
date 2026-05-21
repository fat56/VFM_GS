# 0013 实验结果

## 当前状态

完成。0013 是离线审计，不新增训练、不占 GPU；输入为 0002 / 0009 / 0010 / 0011 已有 full9 结果，统一按 Phase0 baseline 与 QCGI 口径重算。

## 固定策略均值

| policy | dPSNR | dSSIM | dLPIPS | dGS | QCGI |
|---|---:|---:|---:|---:|---:|
| phase0 | +0.0000 | +0.00000 | +0.00000 | +0 | +0.0000 |
| depth_auto_topk | +0.0133 | -0.00013 | +0.00049 | +307 | +0.0065 |
| residual_orientation | +0.0072 | -0.00010 | +0.00019 | -1,651 | +0.0032 |
| dino_i050_full | -0.0027 | +0.00106 | -0.00322 | +53,628 | -0.1015 |
| dino_i050_until8000 | +0.0156 | +0.00095 | -0.00177 | +33,372 | -0.0171 |
| dino_i050_until4000 | -0.0038 | +0.00005 | -0.00080 | +11,992 | -0.0148 |
| oracle_best_qcgi_per_scene | +0.0489 | +0.00111 | -0.00161 | +12,306 | +0.0646 |

## Oracle 逐场景选择

| scene | selected method | dPSNR | dSSIM | dLPIPS | dGS | QCGI |
|---|---|---:|---:|---:|---:|---:|
| bicycle | dino_i050_full | +0.0578 | +0.00559 | -0.00886 | +49,857 | +0.1639 |
| flowers | dino_i050_until8000 | +0.0414 | +0.00104 | +0.00116 | -7,395 | +0.0565 |
| garden | residual_orientation | +0.0186 | +0.00021 | -0.00002 | -9,207 | +0.0228 |
| stump | dino_i050_until8000 | +0.0278 | +0.00223 | -0.00580 | +76,991 | +0.0243 |
| treehill | residual_orientation | +0.0410 | -0.00003 | -0.00014 | +467 | +0.0406 |
| room | residual_orientation | +0.0246 | +0.00012 | +0.00020 | -2,722 | +0.0259 |
| counter | depth_auto_topk | +0.0693 | +0.00022 | -0.00012 | +354 | +0.0739 |
| kitchen | depth_auto_topk | +0.1599 | +0.00058 | -0.00088 | +2,410 | +0.1736 |
| bonsai | phase0 | +0.0000 | +0.00000 | +0.00000 | +0 | +0.0000 |

## 输出文件

- `output/0013/cross_method_scene_oracle/method_comparison_vs_phase0.csv`
- `output/0013/cross_method_scene_oracle/oracle_selection.csv`
- `output/0013/cross_method_scene_oracle/policy_summary.csv`
- `output/0013/cross_method_scene_oracle/summary_stats.json`

## 结论

单一方法仍然不能默认化：最好的固定策略是 0002 `depth_auto_topk`，但 full9 平均只有 +0.0065 QCGI；`residual_orientation` 也只有 +0.0032 QCGI；DINO 变体虽然质量项更好，但容量罚分后仍为负。

交叉方法 oracle 的上界是 +0.0646 QCGI，说明不同 prior 在不同场景确实有互补性；不过这个结果直接使用 test 指标逐场景挑选，只能视为上界，不能视为可部署 selector。下一步若继续这个方向，必须做低成本、训练期可观测的 scene fingerprint，并先证明它在 leave-one-scene-out 下不会复现 0012 的误开/误关问题。
