# 0014 实验结果

## 当前状态

完成。0014 是离线 selector 审计，不新增训练、不占 GPU；只使用 Phase0 checkpoint curve 特征来预测 0013 的 cross-method 选择。

## 策略均值

| policy | dPSNR | dSSIM | dLPIPS | dGS | QCGI |
|---|---:|---:|---:|---:|---:|
| fixed_phase0 | +0.0000 | +0.00000 | +0.00000 | +0 | +0.0000 |
| fixed_depth_auto_topk | +0.0133 | -0.00013 | +0.00049 | +307 | +0.0065 |
| fixed_residual_orientation | +0.0072 | -0.00010 | +0.00019 | -1,651 | +0.0032 |
| fixed_dino_i050_full | -0.0027 | +0.00106 | -0.00322 | +53,628 | -0.1015 |
| fixed_dino_i050_until8000 | +0.0156 | +0.00095 | -0.00177 | +33,372 | -0.0171 |
| fixed_dino_i050_until4000 | -0.0038 | +0.00005 | -0.00080 | +11,992 | -0.0148 |
| oracle_best_qcgi_per_scene | +0.0489 | +0.00111 | -0.00161 | +12,306 | +0.0646 |
| in_sample_phase0_curve_stump | +0.0359 | +0.00073 | -0.00113 | +16,322 | +0.0377 |
| loocv_phase0_curve_stump | +0.0212 | +0.00055 | -0.00055 | +12,864 | +0.0165 |
| loocv_nearest_neighbor_oracle_method | -0.0071 | -0.00006 | +0.00038 | -1,911 | -0.0134 |

## LOOCV 逐场景选择

| scene | stump selected | oracle method | selected QCGI | oracle QCGI |
|---|---|---|---:|---:|
| bicycle | dino_i050_until8000 | dino_i050_full | +0.0731 | +0.1639 |
| flowers | dino_i050_until8000 | dino_i050_until8000 | +0.0565 | +0.0565 |
| garden | residual_orientation | residual_orientation | +0.0228 | +0.0228 |
| stump | dino_i050_until8000 | dino_i050_until8000 | +0.0243 | +0.0243 |
| treehill | dino_i050_until8000 | residual_orientation | -0.0424 | +0.0406 |
| room | depth_auto_topk | residual_orientation | -0.0819 | +0.0259 |
| counter | residual_orientation | depth_auto_topk | +0.0706 | +0.0739 |
| kitchen | dino_i050_until8000 | depth_auto_topk | +0.1047 | +0.1736 |
| bonsai | residual_orientation | phase0 | -0.0792 | +0.0000 |

In-sample 最优树桩为：

```text
i4000_to_i8000_ssim_gain <= 0.0594741 ? residual_orientation : dino_i050_until8000
```

## 输出文件

- `output/0014/phase0_fingerprint_cross_method_selector/feature_table.csv`
- `output/0014/phase0_fingerprint_cross_method_selector/loocv_predictions.csv`
- `output/0014/phase0_fingerprint_cross_method_selector/policy_summary.csv`
- `output/0014/phase0_fingerprint_cross_method_selector/summary_stats.json`

## 结论

Phase0-only fingerprint 有一点可用信号：LOOCV 树桩从固定 `depth_auto_topk` 的 +0.0065 QCGI 提高到 +0.0165 QCGI。但这个增益还很薄，且对少数场景误选很敏感；`treehill`、`room`、`bonsai` 三个误选直接吞掉大部分 oracle 上界。

最近邻 selector 为 -0.0134 QCGI，说明“训练曲线相似”并不自然等价于“prior 响应相似”。因此 cross-method selector 不能只靠 Phase0 曲线默认化；若继续，需要引入更直接的训练期 proxy，例如低频率 residual-orientation probe 或极轻量 descriptor response，而不是继续堆 Phase0-only 规则。
