# 0015 实验结果

## 当前状态

完成。0015 是离线 selector 审计，不新增训练、不占 GPU；在 0014 的 Phase0 curve 特征基础上，加入 0008 residual-orientation proxy scene summary，共 74 个特征。

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
| in_sample_phase0_plus_residual_proxy_stump | +0.0359 | +0.00073 | -0.00113 | +16,322 | +0.0377 |
| loocv_phase0_plus_residual_proxy_stump | +0.0125 | +0.00076 | -0.00112 | +30,930 | -0.0331 |
| loocv_phase0_plus_residual_proxy_nearest_neighbor | -0.0099 | -0.00006 | -0.00005 | -10,677 | -0.0137 |

## LOOCV 逐场景选择

| scene | stump selected | oracle method | selected QCGI | oracle QCGI |
|---|---|---|---:|---:|
| bicycle | dino_i050_until8000 | dino_i050_full | +0.0731 | +0.1639 |
| flowers | dino_i050_until8000 | dino_i050_until8000 | +0.0565 | +0.0565 |
| garden | dino_i050_until8000 | residual_orientation | -0.0554 | +0.0228 |
| stump | dino_i050_full | dino_i050_until8000 | -0.0937 | +0.0243 |
| treehill | dino_i050_until8000 | residual_orientation | -0.0424 | +0.0406 |
| room | depth_auto_topk | residual_orientation | -0.0819 | +0.0259 |
| counter | residual_orientation | depth_auto_topk | +0.0706 | +0.0739 |
| kitchen | dino_i050_until8000 | depth_auto_topk | +0.1047 | +0.1736 |
| bonsai | dino_i050_until8000 | phase0 | -0.3298 | +0.0000 |

In-sample 最优树桩仍然与 0014 相同：

```text
i4000_to_i8000_ssim_gain <= 0.0594741 ? residual_orientation : dino_i050_until8000
```

## 输出文件

- `output/0015/residual_proxy_cross_method_selector/feature_table.csv`
- `output/0015/residual_proxy_cross_method_selector/loocv_predictions.csv`
- `output/0015/residual_proxy_cross_method_selector/policy_summary.csv`
- `output/0015/residual_proxy_cross_method_selector/summary_stats.json`

## 结论

加入 0008 residual proxy 后，LOOCV 没有改善，反而从 0014 Phase0-only 树桩的 +0.0165 QCGI 降到 -0.0331 QCGI。主要失败来自 `garden/stump/treehill/room/bonsai` 的误选，尤其 `bonsai` 被选为 `dino_i050_until8000`，单场景 QCGI 为 -0.3298。

这说明 residual proxy 虽然能解释 residual-orientation 的局部候选方向，但不能直接作为跨 Depth / residual / DINO 的总 selector。当前证据下，不宜继续用小样本 LOOCV 堆特征；更值得回到方法层面，做“不会伤害负场景”的保守触发或容量保护。
