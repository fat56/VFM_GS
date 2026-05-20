# 0007 Held-out Selector Stability Review

## Review Checklist

- [ ] smoke 是否确认 selector / holdout 两组 view_count 均非零。
- [ ] MipNeRF360 和 DB/Tandt 是否都完成。
- [ ] `selector_qcgi` 是否在 heldout 与 test 上同时不低于 baseline。
- [ ] `selector_best_psnr` 的收益是否依赖 Gaussian 大幅增长。
- [ ] 若 selector 失败，是否停止默认化，而不是继续调 QCGI 权重。

## 初始风险

- train 内部 held-out 仍可能离 official test 分布较近或较远，不能完全替代 test 指标。
- even/odd view split 只验证稳定性，不等于最终部署策略。
- partial late-start 候选只覆盖部分 MipNeRF360 场景，选择表需要保留候选缺失的边界条件。
