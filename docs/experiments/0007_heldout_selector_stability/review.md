# 0007 Held-out Selector Stability Review

## Review Checklist

- [x] smoke 是否确认 selector / holdout 两组 view_count 均非零。
- [x] MipNeRF360 和 DB/Tandt 是否都完成。
- [x] `selector_qcgi` 是否在 heldout 与 test 上同时不低于 baseline。
- [x] `selector_best_psnr` 的收益是否依赖 Gaussian 大幅增长。
- [x] 若 selector 失败，是否停止默认化，而不是继续调 QCGI 权重。

## 初始风险

- train 内部 held-out 仍可能离 official test 分布较近或较远，不能完全替代 test 指标。
- even/odd view split 只验证稳定性，不等于最终部署策略。
- partial late-start 候选只覆盖部分 MipNeRF360 场景，选择表需要保留候选缺失的边界条件。

## Review 结论

0007 验证了 0006 的核心风险：train 内部 proxy 即使加 held-out split，仍不能可靠代表 official test。

- MipNeRF360：`selector_qcgi` 的 holdout PSNR 高于 baseline +0.1001，但 test PSNR 低 -0.0067；`selector_best_psnr` 的 LPIPS/SSIM 较好，但多约 222k Gaussian。
- DB/Tandt：`selector_qcgi` 与 `selector_best_psnr` 一致，holdout PSNR 高 +0.0084，但 test PSNR 低 -0.0153，LPIPS 略差。
- 逐场景看，`kitchen/room/playroom` 都出现 holdout 正向、test 负向，是明确的 proxy 泛化失败信号。

## 最终判定

- 不默认化 held-out selector。
- 不继续调 QCGI 权重来挽救 selector 主线。
- 保留 candidate table / held-out metrics 作为离线诊断证据。
- 下一轮优先转向 online residual 的 orientation-aware gating：先判断 depth / inverse-depth / static prior 在每个视角上的方向，再决定是否接入 pruning。
