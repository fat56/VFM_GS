# 0005 MipNeRF360 High-Gain PSNR Push 复盘

## 当前判断

用户目标是当天看到 MipNeRF360 9 场景平均 PSNR 相对 baseline 至少 +0.2。0004 的 prune-side 方向多轮结果显示更像减伤器，平均收益离 +0.2 很远，因此 0005 先转向已有最高增益证据的 DINO token-edge / descriptor densification 引导；但 Round 1 的 high-res token-edge 结果说明这条路不能承担当天 PSNR 目标。

Round 1 已完成 4 场景：`bicycle` +0.0805、`flowers` -0.0266、`room` -0.2330、`counter` +0.0032，平均 -0.0440 PSNR。它的 SSIM/LPIPS 均值正向，说明 token-edge 确实偏向结构/感知一致性，但对 PSNR 是不稳定甚至冲突的信号。继续跑完整 9 场景已经不经济，因为剩余 5 场景必须平均接近 +0.40 PSNR 才能把全局均值拉到 +0.2。

Round 2 改为更直接的质量上限探针：保留 FastGS photometric scorer，不接 DINO/Depth，只把 `densify_until_iter` 从 15k 延到 21k。这个实验检验的问题是：默认 FastGS big 是否在 15k 后过早停止增长，导致边缘、薄结构和高频纹理没有足够容量，而后期 prune/optimization 只能改善已有覆盖，无法新增缺失细节。

## 风险

- 高增益路线很可能增加 Gaussian 数量和训练时间。
- 0001 里部分 +0.2 证据来自较早评测口径；本轮必须用 FastGS big 1.6K baseline 重新对齐。
- 如果 token-edge 在 high-res full recipe 下退化，需要立刻切到 DINO descriptor top-k25 `max` 或 mixed selector。
- 延长 densification 可能显著增加容量，也可能破坏后期 prune 的质量稳定性；若 Round 2 质量不升或只靠大幅增点换来薄提升，应转向 no-prune/long-train 上限诊断，而不是继续调 VFM。

## 下一步

启动 Round 2 双卡 final-only 全 9 场景。若 Round 2 未接近 +0.2，下一优先级是：

1. `0005_fastgs_big_no_prune_floor`：禁用训练期/后期 prune，验证保留容量是否能明显提升 PSNR。
2. `fastgs_big` 60k：不增加新点，只延长优化，验证 30k 是否只是训练步数不足。
3. DINO descriptor top-k25 `max` 只作为补充对照，不再作为主冲刺路线。
