# 0005 MipNeRF360 High-Gain PSNR Push 复盘

## 当前判断

用户目标是当天看到 MipNeRF360 9 场景平均 PSNR 相对 baseline 至少 +0.2。0004 的 prune-side 方向多轮结果显示更像减伤器，平均收益离 +0.2 很远，因此 0005 先转向已有最高增益证据的 DINO token-edge / descriptor densification 引导；但 Round 1 的 high-res token-edge 结果说明这条路不能承担当天 PSNR 目标。

Round 1 已完成 4 场景：`bicycle` +0.0805、`flowers` -0.0266、`room` -0.2330、`counter` +0.0032，平均 -0.0440 PSNR。它的 SSIM/LPIPS 均值正向，说明 token-edge 确实偏向结构/感知一致性，但对 PSNR 是不稳定甚至冲突的信号。继续跑完整 9 场景已经不经济，因为剩余 5 场景必须平均接近 +0.40 PSNR 才能把全局均值拉到 +0.2。

Round 2 改为更直接的质量上限探针：保留 FastGS photometric scorer，不接 DINO/Depth，只把 `densify_until_iter` 从 15k 延到 21k。这个实验检验的问题是：默认 FastGS big 是否在 15k 后过早停止增长，导致边缘、薄结构和高频纹理没有足够容量，而后期 prune/optimization 只能改善已有覆盖，无法新增缺失细节。实际运行时，`bicycle` 与 `room` 都在 18k 左右触发 rasterizer backward 空 SH feature 形状错误，无法产出 metrics。

Round 2 的失败本身也说明一件事：如果要做“后期继续复制/分裂”的正式路线，不能只把 `densify_until_iter` 往后拨。18k 同时位于 extended densify window 和 FastGS late prune schedule，当前点增删逻辑会触发稀有空张量路径。后续如果回到这条路线，应先修训练内核/feature shape 边界，或把 late densify 和 late prune 错开。

Round 3 改为 no-prune 容量上限探针：默认 15k 前增长逻辑不变，只用超大 `prune_min_gaussian_count` 禁用训练期和后期 prune。4 个场景后 early-stop：平均 +0.0280 PSNR、+793,121 GS；`counter` +0.0742 但 `bicycle` -0.0168。结论是保容量能带来薄增益，但代价极大，也不接近 +0.2。

Round 4 改为长训上限探针：保持默认 FastGS big 的 densification/pruning schedule，只把总迭代从 30k 拉到 60k，并同步把 `position_lr_max_steps` 拉到 60k，避免 30k 后位置学习率过早贴近 final。它回答的问题是：当前 30k baseline 是否只是训练预算不足，而不是先验/容量不足。

## 风险

- 高增益路线很可能增加 Gaussian 数量和训练时间。
- 0001 里部分 +0.2 证据来自较早评测口径；本轮必须用 FastGS big 1.6K baseline 重新对齐。
- 如果 token-edge 在 high-res full recipe 下退化，需要立刻切到 DINO descriptor top-k25 `max` 或 mixed selector。
- 延长 densification 当前存在实现边界，18k 会触发 rasterizer backward 空 SH feature 形状错误；除非专门修代码，否则不再直接跑 21k window。
- no-prune 已证明容量不是当天目标的主要瓶颈：点数大幅增加但 PSNR 只有 +0.0280。
- 60k 长训可能比 30k 多花约一倍训练时间；但它不改变模型生命周期逻辑，风险比继续堆 prior 更小。

## 下一步

启动 Round 4 双卡 final-only 全 9 场景。若 Round 4 未接近 +0.2，下一优先级是：

1. softer prune schedule：保留默认 15k densification，只降低 18k/21k/24k/27k 的 prune 强度。
2. 修复 extended densify 的空 feature 形状错误，再回到 late densify。
3. DINO descriptor top-k25 `max` 只作为补充对照，不再作为主冲刺路线。
