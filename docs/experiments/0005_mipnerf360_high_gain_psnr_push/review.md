# 0005 MipNeRF360 High-Gain PSNR Push 复盘

## 当前判断

用户目标是当天看到 MipNeRF360 9 场景平均 PSNR 相对 baseline 至少 +0.2。0004 的 prune-side 方向多轮结果显示更像减伤器，平均收益离 +0.2 很远，因此 0005 先转向已有最高增益证据的 DINO token-edge / descriptor densification 引导。

## 风险

- 高增益路线很可能增加 Gaussian 数量和训练时间。
- 0001 里部分 +0.2 证据来自较早评测口径；本轮必须用 FastGS big 1.6K baseline 重新对齐。
- 如果 token-edge 在 high-res full recipe 下退化，需要立刻切到 DINO descriptor top-k25 `max` 或 mixed selector。

## 下一步

等待 Round 1 双卡 final-only 全 9 场景完成，立即整理结果并提交。
