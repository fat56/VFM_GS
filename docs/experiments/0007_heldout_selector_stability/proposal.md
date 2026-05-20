# 0007 Held-out Selector Stability

## 核心假设

0006 的 train-split selector 暴露了轻微过拟合：用训练视角 proxy 选出来的方法，在 test 均值上没有稳定超过 baseline。0007 只验证一个更窄的问题：

如果把同一批 sampled train views 再切成 `selector` 与 `holdout` 两半，用 `selector` 视角做方法选择，`holdout` 视角和既有 test summary 是否仍支持这个选择。

这不是新训练实验，而是对 0002 / 0004 既有 checkpoint 的二次评估。目标是判断 selector 是否还有继续做 scene-conditioned policy 的价值。

## 变体 / 配置

- 变体：`fastgs_big`
- 候选来源：复用 0006 的 `baseline`、`depth_auto_topk`、`depth_topk010`、`depth_rgb_rerank_start9000`、late-start partial 候选
- 选择器：
  - `selector_best_psnr`：只看 selector views 的 PSNR
  - `selector_qcgi`：用 selector views 的 PSNR / SSIM / LPIPS，并惩罚 Gaussian 增长
- held-out 切分：先按 0006 口径采样 `max_views=16, view_stride=7`，再按 even/odd view index 切成 selector / holdout
- 代码版本：当前主分支 + `scripts/evaluate_0007_heldout_train_selector.py`

## 数据集

- MipNeRF360：9 scenes
- DB/Tandt：4 scenes
- 分辨率：复用原 checkpoint 的 1.6K 自动缩放口径
- 评估：不重新训练，只渲染 train sampled views；test 指标从 0006 candidate table 读取

## 判定标准

- 主看 `selector_qcgi`。
- 至少要同时满足：
  - held-out 均值不低于 baseline；
  - test 均值不低于 baseline；
  - Gaussian 数量不明显增长。
- 若只有 `selector_best_psnr` 赢，且靠 `depth_rgb_rerank_start9000` 增加大量 Gaussian，则不推进默认策略。

## 失败记录

- selector views 赢、holdout/test 输：说明 train proxy 仍然过拟合，不适合默认化。
- holdout 赢、test 输：说明 train 内部 held-out 仍不能代表 official test，需要更接近真实 test 分布的 policy。
- 只有个别场景赢：保留为诊断工具，不做全局 policy。

## 下一步

先完成 held-out selector 稳定性判断。若它仍不能过关，再回到 online residual 的 orientation-aware gating；不要在 selector 未站稳前改 rasterizer。
