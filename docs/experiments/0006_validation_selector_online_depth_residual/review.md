# 0006 Validation Selector + Online Depth Residual Review

## Review Checklist

- [x] `baseline` method 是否在候选表中每个场景至少出现一次。
- [x] selector 输出是否同时覆盖 MipNeRF360 与 DB/Tandt。
- [x] `train_qcgi` 是否真的回退负例，而不是偏好 Gaussian 更多的 prior 分支。
- [x] partial 候选是否只在已有场景参与选择，没有被误当成全量结果。
- [x] online depth residual smoke 是否先验证 proxy 信号，再决定是否修改 CUDA rasterizer。

## 初始风险

- train split proxy 可能过拟合训练视角；若 Round 1 看起来正向，需要追加 held-out view 或更严格的 view-stride 验证。
- 0004 partial 候选只覆盖少数室内场景，不能直接代表全数据集均值。
- online depth residual 的 proxy 版本只适合判断方向，不适合最终报告质量。

## Round 1 Review

Validation-driven selector 的负例回退是有效的，但当前 train-split proxy 不能作为默认策略依据。

- MipNeRF360：`train_best_psnr` 几乎持平 baseline，但额外增加约 222k Gaussian；`train_qcgi` 将容量代价压到约 39k Gaussian，却低于 baseline 0.0084 PSNR。
- DB/Tandt：`train_best_psnr` 与 `train_qcgi` 选择一致，但 test PSNR 低于 baseline 0.0080，LPIPS 也略差。
- 结论：selector 可以保留为 scene-conditioned 分析工具，但不应替换 FastGS 默认 densification / pruning 策略。

## Round 2 Review

Online depth residual proxy 的覆盖率已经足够用于第一轮判断，但信号方向不稳定。

- `residual-depth` 在 indoor / counter 上对 edge 或 RGB error 有轻微优势，但幅度很小。
- `residual-inv` 只在 `stump` 的 RGB error 上明显更好，同时 edge 对齐更弱。
- 结论：不要直接把单一 residual 图接入 pruning。更值得推进的是 orientation-aware residual selector：按场景或视角选择 depth / inverse-depth / prior 的组合，再用 held-out view 验证。

## 最终判定

- 当前可合入：文档、候选构建脚本、selector 实验结果、online residual proxy 诊断脚本。
- 当前不默认化：train-split selector 与单一 online residual map。
- 下一步优先级：先做 held-out selector；如果继续 residual 方向，则先做 orientation-aware gating，再考虑 CUDA rasterizer 内联实现。
