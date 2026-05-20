# 0006 Validation Selector + Online Depth Residual Review

## Review Checklist

- [ ] `baseline` method 是否在候选表中每个场景至少出现一次。
- [ ] selector 输出是否同时覆盖 MipNeRF360 与 DB/Tandt。
- [ ] `train_qcgi` 是否真的回退负例，而不是偏好 Gaussian 更多的 prior 分支。
- [ ] partial 候选是否只在已有场景参与选择，没有被误当成全量结果。
- [ ] online depth residual smoke 是否先验证 proxy 信号，再决定是否修改 CUDA rasterizer。

## 初始风险

- train split proxy 可能过拟合训练视角；若 Round 1 看起来正向，需要追加 held-out view 或更严格的 view-stride 验证。
- 0004 partial 候选只覆盖少数室内场景，不能直接代表全数据集均值。
- online depth residual 的 proxy 版本只适合判断方向，不适合最终报告质量。
