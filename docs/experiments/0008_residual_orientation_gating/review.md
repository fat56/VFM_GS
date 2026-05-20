# 0008 Residual Orientation Gating Review

## Review Checklist

- [x] summary smoke 是否复用 0006 输出跑通。
- [x] full proxy 是否覆盖 MipNeRF360 9 scenes。
- [x] `rgb_oracle_pick` 相对 static prior 是否有足够上限收益。
- [x] `edge_proxy_pick` 是否能在不看 RGB error 的情况下保持或提升 RGB IoU。
- [x] 若 proxy 仍不稳，是否停止接入 pruning，而不是直接改 renderer。

## Round 1 Review

- Smoke 覆盖 39 个 train views，平均 proxy coverage 为 0.5448。
- RGB oracle 相对 static prior 的 RGB IoU 平均 +0.0129，说明 residual orientation 至少有离线上限。
- Edge proxy 相对 static prior 的 RGB IoU 平均 +0.0075、edge IoU 平均 +0.0093，短期内值得扩到全 9 场景。
- Edge proxy 与 RGB oracle match rate 为 0.5641，还不足以作为可部署 selector；它更可能适合后续做 protect/rerank 的软约束。

## Round 2 Review

- Full proxy 覆盖 MipNeRF360 9 scenes / 71 views，平均 proxy coverage 为 0.5849。
- RGB oracle 相对 static prior 的 RGB IoU 平均 +0.0136，说明 orientation-aware residual 的离线上限在全量场景上保持。
- Edge proxy 相对 static prior 的 RGB IoU 平均 +0.0088、edge IoU 平均 +0.0104，且逐场景 RGB delta 全为正。
- Edge proxy 与 RGB oracle match rate 为 0.6479；这个数值比 smoke 更好，但仍不适合作为硬 selector。
- 下一步只接受 late/protect/rerank 级别的训练接入，并保留 Gaussian budget 约束；不让 residual 分支主导 densification。

## 初始风险

- Gaussian center z-buffer 仍是 proxy，不等于 rasterizer alpha-weighted depth。
- RGB oracle 只是上限，不能作为可部署 gating。
- GT edge 也不是纯在线信号；它只能用来判断结构方向，不代表最终策略。
