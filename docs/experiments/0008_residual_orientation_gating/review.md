# 0008 Residual Orientation Gating Review

## Review Checklist

- [x] summary smoke 是否复用 0006 输出跑通。
- [ ] full proxy 是否覆盖 MipNeRF360 9 scenes。
- [ ] `rgb_oracle_pick` 相对 static prior 是否有足够上限收益。
- [ ] `edge_proxy_pick` 是否能在不看 RGB error 的情况下保持或提升 RGB IoU。
- [ ] 若 proxy 仍不稳，是否停止接入 pruning，而不是直接改 renderer。

## Round 1 Review

- Smoke 覆盖 39 个 train views，平均 proxy coverage 为 0.5448。
- RGB oracle 相对 static prior 的 RGB IoU 平均 +0.0129，说明 residual orientation 至少有离线上限。
- Edge proxy 相对 static prior 的 RGB IoU 平均 +0.0075、edge IoU 平均 +0.0093，短期内值得扩到全 9 场景。
- Edge proxy 与 RGB oracle match rate 为 0.5641，还不足以作为可部署 selector；它更可能适合后续做 protect/rerank 的软约束。

## 初始风险

- Gaussian center z-buffer 仍是 proxy，不等于 rasterizer alpha-weighted depth。
- RGB oracle 只是上限，不能作为可部署 gating。
- GT edge 也不是纯在线信号；它只能用来判断结构方向，不代表最终策略。
