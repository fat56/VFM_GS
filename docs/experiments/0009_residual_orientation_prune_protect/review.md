# 0009 Residual Orientation Prune-Protect Review

## Checklist

- [x] 新 backend 是否只影响 `depth_anything_residual_orientation`，不改变旧 depth prior 配置。
- [x] Smoke 是否覆盖 scorer preflight、train、render、metrics。
- [ ] Pilot 是否在双卡 tmux 运行。
- [ ] 是否按轮次 commit/push。
- [ ] 若三场景整体不稳，是否停止训练接入。

## Smoke Review

- 新参数已出现在 `vfm_gs.cli.train --help`，配置加载路径正常。
- 700-step train/render/metrics 全链路通过。
- 直接 scorer smoke 在 `DENSIFY=False` 下打印 late protect 日志，说明 `vfm_prune_protect_mode=rgb_prune_auto_topk` 与 residual orientation counts 已连通。
- Smoke 只证明代码路径健康，不说明 30k 质量收益。

## 风险

- 训练期 center z-buffer depth 仍不是 alpha-weighted renderer depth。
- `edge_iou` selector 使用 GT image edge，是诊断型 proxy，不是纯部署信号。
- Late protect 会改变 pruning，而不是 densification；若 baseline 的后期收益来自 pruning 重分配，保护过强仍可能伤害质量。
