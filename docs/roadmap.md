# Roadmap

## Active

- `0001_vfm_topology_scorer`: 现有 `vfm_loss_thresh/vfm_weight` 预算探针已完成，但不能把 Gaussian count 拉回 baseline。下一步实现显式 VFM importance 控制。

## Queued

- 增加 `vfm_importance_weight` 或 `vfm_importance_mode`，让 VFM densification 强度和 pruning fusion 权重分离。
- 用新的 importance 控制重跑 30k budget-controlled 消融，目标 Gaussian count 接近 baseline。
- 构建 `max_width=518` 或 `640` 的 full-scene `dinov2_vits14` cache，记录 cache build time、disk size 和 validate 结果。
- 增加固定 patch-descriptor projection scorer，对比 token-edge proxy。

## Blocked

- DINOv2 token-edge 在 30k 指标上领先，但默认 Gaussian count 约为 baseline 的 2.04x；t075/w010 仍约为 baseline 的 1.76x。显式 importance 控制出来前不能把收益完全归因于 VFM signal。

## Completed

- 将原生 FastGS 扁平结构迁移到 `src/vfm_gs` 包结构。
- 建立 `vfm_topology_scorer` mock v1，验证 scorer registry 与 FastGS baseline 的输出兼容。
- 建立 `build_vfm_cache` CLI 和 `cached_edge_l1` 后端，验证离线缓存读取链路。
- 为训练输出补充 scorer/backend/阈值/权重等 provenance 日志。
- 增加 `npz_uint8` compact cache storage，将 bicycle edge cache 从约 189MB 降到约 35MB。
- 增加 `validate_vfm_cache` CLI，支持 manifest、checksum、shape、source-image 和 backend 校验。
- 为 cached backend 增加训练前 preflight，提前暴露缺失 cache 或 backend 不匹配。
- 增加 `vfm_backend_probe` CLI，记录当前环境和 DINOv2 cache-size feasibility。
- 增加 optional `dinov2_vits14` / `dinov2_vitb14` cache builder，并在当前 PyTorch 1.12.1 环境完成 4-image ViT-S/14 smoke cache validation。
- 增加 `dinov2_token_edge_l1` scorer backend，完成 194-image DINOv2 cache build/validate 和 220-iteration bicycle smoke train/render/metrics。
- 完成 baseline、compact cached edge、DINOv2 token-edge 的 30k `-r 8` matched ablation；DINO token-edge 指标最好，但点数和渲染成本也最高。
- 完成 t075/w010 budget-control probe；现有阈值/权重 knob 无法充分控制 VFM densification 点数。
