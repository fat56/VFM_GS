# Roadmap

## Active

- `0001_vfm_topology_scorer`: low-score final target prune 已完成 30k probe，但一次性最终裁剪质量低于 baseline。下一步做 staged budget control 或 post-prune fine-tune。

## Queued

- 增加 staged target budget：densification/pruning 后分阶段向目标点数裁剪，让模型继续训练恢复。
- 增加 post-prune fine-tune 选项，对比 staged budget 和最终裁剪后恢复训练。
- 增加 no-effect control：`vfm_importance_mode=rgb_only` + `vfm_weight=0`，测 VFM scorer 开销和 cache overhead。
- 构建 `max_width=518` 或 `640` 的 full-scene `dinov2_vits14` cache，记录 cache build time、disk size 和 validate 结果。
- 增加固定 patch-descriptor projection scorer，对比 token-edge proxy。

## Blocked

- DINOv2 token-edge 在 30k 指标上领先，但默认 Gaussian count 约为 baseline 的 2.04x；importance weight 0.25 和 `rgb_only` 仍约为 baseline 的 1.72x-1.74x。final target prune 能匹配点数但质量低于 baseline，所以还不能把收益完全归因于 VFM signal。

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
- 增加 `vfm_importance_weight` 并完成 i0.25 30k probe；DINO 点数下降但仍未达成 budget matching。
- 增加 `vfm_importance_mode=max|weighted|rgb_only` 并完成 `rgb_only` 30k probe；直接关闭 VFM densification 仍未达成 budget matching。
- 增加 `target_gaussian_count` final-prune control；首版 high-score bulk pruning 是负例，已修正为 low-score bulk pruning。
- 完成 low-score final target-prune 30k probe；点数精确匹配 baseline，但 edge/DINO 质量均低于 baseline，说明一次性最终裁剪不是可用的公平预算方案。
