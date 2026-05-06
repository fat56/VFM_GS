# 路线图

## 进行中

- `0001_vfm_topology_scorer`: staged-budget cached edge 已在 bicycle 和 garden 两个 scene 上超过各自 baseline；no-effect/cadence 控制与 post-prune fine-tune 探测已完成。下一步先补第三个 scene 的 staged edge 复验，再改进 DINO scorer。

## 排队

- 再跑一个 scene 的 staged-budget edge 复验，然后将 edge proxy 固化为 v1 positive control。
- 改进 DINO scorer：从 token-edge proxy 转向 patch descriptor / feature projection 对齐。
- 构建 `max_width=518` 或 `640` 的 full-scene `dinov2_vits14` cache，记录 cache build time、disk size 和 validate 结果。
- 增加固定 patch-descriptor projection scorer，对比 token-edge proxy。
- 设计 dense post-prune recovery schedule，避免 30k 后恢复训练实际更新次数过少。

## 阻塞

- DINOv2 token-edge 默认 30k 指标领先但预算过大；350k staged budget 下只在 LPIPS 上超过 baseline，PSNR/SSIM 仍低。当前 DINO token-edge 还不能作为预算高效的主结果。

## 已完成

- 将原生 FastGS 扁平结构迁移到 `src/vfm_gs` 包结构。
- 建立 `vfm_topology_scorer` mock v1，验证 scorer registry 与 FastGS baseline 的输出兼容。
- 建立 `build_vfm_cache` CLI 和 `cached_edge_l1` 后端，验证离线缓存读取链路。
- 为训练输出补充 scorer/backend/阈值/权重等 provenance 日志。
- 增加 `npz_uint8` compact cache storage，将 bicycle edge cache 从约 189MB 降到约 35MB。
- 增加 `validate_vfm_cache` CLI，支持 manifest、checksum、shape、source-image 和 backend 校验。
- 为 cached backend 增加训练前 preflight，提前暴露缺失 cache 或 backend 不匹配。
- 增加 `vfm_backend_probe` CLI，记录当前环境和 DINOv2 cache-size feasibility。
- 增加 optional `dinov2_vits14` / `dinov2_vitb14` cache builder，并在当前 PyTorch 1.12.1 环境完成 4-image ViT-S/14 cache 快速验证。
- 增加 `dinov2_token_edge_l1` scorer backend，完成 194 张图 DINOv2 cache build/validate，以及 220-iteration bicycle 快速验证的 train/render/metrics。
- 完成 baseline、compact cached edge、DINOv2 token-edge 的 30k `-r 8` matched ablation；DINO token-edge 指标最好，但点数和渲染成本也最高。
- 完成 t075/w010 budget-control probe；现有阈值/权重 knob 无法充分控制 VFM densification 点数。
- 增加 `vfm_importance_weight` 并完成 i0.25 30k probe；DINO 点数下降但仍未达成 budget matching。
- 增加 `vfm_importance_mode=max|weighted|rgb_only` 并完成 `rgb_only` 30k probe；直接关闭 VFM densification 仍未达成 budget matching。
- 增加 `target_gaussian_count` final-prune control；首版 high-score 批量裁剪是负例，已修正为 low-score 批量裁剪。
- 完成 low-score final target-prune 30k probe；点数精确匹配 baseline，但 edge/DINO 质量均低于 baseline，说明单次最终裁剪不是可用的公平预算方案。
- 增加 `target_gaussian_staged` / `target_gaussian_stage_margin` / `target_gaussian_stage_interval`，支持训练期分阶段预算控制。
- 完成 240k staged budget 30k probe；质量较 final-only 大幅恢复，但仍低于 baseline。
- 完成 300k staged budget 30k probe；DINO LPIPS 几乎追平 baseline，但 PSNR/SSIM 仍低。
- 完成 350k staged budget 30k probe；cached edge 在约 340k 点数下超过 baseline，DINO 350k 仅 LPIPS 超过 baseline。
- 完成 garden staged-budget edge 复验；edge 在第二个 scene 上继续超过 baseline。
- 完成 no-effect/cadence control；`fastgs_photometric + densification_interval=100` 与 zero-weight VFM runs 都在约 410k Gaussians，说明此前 no-effect 高点数主要来自 densification cadence。
- 增加 `post_prune_finetune_iterations` 并完成 final-prune-plus-fine-tune 探测；严格 240k 预算下质量明显优于 final-only，但仍低于 baseline 与 350k staged positive control。
