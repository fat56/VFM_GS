# Roadmap

## Active

- `0001_vfm_topology_scorer`: optional DINOv2 cache builder smoke 已通过。下一步接 DINO feature-map scorer，让训练实际消费 `dinov2_patchtokens` cache。

## Queued

- 增加 DINO feature-map scorer，优先固定 RGB/SH0 描述子投影到 DINO patch grid，避免第一版引入 learned adapter。
- 构建 full-scene `dinov2_vits14` cache，记录 cache build time、disk size 和 validate 结果。
- 运行 bicycle 长程 baseline vs cached edge vs DINOv2 scorer 消融。

## Blocked

- DINOv2 已能离线出 cache，但训练 scorer 尚未对齐 rendered feature 与 DINO patch token；这一步完成前不能声称真实 VFM 质量收益。

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
