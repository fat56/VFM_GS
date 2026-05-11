# 实验索引

| ID | 状态 | 变体 | 打分器 | 数据集 | 结果 | 决策 |
|---|---|---|---|---|---|---|
| 0001_vfm_topology_scorer | 收束；DINO descriptor densify-only top-k25 weighted i0.50/i0.70 作为 VFM_GS 初步验证，token-edge selector 与 cached-edge proxy 作为辅助证据 | fastgs_baseline | vfm_topology_scorer / cached_edge_l1 / dinov2_token_edge_l1 / dinov2_descriptor_cosine | mipnerf360 全 9 场景 `-r 8`；tandt_db/db；tandt_db/tandt；部分 high-res 复验 | summary.md / results.md | 保留 DINO descriptor top-k25 `max` 为无回退质量证据，weighted i0.50 为容量受控档，weighted i0.70 为质量折中档；不再扩展 COLMAP sparse proxy、硬 candidate cap 或 staged target-prune |
| 0002_depth_anything_dense_prior | Phase 0 已通过；RTX 5090 high-res FastGS big 全 13 场景对齐 0001/4090D baseline，进入 Depth Anything backend/cache smoke | fastgs_baseline | fastgs_photometric / vfm_topology_scorer / Depth Anything dense depth prior | phase 0 覆盖 MipNeRF360、DB、Tandt 全场景原图 1.6K；Depth Anything 也直接从 high-res bicycle smoke/pilot 开始 | results.md | 实现或确认 dense depth cache/backend，保持 `vfm_weight=0.0` 只影响 densification，并与 high-res 0001 descriptor weighted i0.50/i0.70 对照 |
