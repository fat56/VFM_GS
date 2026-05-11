# 实验索引

| ID | 状态 | 变体 | 打分器 | 数据集 | 结果 | 决策 |
|---|---|---|---|---|---|---|
| 0001_vfm_topology_scorer | 收束；DINO descriptor densify-only top-k25 weighted i0.50/i0.70 作为 VFM_GS 初步验证，token-edge selector 与 cached-edge proxy 作为辅助证据 | fastgs_baseline | vfm_topology_scorer / cached_edge_l1 / dinov2_token_edge_l1 / dinov2_descriptor_cosine | mipnerf360 全 9 场景 `-r 8`；tandt_db/db；tandt_db/tandt；部分 high-res 复验 | summary.md / results.md | 保留 DINO descriptor top-k25 `max` 为无回退质量证据，weighted i0.50 为容量受控档，weighted i0.70 为质量折中档；不再扩展 COLMAP sparse proxy、硬 candidate cap 或 staged target-prune |
| 0002_depth_anything_dense_prior | Phase 0 已通过；Depth Anything V2-S depth-edge prior 已完成 high-res bicycle 620-step cache/build/train/render/metrics smoke，进入 30k pilot | fastgs_baseline | fastgs_photometric / vfm_topology_scorer / depth_anything_depth_edge_prior / depth_anything_depth_prior | phase 0 覆盖 MipNeRF360、DB、Tandt 全场景原图 1.6K；Depth Anything 从 high-res bicycle smoke/pilot 开始 | results.md | 620-step 只证明链路健康且略低于 matched baseline；下一步跑 high-res bicycle 30k，并与 Phase 0 FastGS big 和 0001 descriptor weighted i0.50/i0.70 对照 |
