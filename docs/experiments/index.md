# 实验索引

| ID | 状态 | 变体 | 打分器 | 数据集 | 结果 | 决策 |
|---|---|---|---|---|---|---|
| 0001_vfm_topology_scorer | v1 收束；DINO token-edge i0.50 完成 MipNeRF360 全 9 场景质量候选，边缘代理在 MipNeRF360/DB 正向但 Tandt 不稳 | fastgs_baseline | vfm_topology_scorer / cached_edge_l1 / dinov2_token_edge_l1 / dinov2_descriptor_cosine | mipnerf360 全 9 场景 `-r 8`；tandt_db/db；tandt_db/tandt | results.md | 固化 `cached_edge_l1` 为 proxy 控制组、DINO i0.50 为质量候选；下一版改预算感知 scorer、自动容量保护与 pruning 后局部恢复时序 |
