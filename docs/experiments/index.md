# 实验索引

| ID | 状态 | 变体 | 打分器 | 数据集 | 结果 | 决策 |
|---|---|---|---|---|---|---|
| 0001_vfm_topology_scorer | v1 收束；MipNeRF360 与 DB 边缘代理正向，Tandt 容量保护只部分恢复，DINO descriptor 预算未转正 | fastgs_baseline | vfm_topology_scorer / cached_edge_l1 / dinov2_token_edge_l1 / dinov2_descriptor_cosine | mipnerf360 全 9 场景 `-r 8`；tandt_db/db；tandt_db/tandt | results.md | 固化 `cached_edge_l1` 为 v1 proxy 控制组；下一版改预算感知 scorer、自动容量保护与 pruning 后局部恢复时序 |
