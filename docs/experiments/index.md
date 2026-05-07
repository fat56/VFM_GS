# 实验索引

| ID | 状态 | 变体 | 打分器 | 数据集 | 结果 | 决策 |
|---|---|---|---|---|---|---|
| 0001_vfm_topology_scorer | v1 边缘代理正向控制组完成；DINO descriptor 已跑通 | fastgs_baseline | vfm_topology_scorer / cached_edge_l1 / dinov2_token_edge_l1 / dinov2_descriptor_cosine | mipnerf360/bicycle,garden,counter `-r 8` | results.md | 下一步调 descriptor 阈值/cache width，并设计 dense post-prune recovery |
