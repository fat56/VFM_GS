# 0008 Residual Orientation Gating 结果

## 当前状态

Round 2 full MipNeRF360 proxy 已完成。0008 的离线结论是：orientation-aware residual 有稳定 proxy 上限，`edge_proxy_pick` 在全 9 场景上相对 static prior 的 RGB IoU 与 edge IoU 均为正，但它还只是 per-view proxy，不应直接作为强 pruning selector 接入。

## Round 1：summary smoke

输入：

- `output/0006/online_depth_residual_proxy/indoor_g1/per_view.csv`
- `output/0006/online_depth_residual_proxy/mixed_g0/per_view.csv`

输出：

- `output/0008/residual_orientation_gating_smoke/per_view_orientation.csv`
- `output/0008/residual_orientation_gating_smoke/scene_orientation_summary.csv`
- `output/0008/residual_orientation_gating_smoke/dataset_orientation_summary.csv`
- `output/0008/residual_orientation_gating_smoke/overall_orientation_summary.json`

整体结果：

| views | proxy cov. | prior RGB IoU | RGB oracle IoU | RGB oracle delta | edge proxy RGB IoU | edge proxy RGB delta | edge proxy edge delta | edge/RGB oracle match |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 39 | 0.5448 | 0.0632 | 0.0761 | +0.0129 | 0.0706 | +0.0075 | +0.0093 | 0.5641 |

Pick 分布：

| pick source | prior | residual_depth | residual_inv |
|---|---:|---:|---:|
| RGB oracle | 14 | 8 | 17 |
| edge proxy | 10 | 20 | 9 |
| L1 oracle | 9 | 12 | 18 |

逐场景摘要：

| scene | views | prior RGB IoU | RGB oracle delta | edge proxy RGB delta | edge proxy edge delta | edge/RGB oracle match |
|---|---:|---:|---:|---:|---:|---:|
| bonsai | 8 | 0.1197 | +0.0101 | +0.0075 | +0.0056 | 0.7500 |
| counter | 8 | 0.0477 | +0.0129 | +0.0051 | +0.0096 | 0.5000 |
| kitchen | 8 | 0.0560 | +0.0117 | +0.0078 | +0.0149 | 0.6250 |
| room | 8 | 0.0488 | +0.0158 | +0.0131 | +0.0134 | 0.6250 |
| stump | 7 | 0.0409 | +0.0141 | +0.0034 | +0.0019 | 0.2857 |

初步判断：orientation-aware residual 在五场景上有离线上限，且 edge proxy 对 RGB IoU 没有明显负向；但 `edge_proxy_pick` 偏向 `residual_depth`，而 RGB / L1 oracle 更偏向 `residual_inv`，说明它仍不是可靠 selector。继续跑 MipNeRF360 全 9 场景确认稳定性。

## Round 2：MipNeRF360 full proxy

输入：

- `output/0008/residual_proxy/mip_g0/per_view.csv`
- `output/0008/residual_proxy/mip_g1/per_view.csv`

输出：

- `output/0008/residual_orientation_gating_mipnerf360/per_view_orientation.csv`
- `output/0008/residual_orientation_gating_mipnerf360/scene_orientation_summary.csv`
- `output/0008/residual_orientation_gating_mipnerf360/dataset_orientation_summary.csv`
- `output/0008/residual_orientation_gating_mipnerf360/overall_orientation_summary.json`

整体结果：

| views | proxy cov. | prior RGB IoU | RGB oracle IoU | RGB oracle delta | edge proxy RGB IoU | edge proxy RGB delta | edge proxy edge delta | edge/RGB oracle match |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 71 | 0.5849 | 0.0626 | 0.0762 | +0.0136 | 0.0714 | +0.0088 | +0.0104 | 0.6479 |

Pick 分布：

| pick source | prior | residual_depth | residual_inv |
|---|---:|---:|---:|
| RGB oracle | 18 | 21 | 32 |
| edge proxy | 14 | 28 | 29 |
| L1 oracle | 15 | 22 | 34 |

逐场景摘要：

| scene | views | prior RGB IoU | RGB oracle delta | edge proxy RGB delta | edge proxy edge delta | edge/RGB oracle match |
|---|---:|---:|---:|---:|---:|---:|
| bicycle | 8 | 0.0719 | +0.0113 | +0.0093 | +0.0111 | 0.7500 |
| bonsai | 8 | 0.1197 | +0.0101 | +0.0075 | +0.0056 | 0.7500 |
| counter | 8 | 0.0477 | +0.0129 | +0.0051 | +0.0096 | 0.5000 |
| flowers | 8 | 0.1013 | +0.0026 | +0.0024 | +0.0047 | 0.7500 |
| garden | 8 | 0.0537 | +0.0212 | +0.0071 | +0.0195 | 0.5000 |
| kitchen | 8 | 0.0560 | +0.0117 | +0.0078 | +0.0149 | 0.6250 |
| room | 8 | 0.0488 | +0.0158 | +0.0131 | +0.0134 | 0.6250 |
| stump | 7 | 0.0409 | +0.0141 | +0.0034 | +0.0019 | 0.2857 |
| treehill | 8 | 0.0208 | +0.0230 | +0.0230 | +0.0119 | 1.0000 |

判读：

- `RGB oracle` 上限为 +0.0136 RGB IoU，说明 residual orientation 不是噪声；它确实能找到 static prior 覆盖不到的一批区域。
- `edge_proxy_pick` 在 9/9 场景上 RGB IoU delta 为正，平均 +0.0088；edge IoU 也平均 +0.0104。
- `edge_proxy_pick` 与 `RGB oracle` 的 match rate 为 0.6479，比 smoke 的 0.5641 更好，但仍不足以支撑“硬切换主信号”。
- `residual_inv` 在 RGB/L1 oracle 中更常被选中，`edge_proxy` 则在 `residual_depth` 与 `residual_inv` 间接近均分；后续若接入训练，应保持预算约束和 late/protect 语义，不应扩大 densification 主导权。

决策：进入下一步保守训练实验，只尝试 late edge-orientation residual protect/rerank，不做强 pruning selector，不默认化。
