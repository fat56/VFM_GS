# 0008 Residual Orientation Gating 结果

## 当前状态

Round 1 summary smoke 已完成。脚本可复用 0006 residual proxy 输出，并产出 per-view / scene / dataset / overall 四级汇总。

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

## Round 2：待填

待记录：

- MipNeRF360 全 9 场景整体 summary；
- 逐场景 orientation 分布；
- 是否进入 renderer depth 输出或 pruning 接入。
