# 0008 Residual Orientation Gating

## 核心假设

0006 的 online depth residual proxy 已经说明：`depth` 与 `inverse-depth` 两个方向在不同场景上不稳定，单一 residual map 不能直接接入 pruning。0007 又说明 train-view selector 不可靠。0008 验证一个更保守的方向：

不再让 selector 选择完整训练分支，而是在 residual 层面按视角选择 `static prior`、`depth residual`、`inverse-depth residual` 三者之一，先判断 orientation-aware gating 是否真的比固定 prior 更接近 RGB error / GT edge。

这轮仍是离线诊断，不修改 CUDA rasterizer，也不重新训练。

## 变体 / 配置

- 变体：`fastgs_big` baseline checkpoint
- 输入：Depth Anything V2-S depth cache + Gaussian center z-buffer depth proxy
- 信号：
  - `prior`：Depth Anything static prior top-k
  - `residual_depth`：`abs(norm(proxy_depth) - prior)`
  - `residual_inv`：`abs(norm(1 / proxy_depth) - prior)`
- 诊断选择器：
  - `rgb_oracle_pick`：每个视角按 RGB error IoU 选最强信号，只作上限
  - `edge_proxy_pick`：每个视角按 GT edge IoU 选最强信号，用来近似非 RGB-error 的结构 proxy
  - `l1_oracle_pick`：每个视角按 top-k 区域 RGB L1 选最强信号，只作上限
- 汇总脚本：`scripts/summarize_0008_residual_orientation_gating.py`

## 数据集

- 第一轮 smoke：复用 0006 的 `bonsai/kitchen/room/counter/stump`
- 第二轮全量 proxy：MipNeRF360 9 scenes
- 视角：`max_views=8, view_stride=17`
- top-k：`0.10`
- proxy splat radius：`1`

## 判定标准

- 若 `rgb_oracle_pick` 相比 static prior 的 RGB IoU 提升很小，说明 orientation gating 上限不足。
- 若 `edge_proxy_pick` 经常与 `rgb_oracle_pick` 一致，且 RGB IoU 不低于 static prior，说明结构 proxy 有继续价值。
- 若 `edge_proxy_pick` 只提升 edge IoU、但 RGB IoU 下降，则它适合做保守 protect，不适合作为 pruning/rerank 主信号。

## 决策

只有当全 9 场景显示 orientation-aware gating 对 static prior 有稳定 proxy 优势，才进入下一步 renderer 深度输出或 pruning 接入。否则 residual 分支保留为离线分析工具。
