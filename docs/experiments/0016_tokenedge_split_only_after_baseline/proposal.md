# 0016 DINO Token-Edge Split-Only After Baseline

## 核心问题

0005 说明 DINO token-edge 作为全局强 densification prior 会明显增点且 PSNR 不稳。本实验只问一个更窄的问题：

> 在 FastGS baseline 已经跑到 15K-20K、RGB loss 对新增区域的指导变弱之后，DINO token-edge 能否只通过 split 分支补充结构细节，让后续 5K 的 PSNR/SSIM/LPIPS 继续增长？

## 设计

- 数据集：MipNeRF360 全 9 场景。
- 起点：复用 `output/0002/fastgs_big_baseline_checkpoint_curve/mipnerf360/{scene}/fastgs_big_30k_curve_r_auto` 的 checkpoint-curve PLY。
- 起点 iteration：默认 `16000 / 18000 / 20000`，覆盖 15K-20K window；每个起点续跑 5K。
- VFM cache：`output/0001/vfm_cache_large/{scene}_dinov2_vitl14_token_edge_w1600`。
- DINO 方法：`dinov2_token_edge_l1`，ViT-L/14 1.6K token-edge 2D map。
- 打分器：`vfm_topology_scorer`，多视角 sample camera 和 per-Gaussian `accum_metric_counts` 链路沿用 FastGS。
- importance：`vfm_only`，不混入 RGB importance。
- metric map：token-edge residual top-k25。
- 分支控制：
  - clone disabled。
  - split enabled。
  - densify 内 opacity/size prune disabled。
  - 15K 后 periodic final prune disabled。
  - opacity reset disabled。

这里按“只作为 split 的判定，不判定 clone”实现；split 内部替换原 Gaussian 的行为保留，因为这是 FastGS split 操作本身的一部分。

## 运行

```bash
bash scripts/run_0016_tokenedge_split_only_tmux.sh start
```

默认双卡分组：

- GPU0：`bicycle flowers garden stump treehill`
- GPU1：`room counter kitchen bonsai`

如需只跑一个起点：

```bash
SWITCH_ITERS="16000" bash scripts/run_0016_tokenedge_split_only_tmux.sh start
```

## 输出

- `output/0016/tokenedge_split_only_after_baseline/mip_g0`
- `output/0016/tokenedge_split_only_after_baseline/mip_g1`
- `output/0016/tokenedge_split_only_after_baseline/mipnerf360_combined`

每个 run 会先 render/metric 起点 iteration，再续跑 5K，最后 render/metric final iteration。

## 判读方式

先不做 oracle 或 selector，只看每个起点的：

- `delta_psnr = final - start`
- `delta_ssim = final - start`
- `delta_lpips = final - start`
- `delta_gs_num = final - start`

如果某些场景在 16K/18K/20K 后仍能通过 token-edge split-only 获得稳定质量增长，再进入 0017 或 0016 Round 2 做 scene-adaptive/容量控制。
