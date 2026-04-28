# 0001 VFM Topology Scorer Runbook

## Full Baseline

```bash
uv run --active python -m vfm_gs.cli.train --variant fastgs_baseline -s <dataset>/<scene> -m output/0001_baseline/<scene> --eval
uv run --active python -m vfm_gs.cli.render -m output/0001_baseline/<scene> --skip_train
uv run --active python -m vfm_gs.cli.metrics -m output/0001_baseline/<scene>
```

## Mock VFM Topology v1

```bash
uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_scorer.yaml \
  -s <dataset>/<scene> \
  -m output/0001_vfm/<scene> \
  --eval
```

当前 v1 使用 `vfm_topology_scorer` + `mock_l1` 后端。它不是真实 VFM 质量实验，而是验证 SH0 render、pixel error map、metric map、Gaussian 计数和 FastGS 分数融合链路。

## Cached Edge Proxy v1

`cached_edge_l1` 后端先用 GT 图像的归一化 luminance edge map 作为轻量离线缓存代理。它不是最终 VFM 后端，但能验证 `image_name -> cache entry -> pixel_error_map` 的真实缓存读取流程。
训练入口会在 Scene 加载前自动执行 cache preflight；这里仍显式运行 `validate_vfm_cache`，用于把缓存检查作为实验流程的一部分记录下来。

```bash
uv run --active python -m vfm_gs.cli.build_vfm_cache \
  -s datasets/mipnerf360/bicycle \
  -i images_8 \
  -o output/0001/vfm_cache/bicycle_edge_u8 \
  --max_width 640 \
  --storage npz_uint8

uv run --active python -m vfm_gs.cli.validate_vfm_cache \
  -c output/0001/vfm_cache/bicycle_edge_u8 \
  -s datasets/mipnerf360/bicycle \
  -i images_8 \
  --backend cached_edge_l1

uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_cached_edge_compact.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0001/vfm_cached_edge/bicycle \
  --eval
```

## Optional DINOv2 Cache Smoke

DINOv2 cache building is an offline artifact path for real VFM features. The fast smoke validates `dinov2_patchtokens` cache generation and manifest compatibility before running a training scorer.

When torch.hub remote access is rate-limited, clone the official repository under ignored output state and pass it explicitly:

```bash
git clone https://github.com/facebookresearch/dinov2.git output/0001/external/dinov2

uv run --active python -m vfm_gs.cli.build_vfm_cache \
  -s datasets/mipnerf360/bicycle \
  -i images_8 \
  -o output/0001/vfm_cache/bicycle_dinov2_vits14_smoke \
  --backend dinov2_vits14 \
  --dinov2_repo output/0001/external/dinov2 \
  --max_width 224 \
  --storage npy_float16 \
  --limit 4

uv run --active python -m vfm_gs.cli.validate_vfm_cache \
  -c output/0001/vfm_cache/bicycle_dinov2_vits14_smoke \
  --backend dinov2_vits14
```

For DINOv2, `--storage` defaults to `npy_float16` when omitted. `npz_uint8` is intentionally rejected for DINO patch-token caches.

## DINOv2 Token-Edge Scorer v1

`dinov2_token_edge_l1` 是第一版训练期消费 DINOv2 cache 的 scorer backend。它不在训练循环里跑 DINOv2，而是把离线 `dinov2_patchtokens` 转成 token-edge topology map，再和 SH0 渲染图的 pooled edge map 比较。

```bash
uv run --active python -m vfm_gs.cli.build_vfm_cache \
  -s datasets/mipnerf360/bicycle \
  -i images_8 \
  -o output/0001/vfm_cache/bicycle_dinov2_vits14 \
  --backend dinov2_vits14 \
  --dinov2_repo output/0001/external/dinov2 \
  --max_width 224

uv run --active python -m vfm_gs.cli.validate_vfm_cache \
  -c output/0001/vfm_cache/bicycle_dinov2_vits14 \
  --backend dinov2_vits14

uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_dinov2_token_edge.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0001/vfm_dinov2_token_edge_bicycle_smoke \
  --eval \
  --iterations 220 \
  --densify_from_iter 50 \
  --densify_until_iter 220 \
  --densification_interval 50 \
  --test_iterations 220 \
  --save_iterations 220 \
  --checkpoint_iterations 220 \
  -r 8

uv run --active python -m vfm_gs.cli.render -m output/0001/vfm_dinov2_token_edge_bicycle_smoke --skip_train
uv run --active python -m vfm_gs.cli.metrics -m output/0001/vfm_dinov2_token_edge_bicycle_smoke
```

## 30k Matched Ablation

220-iteration runs are smoke checks only. Use this 30k `-r 8` set as the minimum quality gate while iterating on scorer behavior:

```bash
uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0001/baseline_bicycle_30k_r8 \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  -r 8

uv run --active python -m vfm_gs.cli.render -m output/0001/baseline_bicycle_30k_r8 --skip_train
uv run --active python -m vfm_gs.cli.metrics -m output/0001/baseline_bicycle_30k_r8

uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_cached_edge_compact.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0001/vfm_cached_edge_compact_bicycle_30k_r8 \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  -r 8

uv run --active python -m vfm_gs.cli.render -m output/0001/vfm_cached_edge_compact_bicycle_30k_r8 --skip_train
uv run --active python -m vfm_gs.cli.metrics -m output/0001/vfm_cached_edge_compact_bicycle_30k_r8

uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_dinov2_token_edge.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0001/vfm_dinov2_token_edge_bicycle_30k_r8 \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  -r 8

uv run --active python -m vfm_gs.cli.render -m output/0001/vfm_dinov2_token_edge_bicycle_30k_r8 --skip_train
uv run --active python -m vfm_gs.cli.metrics -m output/0001/vfm_dinov2_token_edge_bicycle_30k_r8
```

## Budget-Control Probe

Existing knobs can be overridden from the command line. The first probe used a higher VFM threshold and lower pruning fusion weight:

```bash
uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_cached_edge_compact.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0001/vfm_cached_edge_t075_w010_bicycle_30k_r8 \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --vfm_loss_thresh 0.75 \
  --vfm_weight 0.10 \
  -r 8

uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_dinov2_token_edge.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0001/vfm_dinov2_token_edge_t075_w010_bicycle_30k_r8 \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --vfm_loss_thresh 0.75 \
  --vfm_weight 0.10 \
  -r 8
```

This did not bring Gaussian counts near the baseline, which is why the next implementation step is an explicit VFM importance control rather than more smoke-grid runs.

## Explicit Importance Weight Probe

`vfm_importance_weight` scales VFM densification counts before they are fused with RGB importance. It is separate from `vfm_weight`, which controls pruning-score fusion.

```bash
uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_cached_edge_compact.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0001/vfm_cached_edge_i025_bicycle_30k_r8 \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --vfm_importance_weight 0.25 \
  -r 8

uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_dinov2_token_edge.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0001/vfm_dinov2_token_edge_i025_bicycle_30k_r8 \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --vfm_importance_weight 0.25 \
  -r 8
```

## 2026-04-28 Smoke Validation

同条件低分辨率短跑，用于确认 densification 分支实际触发 scorer：

```bash
uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0001/baseline_bicycle_smoke \
  --eval \
  --iterations 220 \
  --densify_from_iter 50 \
  --densify_until_iter 220 \
  --densification_interval 50 \
  --test_iterations 220 \
  --save_iterations 220 \
  --checkpoint_iterations 220 \
  -r 8

uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_scorer.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0001/vfm_mock_bicycle_smoke \
  --eval \
  --iterations 220 \
  --densify_from_iter 50 \
  --densify_until_iter 220 \
  --densification_interval 50 \
  --test_iterations 220 \
  --save_iterations 220 \
  --checkpoint_iterations 220 \
  -r 8

uv run --active python -m vfm_gs.cli.render -m output/0001/baseline_bicycle_smoke --skip_train
uv run --active python -m vfm_gs.cli.metrics -m output/0001/baseline_bicycle_smoke

uv run --active python -m vfm_gs.cli.render -m output/0001/vfm_mock_bicycle_smoke --skip_train
uv run --active python -m vfm_gs.cli.metrics -m output/0001/vfm_mock_bicycle_smoke

uv run --active python -m vfm_gs.cli.build_vfm_cache \
  -s datasets/mipnerf360/bicycle \
  -i images_8 \
  -o output/0001/vfm_cache/bicycle_edge_u8 \
  --max_width 640 \
  --storage npz_uint8

uv run --active python -m vfm_gs.cli.validate_vfm_cache \
  -c output/0001/vfm_cache/bicycle_edge_u8 \
  -s datasets/mipnerf360/bicycle \
  -i images_8 \
  --backend cached_edge_l1

uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_cached_edge_compact.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0001/vfm_cached_edge_compact_bicycle_smoke \
  --eval \
  --iterations 220 \
  --densify_from_iter 50 \
  --densify_until_iter 220 \
  --densification_interval 50 \
  --test_iterations 220 \
  --save_iterations 220 \
  --checkpoint_iterations 220 \
  -r 8

uv run --active python -m vfm_gs.cli.render -m output/0001/vfm_cached_edge_compact_bicycle_smoke --skip_train
uv run --active python -m vfm_gs.cli.metrics -m output/0001/vfm_cached_edge_compact_bicycle_smoke
```
