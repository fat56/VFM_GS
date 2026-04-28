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
```
