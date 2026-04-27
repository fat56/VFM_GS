# 0001 VFM Topology Scorer Runbook

## Baseline

```bash
python -m vfm_gs.cli.train --variant fastgs_baseline -s <dataset>/<scene> -m output/0001_baseline/<scene> --eval
python -m vfm_gs.cli.render -m output/0001_baseline/<scene> --skip_train
python -m vfm_gs.cli.metrics -m output/0001_baseline/<scene>
```

## Planned Experiment

```bash
python -m vfm_gs.cli.train --variant fastgs_baseline --config configs/experiments/0001_vfm_topology_scorer.yaml -s <dataset>/<scene> -m output/0001_vfm/<scene> --eval
```

当前配置仍回落到 `fastgs_photometric` scorer。真实 VFM scorer 接入后再更新命令和配置。
