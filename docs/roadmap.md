# Roadmap

## Active

- `0001_vfm_topology_scorer`: 梳理 VFM 拓扑打分器的 v1 落地边界，当前处于方案阶段。

## Queued

- 建立 mock scorer，验证 scorer registry 与 FastGS baseline 的输出兼容。
- 接入真实 VFM 缓存前，先定义 cache manifest 和数据校验脚本。

## Blocked

- VFM 后端依赖尚未确认，需要根据 Python 3.7、PyTorch 1.12.1、CUDA 11.6 兼容性筛选。

## Completed

- 将原生 FastGS 扁平结构迁移到 `src/vfm_gs` 包结构。
