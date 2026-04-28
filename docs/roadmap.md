# Roadmap

## Active

- `0001_vfm_topology_scorer`: mock v1 已接入 scorer registry 并完成 bicycle smoke。下一步接真实缓存后端。

## Queued

- 接入真实 VFM 缓存前，先定义 cache manifest 和数据校验脚本。
- 为训练输出补充 scorer/backend/阈值/权重等 provenance 日志。
- 运行 bicycle 长程 baseline vs VFM cached backend 消融。

## Blocked

- 真实 VFM 后端依赖尚未确认，需要根据 Python 3.10、PyTorch 1.12.1、CUDA 11.6 兼容性和显存占用筛选。

## Completed

- 将原生 FastGS 扁平结构迁移到 `src/vfm_gs` 包结构。
- 建立 `vfm_topology_scorer` mock v1，验证 scorer registry 与 FastGS baseline 的输出兼容。
