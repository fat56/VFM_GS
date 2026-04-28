# Roadmap

## Active

- `0001_vfm_topology_scorer`: cached edge proxy 已完成 bicycle smoke。下一步压缩缓存格式并接真实 VFM 后端。

## Queued

- 增加 cache validation 命令，训练前检查缺失 entry、shape、checksum 和 backend 兼容性。
- 增加紧凑缓存格式，降低 `.npy` edge/feature map 的磁盘占用。
- 运行 bicycle 长程 baseline vs VFM cached backend 消融。

## Blocked

- 真实 VFM 后端依赖尚未确认，需要根据 Python 3.10、PyTorch 1.12.1、CUDA 11.6 兼容性和显存占用筛选。

## Completed

- 将原生 FastGS 扁平结构迁移到 `src/vfm_gs` 包结构。
- 建立 `vfm_topology_scorer` mock v1，验证 scorer registry 与 FastGS baseline 的输出兼容。
- 建立 `build_vfm_cache` CLI 和 `cached_edge_l1` 后端，验证离线缓存读取链路。
- 为训练输出补充 scorer/backend/阈值/权重等 provenance 日志。
