# Roadmap

## Active

- `0001_vfm_topology_scorer`: compact cache、validation 与训练 preflight 已完成。下一步做真实 VFM 后端依赖/显存评估。

## Queued

- 调研并验证 DINOv2 / monocular depth 后端在当前 Python 3.10、PyTorch 1.12.1、CUDA 11.6 环境下的可行性。
- 运行 bicycle 长程 baseline vs VFM cached backend 消融。

## Blocked

- 真实 VFM 后端依赖尚未确认，需要根据 Python 3.10、PyTorch 1.12.1、CUDA 11.6 兼容性和显存占用筛选。

## Completed

- 将原生 FastGS 扁平结构迁移到 `src/vfm_gs` 包结构。
- 建立 `vfm_topology_scorer` mock v1，验证 scorer registry 与 FastGS baseline 的输出兼容。
- 建立 `build_vfm_cache` CLI 和 `cached_edge_l1` 后端，验证离线缓存读取链路。
- 为训练输出补充 scorer/backend/阈值/权重等 provenance 日志。
- 增加 `npz_uint8` compact cache storage，将 bicycle edge cache 从约 189MB 降到约 35MB。
- 增加 `validate_vfm_cache` CLI，支持 manifest、checksum、shape、source-image 和 backend 校验。
- 为 cached backend 增加训练前 preflight，提前暴露缺失 cache 或 backend 不匹配。
