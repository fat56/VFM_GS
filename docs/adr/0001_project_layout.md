# ADR 0001: 项目结构与实验迭代方式

## Context

原生 FastGS 仓库以根目录脚本为入口，适合复现实验，但不适合长期维护多个 scorer、训练变体和消融实验。VFM_GS 后续会反复试验不同拓扑打分器，需要稳定的包结构、配置矩阵和文档化实验台账。

## Decision

项目采用 `src/vfm_gs` 包结构，root 只保留配置、脚本、docs、assets、submodules 和项目元数据。训练中的可替换逻辑通过 registry 暴露稳定名称，实验差异通过 `configs/variants` 与 `configs/experiments` 管理。

## Consequences

正向影响是入口清晰、实验可复现、后续 scorer 可并存。代价是旧命令 `python train.py` 不再作为主入口，需要改用 `python -m vfm_gs.cli.train` 或安装后的 `vfm-gs-train`。
