# VFM_GS

VFM_GS 是基于 FastGS 重组出的实验型 3D Gaussian Splatting 工作区。当前代码仍保留 FastGS baseline 的训练语义，但项目结构已经改成适合长期模型迭代的形态：源码在 `src/vfm_gs`，实验配置在 `configs`，方案、记录和复盘在 `docs`。

这个仓库的目标不是再复制一份原生 FastGS，而是把 FastGS 变成可切换 scorer、可做消融、可追踪实验结果的研究工程底座。VFM 拓扑打分器的第一版方案见 [docs/experiments/0001_vfm_topology_scorer/proposal.md](docs/experiments/0001_vfm_topology_scorer/proposal.md)。

## Project Layout

```text
.
├── configs/                 # 训练变体和实验配置
├── docs/                    # 实验方案、结果、复盘和架构决策
├── scripts/                 # 批量训练、评测和 smoke test
├── src/vfm_gs/              # Python package
│   ├── cli/                 # train/render/metrics/convert/full-eval 入口
│   ├── config/              # legacy argparse + YAML 配置加载
│   ├── scorers/             # scorer registry
│   ├── gaussian_renderer/   # FastGS renderer wrapper
│   ├── scene/
│   ├── utils/
│   └── lpips_pytorch/
├── submodules/              # CUDA extensions, 保持原 FastGS 路径
├── pyproject.toml
└── environment.yml          # 旧 conda 环境参考，不再作为主包管理入口
```

## Environment With uv

包管理统一使用 `uv`。`environment.yml` 只保留为 FastGS 原 CUDA/PyTorch 组合的参考，日常安装不要再走 `conda env create`。

系统侧仍需要可用的 NVIDIA 驱动、CUDA 编译工具链和 C++ 编译器。FastGS CUDA 扩展会在安装 `submodules/` 时本地编译，`nvcc -V` 与 PyTorch CUDA wheel 的版本需要匹配。

```bash
# Install uv if the machine does not have it yet.
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate a project-local virtual environment.
uv venv .venv --python 3.10
source .venv/bin/activate

# Install PyTorch CUDA 11.6 wheels used by the original FastGS environment.
uv pip install \
  torch==1.12.1+cu116 \
  torchvision==0.13.1+cu116 \
  torchaudio==0.12.1 \
  --extra-index-url https://download.pytorch.org/whl/cu116

# Install the project package and Python dependencies declared in pyproject.toml.
uv pip install -e .

# PyTorch 1.12 still imports pkg_resources while building CUDA extensions.
uv pip install "setuptools<81" wheel

# Build local CUDA extensions.
TORCH_CUDA_ARCH_LIST="8.6+PTX" uv pip install --no-build-isolation \
  submodules/diff-gaussian-rasterization_fastgs \
  submodules/simple-knn \
  submodules/fused-ssim
```

`TORCH_CUDA_ARCH_LIST="8.6+PTX"` is needed on RTX 40/Ada GPUs when using
PyTorch 1.12, because that PyTorch release does not recognize `sm_89`
directly. On older GPUs, set `TORCH_CUDA_ARCH_LIST` to the local compute
capability instead.

Windows 需要先准备 MSVC/CUDA 编译环境；如果使用 PowerShell，多条命令用 `;` 分隔。

## Quick Checks

不需要数据集即可做入口检查：

```bash
uv run --active python -m compileall src/vfm_gs
uv run --active python -m vfm_gs.cli.train --help
uv run --active python -m vfm_gs.cli.render --help
uv run --active python -m vfm_gs.cli.metrics --help
bash scripts/smoke_test.sh
```

`scripts/smoke_test.sh` 会优先使用环境变量 `PYTHON`，否则回退到 `python3`。如果已经激活 `.venv`，直接运行即可。

## Dataset Layout

数据集默认放在 `datasets/`，该目录已被 `.gitignore` 忽略。

```text
datasets/
├── mipnerf360/
│   ├── bicycle/
│   ├── flowers/
│   └── ...
├── db/
│   ├── playroom/
│   └── ...
└── tanksandtemples/
    ├── truck/
    └── ...
```

Mip-NeRF 360 数据集来自原作者页面。Tanks&Temples 与 Deep Blending 可使用 3DGS/FastGS 常用的 COLMAP 格式数据。

## Train / Render / Metrics

推荐使用 `uv run --active` 调用包入口，确保命令运行在当前 `.venv` 中。

```bash
uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/bicycle_baseline \
  --eval

uv run --active python -m vfm_gs.cli.render \
  -m output/bicycle_baseline \
  --skip_train

uv run --active python -m vfm_gs.cli.metrics \
  -m output/bicycle_baseline
```

安装 editable 包后也可以直接使用 console scripts：

```bash
vfm-gs-train --variant fastgs_baseline -s datasets/mipnerf360/bicycle -i images -m output/bicycle_baseline --eval
vfm-gs-render -m output/bicycle_baseline --skip_train
vfm-gs-metrics -m output/bicycle_baseline
vfm-gs-build-vfm-cache -s datasets/mipnerf360/bicycle -i images_8 -o output/0001/vfm_cache/bicycle_edge_u8 --max_width 640 --storage npz_uint8
vfm-gs-validate-vfm-cache -c output/0001/vfm_cache/bicycle_edge_u8 -s datasets/mipnerf360/bicycle -i images_8 --backend cached_edge_l1
vfm-gs-probe-vfm-backend --width 640 --height 426 --num_images 194
vfm-gs-build-vfm-cache -s datasets/mipnerf360/bicycle -i images_8 -o output/0001/vfm_cache/bicycle_dinov2_vits14_smoke --backend dinov2_vits14 --dinov2_repo output/0001/external/dinov2 --max_width 224 --limit 4
vfm-gs-train --variant fastgs_baseline --config configs/experiments/0001_vfm_topology_dinov2_token_edge.yaml -s datasets/mipnerf360/bicycle -i images -m output/0001/vfm_dinov2_token_edge/bicycle --eval
```

批量脚本仍保留，但已经指向新的包入口：

```bash
bash scripts/train_base.sh
bash scripts/train_big.sh
```

## Variants And Scorers

训练入口支持两层切换：

- `--variant <name>` 读取 `configs/variants/<name>.yaml`。
- `--config <path>` 在 variant 之上叠加某次实验的配置。

当前可用 variant：

| Variant | Config | Scorer | Intent |
|---|---|---|---|
| `fastgs_baseline` | `configs/variants/fastgs_baseline.yaml` | `fastgs_photometric` | 原 FastGS 标准训练设置 |
| `fastgs_big` | `configs/variants/fastgs_big.yaml` | `fastgs_photometric` | 更频繁 densification 的高质量设置 |

scorer registry 位于 `src/vfm_gs/scorers/`。当前注册了 `fastgs_photometric` 和 `vfm_topology_scorer`。`vfm_topology_scorer` 的 v1 默认使用 `mock_l1` 后端：它用 SH0 渲染图与 GT 生成 VFM-style pixel error map，再通过 FastGS 既有 `metric_map` 计数器融合到 Gaussian 级评分。`cached_edge_l1` 后端可先验证离线缓存流程，后续真实 VFM 后端会复用同一 cache manifest 入口。cached backend 会在训练前执行 preflight，提前检查 manifest、backend 和 source image entry。`vfm_weight` 控制 VFM pruning-score 融合强度；`vfm_importance_mode` 和 `vfm_importance_weight` 控制 VFM densification importance，默认 `max` / `1.0` 保持旧行为。`target_gaussian_count` 默认关闭，可在训练结束时按最低 pruning/support score 裁到固定 Gaussian 预算并重新保存最终 PLY。

离线缓存后端可以先用轻量 edge proxy 验证完整 cache 流程：

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

真实 VFM cache 的第一条可选路径是 DINOv2 patch-token builder。若 torch.hub 远程访问被 GitHub rate limit，先把官方 DINOv2 仓库 clone 到被忽略的输出目录，再用 `--dinov2_repo` 指向本地路径：

```bash
git clone https://github.com/facebookresearch/dinov2.git output/0001/external/dinov2

uv run --active python -m vfm_gs.cli.build_vfm_cache \
  -s datasets/mipnerf360/bicycle \
  -i images_8 \
  -o output/0001/vfm_cache/bicycle_dinov2_vits14_smoke \
  --backend dinov2_vits14 \
  --dinov2_repo output/0001/external/dinov2 \
  --max_width 224 \
  --limit 4

uv run --active python -m vfm_gs.cli.validate_vfm_cache \
  -c output/0001/vfm_cache/bicycle_dinov2_vits14_smoke \
  --backend dinov2_vits14
```

DINOv2 cache 默认写 `npy_float16`，manifest feature 为 `dinov2_patchtokens`。`dinov2_token_edge_l1` 是第一版消费该 cache 的训练后端：它把 DINO patch tokens 投影成 token-edge topology map，再与 SH0 渲染图的 pooled edge map 形成 pixel error map。

```bash
uv run --active python -m vfm_gs.cli.build_vfm_cache \
  -s datasets/mipnerf360/bicycle \
  -i images_8 \
  -o output/0001/vfm_cache/bicycle_dinov2_vits14 \
  --backend dinov2_vits14 \
  --dinov2_repo output/0001/external/dinov2 \
  --max_width 224

uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_dinov2_token_edge.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0001/vfm_dinov2_token_edge/bicycle \
  --eval
```

实验配置示例：

```bash
uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_scorer.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0001/bicycle \
  --eval
```

## Experiment Docs

`docs/` 是实验统筹目录，不存放大型运行产物。

- `docs/roadmap.md`：实验队列和状态。
- `docs/experiments/index.md`：实验总表。
- `docs/experiments/_template.md`：新实验模板。
- `docs/experiments/<id>/proposal.md`：方案。
- `docs/experiments/<id>/runbook.md`：命令和流程。
- `docs/experiments/<id>/results.md`：指标摘要。
- `docs/experiments/<id>/review.md`：结论和下一步。
- `docs/adr/`：架构决策记录。

运行输出放在 `output/`、`eval/`、`runs/` 或外部存储；docs 只记录关键指标、artifact 路径和决策。

## Conversion

COLMAP 转换入口也迁到了包内：

```bash
uv run --active python -m vfm_gs.cli.convert \
  -s datasets/custom_scene \
  --resize
```

## Upstream FastGS

本项目基于 FastGS 代码改造。FastGS 原项目、论文和许可证信息保留在：

- FastGS homepage: <https://fastgs.github.io/>
- FastGS paper: <https://arxiv.org/abs/2511.04283>
- Original license notes: [LICENSE_ORIGINAL.md](LICENSE_ORIGINAL.md)

FastGS 构建于 3DGS、Taming-3DGS、Speedy-Splat、Abs-GS 等工作之上。继续使用或发布结果时，需要同时遵守上游项目和本仓库的许可证要求。
