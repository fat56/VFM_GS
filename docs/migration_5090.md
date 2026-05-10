# 双卡 RTX 5090 服务器迁移规划

## 迁移结论

推荐方案：在新服务器上 `git clone` 当前仓库，重新创建虚拟环境并编译 CUDA 扩展；只复制数据集、VFM cache、必要权重缓存和关键实验产物。不建议打包整个工作区原样搬迁，也不建议复制 `.venv`。

原因：

- 当前环境是 PyTorch 1.12.1 + CUDA 11.6，适配的是旧 FastGS/Ada 路径；RTX 5090 属于 Blackwell 架构，旧 cu116 wheel 和已编译扩展不适合作为新机基础。
- `submodules/diff-gaussian-rasterization_fastgs`、`submodules/simple-knn`、`submodules/fused-ssim` 都是本地 CUDA extension，应在新机器上按新 PyTorch/CUDA ABI 重新编译。
- 当前 `.venv` 绑定本机路径和 CUDA/PyTorch ABI，复制过去更容易引入不可解释的编译和运行问题。

## 环境清单

当前项目主包管理器是 `uv`。`environment.yml` 只保留为旧 FastGS conda 环境参考，日常安装以 `pyproject.toml` 和 README 中的 `uv` 流程为准。

当前旧机环境参考：

```text
Ubuntu 20.04.5
Python 3.10.20
PyTorch 1.12.1+cu116
torchvision 0.13.1+cu116
numpy 1.26.4
pillow 12.2.0
pyyaml 6.0.3
plyfile 1.1.3
websockets 16.0
tqdm 4.67.3
```

新 RTX 5090 机器建议环境：

```text
Linux
NVIDIA Driver: 支持 Blackwell / RTX 5090
CUDA Toolkit / nvcc: 使用 PyTorch 当前官方 wheel 对应的 CUDA 主版本
Python: 3.10
PyTorch: 使用官网当前推荐的 Blackwell 可用版本
uv
gcc/g++、build-essential、python3-dev
```

关键 Python 依赖：

```text
torch / torchvision / torchaudio
numpy<2
pillow
plyfile
pyyaml
tqdm
websockets
setuptools
wheel
packaging
```

需要重新编译的本地 CUDA extension：

```text
submodules/diff-gaussian-rasterization_fastgs
submodules/simple-knn
submodules/fused-ssim
```

可选系统工具：

```text
colmap       # 只有运行 vfm-gs-convert / COLMAP 转换时需要
imagemagick  # convert.py 的 resize 流程可能使用 magick
```

## 目录迁移策略

必须迁移或重新获取：

```text
代码：通过 git clone 获取
数据：/root/autodl-tmp/datasets，或在新机保持同样 datasets 软链接
```

建议迁移：

```text
output/0001/vfm_cache
output/0001/vfm_cache_large
output/0001/external/dinov2
/root/.cache/torch/hub/checkpoints
```

按价值选择迁移：

```text
output/0001/full_mipnerf360_v1
output/0001/full_tandt_db_v1
output/0001/large_res_vitl_full
output/0001/large_res_fastgs_big_baseline
output/0001/large_res_vitl_big_overrides
output/0001/weighted_*
output/0001/descriptor_*
其他包含 summary.csv、results.json、point_cloud、logs 的实验目录
```

不建议迁移：

```text
.venv
submodules/**/build
*.egg-info
__pycache__
output 中临时 smoke/test 目录
/root/.cache/uv
```

## 新服务器配置命令草案

先检查系统状态：

```bash
nvidia-smi
nvcc -V
gcc --version
g++ --version
```

准备代码：

```bash
cd /root/autodl-tmp
git clone git@github.com:fat56/VFM_GS.git
cd VFM_GS
```

准备 Python 环境：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv .venv --python 3.10
source .venv/bin/activate
```

安装 PyTorch。以 PyTorch 官网当前选择器为准；CUDA 12.8 wheel 示例：

```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

说明：PyTorch 2.7 起已有 Blackwell 与 CUDA 12.8 wheel 支持；后续版本的默认 CUDA wheel 可能变化。新机实际配置时优先使用 PyTorch 官网当前 selector 生成的命令，如果需要锁定 CUDA 12.8，则使用 PyTorch previous versions 页面中仍提供的 `cu128` 版本。

安装项目和编译依赖：

```bash
uv pip install -e .
uv pip install "numpy<2" setuptools wheel packaging
```

重新编译本地 CUDA extension。RTX 5090 通常对应 Blackwell `sm_120`，如本机确认 capability 不是 12.0，应按实际值调整：

```bash
export TORCH_CUDA_ARCH_LIST="12.0"
uv pip install --no-build-isolation submodules/diff-gaussian-rasterization_fastgs
uv pip install --no-build-isolation submodules/simple-knn
uv pip install --no-build-isolation submodules/fused-ssim
```

验证双卡与 PyTorch：

```bash
python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda)
print(torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))
PY
```

项目入口检查：

```bash
uv run --active python -m compileall src/vfm_gs
uv run --active python -m vfm_gs.cli.train --help
uv run --active python -m vfm_gs.cli.render --help
uv run --active python -m vfm_gs.cli.metrics --help
bash scripts/smoke_test.sh
```

## 数据和缓存同步命令草案

假设旧机到新机可 SSH，`NEW` 替换为新服务器地址：

```bash
rsync -aH --info=progress2 /root/autodl-tmp/datasets/ NEW:/root/autodl-tmp/datasets/
ln -sfn /root/autodl-tmp/datasets /root/autodl-tmp/VFM_GS/datasets

rsync -aH --info=progress2 /root/autodl-tmp/VFM_GS/output/0001/vfm_cache/ NEW:/root/autodl-tmp/VFM_GS/output/0001/vfm_cache/
rsync -aH --info=progress2 /root/autodl-tmp/VFM_GS/output/0001/vfm_cache_large/ NEW:/root/autodl-tmp/VFM_GS/output/0001/vfm_cache_large/
rsync -aH --info=progress2 /root/autodl-tmp/VFM_GS/output/0001/external/dinov2/ NEW:/root/autodl-tmp/VFM_GS/output/0001/external/dinov2/
rsync -aH --info=progress2 /root/.cache/torch/hub/checkpoints/ NEW:/root/.cache/torch/hub/checkpoints/
```

实验结果目录建议等当前旧机训练完全结束后再统一同步，避免搬到半写入状态。

## 双卡使用方式

当前训练代码没有 DDP 或单实验多卡入口。双卡服务器建议先按“每张卡跑不同场景/实验”使用：

```bash
CUDA_VISIBLE_DEVICES=0 uv run --active python -m vfm_gs.cli.train ...
CUDA_VISIBLE_DEVICES=1 uv run --active python scripts/run_0001_descriptor_quality_probe.py ...
```

后续如果要做多卡单实验，需要单独改训练调度、随机种子、输出目录锁和日志汇总；不建议作为迁移第一步。

## 风险点

- PyTorch/CUDA/Blackwell 兼容性必须在新机先通过 `torch.cuda.get_device_capability()` 和 smoke test 验证。
- 三个 CUDA extension 可能在 PyTorch 2.x + CUDA 12.8 下暴露源码兼容问题。若编译失败，先保留完整 build log，再逐个修补。
- 不要复制 `.venv`。旧环境里的 cu116 wheel 和编译产物不应进入新机。
- 迁移实验结果时不要覆盖新机正在运行的 `output/` 目录，建议按日期或实验批次分目录同步。
