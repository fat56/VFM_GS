# VFM 后端可行性

## 环境探测

命令：

```bash
uv run --active python -m vfm_gs.cli.vfm_backend_probe --width 640 --height 426 --num_images 194
uv run --active python -m vfm_gs.cli.vfm_backend_probe --width 1600 --height 1066 --num_images 194 --json
```

2026-04-28 观测到的环境：

| 项目 | 值 |
|---|---|
| Python | 3.10.20 |
| PyTorch | 1.12.1+cu116 |
| CUDA | 11.6 |
| GPU | NVIDIA GeForce RTX 4090 D |
| GPU 显存 | 23.52GB |
| 可选包 | `transformers`、`timm`、`xformers`、`opencv-python` 均未安装 |

## DINOv2 缓存体积估算

DINOv2 model card 和仓库文档提供了 ViT-S/B/L/g 变体和 patch size 14 模型。对于只使用缓存的后端，优先可行目标是把特征提取放在训练循环外离线完成，而不是训练期在线运行 VFM。

来源：

- DINOv2 repository: https://github.com/facebookresearch/dinov2
- DINOv2 model card: https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md

原始 float32 特征缓存体积估算：

| 分辨率 | 变体 | 每图 token 数 | 每图 MB | 194 张图 GB |
|---|---|---:|---:|---:|
| 640x426 | ViT-S/14 | 1,426 | 2.09 | 0.40 |
| 640x426 | ViT-B/14 | 1,426 | 4.18 | 0.79 |
| 640x426 | ViT-L/14 | 1,426 | 5.57 | 1.06 |
| 640x426 | ViT-g/14 | 1,426 | 8.36 | 1.58 |
| 1600x1066 | ViT-S/14 | 8,855 | 12.97 | 2.46 |
| 1600x1066 | ViT-B/14 | 8,855 | 25.94 | 4.91 |
| 1600x1066 | ViT-L/14 | 8,855 | 34.59 | 6.55 |
| 1600x1066 | ViT-g/14 | 8,855 | 51.88 | 9.83 |

## 决策

- 优先走 `torch.hub` DINOv2 路径，因为当前 FastGS 运行时固定在 PyTorch 1.12.1，而官方仓库提供 `torch.hub` 加载方式。实际探测时远程 `torch.hub` listing 触发 GitHub 403 限流，所以更可靠的做法是先 clone 官方 DINOv2 仓库，再通过 `--dinov2_repo` 显式指定。
- full-scene 缓存优先从 `dinov2_vits14` 或 `dinov2_vitb14`、`max_width` 518-640 开始。`max_width` 224 配合 `--limit` 继续作为快速 smoke 目标。
- 缓存降采样后的 patch-token map，而不是原始全分辨率图像特征。当前 builder 保存 L2 归一化后的 DINOv2 patch tokens，形状为 `[grid_h, grid_w, dim]`。
- 保留 `cached_edge_l1` 作为确定性 fallback 和回归测试后端。
- 首个训练期 DINO 消费端使用 `dinov2_token_edge_l1`，因为它避免在线 DINO 推理，并能把 patch tokens 转成适配现有 FastGS metric-map 打分路径的标量拓扑图。

## DINOv2 缓存冒烟

2026-04-28 观测结果：

- 本地 clone：`output/0001/external/dinov2`。
- 后端：`dinov2_vits14`，pretrained，`max_width=224`，`--limit 4`，storage 为 `npy_float16`。
- 输出：`output/0001/vfm_cache/bicycle_dinov2_vits14_smoke`，4 个 entry，约 500K。
- 首个 entry 形状：`10x16x384`。
- 校验：`vfm_gs.cli.validate_vfm_cache --backend dinov2_vits14` 通过。
- `xformers` 未安装；DINOv2 输出 fallback warning，但缓存生成完成。
- 当前 DINOv2 期望公开的 `torch.nn.functional.scaled_dot_product_attention`，而 PyTorch 1.12.1 没有该 API。builder 为这个可选缓存路径增加了隔离的兼容 shim，底层调用 1.12 的私有 attention 函数。

## DINOv2 令牌边缘打分器冒烟

2026-04-28 观测结果：

- Full-scene 缓存：`output/0001/vfm_cache/bicycle_dinov2_vits14`，194 个 entry，`max_width=224`，`npy_float16`，约 24M。
- 缓存构建时间：在本地 DINOv2 仓库和 pretrained weights 已可用后耗时 15s。
- 训练配置：`configs/experiments/0001_vfm_topology_dinov2_token_edge.yaml`。
- 冒烟结果：PSNR 20.2913，SSIM 0.4272，LPIPS 0.6006，77,761 个 Gaussians，训练时间 1.72s。
- 渲染结果：25 个 test frames，410.94 FPS。
- 解读：稳定消费真实 DINO cache 的链路已验证，但标量 token-edge projection 仍只是语义特征一致性的代理。

## 下一步实现

先为 `dinov2_token_edge_l1` 跑一个小阈值/权重网格，再和第二个确定性的 patch-descriptor projection 对比；之后再推进更大的 `max_width` 缓存或更长训练 schedule。
