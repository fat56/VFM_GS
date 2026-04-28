# VFM Backend Feasibility

## Environment Probe

Command:

```bash
uv run --active python -m vfm_gs.cli.vfm_backend_probe --width 640 --height 426 --num_images 194
uv run --active python -m vfm_gs.cli.vfm_backend_probe --width 1600 --height 1066 --num_images 194 --json
```

Observed environment on 2026-04-28:

| Item | Value |
|---|---|
| Python | 3.10.20 |
| PyTorch | 1.12.1+cu116 |
| CUDA | 11.6 |
| GPU | NVIDIA GeForce RTX 4090 D |
| GPU memory | 23.52GB |
| Optional packages | `transformers`, `timm`, `xformers`, `opencv-python` all missing |

## DINOv2 Cache Size Estimate

The DINOv2 model card and repository document ViT-S/B/L/g variants and patch-size-14 models. For a cache-only backend, the first feasible target is feature extraction outside the training loop, not online VFM inference.

Sources:

- DINOv2 repository: https://github.com/facebookresearch/dinov2
- DINOv2 model card: https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md

Estimated raw float32 feature cache size:

| Resolution | Variant | Tokens/Image | MB/Image | GB/194 Images |
|---|---|---:|---:|---:|
| 640x426 | ViT-S/14 | 1,426 | 2.09 | 0.40 |
| 640x426 | ViT-B/14 | 1,426 | 4.18 | 0.79 |
| 640x426 | ViT-L/14 | 1,426 | 5.57 | 1.06 |
| 640x426 | ViT-g/14 | 1,426 | 8.36 | 1.58 |
| 1600x1066 | ViT-S/14 | 8,855 | 12.97 | 2.46 |
| 1600x1066 | ViT-B/14 | 8,855 | 25.94 | 4.91 |
| 1600x1066 | ViT-L/14 | 8,855 | 34.59 | 6.55 |
| 1600x1066 | ViT-g/14 | 8,855 | 51.88 | 9.83 |

## Decision

- Prefer a torch.hub DINOv2 path first, because the current FastGS runtime is pinned to PyTorch 1.12.1 and the official repository exposes torch.hub loading. In practice, remote torch.hub listing hit a GitHub 403 rate limit during probing, so the reliable path is to clone the official DINOv2 repo once and pass `--dinov2_repo`.
- Start with `dinov2_vits14` or `dinov2_vitb14` at `max_width` 518-640 for full-scene cache builds. Keep `max_width` 224 with `--limit` as a fast smoke target.
- Cache reduced patch-token maps rather than raw full-resolution image features. The implemented builder stores L2-normalized DINOv2 patch tokens as `[grid_h, grid_w, dim]`.
- Keep `cached_edge_l1` as the deterministic fallback and regression test backend.
- Use `dinov2_token_edge_l1` as the first training-time DINO consumer because it avoids online DINO inference and turns patch tokens into a scalar topology map that fits the existing FastGS metric-map scorer path.

## DINOv2 Cache Smoke

Observed on 2026-04-28:

- Local clone: `output/0001/external/dinov2`.
- Backend: `dinov2_vits14`, pretrained, `max_width=224`, `--limit 4`, storage `npy_float16`.
- Output: `output/0001/vfm_cache/bicycle_dinov2_vits14_smoke`, 4 entries, about 500K.
- First entry shape: `10x16x384`.
- Validation: `vfm_gs.cli.validate_vfm_cache --backend dinov2_vits14` passed.
- `xformers` is not installed; DINOv2 emitted fallback warnings but cache generation completed.
- Current DINOv2 expects public `torch.nn.functional.scaled_dot_product_attention`, which PyTorch 1.12.1 does not provide. The builder adds an isolated compatibility shim over the private 1.12 attention function for this optional cache path.

## DINOv2 Token-Edge Scorer Smoke

Observed on 2026-04-28:

- Full-scene cache: `output/0001/vfm_cache/bicycle_dinov2_vits14`, 194 entries, `max_width=224`, `npy_float16`, about 24M.
- Cache build time: 15s after the local DINOv2 repo and pretrained weights were already available.
- Training config: `configs/experiments/0001_vfm_topology_dinov2_token_edge.yaml`.
- Smoke result: PSNR 20.2913, SSIM 0.4272, LPIPS 0.6006, 77,761 Gaussians, 1.72s training time.
- Render result: 25 test frames at 410.94 FPS.
- Interpretation: stable real-DINO cache consumption is validated, but the scalar token-edge projection is still a proxy for semantic feature consistency.

## Next Implementation Step

Run a small threshold/weight grid for `dinov2_token_edge_l1`, then compare it with a second deterministic patch-descriptor projection before moving to a larger `max_width` cache or a longer training schedule.
