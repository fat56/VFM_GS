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

- Prefer a torch.hub DINOv2 path first, because the current FastGS runtime is pinned to PyTorch 1.12.1 and the official repository exposes torch.hub loading.
- Start with `dinov2_vits14` or `dinov2_vitb14` at `max_width` 518-640.
- Cache reduced maps or projected features rather than raw full-resolution token tensors.
- Keep `cached_edge_l1` as the deterministic fallback and regression test backend.

## Next Implementation Step

Add a real DINOv2 cache builder behind an optional dependency path. It should fail gracefully when weights or dependencies are unavailable, and it should write the same manifest shape used by `cached_edge_l1`.
