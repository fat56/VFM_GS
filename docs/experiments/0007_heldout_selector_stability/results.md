# 0007 Held-out Selector Stability 结果

## 当前状态

Round 1 准备中。先用 0006 候选表做 held-out train-view selector：

- selector views：采样 train views 的 even index
- holdout views：采样 train views 的 odd index
- test metrics：复用 0006 candidate table 中的 official test summary

## Round 1：待填

待 tmux 任务完成后记录：

- MipNeRF360 baseline / `selector_best_psnr` / `selector_qcgi` 的 selector、holdout、test 均值；
- DB/Tandt baseline / `selector_best_psnr` / `selector_qcgi` 的 selector、holdout、test 均值；
- 逐场景选择表；
- 是否比 0006 train-split selector 更少过拟合。
