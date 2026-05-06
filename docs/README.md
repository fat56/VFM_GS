# VFM_GS 文档

这个目录负责管理实验计划、运行记录、结果复盘和架构决策。代码只保存可执行能力，实验背景、假设、失败原因和结论都写在这里，注意文档使用中文。

推荐流程：

1. 在 `docs/experiments/index.md` 登记实验 ID。
2. 复制 `docs/experiments/_template.md` 到新的实验目录。
3. 在 `configs/experiments/` 增加同名配置。
4. 运行实验后只把摘要、指标表和输出路径写回 docs，原始 artifact 留在 `output/`、`eval/` 或外部存储。
5. 实验关闭时更新 `review.md` 和 `docs/roadmap.md`。

日常代码版本通过 registry 和配置切换；真正需要回退实现时使用 Git。
