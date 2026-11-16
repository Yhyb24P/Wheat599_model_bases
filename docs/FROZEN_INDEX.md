# 冻结证据索引

已结束实验的代码、配置与脚本不在现役树保存，而是由 **58 个 Git 标签**逐字保存（原路径与 blob SHA 不变）。本文件是导航入口：实验 → 标签 → 结果目录。关键数字统一见 [RESULTS](RESULTS.md)，不在此重复。

## 保存与复现机制

- **标签**：每个冻结实验对应一个或多个标签，保存当时的完整源码、配置、脚本与测试。`git tag` 列出全部 58 个。
- **结果**：`results/<数据集>/<家族>/<run>/` 保留原始运行产物（2026-11-16 按数据集/家族归档，旧→新路径见 [PATH_MIGRATION](PATH_MIGRATION.md)）；其中 `provenance.json` 固定数据 SHA、源码 SHA、配置、种子与划分，是运行与标签的对应凭证。
- **精确复现**：`git checkout <标签>` 检出原提交，按该标签内的配置与入口运行；完成后 `git checkout main` 返回。被移除的模块/配置在标签中原路径仍在。
- 现役树只保留共享核心、当前候选、统一入口与现役配置/测试；冻结配方文档见 `docs/history/methods/`。

## 现役主候选

四独立分位数 TabM 与多环境 Huber ICM 的预固定等权融合；最新形状收缩分位数 TabM 四臂比较已完成（有限收缩外层 PCC 0.511737，未超过自由/共享形状或现役参照）。数字与停止原因见 [RESULTS](RESULTS.md)。现役入口 `scripts/run_background_nested.py`，`RUNNERS` 映射 25 条现役路由。

## 标签按家族

| 家族 | 标签 | 结果目录前缀 |
| --- | --- | --- |
| 现役候选谱系 | `quantile-tabm-dev-20261106`、`quantile-tabm-final-20261106`、`independent-tabm-final-protocol/results/refit-manifest-20261105`、`wheat599-huber-icm-20261103`、`fusion-refit-20261106` | `quantile_tabm_*`、`independent_tabm_final_*`、`fusion_*` |
| 分位数变体 | `anchored-quantile-dev`、`conditional-quantile-dev-v1/v2`、`distill-correlation-dev/outer`、`distill-quantile-dev-9ad8424/corrected/parallel`、`mean-quantile-dev-20261106`、`menet-quantile-dev-97c9f0f`、`midpoint-quantile-dev-20261106`、`relation-quantile-dev-20261107/-v2/controls`、`multihead-mse-dev-20261106` | `anchored_quantile_*`、`conditional_quantile_*`、`distill_*`、`mean_quantile_*`、`menet_quantile_*`、`midpoint_quantile_*`、`relation_quantile_*`、`multihead_mse_*` |
| 核 / GP | `adaptive-anova-dev-20261114`、`grouped-kernel-stage1/outer-20261105`、`grouped-weights-stage2-20261105`、`matrix-inference-prototype-20261104`、`matrix-kernel-reference-20261104`、`locus-context-protocol-20261104`、`joint-inference-protocol-20261104`、`joint-tabm-icm-dev-20261106`、`ng-icm-selection-20261103`、`structured-coefficient-dev/-v2` | `adaptive_anova_*`、`grouped_kernel_*`、`grouped_weights_*`、`matrix_*`、`locus_context_*`、`joint_*`、`structured_coefficient_*` |
| TabM 架构 | `environment-tabm-protocol-20261105`、`genomic-tabm-development-20261104`、`retrieval-tabm-protocol-20261105`、`tabicl-genomic-dev-20261106` | `environment_tabm_*`、`genomic_tabm_*`、`retrieval_*`、`tabicl_*` |
| 稀疏 | `sparse-interaction-diagnostic-20261104`、`sparse-polygenic-bart-dev-v1/v2`、`sparse-residual-quantile-dev-v1` | `sparse_interaction_*`、`sparse_polygenic_bart_*`、`sparse_residual_quantile_*` |
| Mul-PheG2P | `mulphe-reevaluation-20261105/-corrected/-v2`、`mulphe-unified-dev-20261112/-parallel/-parallel-v2` | `mulphe_*` |
| Wheat2000 | `wheat2000-deepgs-protocol-20261105`、`wheat2000-gblup-selection-20261105`、`wheat2000-tabm-gblup-protocol-20261105` | `wheat2000_*` |
| 其他 | `dnawhisper-protocol/selection-20261104`、`phenotype-rotation-protocol-20261105`、`ragp-protocol-20261105`、`rfm-dev-20261106`、`vector-readout-diagnostic-20261104`、`g0-stage1-20261029` | `dnawhisper_*`、`phenotype_rotation_*`、`ragp_*`、`rfm_*`、`vector_diagnostic_*` |

表中"结果目录前缀"为 run 名模式；2026-11-16 起这些 run 已按数据集/家族归档到 `results/wheat599/{baselines,kernels,neural,diagnostics}/` 与 `results/wheat2000/` 下（固定划分在 `protocols/`），旧→新逐条映射见 [PATH_MIGRATION](PATH_MIGRATION.md)。标签内仍保存原路径快照。

中断记录（Wheat2000、DeepGS、中点分位数）不作完整结果或算法失败；BiMSGP 为硬边界，不重启，历史适配记录保留。具体数字、失败与停止原因以 [RESULTS](RESULTS.md) 为准。