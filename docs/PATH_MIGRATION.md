# 路径迁移索引（2026-11-16 结构性重构）

经用户直接授权，按数据集/方法家族用 `git mv` 迁移源码/配置/结果到结构化目录（包名 `genopred`）。**保留文件内容与 SHA 值、provenance 字节不变**；冻结清单路径键与现役引用同步到新路径。历史精确复现经 58 个 Git 标签＋本索引。机器可读映射见 [PATH_MIGRATION.json](PATH_MIGRATION.json)。

## 目标结构

```
src/genopred/{data,models/{classical,kernels,neural,structured},evaluation}
configs/{wheat599/{baselines,kernels,tabm,structured},wheat2000,diagnostics}
protocols/                      # 固定划分与协议输入
results/{wheat599/{baselines,kernels,neural,diagnostics},wheat2000}/<原run名>
third_party/                    # 上游源码（WheatGP/dnawhisper/bayesian_trees）
references/                     # 文献源与 BibTeX
site/{models.html,runs.html,references.html}
checkpoints/  workspace/        # 本地权重 / 临时材料（不入 Git）
```

## 规模

- 源码文件 67（含子包）；import 映射 53
- 配置 100；结果目录 184；展示页 3

## 结果目录家族分布

- `wheat2000`：13
- `wheat2000_protocol_20261105`：1
- `wheat599/baselines`：56
- `wheat599/diagnostics`：21
- `wheat599/kernels`：31
- `wheat599/neural`：62

## 源码模块映射（旧 → 新）

| 旧 | 新 |
|---|---|
| `src/wheat599/__init__.py` | `src/genopred/__init__.py` |
| `src/wheat599/background_rbf.py` | `src/genopred/models/classical/background_rbf.py` |
| `src/wheat599/bh_lmc.py` | `src/genopred/models/kernels/bh_lmc.py` |
| `src/wheat599/classical_baselines.py` | `src/genopred/models/classical/classical_baselines.py` |
| `src/wheat599/conditional_quantile.py` | `src/genopred/models/neural/conditional_quantile.py` |
| `src/wheat599/data.py` | `src/genopred/data/wheat599.py` |
| `src/wheat599/deep_baselines.py` | `src/genopred/models/neural/deep_baselines.py` |
| `src/wheat599/dnngp_worker.py` | `src/genopred/models/neural/dnngp_worker.py` |
| `src/wheat599/environment_selection.py` | `src/genopred/evaluation/environment_selection.py` |
| `src/wheat599/environment_tabm.py` | `src/genopred/models/neural/environment_tabm.py` |
| `src/wheat599/extrapolation_diagnostic.py` | `src/genopred/evaluation/extrapolation_diagnostic.py` |
| `src/wheat599/genomic_tabm.py` | `src/genopred/models/neural/genomic_tabm.py` |
| `src/wheat599/genomic_tabm_run.py` | `src/genopred/models/neural/genomic_tabm_run.py` |
| `src/wheat599/get_icm.py` | `src/genopred/models/kernels/get_icm.py` |
| `src/wheat599/gsformer_unified.py` | `src/genopred/models/neural/gsformer_unified.py` |
| `src/wheat599/heterogeneous_icm.py` | `src/genopred/models/kernels/heterogeneous_icm.py` |
| `src/wheat599/independent_final.py` | `src/genopred/models/neural/independent_final.py` |
| `src/wheat599/induced_ftt.py` | `src/genopred/models/neural/induced_ftt.py` |
| `src/wheat599/interaction_tabm.py` | `src/genopred/models/neural/interaction_tabm.py` |
| `src/wheat599/kernel_baselines.py` | `src/genopred/models/kernels/kernel_baselines.py` |
| `src/wheat599/kernel_ridge.py` | `src/genopred/models/classical/kernel_ridge.py` |
| `src/wheat599/member_interaction_tabm.py` | `src/genopred/models/neural/member_interaction_tabm.py` |
| `src/wheat599/metrics.py` | `src/genopred/evaluation/metrics.py` |
| `src/wheat599/models/__init__.py` | `src/genopred/models/library/__init__.py` |
| `src/wheat599/models/base.py` | `src/genopred/models/base.py` |
| `src/wheat599/models/ft_transformer.py` | `src/genopred/models/neural/ft_transformer.py` |
| `src/wheat599/models/kernel_ridge_gblup.py` | `src/genopred/models/classical/kernel_ridge_gblup.py` |
| `src/wheat599/models/linear.py` | `src/genopred/models/classical/linear.py` |
| `src/wheat599/models/rkhs_gaussian.py` | `src/genopred/models/kernels/rkhs_gaussian.py` |
| `src/wheat599/models/svr_rbf.py` | `src/genopred/models/classical/svr_rbf.py` |
| `src/wheat599/models/tabm.py` | `src/genopred/models/neural/tabm.py` |
| `src/wheat599/models/tabpfn_v3_5.py` | `src/genopred/models/neural/tabpfn_v3_5.py` |
| `src/wheat599/moegp_worker.py` | `src/genopred/models/neural/moegp_worker.py` |
| `src/wheat599/mtcro_unified.py` | `src/genopred/models/neural/mtcro_unified.py` |
| `src/wheat599/multienv_kernel.py` | `src/genopred/models/kernels/multienv_kernel.py` |
| `src/wheat599/polygenic_simulation.py` | `src/genopred/evaluation/polygenic_simulation.py` |
| `src/wheat599/quantile_tabm.py` | `src/genopred/models/neural/quantile_tabm.py` |
| `src/wheat599/quantile_tabm_final.py` | `src/genopred/models/neural/quantile_tabm_final.py` |
| `src/wheat599/query_quantile.py` | `src/genopred/models/neural/query_quantile.py` |
| `src/wheat599/ragp.py` | `src/genopred/models/neural/ragp.py` |
| `src/wheat599/ragp_run.py` | `src/genopred/models/neural/ragp_run.py` |
| `src/wheat599/reml_icm.py` | `src/genopred/models/kernels/reml_icm.py` |
| `src/wheat599/robust_icm.py` | `src/genopred/models/kernels/robust_icm.py` |
| `src/wheat599/shape_quantile.py` | `src/genopred/models/neural/shape_quantile.py` |
| `src/wheat599/shape_quantile_final.py` | `src/genopred/models/neural/shape_quantile_final.py` |
| `src/wheat599/shared_basis_tabm.py` | `src/genopred/models/neural/shared_basis_tabm.py` |
| `src/wheat599/supervised_ising.py` | `src/genopred/models/structured/supervised_ising.py` |
| `src/wheat599/tree_tensor.py` | `src/genopred/models/structured/tree_tensor.py` |
| `src/wheat599/treewidth_ising.py` | `src/genopred/models/structured/treewidth_ising.py` |
| `src/wheat599/wheat2000_candidate.py` | `src/genopred/models/neural/wheat2000_candidate.py` |
| `src/wheat599/wheat2000_classical.py` | `src/genopred/models/classical/wheat2000_classical.py` |
| `src/wheat599/wheat2000_data.py` | `src/genopred/data/wheat2000.py` |
| `src/wheat599/wheat2000_gblup.py` | `src/genopred/models/kernels/wheat2000_gblup.py` |
| `src/wheat599/wheat2000_gsformer.py` | `src/genopred/models/neural/wheat2000_gsformer.py` |
| `src/wheat599/wheat2000_huber.py` | `src/genopred/models/kernels/wheat2000_huber.py` |
| `src/wheat599/wheat2000_moegp.py` | `src/genopred/models/neural/wheat2000_moegp.py` |
| `src/wheat599/wheat2000_mtcro.py` | `src/genopred/models/neural/wheat2000_mtcro.py` |
| `src/wheat599/wheat2000_wheatgp.py` | `src/genopred/models/neural/wheat2000_wheatgp.py` |
| `src/wheat599/wheatgp_unified.py` | `src/genopred/models/neural/wheatgp_unified.py` |

子包：`models/`→`models/`（base.py 注册表）＋按家族分发适配器；`dnawhisper/`、`bayesian_trees/`→`third_party/`。逐文件见 JSON。

## 配置与结果映射

逐条旧→新见 [PATH_MIGRATION.json](PATH_MIGRATION.json) 的 `configs`/`results` 键。结果保留原 run 目录名，仅加数据集/家族前缀；固定划分移入 `protocols/`。

## 展示页迁移

| 旧 | 新 |
|---|---|
| `docs/models.html` | `site/models.html` |
| `results/index.html` | `site/runs.html` |
| `ref/index.html` | `site/references.html` |

## 复现说明

- 精确历史复现：`git checkout <标签>` 检出原提交（原路径仍在），按标签内配置运行。
- 现役复现：用新路径＋更新后的清单路径键（SHA 值不变）；`validate_manifest` 对演化源文件按历史 Git blob 核验。
- provenance.json 保留原始字节（记录旧路径），经本索引解析到新位置。
