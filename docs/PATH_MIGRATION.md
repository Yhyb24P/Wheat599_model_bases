# 路径迁移索引

机器可读映射见 [PATH_MIGRATION.json](PATH_MIGRATION.json)。第一轮将包名由 `wheat599` 改为 `genopred`，按数据集和方法家族整理目录；第二轮以 `bc724f7` 为基点，分开模型机制、训练方法与实验编排，沿用第一轮结果位置，不重新搬迁冻结运行。

运行目录保留原名，provenance 保留原始字节，冻结清单保留历史 SHA 值。源码和配置随 import、路径及职责调整而演化，其现役文件不承诺仍有历史 SHA；精确复现检出对应标签或提交。

## 当前目录

```text
src/genopred/
  models/{tabm,comparators,classical,kernels,structured,neural}/
  experiments/{wheat599,wheat2000}/
  data/  evaluation/  cli.py  __main__.py
configs/{wheat599/{baselines,kernels,tabm,structured},wheat2000,diagnostics}/
protocols/                     # 固定划分与协议输入
results/{wheat599/{baselines,kernels,neural,diagnostics},wheat2000}/<原run名>
results/run_index.json         # 从运行元数据派生的索引
third_party/  references/      # 上游原件 / 文献源
site/{models.html,runs.html,references.html}
checkpoints/  workspace/       # 本机权重 / 临时材料，不入 Git
```

`models/tabm/` 保存骨干、分位数训练、形状变换和互作机制；`models/comparators/` 保存公开对照及其紧耦合上游配方。`kernels/`、`classical/`、`structured/` 按算法家族保留，`neural/` 保留注册适配器。`experiments/` 负责划分、开发选模、重拟合、清单及产物保存，统一入口为 `PYTHONPATH=src python -m genopred run`。

## 两轮映射的读取方式

JSON 顶层的 `src`、`imports`、`configs`、`results`、`site` 保存第一轮映射，逐项保留。第二轮映射放在 `round2`：

- `base_commit`：第一轮完成时的 `bc724f7`。
- `src`：28 个第一轮源码路径到当前主要实现路径的映射；多个旧模块可以合入同一文件。
- `scripts`：9 个旧脚本到当前承接职责的源码文件；该表记录实现去向，不表示保留旧命令。
- `extracted`：从原文件另外抽出的实验编排。旧 `quantile_tabm.py` 的主要实现归分位数训练，`run`、worker 初始化和任务执行另归 `experiments/wheat599/quantile_tabm_dev.py`；两个背景岭脚本的编排归 `experiments/wheat599/background_ridge.py`。

从旧包定位当前源码时，先查顶层 `src`，再查 `round2.src`；查运行入口还需看 `round2.extracted`。例如 `src/wheat599/quantile_tabm.py` 先对应 `src/genopred/models/neural/quantile_tabm.py`，再分为 `models/tabm/quantile.py` 与 `experiments/wheat599/quantile_tabm_dev.py`。

## 第二轮源码职责

以下路径均相对 `src/genopred/`；旧列均为第一轮路径。

| 原模块 | 当前实现 |
| --- | --- |
| `models/neural/{genomic_tabm,environment_tabm,induced_ftt}.py` | `models/tabm/backbone.py` |
| `models/neural/{quantile_tabm,query_quantile,conditional_quantile}.py` | `models/tabm/quantile.py`；实验编排另见上述 `extracted` |
| `models/neural/shape_quantile.py` | `models/tabm/shape.py` |
| `models/neural/{interaction_tabm,member_interaction_tabm,shared_basis_tabm}.py` | `models/tabm/interactions.py` |
| `models/neural/quantile_tabm_final.py` | `experiments/wheat599/quantile_tabm_outer.py` |
| `models/neural/shape_quantile_final.py` | `experiments/wheat599/shape_quantile_outer.py` |
| `models/neural/independent_final.py` | `experiments/wheat599/independent_tabm_outer.py` |
| `models/neural/genomic_tabm_run.py` | `experiments/wheat599/genomic_tabm_dev.py` |
| `models/neural/ragp_run.py` | `experiments/wheat599/ragp_dev.py` |
| `models/classical/wheat2000_classical.py` | `experiments/wheat2000/classical.py` |

公开对照的 12 个模块由 `models/neural/` 原名移入 `models/comparators/`：`deep_baselines`、`gsformer_unified`、`mtcro_unified`、`wheatgp_unified`、`ragp`、`wheat2000_gsformer`、`wheat2000_moegp`、`wheat2000_mtcro`、`wheat2000_wheatgp`、`wheat2000_candidate`、`dnngp_worker`、`moegp_worker`。保留方法名、数值计算和紧耦合配方；文件名中的历史 `unified` 不表示另建一套算法。

## 脚本、配置与结果

`scripts/run_background_nested.py` 的参数解析和路由由 `src/genopred/cli.py` 承接；`run_background_ridge.py` 的核与候选求解由 `models/classical/background_ridge.py` 承接。背景岭的实验编排合入 `experiments/wheat599/background_ridge.py`。七个 quantile/Wheat2000 薄转发的实现去向逐项见 `round2.scripts`，可用路由以 `genopred.cli.RUNNERS` 和各执行模块参数为准。

配置与结果的旧位置按顶层 `configs`、`results` 读取，第二轮不更名冻结运行。固定划分在 `protocols/`；展示页按顶层 `site` 映射到 `site/`。运行索引由 `PYTHONPATH=src python -m genopred index` 派生，不改历史 provenance。

`unmapped_configs` 为空，当前 99 个已跟踪配置均有第一轮映射；映射中的 `wheat2000_moegp.yaml` 当时未跟踪，不能以条目存在证明文件已交付。已冻结配方和现役公开对照配置继续保留，未因缺少文件名引用而删除。

## 复现

精确历史复现按 [FROZEN_INDEX](FROZEN_INDEX.md) 检出原标签或 provenance 指定提交，使用其中的原路径、入口和配方。58 个冻结标签不因迁移改写。现役文件定位沿两轮映射；冻结源校验使用历史 Git blob，不能用最新文件的 SHA 覆盖历史值来通过校验。数据、权重与大预测仍需在本机准备。
