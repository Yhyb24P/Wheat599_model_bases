# Wheat599 植物基因组表型预测研究

现役主候选是**四独立分位数 TabM 与多环境 Huber ICM 的预固定等权融合**。统一协议外层宏 PCC 为 **0.526506**；项目内结果不代表跨数据集领先。主比较沿用既有材料级十外折、三内折，随机模型使用种子 42、2026 的平均预测；不根据外层结果更改配置或融合权重。

## 阅读结果与模型

- [结果与架构 HTML](site/models.html)：可搜索、筛选、排序并导出 CSV 的模型总表；默认候选与公开强模型置前，最优候选提供可按外折/环境切换的真实网络层级图，以及参数共享、分位数监督、ICM 核与损失机理图。
- [参考文献与逐篇讲解](site/references.html)：文献、来源及其与本项目的关系。
- [运行目录索引](site/runs.html)：按数据集/家族归档的运行目录查找摘要、配置、划分与状态。
- [结果原文](docs/RESULTS.md)、[评价协议](docs/EVALUATION_PROTOCOL.md)、[历史模型对比](docs/MODEL_COMPARISON.md)：可视化的事实来源。
- [冻结证据索引](docs/FROZEN_INDEX.md)：已结束实验的代码/配置由 58 个 Git 标签逐字保存，按家族导航到标签与 `results/`；精确复现检出原标签。
- [路径迁移索引](docs/PATH_MIGRATION.md)：包名与目录迁移，以及模型机制／实验编排拆分的两轮旧→新路径映射。

HTML 可直接用浏览器打开，不需要联网或启动服务；外部论文链接需要联网。页面是生成视图，修订证据后在仓库根目录重建：

```bash
python3 scripts/build_report.py
python3 references/build.py
PYTHONPATH=src python3 -m pytest tests/test_report.py -q
```

结果页构建使用 Python 标准库和 Pandoc（含原生 MathML 输出），文献页仅需标准库。上述命令不调用训练入口、不读取原始表型、不重新计算模型成绩；测试在安装 BibTeX 时检查书目能否实际解析。文献内容统一维护在 `references/build.py`。

近期完成的比较（[Mul-PheG2P 同协议适配](docs/methods/MULPHE_UNIFIED.md)、形状收缩分位数 TabM、分阶自适应互作核 GP、[查询条件分位数](docs/methods/QUERY_QUANTILE.md)、[监督条件 Ising](docs/methods/SUPERVISED_ISING.md)）的数字、停止原因与适配差异统一见[结果记录](docs/RESULTS.md)；各批已结束，不自动追加实验。

## 环境

基础依赖：

```bash
python -m pip install -r env/requirements-base.txt
python -m pip install -r env/requirements-bench.txt
```

精确版本见 `env/lock.txt`。FT-Transformer 来自 `rtdl-revisiting-models`（Apache-2.0）。

Dell 通过 `ssh dell` 经 omen 跳转接入，项目操作限于 `/home/dell/mnt/data/projects/intern/active/pangjy`。登录后启用独立环境：

```bash
source /home/dell/miniconda3/etc/profile.d/conda.sh
conda activate /home/dell/mnt/data/projects/intern/active/pangjy/conda-envs/wheat599
cd /home/dell/mnt/data/projects/intern/active/pangjy/Wheat599-research
```

接入记录中的该私有仓库当时只同步了结果与部分模型代码，并非完整可运行副本；环境版本及接入说明见 [env/dell_environment.yaml](env/dell_environment.yaml)。

## 数据

```bash
python scripts/fetch_data.py
```

从 `gdlc/BGLR-R` 固定提交下载 `wheat.RData` 到 `data/raw/`，校验 SHA-256：

```text
8710523389007dd8b7c88a5deb74ee16aea0bae158496bc11f0626fb43b5ea21
```

数据事实、R 列主序读取要求与旧 C-order 故障见 [docs/DATA_FACTS.md](docs/DATA_FACTS.md)。`data/raw/` 不入 Git。固定划分与实验协议输入在 `protocols/`。

## 运行与验证

正式实现在 `src/genopred/`（机制 `models/` 与实验编排 `experiments/` 分离，另有 `data/`、`evaluation/`），运行配置在 `configs/`（按数据集/家族归档）；统一入口是 `python -m genopred run`（`genopred.cli` 的 `RUNNERS` 路由，`cfg['model']` 键不变）。以下命令供复现使用，不表示本轮整理会启动训练。输出必须指定新目录；不要覆盖历史运行。涉及冻结清单的精确复现，先在对应提交/标签的检出中准备数据和既有产物，再使用以下命令；当前主分支不改写历史 SHA 来绕过版本校验。

```bash
# Huber ICM：复用已冻结选择，完整开发集重拟合
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src python -m genopred run --config configs/wheat599/kernels/robust_icm.yaml --selection results/wheat599/kernels/robust_icm_development_20261103/selection.json --out results/wheat599/kernels/robust_icm_reproduction

# 分位数 TabM：使用已冻结外层清单（需要本地数据与基线 OOF）
PYTHONPATH=src python -m genopred run --config configs/wheat599/tabm/quantile_tabm_final.yaml --out results/wheat599/neural/quantile_tabm_reproduction

# 机器可读运行索引（只读派生自 provenance/execution_status，不动冻结目录）
PYTHONPATH=src python -m genopred index

# 现有代码测试
PYTHONPATH=src python -m pytest tests/ -q
```

分位数训练配方见 [开发说明](docs/methods/QUANTILE_TABM.md) 与 [最终重拟合](docs/methods/QUANTILE_TABM_FINAL.md)；其他方法的运行命令在各自方案中，可从架构 HTML 查找。大预测、数据与权重不随 Git 分发，因此仅克隆仓库不足以复现依赖历史预测的运行。当前任务不跑 TabPFN、BiMSGP 或新数据集；TabPFN 测试依赖缺失时会跳过。

## 入口、实现与配置的对应

| 用途 | 唯一入口 / 实现 | 配置 |
|---|---|---|
| 当前模型开发、重拟合与汇总 | `python -m genopred run`（`genopred.cli`）；`RUNNERS` 映射到 `models/`（机制）与 `experiments/`（编排），只导入所选模型 | `model: <名称>` 的运行配置，如 `robust_icm.yaml`、`quantile_tabm_final.yaml`、`heterogeneous_icm.yaml` |
| 原背景岭回归 / RR / RBF | 同一入口；`model` 缺省→嵌套 ridge/RR，`background_rbf`→hybrid；编排在 `experiments/wheat599/background_ridge.py`，数学核心在 `models/classical/background_ridge.py` | `background_ridge_nested.yaml`、`background_rbf_nested.yaml` |
| 运行索引 | `python -m genopred index`（`evaluation/run_index.py`） | 只读派生 `results/run_index.json` |
| 数据下载 | `scripts/fetch_data.py` | 固定来源与 SHA，不改 `data/raw/` |
| 展示与文献 | `scripts/build_report.py`、`references/build.py` | 只读取已有结果、配置及文献，不训练 |
| 专用命令 | `scripts/run_moegp_unified.py`（独立两阶段参数/外部依赖） | 不并入统一 CLI |
| 历史独立入口 | `run_bimsgp*`、`run_a_rr_nested.py`、`run_stage1.py`、`run_benchmark.py`、`run_background_nested.py`、`run_background_ridge.py` 及 wheat2000 薄转发等已移出现役树，由冻结标签保存 | 检出对应标签后沿用各自配置结构；不是统一入口的新运行许可 |
| 已有独立核验 | `verify_*.py` 已移出现役树，由冻结标签保存 | 检出对应标签后保留独立复算口径，不改成调用被核验的汇总函数 |

`configs/` 中的运行配置、搜索空间和架构配方职责不同，不能把所有 YAML 都直接交给训练入口。统一入口拒绝未知模型名及误传的嵌套架构字典，`--resume-summary` 仅限 independent TabM。`src/genopred/models/` 含通用基线适配器框架（`base.py` 注册表）、`tabm/`（backbone/quantile/shape/interactions 机制）、`comparators/`（公开方法对照）与按家族归档的 kernels/classical/structured；实验编排（划分/选模/重拟合/清单/输出）在 `experiments/{wheat599,wheat2000}/`。

2026-11-16 经用户授权按数据集/方法家族用 `git mv` 迁移源码/配置/结果到结构化目录（包名 `genopred`）：保留结果内容、provenance 字节与历史 SHA 值；源码的 import 和配置中的路径随迁移更新，冻结清单仅同步路径键，旧→新映射见 [PATH_MIGRATION](docs/PATH_MIGRATION.md)。历史精确复现仍检出原记录的提交/标签，不改写旧清单 SHA 值来适应最新代码。完整 Mul-PheG2P 实现及测试由冻结标签 `mulphe-unified-*`（源自 `742132f`）保存，结果留在 `results/wheat599/baselines/mulphe_*`。

第二轮重构（基于 `bc724f7`）：把模型机制（`models/tabm` 四拆、`models/comparators`）与实验编排（`experiments/{wheat599,wheat2000}`）分离，实验运行与索引分别使用 `python -m genopred run`、`python -m genopred index`；冻结运行目录与 provenance 字节/SHA 不动，冻结清单经历史 blob 链式校验，旧→新路径与拆出函数记录在映射的 `round2` 节。不重训、不改 PCC/选模规则。

## 文件与结果约定

| 路径 | 职责 |
|---|---|
| `src/genopred/` | 唯一正式实现：`data/` 数据读取；`models/{tabm,comparators,classical,kernels,structured,neural}` 模型机制（tabm 按 backbone/quantile/shape/interactions，comparators 公开方法对照，neural 注册适配器）；`experiments/{wheat599,wheat2000}/` 实验编排（划分/选模/重拟合/清单/输出）；`evaluation/` 划分/选模/指标/诊断/运行索引；`cli.py` 统一入口 |
| `tests/` | 实现的验证测试（含基线配套测试），保留，非可删 |
| `third_party/` | 上游原始代码与 LICENSE（WheatGP、dnawhisper、bayesian_trees）；与 `src/` 的项目适配分层，不重复 |
| `scripts/` | 现有 CLI、数据获取及离线报告构建 |
| `configs/` | 运行配置，按 `wheat599/{baselines,kernels,tabm,structured}`、`wheat2000/`、`diagnostics/` 归档；由原运行 provenance/manifest 确定版本 |
| `protocols/` | 固定划分与实验协议输入（splits_manifest 等） |
| `results/<数据集>/<家族>/<run>/` | 原始运行产物；保留 run 命名与来源，按数据集/家族归档 |
| `site/` | 生成的离线展示：`models.html`、`runs.html`、`references.html` |
| `docs/RESULTS.md` | 结果、失败、停止原因和证据限制的主记录 |
| `references/` | 参考文献源（`build.py`）与 BibTeX 书目 |
| `docs/` | 评价协议、数据事实、结果、许可和冻结索引 |
| `docs/methods/` | 各模型的实际配方、输入、训练与适配说明 |
| `docs/history/` | 已结束阶段、旧交接、迁移与原始 ARS 记录；不作为当前运行要求 |
| `data/raw/`、`checkpoints/` | 本地只读数据与权重，不入 Git |
| `env/` | 依赖与已授权机器环境 |
| `workspace/` | 本机附件、旧未整合副本、交接包与测试残留；整体不入 Git |

`src/genopred/models/tabm/quantile.py` 的 `train()` 是统一变体分发器（训练方法核心）：按 `optimizer_variant` 条件加载 ising/query/conditional/shape/gated/member_conditional 等历史变体；这些变体已冻结到 Git 标签（见 [FROZEN_INDEX](docs/FROZEN_INDEX.md)），分发分支仍留现役树，并非冗余副本。实验编排（`run()`/workers/选模/重拟合/清单）在 `experiments/wheat599/`，与机制分离。`models/`（适配器框架）、`third_party/`（上游原件）与家族实现是分层而非重复；`tests/` 是验证这些实现的测试，应保留。

命名三维：**算法身份**用稳定方法名（`quantile_tabm`、`robust_icm`、`gsformer`）；**实验阶段**用 `development`/`selection`/`refit`/`outer_eval`/`diagnostic`（写入运行配置与结果，`nested`/`outer` 保留评价语义）；**执行身份**用日期＋必要运行号，状态与来源写入元数据。本轮将独立实验模块的 `*_final` 改为 `*_outer`；公开对照移入 `comparators/` 后保留原名，包括 `*_unified`。新运行沿用 `results/<数据集>/<家族>/<方法>_<阶段>_<日期>_r<N>/`；`python -m genopred index` 派生 `results/run_index.json`（dataset/method/phase/status/role/source_commit/protocol/result_path），冻结历史目录不再物理搬迁，role/phase 由索引表达。主表只接收已核对统一协议的结果；开发选模、历史不同协议、模拟、其他数据集、停止与失败记录各保留原口径。常见 `summary` 保存指标，`provenance` / `manifest` 保存数据 SHA、源码、配置、种子和划分，`execution_status` / `failures` / `interruption` 保存执行状态；不同历史文件格式不强制重写。逐材料与逐种子预测保留原位置，不能因导航整理而丢弃。

[AGENTS.md](AGENTS.md) 保存执行边界；[DATA_FACTS](docs/DATA_FACTS.md) 保存数据事实；[LICENSE_NOTES](docs/LICENSE_NOTES.md) 保存许可；[REFERENCES](docs/REFERENCES.md) 保留历史文献核验记录。旧评审、G0、Stage 1/2 与交接文档只作历史，不重新启用其流程。
