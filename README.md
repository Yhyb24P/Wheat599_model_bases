# Wheat599 植物基因组表型预测研究

当前仅推进 **Wheat599**。现役主候选为四独立分位数 TabM 与多环境 Huber ICM 的固定等权融合：同协议外层宏 PCC **0.526506**，高于 Huber 的 **0.521745**。分位数 TabM 单独为 **0.518829**，高于原独立 MSE TabM 的 **0.508190**；融合权重在外层评价前固定。结果与验证见 [RESULTS](docs/RESULTS.md)，规则见 [QUANTILE_TABM_FINAL](docs/QUANTILE_TABM_FINAL.md)。

当前只在开发层比较 99 输出 MSE TabM 与分位数目标，以研究配方差异；已冻结外层主候选不变。Wheat2000 维持停止，BiMSGP 不跑。已有划分曾多轮用于研究，当前领先为项目内同协议结果，跨数据集表现尚未确认。

目标是形成有可靠预测与效率证据的植物基因组表型预测方法论文；成熟神经架构的有效任务适配可以成为贡献。Wheat599是现阶段主要任务；SSH omen和已授权的本机算力按当前范围使用。跨数据集执行暂缓，已有记录见 [WHEAT2000](docs/WHEAT2000.md)，Wheat599证据见 [RESULTS](docs/RESULTS.md)。

按用户最新提案实现的[材料—位点双向推断原型](docs/MATRIX_INFERENCE.md)在首轮模拟学习中整体未超过固定核，当前参数化开发已结束，未进入 Wheat599 评价。 随后完成的稀疏交互模拟诊断已建立其具体推断缺口：未知位置统计方法在全部 16 个成对任务上超过首版网络，但加性任务族表现变差；尚无新的神经模型或真实数据增益，见 [结果](docs/RESULTS.md)。

Huber ICM 保留为强统计参照；完整比较见 [RESULTS](docs/RESULTS.md)，当前任务见 [RESEARCH_PLAN](docs/RESEARCH_PLAN.md)。

以下为先前阶段记录；现役状态以上方和 RESULTS 为准。

背景 + RBF 模型完成 40/40 环境折，折均宏 PCC **0.4919**、RMSE **0.8680**、MAE **0.6676**；同折 Gaussian kernel ridge 和加性＋RBF kernel ridge 的宏 PCC 分别为 **0.5025 / 0.5092**，均高于三核。结果为探索性比较，详见 [docs/RESULTS.md](docs/RESULTS.md)。

运行新候选：`python scripts/run_background_nested.py --config configs/background_rbf_nested.yaml --out results/<开发选模目录>`；固定选择后追加 `--selection results/<开发选模目录>/selection.json` 并指定新的 `--out` 进行外层重拟合与评价。

背景条件化联合收缩候选已完成 40/40 次：折均宏平均 PCC 0.4760、RMSE 0.8851、MAE 0.6868；相对同折 RR-BLUP 的 PCC 差 +0.0232、RMSE 差 −0.0106。详见 [docs/RESULTS.md](docs/RESULTS.md)。

复现命令：`python scripts/run_background_ridge.py --config configs/background_ridge.yaml --out results/<新运行目录>`。

MoEGP 独立专家固定配置已完成同折评价：宏均 PCC 0.4454、RMSE 0.8966、MAE 0.6998；相对 RR-BLUP 的 PCC 差 −0.0074。配置和完整结果见 [docs/RESULTS.md](docs/RESULTS.md)。

复现 MoEGP 时，将官方仓库检出至 `configs/moegp.yaml` 的 `source` 路径，并保持配置中指定的提交及干净工作树；然后运行 `python scripts/run_moegp.py --out results/<新运行目录>`。原始数据和 OOF 预测不随 Git 分发。

Direct FT-T 固定配置已完成 40/40 次：折均宏平均 PCC 0.4193、RMSE 0.9387、MAE 0.7317；预设 GO=False。与固定初始化 A 及同折 RR-BLUP 的比较见 [docs/RESULTS.md](docs/RESULTS.md)。

**状态（2026-10-30）：** 已固定 `build_model()` 前的初始化种子；完成 A 架构的 40 次正式训练、同一训练轨迹内的 val-PCC / val-RMSE 双 checkpoint 比较、同折 RR-BLUP 对照、固定 RR-BLUP + A 骨干的残差修正首轮，以及共享低秩编码 + 双向诱导注意力 + 多槽读出的完整 FT-T 重构首轮。val-PCC 规则宏均 test PCC 0.4342；残差修正未观察到相对 RR-BLUP 的可复现增量（点估计 −0.0011 / +0.0218）；完整重构宏均 PCC 0.3382、RMSE 0.9764，预注册 GO=False。该 NO-GO 只针对本配置/单 seed/固定十折，不构成对重构方向的一般性否定，也未做模块消融。预测 SD 小本身不是退化，MSE 最优尺度为 k=r。完整数字与判读见 [docs/RESULTS.md](docs/RESULTS.md)。

当前任务不跑 TabPFN；不做测试集调参。Stage 1（TabPFN-3.5 / FT-T / Ridge 门控）与六方法 Stage 2 协议已暂停，相关文档保留为历史。

**规范：** [AGENTS.md](AGENTS.md)。旧仓库 `RC_CKRR_semisynth` 已按用户授权清空，原路径现为指向本仓库的软链接；备份与会话史见 [docs/MIGRATION.md](docs/MIGRATION.md)。本项目不延续其 C1 半合成路径。

## 环境

基础依赖：

```bash
python -m pip install -r env/requirements-base.txt
python -m pip install -r env/requirements-bench.txt
```

精确版本见 `env/lock.txt`。FT-Transformer 来自 `rtdl-revisiting-models`（Apache-2.0）。

## 数据

```bash
python scripts/fetch_data.py
```

从 `gdlc/BGLR-R` 固定提交下载 `wheat.RData` 到 `data/raw/`，校验 SHA-256：

```text
8710523389007dd8b7c88a5deb74ee16aea0bae158496bc11f0626fb43b5ea21
```

数据事实、R 列主序读取要求与旧 C-order 故障见 [docs/DATA_FACTS.md](docs/DATA_FACTS.md)。`data/raw/` 不入 Git。

## 运行

复现本轮 Huber ICM 的冻结选择与完整开发集重拟合（指定新的输出目录）：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/run_background_nested.py --config configs/robust_icm.yaml --selection results/robust_icm_development_20261103/selection.json --out results/robust_icm_reproduction
```

本轮实际使用 CPU、Python 3.12.15、NumPy 1.26.4、SciPy 1.13.1、PyTorch 2.7.1+cu128，详见结果目录的 `runtime.json`。以下为历史 FT-T 及其基线入口。

正式 10 折 FT-Transformer（seed 42；test 9×60 + 1×59、val 54、train 485/486；test 只评一次）：

```bash
python scripts/run_bimsgp.py --config configs/bimsgp.yaml --out results/bimsgp_ftt_fixed_20261029
```

同折 RR-BLUP 与 FT-T 训练/验证/测试诊断：

```bash
python scripts/run_bimsgp_diag.py --config configs/bimsgp.yaml --parts rrblup --out results/bimsgp_diag_fixed_20261029
python scripts/run_bimsgp_diag.py --config configs/bimsgp.yaml --parts ftt --out results/bimsgp_diag_ftt_fixed_20261029
```

逐 epoch 轨迹与验证集变体选择（后者默认在选中后只评一次 test）：

```bash
python scripts/run_trajectory.py --config configs/bimsgp.yaml --out results/bimsgp_trajectory_20261030
python scripts/run_variants.py --config configs/bimsgp.yaml --out results/bimsgp_variants_20261030
```

同一条训练轨迹内比较两种预先固定的 checkpoint 规则（最高验证 PCC / 最低验证 RMSE），test 只作评价、不选规则：

```bash
python scripts/run_ckpt_rules.py --config configs/bimsgp.yaml --out results/bimsgp_ckpt_rules_20261030
```

固定 RR-BLUP + A 骨干残差修正（末层零初始化；RR-only 与 residual 按验证 PCC 选分支；test 只评一次）：

```bash
python scripts/run_residual.py --config configs/bimsgp_residual.yaml --out results/bimsgp_residual_20261030
```

完整诱导注意力重构（共享低秩 tokenizer + 双向诱导块 + 多槽读出；首轮完整模型）：

```bash
python scripts/run_induced.py --config configs/bimsgp_induced.yaml --out results/bimsgp_induced_20261030
```

原 FT 编码 + 一层完整标记交互 + 因子化线性读出（第四候选，正式运行待执行）：

```bash
python scripts/run_direct.py --config configs/bimsgp_direct.yaml --out results/bimsgp_direct_20261030
```

测试：

```bash
pytest tests/ -q
```

`tests/test_torch_models.py` 的 TabPFN 用例需要 TabPFN 权重与 `TABPFN_TOKEN`，缺失时自动跳过；当前任务不跑 TabPFN。

## 布局

| 路径 | 内容 |
|------|------|
| `src/wheat599/` | 数据读取、指标与历史 Stage 1 代码 |
| `scripts/` | 命令行入口（数据获取、10 折运行、诊断、变体选择） |
| `configs/` | 运行配置；当前为 `bimsgp.yaml` 与 `bimsgp_selected_20261030.yaml` |
| `data/raw/` | 只读数据（不入 Git；`SHA256SUMS` 固定版本） |
| `results/<run>/` | 配置快照、划分、OOF 预测、指标、manifest |
| `docs/` | 数据事实、当前结果、许可与历史方案 |
| `env/` | 依赖锁 |
| `ars-output/` | 历史 ARS Phase 1 中间产物，不参与当前运行 |

## 文档

| 文档 | 职责 |
|------|------|
| [AGENTS.md](AGENTS.md) | Agent 行为约束、授权与协作、硬边界 |
| [docs/RESULTS.md](docs/RESULTS.md) | 当前有效结果与限制 |
| [docs/DATA_FACTS.md](docs/DATA_FACTS.md) | BGLR 数据核验、R 列主序读取与文档差异 |
| [docs/LICENSE_NOTES.md](docs/LICENSE_NOTES.md) | 数据、代码与权重许可 |
| [docs/REFERENCES.md](docs/REFERENCES.md) | 文献与代码核验状态 |
| [docs/STAGE1_PLAN.md](docs/STAGE1_PLAN.md) | 历史 Stage 1 门控方案，暂停 |
| [docs/STAGE2_PLAN.md](docs/STAGE2_PLAN.md) | 历史六方法协议，未冻结 |
| [docs/REVIEW_PROTOCOL.md](docs/REVIEW_PROTOCOL.md) | 历史评审流程记录 |
| [docs/MIGRATION.md](docs/MIGRATION.md) | 旧仓库清空与 AIA 迁移 |
