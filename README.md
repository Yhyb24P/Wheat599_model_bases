# Wheat599 FT-Transformer 10 折评价与失败诊断

**状态（2026-10-30）：** 正式 10 折评价、同折 RR-BLUP 对照、过拟合诊断与验证集变体选择完成。FT-Transformer 宏均 test PCC 0.4341，同折 RR-BLUP 0.4528；验证集选中的变体 test PCC 0.4324，未缩小差距。完整数字与判读见 [docs/RESULTS.md](docs/RESULTS.md)。

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

测试：

```bash
pytest tests/ -q
```

`tests/test_torch_models.py` 的 TabPFN 用例需要 TabPFN 权重与 `TABPFN_TOKEN`；当前任务不跑 TabPFN。

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
