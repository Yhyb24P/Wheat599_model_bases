# Wheat599 模型基座比较

**Stage 1（active，first glance）：** 在 BGLR Wheat599（599 份 CIMMYT 小麦纯系、1279 个 DArT 0/1 标记、4 个环境平均产量）上，比较冻结的 TabPFN-3.5 ICL、小型 FT-Transformer 与 Ridge 参照；random 外层 5 折 × 3 重复；PCC/RMSE；2 GPU-hours 上限。Stage 1 是门控，不是发表推断；信号门见 [docs/STAGE1_PLAN.md](docs/STAGE1_PLAN.md) §6。

**Stage 2（条件执行，未冻结）：** 六方法完整协议（含 RKHS、RBF-SVR、TabM、IBS 亲缘臂、10/5 重复、8 主对比、LightGBM secondary）保留在 [docs/STAGE2_PLAN.md](docs/STAGE2_PLAN.md)；只有 Stage 1 触发正向信号后才启动 Stage 2 G0。

**状态（2026-10-29）：协议草案，未开始正式运行，无结果。** 项目规范见 [AGENTS.md](AGENTS.md)，当前阶段见 [docs/STAGE1_PLAN.md](docs/STAGE1_PLAN.md)，评审流程见 [docs/REVIEW_PROTOCOL.md](docs/REVIEW_PROTOCOL.md)。旧仓库 `RC_CKRR_semisynth` 已按用户授权清空，原路径现为指向本仓库的软链接；备份与会话史见 [docs/MIGRATION.md](docs/MIGRATION.md)。本项目不延续其 C1 半合成路径。

## 布局

| 路径 | 内容 |
|------|------|
| `src/wheat599/` | 数据读取、模型适配、CV 编排、指标 |
| `scripts/` | 命令行入口（数据获取、运行基准） |
| `configs/` | 运行配置与模型网格 |
| `data/raw/` | 只读数据（不入 Git；`SHA256SUMS` 固定版本） |
| `results/<run>/` | 配置快照、划分、OOF 预测、指标、manifest |
| `docs/` | 研究方案、数据事实、评审流程、许可记录、当前结果 |
| `env/` | 依赖锁 |
| `ars-output/` | ARS 工作流的 Phase 1 中间产物（研究问题简报、方法蓝图） |

## 数据

```bash
python scripts/fetch_data.py
```

从 `gdlc/BGLR-R` 固定提交下载 `wheat.RData` 到 `data/raw/`，校验 SHA-256：

```text
8710523389007dd8b7c88a5deb74ee16aea0bae158496bc11f0626fb43b5ea21
```

数据事实与官方文档差异见 [docs/DATA_FACTS.md](docs/DATA_FACTS.md)。

## 运行

正式实现尚未完成。计划入口：

```bash
python scripts/run_benchmark.py --config configs/stage1.yaml --out-dir results/stage1
```

Stage 2 配置保留在 `configs/stage2.yaml`，不执行。

## 文档

| 文档 | 职责 |
|------|------|
| [AGENTS.md](AGENTS.md) | Agent 行为约束、授权与协作、硬边界 |
| [docs/STAGE1_PLAN.md](docs/STAGE1_PLAN.md) | 当前 Stage 1：方法、划分、指标、信号门、预算 |
| [docs/STAGE2_PLAN.md](docs/STAGE2_PLAN.md) | 条件执行的完整六方法协议 |
| [docs/RESEARCH_PLAN.md](docs/RESEARCH_PLAN.md) | 阶段索引与快速导航 |
| [docs/REVIEW_PROTOCOL.md](docs/REVIEW_PROTOCOL.md) | 协议冻结、结果审计、ARS 评审面板、监管纪律 |
| [docs/DATA_FACTS.md](docs/DATA_FACTS.md) | BGLR 数据核验与文档差异 |
| [docs/LICENSE_NOTES.md](docs/LICENSE_NOTES.md) | 数据、代码与权重许可 |
| [docs/REFERENCES.md](docs/REFERENCES.md) | 文献、代码与权重许可的核验状态 |
| [docs/RESULTS.md](docs/RESULTS.md) | 当前有效结果（目前为空） |
