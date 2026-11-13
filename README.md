# Wheat599 植物基因组表型预测研究

现役主候选是**四独立分位数 TabM 与多环境 Huber ICM 的预固定等权融合**。统一协议外层宏 PCC 为 **0.526506**；项目内结果不代表跨数据集领先。主比较沿用既有材料级十外折、三内折，随机模型使用种子 42、2026 的平均预测；不根据外层结果更改配置或融合权重。

## 阅读结果与模型

- [结果与架构 HTML](docs/models.html)：可搜索、筛选、排序并导出 CSV 的模型总表；默认候选与公开强模型置前，最优候选提供可按外折/环境切换的真实网络层级图，以及参数共享、分位数监督、ICM 核与损失机理图。
- [参考文献与逐篇讲解](ref/index.html)：文献、来源及其与本项目的关系。
- [运行目录索引](results/index.html)：按原始运行目录查找摘要、配置、划分与状态。
- [结果原文](docs/RESULTS.md)、[评价协议](docs/EVALUATION_PROTOCOL.md)、[历史模型对比](docs/MODEL_COMPARISON.md)：可视化的事实来源。

HTML 可直接用浏览器打开，不需要联网或启动服务；外部论文链接需要联网。页面是生成视图，修订证据后在仓库根目录重建：

```bash
python3 scripts/build_report.py
python3 ref/build.py
python3 tests/test_report.py
```

结果页构建使用 Python 标准库和 Pandoc（含原生 MathML 输出），文献页仅需标准库。上述命令不调用训练入口、不读取原始表型、不重新计算模型成绩；测试在安装 BibTeX 时检查书目能否实际解析。文献内容统一维护在 `ref/build.py`。

最新完成的是[完整 Mul-PheG2P 同协议适配](docs/MULPHE_UNIFIED.md)：1,920 个开发单元、320 次外层重拟合、0 运行失败；调优标量融合 PCC 0.438225，默认配置 0.443596。目标编码器已包含；适配差异见原记录。该批结束，不自动追加实验。

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

## 运行与验证

正式实现保留在 `src/wheat599/`，运行配置保留在 `configs/`；多数当前方法复用 `scripts/run_background_nested.py`。以下命令供复现使用，不表示本轮整理会启动训练。输出必须指定新目录；不要覆盖历史运行。

```bash
# Huber ICM：复用已冻结选择，完整开发集重拟合
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/run_background_nested.py --config configs/robust_icm.yaml --selection results/robust_icm_development_20261103/selection.json --out results/robust_icm_reproduction

# 分位数 TabM：使用已冻结外层清单（需要本地数据与基线 OOF）
python scripts/run_background_nested.py --config configs/quantile_tabm_final.yaml --out results/quantile_tabm_reproduction

# 现有代码测试
PYTHONPATH=src python -m pytest tests/ -q
```

分位数训练配方见 [开发说明](docs/QUANTILE_TABM.md) 与 [最终重拟合](docs/QUANTILE_TABM_FINAL.md)；其他方法的运行命令在各自方案中，可从架构 HTML 查找。大预测、数据与权重不随 Git 分发，因此仅克隆仓库不足以复现依赖历史预测的运行。当前任务不跑 TabPFN、BiMSGP 或新数据集；TabPFN 测试依赖缺失时会跳过。

## 文件与结果约定

| 路径 | 职责 |
|---|---|
| `src/wheat599/` | 唯一正式模型、训练与汇总实现 |
| `scripts/` | 现有 CLI、数据获取及离线报告构建 |
| `configs/` | 各次运行配置；由原运行 provenance/manifest 确定版本 |
| `results/<run>/` | 原始运行产物；保留路径、命名与来源，不为展示搬移或重复复制 |
| `results/index.html` | 从本地运行目录生成的导航，不作运行成功判定 |
| `docs/RESULTS.md` | 结果、失败、停止原因和证据限制的主记录 |
| `docs/models.html` | 从结果与方案生成的离线展示 |
| `ref/` | 参考文献和简要 HTML 讲解 |
| `docs/` | 使用协议、数据事实、模型方案、许可与历史记录 |
| `data/raw/`、`models/` | 本地只读数据与权重，不入 Git |
| `env/` | 依赖与已授权机器环境 |
| `ars-output/`、`unused/`、`临时文档/` | 既有历史或本地材料，不作为现役入口，不自动删除 |

新运行沿用 `results/<方法>_<阶段>_<日期>/`，无需重命名既有目录。主表只接收已核对统一协议的结果；开发选模、历史不同协议、模拟、其他数据集、停止与失败记录各保留原口径。常见 `summary` 保存指标，`provenance` / `manifest` 保存数据 SHA、源码、配置、种子和划分，`execution_status` / `failures` / `interruption` 保存执行状态；不同历史文件格式不强制重写。逐材料与逐种子预测保留原位置，不能因导航整理而丢弃。

[AGENTS.md](AGENTS.md) 保存执行边界；[DATA_FACTS](docs/DATA_FACTS.md) 保存数据事实；[LICENSE_NOTES](docs/LICENSE_NOTES.md) 保存许可；[REFERENCES](docs/REFERENCES.md) 保留历史文献核验记录。旧路径 `RC_CKRR_semisynth` 是本仓库软链接，见 [迁移记录](docs/MIGRATION.md)。旧评审、G0、Stage 1/2 与交接文档只作历史，不重新启用其流程。
