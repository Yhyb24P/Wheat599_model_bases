# License Notes

**核验日期：** 2026-10-29。只记录已实际查询到的许可；未确认项标为待核实。

## 数据

| 项 | 许可 / 状态 | 说明 |
|----|-------------|------|
| BGLR 包（含 `wheat.RData`） | GPL-3 | `DESCRIPTION` 字段 `License: GPL-3`。数据通过 `scripts/fetch_data.py` 从官方仓库获取，不在本仓库重新分发。 |
| wheat 数据本身 | 未发现独立许可声明 | 数据来自 CIMMYT，经 BGLR 包发布；使用与再分发遵循 BGLR 的 GPL-3 与来源方要求。发表时引用 BGLR 与 CIMMYT 来源。 |

## 代码与权重

材料—位点推断及普通上下文对照的四个 checkpoint 由本项目独立模拟任务训练，未使用第三方预训练权重，未指定独立权重分发许可；仅保留本地，不入 Git。训练来源 `e3c7369`，目录 `results/matrix_inference_learning_20261104/`：

| 本地文件 | SHA-256 |
|---|---|
| `matrix_42.pt` | `09be932e3c86f954c3214696385bc32f34da99dd5cf7f714ac26e0356b7175ed` |
| `matrix_2026.pt` | `8c4ac599026d93a3e9db8a4f1b9c803238b68ba4ed28e2edec7f6b0ff64d0345` |
| `direct_42.pt` | `30c18195a70605b5aa986751b3ee9fbc608ec327f3ef55894703862233064015` |
| `direct_2026.pt` | `4816e2d527e38e23cd59c6521d7d510867ee353ba1f64d474056e1fa0dbceafc` |

DNAwhisper 从外部固定源码 `Marxin1992/Whisperer_of_DNA@7a657cc00d44263c4b1085d3991ecc0e935c14e5` 加载，2026-11-04 核对本地固定副本的 `LICENSE` 为 Apache-2.0；本仓库只提交协议适配器，不复制上游代码或权重。所导入源码及配置的 Git blob 摘要固定在 `configs/dnawhisper_nested.yaml`，运行时逐项核验。没有使用上游预训练权重，各折从头执行训练子集内预训练；如后续保存本项目权重，权重不入 Git，其 SHA 另行记录。

MoEGP 官方仓库在提交 `d3a3c44f649fadf95af34d63f069fc8dfd85af38` 未发现许可证文件；本项目从本地官方副本加载代码，不复制或重新分发该源码。

BiMSGP 官方仓库提交 `71084c8fa5444fbc747b0291ace627b1f2d6ef33` 使用 MIT 许可证；本项目从本地官方副本加载源码。本次安装的 `mamba-ssm` 2.3.2.post1 为 Apache-2.0，`causal-conv1d` 1.6.2.post1 为 BSD 3-Clause，均以安装包附带 LICENSE 为准。

| 项 | 许可 / 状态 | 说明 |
|----|-------------|------|
| TabPFN 推理代码（`PriorLabs/TabPFN`） | Apache-2.0 | GitHub API 核实。 |
| TabPFN-3.5 权重（`Prior-Labs/tabpfn_3_5`） | `tabpfn-3-5-license-v1.0`（非商用） | 明确允许测试、评估、内部基准与学术研究；禁止商业/生产用途、商业决策、客户交付物、付费产品。许可证全文保存在 `docs/licenses/tabpfn-3-5-license-v1.0.txt`。权重不进入本仓库。 |
| `rtdl-revisiting-models` | Apache-2.0 | `yandex-research/rtdl-revisiting-models` 仓库含 LICENSE；PyPI wheel 的 `dist-info` 含 Apache-2.0 LICENSE 与 `License :: OSI Approved :: Apache Software License`。此前的“未检测到许可证”是查错了 404 路径 `tabular-dl-revisiting-models`。 |
| TabM（`yandex-research/tabm`） | Apache-2.0 | GitHub API 核实。 |
| LassoNet（`lasso-net/lassonet`，扩展比较） | MIT | GitHub API 核实。 |
| numpy / scipy / scikit-learn / PyTorch | 各自发行版许可（常见为 BSD 系） | 发布前在 `env/` 锁文件中核对具体版本与许可。 |

## 使用边界

- TabPFN-3.5 权重只用于本项目的非商业研究、内部评估与学术比较；不用于采购竞标基准、客户交付或任何生产决策。
- 不在仓库、论文附件或公开镜像中分发 TabPFN 权重；只记录 checkpoint 名称、SHA 与获取方式。
- 若研究结果对外发布，确认输出（预测、指标、图表）不构成许可证禁止的“客户交付物”或商业决策依据；必要时在论文中声明许可与使用边界。
- `rtdl-revisiting-models` 为 Apache-2.0；保留 LICENSE 与版本记录，不再列为 G4 阻断项。
