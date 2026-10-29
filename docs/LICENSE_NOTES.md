# License Notes

**核验日期：** 2026-10-29。只记录已实际查询到的许可；未确认项标为待核实。

## 数据

| 项 | 许可 / 状态 | 说明 |
|----|-------------|------|
| BGLR 包（含 `wheat.RData`） | GPL-3 | `DESCRIPTION` 字段 `License: GPL-3`。数据通过 `scripts/fetch_data.py` 从官方仓库获取，不在本仓库重新分发。 |
| wheat 数据本身 | 未发现独立许可声明 | 数据来自 CIMMYT，经 BGLR 包发布；使用与再分发遵循 BGLR 的 GPL-3 与来源方要求。发表时引用 BGLR 与 CIMMYT 来源。 |

## 代码与权重

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
