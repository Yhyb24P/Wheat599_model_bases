# Research Plan — Wheat599 FT-Transformer 评价

**当前任务（2026-10-30）：** FT-Transformer 在 BGLR Wheat599 上的 10 折评价与失败诊断。seed 42；test 9×60 + 1×59、val 54、train 485/486；四环境共享；Train-only scaler；PCC/RMSE/MAE；test 只评一次。已完成正式结果、同折 RR-BLUP 对照、过拟合诊断与验证集变体选择；结果见 [RESULTS.md](RESULTS.md)。

当前结论：FT-Transformer 宏均 test PCC 0.4341，低于同折 RR-BLUP 0.4528；验证集选中的变体 test PCC 0.4324；主要问题是训练集到验证集的泛化差距，不是数据行对应错误或欠拟合。10 折不独立，只作描述，不做 p 值/CI。

**历史方案（暂停，不执行）：**

- [STAGE1_PLAN.md](STAGE1_PLAN.md)：冻结 TabPFN-3.5 ICL / 小型 FT-Transformer / Ridge 的门控方案。
- [STAGE2_PLAN.md](STAGE2_PLAN.md)：六方法完整协议，未冻结。
- [REVIEW_PROTOCOL.md](REVIEW_PROTOCOL.md)：G0–G4 与旧评审纪律，保留为记录。

## 快速索引

| 文档 | 职责 |
|------|------|
| [RESULTS.md](RESULTS.md) | 当前有效结果与限制 |
| [DATA_FACTS.md](DATA_FACTS.md) | BGLR 数据核验、R 列主序读取与文档差异 |
| [LICENSE_NOTES.md](LICENSE_NOTES.md) | 数据、代码与权重许可 |
| [REFERENCES.md](REFERENCES.md) | 文献与代码核验状态 |
| [MIGRATION.md](MIGRATION.md) | 旧仓库清空与 AIA 迁移记录 |

当前配置：`configs/bimsgp.yaml`；验证集选中变体的测试配置：`configs/bimsgp_selected_20261030.yaml`。
