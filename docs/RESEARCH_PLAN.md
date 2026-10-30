# Research Plan — Wheat599 FT-Transformer 评价

**当前任务（2026-10-30）：** FT-Transformer 在 BGLR Wheat599 上的 10 折评价与失败诊断。seed 42；test 9×60 + 1×59、val 54、train 485/486；四环境共享；Train-only scaler；PCC/RMSE/MAE；test 只评一次。已修正 `build_model()` 前的初始化种子；新增同一训练轨迹内的双 checkpoint 规则比较（最高验证 PCC / 最低验证 RMSE），test 只作评价、不选规则。正式结果、同折 RR-BLUP 对照、过拟合诊断与探索性变体选择见 [RESULTS.md](RESULTS.md)。

当前结论：固定种子的 val-PCC 规则 test PCC 宏均 0.4342，val-RMSE 规则 0.4302、test RMSE 0.9242；两者都低于同折 RR-BLUP（PCC 0.4528、RMSE 0.8957）。切换 checkpoint 规则改善 RMSE 但降低 PCC，且没有恢复预测幅度（预测 SD/真实 SD 宏均由 0.556 降到 0.446）。主要问题仍是训练集到验证集的泛化差距和预测收缩，checkpoint 选择只解释少数折的近常数预测。10 折不独立，只作描述，不做 p 值/CI。

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
