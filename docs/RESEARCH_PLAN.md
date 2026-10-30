# Research Plan — Wheat599 FT-Transformer 评价

**当前任务（2026-10-30）：** FT-Transformer 在 BGLR Wheat599 上的 10 折评价与失败诊断。seed 42；test 9×60 + 1×59、val 54、train 485/486；四环境共享；Train-only scaler；PCC/RMSE/MAE；test 只评一次。已修正 `build_model()` 前的初始化种子；完成同一训练轨迹内的双 checkpoint 规则比较（最高验证 PCC / 最低验证 RMSE），以及固定 RR-BLUP + A 骨干残差修正首轮（末层零初始化；RR-only/residual 用合成验证预测选分支）。test 只作评价、不选规则。正式结果、同折 RR-BLUP 对照、过拟合诊断与探索性变体选择见 [RESULTS.md](RESULTS.md)。

当前结论：固定种子的 val-PCC 规则 test PCC 宏均 0.4342，val-RMSE 规则 0.4302、test RMSE 0.9242；两者都低于同折 RR-BLUP（PCC 0.4528、RMSE 0.8957）。RMSE 规则把预测尺度 k 从 0.5555 拉近 r=0.43，改善均值/尺度校准（MSE 0.9017→0.8632），但方向项略差（0.7852→0.7868），未提高跨材料相关性；预测 SD 小本身不是退化，MSE 最优尺度为 k=r。固定 RR-BLUP + A 骨干残差修正首轮已完成：40/40 折选中 residual，test PCC 0.4517、RMSE 0.9176，相对 A 的点估计改善为 +0.0175 / −0.0279，但未观察到相对 RR-BLUP 的可复现增量（点估计 −0.0011 / +0.0218），MSE 点估计增加主要来自尺度失配（+0.0369），方向项几乎不变；修正量约为验证残差 SD 的 0.243，仍未能提高可泛化相关性。40/40 是选择规则的合法结果，不是有效性证据；val 偏好未迁移到 test。10 折不独立，只作描述，不做 p 值/CI。

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

当前配置：`configs/bimsgp.yaml`；验证集选中变体的测试配置：`configs/bimsgp_selected_20261030.yaml`；残差修正首轮：`configs/bimsgp_residual.yaml`。
