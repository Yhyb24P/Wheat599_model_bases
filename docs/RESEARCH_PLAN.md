# Research Plan — Wheat599 FT-Transformer 评价

**当前主线（2026-11-02）：** 优化背景条件化联合收缩完整模型，以 PCC 为主指标，与合理优化的 BiMSGP、MoEGP 和基础 RR-BLUP 在同一外层折比较。每个外层折的 90% 开发数据内用三折选模，完成后用完整开发数据重拟合，外层 test 只用于最终比较；PCA 与标准化仅在各次训练子集拟合。旧的 81% 训练结果保留为首轮记录，不作为新协议的公平对照。不同模型可使用适合各自的配置和预算；不以测试结果改配置，也不在当前阶段做模块消融。已有外层折曾参与多轮开发，新结果仅作描述性比较。

当前开发选模：背景模型预设 48 个完整配置（背景维数、门控数、两类收缩、训练步数和学习率），每配置三折，以均值 PCC 选择；RR 在同一内部折独立选择 alpha。BiMSGP 使用作者公开实现及三组候选，验证 PCC 选择配置与训练轮数。MoEGP 同协议与预算正在整理；所有外层结果须在比较规则冻结后集中评价。

背景模型及 RR 的 40 个开发单元已完成选模，未作本轮外层测试预测；选择记录位于 `results/background_nested_selection_20261102/`。BiMSGP 官方源码 `71084c8fa5444fbc747b0291ace627b1f2d6ef33`，预设 3 组候选、每组最多 100 轮；每个 epoch 汇总三折验证 PCC 选配置和轮数，完整开发集重拟合使用三折中位学习率轨迹。首次 FP16 开发运行出现非有限训练损失／验证 PCC 后停止，改用官方公开训练器支持的 FP32，失败记录保留在本地；未使用外层测试预测。作者论文表格使用 test 选 checkpoint，本项目不采用该选择方式。

协作方另行进行的 A/RR 与 MoEGP 冒烟曾分别预测并查看外层折 0、环境 1 的测试指标；据其报告未据此改动配置。预测记录保留在本地，不能称该折未被观察；整个划分此前亦已用于多轮开发，本轮比较不作为独立确认。最终汇总保留全部折，显式披露测试接触。

以下 2026-10-30 内容是历史 FT-T 任务记录。

**当前任务（2026-10-30）：** FT-Transformer 在 BGLR Wheat599 上的 10 折评价与失败诊断。seed 42；test 9×60 + 1×59、val 54、train 485/486；四环境共享；Train-only scaler；PCC/RMSE/MAE；test 只评一次。已修正 `build_model()` 前的初始化种子；完成同一训练轨迹内的双 checkpoint 规则比较（最高验证 PCC / 最低验证 RMSE）、固定 RR-BLUP + A 骨干残差修正首轮，以及共享低秩编码 + 双向诱导注意力 + 多槽读出的完整 FT-T 重构首轮。test 只作评价、不选规则。正式结果、同折 RR-BLUP 对照、过拟合诊断与探索性变体选择见 [RESULTS.md](RESULTS.md)。

当前结论：固定种子的 val-PCC 规则 test PCC 宏均 0.4342，val-RMSE 规则 0.4302、test RMSE 0.9242；两者都低于同折 RR-BLUP（PCC 0.4528、RMSE 0.8957）。RMSE 规则把预测尺度 k 从 0.5555 拉近 r=0.43，改善均值/尺度校准（MSE 0.9017→0.8632），但方向项略差（0.7852→0.7868），未提高跨材料相关性；预测 SD 小本身不是退化，MSE 最优尺度为 k=r。固定 RR-BLUP + A 骨干残差修正首轮已完成：40/40 折选中 residual，test PCC 0.4517、RMSE 0.9176，相对 A 的点估计改善为 +0.0175 / −0.0279，但未观察到相对 RR-BLUP 的可复现增量（点估计 −0.0011 / +0.0218）。完整诱导注意力重构首轮（共享低秩 tokenizer + 双向诱导块 + 多槽读出）宏均 PCC 0.3382、RMSE 0.9764，vs A −0.0961/+0.0309，vs RR −0.1147/+0.0806；预注册 GO=False，未做消融或重试。该 NO-GO 只针对本配置/单 seed/固定十折，不构成对重构方向的一般性否定，也未做模块级归因。10 折不独立，只作描述，不做 p 值/CI。

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

当前配置：`configs/bimsgp.yaml`；验证集选中变体的测试配置：`configs/bimsgp_selected_20261030.yaml`；残差修正首轮：`configs/bimsgp_residual.yaml`；完整诱导重构：`configs/bimsgp_induced.yaml`。
