# Research Plan — Wheat599 模型基座比较

**当前 active 阶段：Stage 1（first glance）** — [docs/STAGE1_PLAN.md](STAGE1_PLAN.md)

在 BGLR Wheat599 上比较冻结的 TabPFN-3.5 ICL 与小型 FT-Transformer，相对 Ridge 参照是否有值得进入 Stage 2 的信号。Stage 1 是门控，不是发表推断；结果不作稳定改进声明。

**Stage 2（条件执行，未冻结）** — [docs/STAGE2_PLAN.md](STAGE2_PLAN.md)

六方法完整协议（线性收缩、Gaussian RKHS、RBF-SVR、TabM、FT-Transformer、TabPFN；random + IBS 亲缘臂；10/5 重复；8 主对比；LightGBM secondary）。只有 Stage 1 触发正向信号后，才启动 Stage 2 G0（gb/iq 签核 + 用户授权提交）。

## 快速索引

| 文档 | 职责 |
|------|------|
| [STAGE1_PLAN.md](STAGE1_PLAN.md) | 当前冻结对象：方法、划分、指标、信号门、预算 |
| [STAGE2_PLAN.md](STAGE2_PLAN.md) | 条件执行的完整六方法协议 |
| [REVIEW_PROTOCOL.md](REVIEW_PROTOCOL.md) | G0–G4、监管与冻结纪律 |
| [DATA_FACTS.md](DATA_FACTS.md) | BGLR 数据核验与文档差异 |
| [REFERENCES.md](REFERENCES.md) | 文献与许可核验状态 |
| [RESULTS.md](RESULTS.md) | 当前有效结果（目前为空） |
| [MIGRATION.md](MIGRATION.md) | 旧仓库清空与 AIA 迁移记录 |

配置：`configs/stage1.yaml`（active）；`configs/stage2.yaml`（保留，不执行）。