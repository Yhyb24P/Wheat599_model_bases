> **历史产物：** 本文件是 ARS Phase 1 的早期产物，部分设计（3 次重复、4 个主对比、15 fold 值 t 区间）已被 2026-10-29 G0 修订取代；当前有效协议以 `docs/STAGE1_PLAN.md` 与 `configs/stage1.yaml` 为准（Stage 2 见 `docs/STAGE2_PLAN.md`）。

## Research Question Brief

### Topic Area

BGLR Wheat599 上的 marker-only 表型预测模型基座比较：线性收缩、Gaussian RKHS、RBF-SVR、TabM、小型 FT-Transformer 与冻结的 TabPFN-3.5 ICL。

### Primary Research Question

在 BGLR Wheat599 的四个环境上，采用无泄漏的嵌套交叉验证，冻结的 TabPFN-3.5 上下文学习相对从头训练的小型 FT-Transformer、TabM、Gaussian RKHS、RBF-SVR 与线性收缩，是否在样本外 PCC 与 RMSE 上取得可重复的、超过方法间噪声的改进？

### FINER Assessment

| Criterion | Score | Justification |
|-----------|-------|---------------|
| Feasible | 5/5 | 数据公开且已核验（599×1279 二值标记、4 环境、A 矩阵）；六类方法均有官方或成熟实现；单卡 32GB 与 CPU 足以完成 5×3×4×3 的嵌套 CV；无人类受试者。 |
| Interesting | 4/5 | 预训练表格基础模型在约 480 个训练材料上的表现，与"小样本高维"传统基因组预测经验存在张力；Wheat599 是领域常用基准，但缺少同一划分下的无泄漏多基座比较。 |
| Novel | 4/5 | 新意不在新算法，而在同一数据、同一划分、同一指标下，把预训练先验、逐标记注意力、非注意力神经网络与核方法放在一起，并区分"超过线性"与"超过非线性对照"两种证据等级。 |
| Ethical | 5/5 | 公开数据；无人类受试者；TabPFN 代码与权重许可分别核查；不涉及育种部署承诺。 |
| Relevant | 4/5 | 直接服务于本项目后续模型基座选择；对基因组预测中"是否值得引入预训练/注意力"的判断有方法学价值，但不直接等于育种应用收益。 |
| **Average** | **4.4/5** | 达到 FINER 阈值。 |

### Scope Boundaries

**In Scope:**

- BGLR `wheat.RData` 标准公开版本，599 份 CIMMYT 小麦纯系，1279 个 DArT 0/1 标记，4 个环境的平均产量。
- marker-only 预测；四个环境分别建模、分别报告，保留实际列名 `"1"`、`"2"`、`"4"`、`"5"`。
- 六类核心基座：线性收缩（Ridge/RR-BLUP/GBLUP 取一种实现）、Gaussian RKHS、RBF-SVR、TabM、小型 FT-Transformer、TabPFN-3.5 冻结 ICL。
- 无泄漏嵌套交叉验证；随机划分与亲缘簇划分两种协议；PCC、RMSE、MAE、R²；材料级配对不确定性。
- 数据核验、代码、配置、依赖锁、随机种子与 OOF 预测的可复现记录。

**Out of Scope:**

- 其它小麦群体、其它物种、真实育种项目数据；本项目不声称外推到 CIMMYT 之外。
- 基因型×环境互作建模、环境协变量、多性状联合建模。
- 生物学上位互作、因果基因发现、标记效应解释；基准排名不构成这些证据。
- TabPFN 微调、从零预训练表格基础模型；TabICLv2 等开放预训练平台只在后续单独立项。
- GPformer、GP-WAITER 作为工程基座；如作为领域对照须重新实现并单列。
- 谱系矩阵 A 进入核心比较；A 只用于亲缘划分敏感性分析或单列扩展。
- 旧仓库 RC_CKRR_semisynth 的 C1 半合成审计路径。

**Domain:** 统计遗传学 / 基因组预测 / 表格机器学习基准。

**Timeframe:** 2026-10-29 起；数据与文献证据截至 2026-10-29。

**Geography:** 数据来源为 CIMMYT 历史小麦材料；分析在本地完成。

**Population:** 599 份小麦纯系；无人类或动物受试者。

**Key Assumptions:**

- BGLR 发布的 Y 是该基准的标准目标；四个环境虽有相关，但按四个任务分别建模是合理的。
- X 的 0/1 编码按发布方定义使用；MAF<0.05 列的存在是数据事实，不擅自删除。
- 嵌套 CV 的材料级划分能近似"对未见材料预测"的群体内目标；亲缘簇划分提供更严格的外推压力测试。
- 六类方法的实现差异（正则化参数化、核尺度、早停）可以在统一协议内被公平处理；无法完全消除的实现差异作为限制记录。

### Sub-questions

1. 四类基座（线性、核、非注意力神经、逐标记注意力、预训练 ICL）在四个环境与两种划分协议下的样本外 PCC/RMSE 排序是否一致？哪些差异超过材料级配对不确定性？
2. 若小型 FT-Transformer 优于线性模型，其增益是否也超过 TabM 与 Gaussian RKHS？即增益应解释为一般非线性，还是逐标记表示/注意力？
3. TabPFN-3.5 的增益是否稳定于默认的 8 成员 × 768 列 balanced 子采样与二值标记类别化处理？在强制数值化或全列覆盖变体下是否消失？

### Sub-Question Bindings

1. inherits: population=Wheat599 599 lines; timeframe=2026-10-29; geography=CIMMYT historical lines; domain=genomic prediction benchmark; deviations: none
2. inherits: population=Wheat599 599 lines; timeframe=2026-10-29; geography=CIMMYT historical lines; domain=genomic prediction benchmark; deviations: none
3. inherits: population=Wheat599 599 lines; timeframe=2026-10-29; geography=CIMMYT historical lines; domain=genomic prediction benchmark; deviations: none

### Candidate Questions Considered

| # | Candidate | FINER Avg | Why not selected |
|---|-----------|-----------|-----------------|
| 1 | 六类基座在 Wheat599 四环境、两种划分下的无泄漏比较（本 RQ） | 4.4 | 选中：直接回答基座选择问题，且可在现有资源内完成。 |
| 2 | 逐标记注意力是否发现生物学上位互作 | 2.6 | 不可行：599 份材料、1279 个二值标记、无坐标，无法支持机制性因果主张。 |
| 3 | 预训练表格模型能否替代 GBLUP 用于育种决策 | 2.8 | 范围过大：需要多群体、多性状、决策损失与经济评估；本项目只提供基准证据。 |
| 4 | 在 Wheat599 上复现 Fan & Waldmann 的模型排名 | 2.4 | 公开代码存在泄漏与划分不一致；复现其排名不能提供可靠证据。 |
| 5 | 为 DArT 标记设计新的位置感知注意力编码 | 2.2 | 缺少标记坐标；列序号不能解释为遗传距离，机制假设不成立。 |
