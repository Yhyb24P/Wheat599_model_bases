> **历史产物：** 本文件是 ARS Phase 1 的早期产物，部分设计（3 次重复、4 个主对比、15 fold 值 t 区间）已被 2026-10-29 G0 修订取代；当前有效协议以 `docs/STAGE1_PLAN.md` 与 `configs/stage1.yaml` 为准（Stage 2 见 `docs/STAGE2_PLAN.md`）。

## Methodology Blueprint

### Research Paradigm

**Selected:** 实证主义 / 计算科学（positivist, computational）。

**Justification:** 研究问题是可观测、可重复的样本外预测比较：固定数据、固定划分、固定指标，比较不同模型基座的预测风险。不涉及意义建构或干预效果；适合以预注册协议和可复现计算来约束推断。

### Method

**Type:** 定量（quantitative）。

**Specific Method:** 比较基准（comparative benchmarking）＋无泄漏嵌套交叉验证（nested cross-validation）＋材料级配对不确定性。

**Justification:** RQ 问的是"在未见材料上，哪类基座的预测风险更低"，而不是"某个模型能否拟合训练数据"。嵌套 CV 把超参数选择限制在外层训练折内，直接对应这个目标；配对 bootstrap 以材料为抽样单位，避免把 CV 折当作独立重复。

### Data Strategy

**Data Type:** 二手公开数据（secondary data）。

**Sources:**

- `wheat.X`：599×1279 DArT 0/1 标记；官方文档说明为纯系 presence/absence，发布方做过 MAF<0.05 过滤与边际 Bernoulli 插补；实测仍有 96/1279 列 MAF<0.05。
- `wheat.Y`：599×4 环境平均产量，已按环境标准化；列名 `"1"`、`"2"`、`"4"`、`"5"`。
- `wheat.A`：599×599 ICIS 谱系 numerator relationship matrix；对角 1.5572–2，非对角 0–1.9978，与 $XX^\top/p$ 的相关系数约 0.29。核心分析不用 A。
- `wheat.sets`：发布方提供的随机 10 折向量；不作为主协议，可作为与 BGLR 示例衔接的敏感性分析。
- 来源：`gdlc/BGLR-R` 的 `data/wheat.RData`，SHA-256 `8710523389007dd8b7c88a5deb74ee16aea0bae158496bc11f0626fb43b5ea21`。

**Sampling:** 全量 599 份材料；不抽样。外层训练折约 479–480 份，内层训练折约 319–320 份。

**Time Frame:** 数据版本固定为上述 SHA-256；文献与实现证据截至 2026-10-29。

### Analytical Framework

**Technique:** 嵌套交叉验证＋预注册模型网格＋OOF 预测的材料级配对比较。

**Steps:**

1. **数据固定。** 下载并校验 `wheat.RData`；记录 SHA-256、来源、R 版本与读取脚本。X 按发布版使用；MAF 过滤只作为单独敏感性分析。
2. **划分生成。** 外层 5 折，重复 3 次，种子固定；内层 3 折。随机协议按环境 Y 的分位数组分层，并兼顾每行标记均值（技术性协变量）；亲缘簇协议先用标记 IBS 距离层次聚类得到簇（例如 $K=10$），再按簇分组做 5 折外层与 3 折内层。划分文件写入 `results/<run>/splits/`。
3. **模型网格。** 每个模型的超参数网格在运行前写入 `configs/`；选择只在内层 CV 内完成。早停只使用内层验证折。
4. **外层评估。** 每个外层折只用训练折拟合/选择，预测留出折；每个材料在每个重复中恰好得到一个 OOF 预测。
5. **指标。** 每个环境、每个方法、每个重复计算 PCC、RMSE、MAE、R²；报告重复间均值与范围。主指标为 PCC 与 RMSE，二者同时报告。
6. **配对比较。** 以线性收缩为参照，计算材料级配对差值（如 $\Delta\mathrm{RMSE}$、$\Delta\mathrm{PCC}$）；用 10,000 次材料级 bootstrap 给出 95% 区间。亲缘簇协议按簇 bootstrap，避免把同簇材料当独立样本。
7. **多重性。** 预注册主要对比（TabPFN vs 线性；FT-Transformer vs 线性；FT-Transformer vs TabM；FT-Transformer vs RKHS）；其余对比作为探索性，报告全部结果与区间；若使用 p 值，在同一对比族内做 Holm 校正，不以显著性阈值代替效应量。
8. **敏感性。** (a) MAF≥0.05 列过滤；(b) BGLR `wheat.sets` 10 折（单层，不调参）作为与文献衔接的参考；(c) TabPFN 强制数值化（`MIN_UNIQUE_FOR_NUMERICAL_FEATURES=2`）与默认类别化对比；(d) 外层重复种子间的排序稳定性。
9. **记录。** 每次运行保存配置快照、划分、OOF、指标、依赖锁、代码提交、数据 SHA 与 manifest；不得用外层测试结果回头选择模型。

**Tools:** Python（numpy、scipy、scikit-learn、PyTorch、tabpfn、rtdl-revisiting-models、tabm）；R 仅用于数据读取与 BGLR 文档核对；结果以 CSV/JSON/Markdown 保存。

### Validity Criteria

| Criterion | Strategy to Ensure |
|-----------|-------------------|
| Construct validity | 指标只声称预测精度（PCC/RMSE/MAE/R²），不声称生物学机制或育种决策收益；TabPFN 的增益归因于预训练先验＋ICL＋集成/子采样，FT-Transformer 的增益只有在超过 TabM 与 RKHS 时才解释为逐标记表示/注意力。 |
| Internal validity | 全部预处理、特征筛选、核带宽、早停、超参数选择、预训练配置选择都在外层训练折内完成；划分文件固定；OOF 预测逐材料检查无重复、无泄漏。 |
| External validity | 随机划分对应群体内未见材料；亲缘簇划分对应更远的遗传背景外推。单数据集限制明确写入结论，不外推到其它物种或育种项目。 |
| Statistical validity | 以材料（或簇）为抽样单位做配对 bootstrap；外层重复只用于稳定性，不把折当独立重复；多重比较按预注册对比族处理；报告效应量与区间。 |
| Reliability / reproducibility | 数据 SHA-256、代码提交、依赖锁、配置文件、随机种子、OOF 预测与 manifest 全部落盘；任何结果可由记录重建。 |
| Implementation fairness | 同一划分、同一指标、同一预处理；每个算法族只保留一套正式实现；优先官方实现；神经网络与核方法使用相同的内层 CV 预算；实现差异写入限制。 |

### Limitations (By Design)

- **单数据集。** 只有 Wheat599；结论限定在该数据与四个环境，不外推到其它小麦群体或物种。
- **样本量与标记类型。** 599 份纯系、1279 个二值 DArT 标记；不能代表高密度 SNP、多倍体或杂交群体。
- **环境相关。** 四个环境产量相关（例如环境 2 与环境 4 的样本相关约 0.66），不能当作四个独立重复；按环境分别报告，不把环境数当独立样本量。
- **无标记坐标。** 不能构造有生物学依据的位置编码；FT-Transformer 不使用位置编码，避免把列序号误当遗传距离。
- **预训练模型许可。** TabPFN 代码 Apache-2.0，3.5 权重另有非商用许可；结果只用于研究，不构成商业部署许可。
- **实现差异。** 不同包的正则化参数化、标准化默认值与数值求解器不同；统一协议只能减小、不能消除这些差异；敏感实现细节写入 manifest。
- **基准不等于育种收益。** PCC/RMSE 的改进不自动转化为育种选择响应或经济收益；本项目不做该推断。

### Ethical Considerations

- 无人类受试者、无动物实验、无个人数据；不涉及 IRB。
- 公开数据与开源实现的使用遵守各自许可；TabPFN 权重许可单独记录。
- 不把基准排名表述为"最佳育种模型"；不对外发布未经复核的中间结果。
- 计算资源与其它项目共享；GPU 使用遵循项目规范，任务结束释放显存。

### Reporting Standard

- Recommended guideline: 本项目是计算基准而非临床/流行病学研究，PRISMA、CONSORT、STROBE 等不适用。采用结构化基准报告：数据来源与 SHA、划分协议、模型与网格、调参预算、指标定义、OOF 与配对区间、实现版本、负结果与限制。参照 DOME（Data, Optimization, Model, Evaluation）四项记录框架；具体文献引用在写作阶段核实后补入。

### Preregistration

- Recommended: Yes（确认性比较、多模型、多环境、多次外层重复，存在选择与多重比较风险）。
- Platform: OSF Registries 或 Git 协议冻结标签（二者择一；优先 Git 标签＋`docs/RESEARCH_PLAN.md` 冻结提交，便于与代码同版本）。
- Status: Planned（在首次正式运行前冻结；任何偏离记录为 amendment）。
- Completed artifact declaration: not_provided
- Companion handle: none
- Sidecar ownership: dispatching layer only; do not populate a digest here
