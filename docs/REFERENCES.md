# References and Verification Status

**更新：** 2026-10-29。只列能核实的文献；未核实项明确标注。引用格式在写作阶段统一为 APA 7。

## 方法与数据

| # | 文献 | 状态 |
|---|------|------|
| 1 | Gorishniy, N., Rubachev, I., Khrulkov, V., & Babenko, A. (2021). Revisiting Deep Learning Models for Tabular Data. *NeurIPS 2021*. arXiv:2106.11959. 代码 `yandex-research/rtdl-revisiting-models`（Apache-2.0）。**论文默认 FT-Transformer 为 3 层、embedding 192、8 头（Appendix E.2 Table 12）；调参空间层数 $[1,4]$、embedding $[64,512]$（Table 13）。本项目的 d=16/32、1–2 层是自主小模型选择，不是官方默认。** | gb 在线核验；本地核验 wheel LICENSE；law 核验默认/调参空间。 |
| 2 | Yu, W., Chen, Y., Liu, B., Huang, E., & Guo, F. (2026). FT-Transformer 预测 STI 风险（Natsal-3）. *BMC Public Health, 26*(1), 1097. PMID 41761163（PMC13049863）。 | gb 在线核验；**只作应用引用**，不是 FT-Transformer 架构出处。 |
| 3 | Gorishniy, N., Kotelnikov, A., & Babenko, A. (2025). TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling. *ICLR 2025*. arXiv:2410.24210. 代码 `yandex-research/tabm`（Apache-2.0）。 | gb 在线核验。 |
| 4 | Qu, J. T. G., Holzmüller, D., Varoquaux, G., & Le Morvan, Z. (2025). TabICL: A Tabular Foundation Model for In-Context Learning on Tables. *ICML 2025*. PMLR v267 (qu25d). 代码 `soda-inria/tabicl`（含 TabICLv2, ICML 2026）。 | gb 在线核验。 |
| 5 | Wu, C., Zhang, Y., Ying, Z., et al. (2024). A transformer-based genomic prediction method fused with knowledge-guided module (GPformer). *Briefings in Bioinformatics, 25*(1), bbad438. PMC10701102. | gb 在线核验；**无公开代码**（Data Availability：“reasonable request”）。与 GPTransformer（大麦, Front Plant Sci 2021）、Cropformer 区分，引用写全称。 |
| 6 | Li, J., Yu, L., Li, M., et al. (2026). Leveraging weighted embedding and Transformer architecture (GP-WAITER). *Nature Communications*. DOI 10.1038/s41467-026-71035-5. 代码 `snowo-w/GP-WAITER`。 | gb 在线核验；核心 `.so`；`train-test.py` L64/82 `reshape(180,4589)`、L212–218 `test_loss` 早停；正文每训练子集重算加权。 |
| 7 | Fan, Y., & Waldmann, P. (2024). Tabular deep learning: a comparative study applied to multi-task genome-wide prediction. *BMC Bioinformatics, 25*, 322. PMC11452967. 代码 `angelYHF/Tabular-deep-learning-for-GWP`。 | gb 在线核验论文与数字；law 逐文件核验五项缺陷 + 两项新发现（vendored lassonet `score()` 正 MSE + `argmax`；LightGBM 测试集早停/调参），证据见 `ars-output/supervision_factcheck_20261029.md`。LassoNet 0.691 / LightGBM 0.657 / FT-Transformer 0.602 是合并多任务 $r$，不是四环境各自精度。不得写“评审者指出全样本标准化”（无公开出处）。 |
| 8 | Yang, T., Wu, W., Xue, Y., Zhou, L., Kang, H., & Liu, J. (2026). Gsformer. *Genetics Selection Evolution*. DOI 10.1186/s12711-026-01055-8（PMC13455463）。代码 `hajudien/GSformer`。 | gb 在线核验；五折外层/三折内层；SVR 最好于 GY1/GY3/GY4，GBLUP 最好于 GY2；目标为 DNNGP 预处理 EBV；输入为 SNP/PC，**不含谱系**。**GY1–GY4 与本仓列名 `"1","2","4","5"` 的映射未核**；跨论文引用其排名时不假设对应。 |
| 9 | Prior Labs (2026). *TabPFN-3.5 technical report*. arXiv:2609.17895. 权重 `Prior-Labs/tabpfn_3_5`（`tabpfn-v3.5-20260909.safetensors`，876MB，许可 `tabpfn-3-5-license-v1.0`）；代码 `tabpfn==9.2.0`（Apache-2.0）。 | 本地核验 checkpoint 元数据（8 成员/balanced/768/无全局投影）与许可；gb 在线核验技术报告 §3.2 与 <1000 行微调警告原文。 |
| 10 | 数据出处：CIMMYT Global Wheat Program；BGLR-R 随包文档。`gdlc/BGLR-R`@`7e92aa2d`，`data/wheat.RData`，SHA-256 见 `DATA_FACTS.md`。 | 本地核验；man/wheat.Rd 载明 MAF≥0.05 过滤声明与 Bernoulli 插补；“实测 96 列 MAF<0.05”与文档差异并列记录。 |
| 11 | LassoNet. Lemhadri, I., et al. (2021). LassoNet: A Neural Network with Feature Sparsity. *JMLR, 22*(20-848). 代码 `lasso-net/lassonet`（MIT）。 | 待核验（扩展比较使用；正式引用前核对）。 |

## 代码与权重许可

| 项 | 许可 / 状态 |
|----|-------------|
| `PriorLabs/TabPFN` | Apache-2.0（本地核验） |
| `Prior-Labs/tabpfn_3_5` 权重 | `tabpfn-3-5-license-v1.0`，非商用；允许研究/内部基准，禁止商业决策、客户交付、付费产品；全文在 `docs/licenses/` |
| `yandex-research/rtdl-revisiting-models` | Apache-2.0（仓库 LICENSE 与 wheel `dist-info` 均核验） |
| `yandex-research/tabm` | Apache-2.0（GitHub API） |
| `lasso-net/lassonet` | MIT（GitHub API，扩展比较） |
| BGLR | GPL-3 |

## 独立监管核验（law, 2026-10-29）

- 报告：`ars-output/supervision_factcheck_20261029.md`（SHA-256 `62697f9f187fbb0b20cd15fee09be9539459df8b32812917c6f554aac1475208`，11435 字节）。
- 升级：Fan & Waldmann 代码缺陷从“待复核”升为“逐文件已核验”；新增 vendored lassonet 的 `score()` 正 MSE + `argmax` 与 LightGBM 测试集早停两项。
- 状态更新（law/gb 核验）：`tabpfn==9.2.0` 在 PyPI 存在（2026-10-12，trusted-publishing provenance 齐全）；arXiv:2609.17895 与 TabPFN-3.5 技术报告的对应关系由 PyPI 官方 BibTeX 闭合（Jäger et al., 2026）。此前“arXiv 未核实”条目可升级。
- rtdl 仓库名：arXiv 页脚旧链接 `yandex-research/tabular-dl-revisiting-models` 已 404；现行官方仓库为 `yandex-research/rtdl-revisiting-models`，引用 URL 用后者。
- 监管章程（冻结先于运行、网格即预算、再来一轮须命名决策、M1/M2 无稳定差异即负结论）已写入 `docs/REVIEW_PROTOCOL.md` 与 `AGENTS.md`。

## GP 基础与方法学文献（law 在线核验，写作引用前复核）

| 主题 | 文献 |
|------|------|
| 基因组预测基础 | Meuwissen, T. H. E., Hayes, B. J., & Goddard, M. E. (2001). Prediction of total genetic value using genome-wide dense marker maps. *Genetics, 157*(4), 1819–1829. |
| GBLUP / 关系矩阵 | VanRaden, P. M. (2008). Efficient methods to compute genomic predictions. *Journal of Dairy Science, 91*(11), 4414–4423. |
| 小麦/植物 GP | Crossa, J., et al. (2010). Prediction of genetic values of quantitative traits in plant breeding using pedigree and molecular markers. *Genetics, 186*(2), 713–724. |
| 全基因组回归 | de los Campos, G., Hickey, J. M., Pong-Wong, R., Daetwyler, H. D., & Calus, M. P. L. (2013). Whole-genome regression and prediction methods applied to plant and animal breeding. *Genetics, 193*(2), 327–345. |
| GP 评估规范 | Crossa, J., et al. (2017). Genomic selection in plant breeding: methods, models, and perspectives. *Trends in Plant Science, 22*(11), 961–975. |
| 选择偏差 / 折内完成 | Cawley, G. C., & Talbot, N. L. C. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. *JMLR, 11*, 2079–2107. |
| 小样本验证 | Vabalas, A., Gowen, E., Poliakoff, E., & Casson, A. J. (2019). Machine learning algorithm validation with a limited sample size. *PLOS ONE, 14*(11), e0224365. |
| CV 方差不可无偏估计 | Bengio, Y., & Grandvalet, Y. (2004). No unbiased estimator of the variance of K-fold cross-validation. *JMLR, 5*, 1089–1105. |
| 泛化误差推断 | Nadeau, C., & Bengio, Y. (2003). Inference for the generalization error. *Machine Learning, 52*(3), 239–281. |
| 配对检验 | Dietterich, T. G. (1998). Approximate statistical tests for comparing supervised classification learning algorithms. *Neural Computation, 10*(7), 1895–1923. |
| 多模型比较 | Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets. *JMLR, 7*, 1–30. |
| 贝叶斯配对比较 | Benavoli, A., Corani, G., Demšar, J., & Zaffalon, M. (2017). Time for a change: a tutorial for comparing multiple classifiers through Bayesian analysis. *JMLR, 18*(77), 1–36. |
| 重复 CV 检验 | Bouckaert, R. R., & Frank, E. (2004). Evaluating the replicability of significance tests for comparing learning algorithms. *LNCS, 3056*, 3–12. |
| 反方视角 | Wainer, J., & Cawley, G. (2021). Nested cross-validation when selecting classifiers is overzealous for most practical applications. *Expert Systems with Applications, 171*, 115222. 该文支撑“嵌套 CV 给无偏泛化估计”，不支撑“嵌套 CV 总是必要”；本项目超参网格大，嵌套属必要侧。 |

## 不引用 / 易错项

- `PMC13049863` 是 FT-Transformer 用于性传播感染风险预测的应用论文（Natsal-3），不是 FT-Transformer 原始论文；只作应用引用。
- TabICL 是 ICML 2025，不是 ICLR；TabICLv2 才是 ICML 2026。
- TabM 作者为 Gorishniy, Kotelnikov, Babenko；不要写成 Akhanov 等。
- GPformer 与 GPTransformer（大麦 2021）、Cropformer 是不同工作；引用写全称。
- Gsformer 的 GY1–GY4 与本仓列名映射未核；不要假设其四任务顺序与 `"1","2","4","5"` 相同。
- Fan & Waldmann 公开代码的四项问题（全样本标准化顺序、90/10 与正文不符、负 MSE 回传、`score()` 与 CV argmax 方向）目前是简报审查结论；`Regression.zip` 尚未逐项复核。冻结文本标“待复核”，不写成已证实事实。
- “评审者曾指出全样本标准化”没有找到公开出处；不写入论文或仓库文本，除非给出可引用来源。
