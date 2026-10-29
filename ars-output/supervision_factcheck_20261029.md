# 独立监管核查报告（law）

**日期：** 2026-10-29　**角色：** 独立第三视角监管（用户指定，防止无限测试循环）
**范围：** Wheat599 模型基座比较项目 G0 冻结前的协议与承重事实。本轮只读：未改任何仓库文件、未训练、未下载权重。
**方法：** 对研讨简报与 `docs/` 现行文本的每条承重声明做独立来源核验（官方文档、论文全文、代码仓库逐文件阅读），不采用团队既有结论作为依据。

## 1. 已独立核实（与团队文档一致，可作基石）

### 1.1 BGLR wheat 数据事实

- 官方文档（BGLR 参考手册 wheat 页）与 Pérez & de los Campos (2014)（*Genetics*, 198(2), 483–495，https://doi.org/10.1534/genetics.114.164442 ）：599 个 CIMMYT 品系、DArT 标记编辑后 1279 个、presence/absence 0/1 编码、缺失按 Bernoulli(p̂) 插补、Y 为 4 环境平均产量并按环境标准化、A 为系谱导出的 numerator relationship matrix、无标记坐标表。
- 本次对官方数据文件直接加载实测：`wheat.X` 599×1279、取值严格 {0,1}、无缺失、无常数列；`wheat.Y` 599×4、列名 `"1"`、`"2"`、`"4"`、`"5"`、各列均值 0 标准差 1；`wheat.A` 599×599 对称；文件内对象仅 wheat.A / wheat.sets / wheat.X / wheat.Y。与 `docs/DATA_FACTS.md` 一致。
- "用 A 的模型比只用 X 多用了信息"：A（系谱）与 G/X（标记）是两个信息源；Pérez & de los Campos (2014) 方差分解 σa²=0.24 对 σg²=0.42，PM 对 P 的配对检验各环境 p<0.001，可作引用依据。

### 1.2 TabPFN-3.5

PriorLabs/TabPFN 官方仓库与文档核实：checkpoint 文件名 `tabpfn-v3.5-20260909.safetensors`；代码 Apache-2.0、权重单独许可（TABPFN-3-5 License v1.0，非商用）；默认 8 成员集成、每成员最多 768 列、`FEATURE_SUBSAMPLING_METHOD='balanced'`、无全局投影；官方 fine-tuning 文档明确"<1000 行微调过拟合风险常超过收益，应先以默认模型建立 ICL 基线"。与 `docs/LICENSE_NOTES.md`、AGENTS.md 硬边界一致。

注意表述：官方文档未出现"冻结权重（freeze）"措辞，其原意是"小数据不微调、以默认 ICL 为基线"。对外引用建议用原意。

### 1.3 Gsformer 论文

Yang, T., Wu, W., Xue, Y., Zhou, L., Kang, H., & Liu, J. (2026). Gsformer: A dual-architecture deep learning framework with CNN-self-attention and sparse-attention for genomic selection. *Genetics Selection Evolution*, 58(1). https://doi.org/10.1186/s12711-026-01055-8 核实：五折外层×三折内层嵌套 CV（内层网格搜索、外层无偏估计）；小麦 599 品种×4 环境、1279 标记、251 个 PC；SVR 在 GY1/GY3/GY4 最好、GBLUP 在 GY2 最好；目标为 DNNGP 预处理 EBV。与 REFERENCES.md 一致。架构是 CNN-self-attention + native sparse attention，**无图组件**（团队称 Gsformer-CSA 正确）；论文未用 "Wheat599" 名称；GY1–GY4 与本仓列名映射未核（团队已标注）。

## 2. 冻结前必须修正（协议文本，非 amendment）

### 2.1 FT-Transformer 小模型配置的出处错误

`docs/RESEARCH_PLAN.md` §3 表格与 §5 网格写"`rtdl-revisiting-models` 官方包；d=16/32、1–2 层"，该归属不成立：

- Gorishniy, Y., Rubachev, I., Khrulkov, V., & Babenko, A. (2021). Revisiting deep learning models for tabular data. *NeurIPS 34*. arXiv:2106.11959 的 FT-Transformer **默认配置为 3 层、embedding 192、8 头（约 929K 参数）**（附录 E.2 Table 12）；调参空间为层数 UniformInt[1,4]（A 类数据集）/ [1,6]（B 类）、embedding 大小 UniformInt[64,512]（Table 13）。论文与仓库 README 均无 d=16/32 或 1–2 层的推荐。
- d∈{16,32} 低于论文调参空间下限（64）。若团队基于 n≈480 的小样本 regime 有意选小模型，应在协议中写成**自主选择并给出理由**（如参数量与训练材料量级、过拟合防范），同时正确引用论文的实际默认值与调参空间；不能写成官方包/论文默认。

此项是预注册网格与出处的协议文本问题，须在 G0 冻结标签前修正；冻结后按 amendment 流程处理会违反"amendment 不得修改主对比/网格"的约定。

### 2.2 （附带）二值 embedding 等价表述

t_j(0)=b_j、t_j(1)=b_j+w_j 是论文 §3.3 公式（类别特征 T_j^(cat)=b_j+e_j^T W_j）在 {0,1} 输入下的直接推论，可写，但引用时注明为公式推论而非论文逐字讨论。

## 3. 已补核：Fan & Waldmann 代码缺陷（可升级 REFERENCES.md 状态）

Fan, Y., & Waldmann, P. (2024). Tabular deep learning: a comparative study applied to multi-task genome-wide prediction. *BMC Bioinformatics*, 25(1), 322. https://doi.org/10.1186/s12859-024-05940-1 （代码：https://github.com/angelYHF/Tabular-deep-learning-for-GWP ）。本次对仓库源码逐文件阅读，简报五项缺陷全部属实，另有两项新发现：

| # | 缺陷 | 证据（文件/行级） |
|---|------|------|
| a | 全样本标准化先于划分 | `LassoNet/data.py` `load_wheat_csv_ds()` 先 `in_place_normalize(features)` 后 `train_test_indices()`；且自带 `wheat-hitopt.csv` 的 4 个性状列本身已标准化（mean≈0, sd≈1） |
| b | 划分比例不符 | 论文 70/10/20；`LassoNet/entry.py` 实调 `testing_set_ratio=0.1`（90/10、无独立验证集）；LightGBM/NODE/TabNet notebook 用 `train_test_split(test_size=0.2)`（80/20） |
| c | 测试集参与选型 | `LassoNet/model.py` `cv_eval_model()` 返回 `-testing_mse`；`LassoNet/bo.py` 以 bayes_opt `maximize` 该目标；与论文"在训练折验证集上调参"的声明矛盾 |
| d | score() 与 argmax 方向相反 | vendored `lassonet/interfaces.py` 为 `LassoNetRegressor` 新增 `score()` 返回正 MSE，而 `BaseLassoNetCV.path()` 仍用 `np.nanargmax` → **路径内 lambda 按最差 MSE 选择**；上游 lasso-net/lassonet 各版本均无此 score() 覆盖（继承 R²，与 argmax 一致） |
| e | FT-Transformer 脚本无小麦适配 | `FT-T.py` 为 torch_frame 通用示例（Yandex 数据集、`args.dataset`），全文无 wheat/GY/599；`TabT.py` 同 |

新发现：

1. 缺陷 d 的后果比"选型泄漏"更重：LassoNetRegressorCV 的正则路径内部 lambda 选择也被反转。
2. 缺陷 c 不限于 LassoNet：`LightGBM.ipynb` 同样以测试集为 `eval_set` 早停、以测试集 l2 为 hyperopt 目标。

另：Table 1 的 0.691 / 0.657 / 0.602 是 Pearson r（对应 MSE 0.149 / 0.160 / 0.206）。团队 REFERENCES.md 已正确标注为"合并多任务 r"，可保留；上述代码项可从"待 M2 前解 ZIP 对照"升级为已逐项核验（本报告即出处）。

## 4. 待本地核验项（不阻断冻结，列入 G1）

1. `tabpfn==9.2.0` 版本存在性与 `pip download` 可达性；`arXiv:2609.17895`（TabPFN-3.5 技术报告）可打开且与 checkpoint 对应。本次未能独立打开该报告页面（验证墙拦截），状态记**未核实**。
2. CRAN 正式版 BGLR 的 `wheat.RData` 与固定提交 `7e92aa2d` 的 SHA-256 一致性（本次实测基于 GitHub 数据文件；`scripts/fetch_data.py` 重跑即覆盖，G1 已有该项）。
3. rtdl 包默认架构参数（`d_token`/`n_blocks`）本地读数，与论文 Table 12/13 对照，作为 §2.1 修正的落实证据。

## 5. 方法论支撑文献（供协议与报告引用，APA 7）

| 要点 | 文献 |
|------|------|
| 选择偏差与折内完成所有 data-dependent 步骤 | Cawley, G. C., & Talbot, N. L. C. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. *JMLR, 11*, 2079–2107. https://jmlr.org/papers/v11/cawley10a.html ；Vabalas, A., et al. (2019). Machine learning algorithm validation with a limited sample size. *PLOS ONE, 14*(11), e0224365. https://doi.org/10.1371/journal.pone.0224365 |
| CV 折不独立、方差无无偏估计 | Bengio, Y., & Grandvalet, Y. (2004). No unbiased estimator of the variance of K-fold cross-validation. *JMLR, 5*, 1089–1105. https://www.jmlr.org/papers/volume5/grandvalet04a/grandvalet04a.pdf ；Nadeau, C., & Bengio, Y. (2003). Inference for the generalization error. *Machine Learning, 52*(3), 239–281. https://doi.org/10.1023/a:1024068626366 |
| 配对检验与效应量报告 | Dietterich, T. G. (1998). *Neural Computation, 10*(7), 1895–1923. https://doi.org/10.1162/089976698300017197 ；Demšar, J. (2006). *JMLR, 7*, 1–30. https://www.jmlr.org/papers/v7/demsar06a.html ；Benavoli, A., et al. (2017). *JMLR, 18*(77), 1–36. https://www.jmlr.org/papers/v18/16-305.html ；Bouckaert, R. R., & Frank, E. (2004). *LNCS, 3056*, 3–12. https://doi.org/10.1007/978-3-540-24775-3_3 |
| 反方视角（引用时注意立场） | Wainer, J., & Cawley, G. (2021). Nested cross-validation when selecting classifiers is overzealous for most practical applications. *Expert Systems with Applications, 171*, 115222. https://doi.org/10.1016/j.eswa.2021.1152220 —— 它支撑"嵌套 CV 给无偏泛化估计"，不支撑"嵌套 CV 总是必要"；本项目超参网格大，嵌套属必要侧 |
| 基因组预测评估规范 | de los Campos, G., et al. (2013). *Genetics, 193*(2), 327–345. https://doi.org/10.1534/genetics.112.143313 ；Crossa, J., et al. (2017). *Trends in Plant Science, 22*(11), 961–975. https://doi.org/10.1016/j.tplants.2017.08.001 |

未找到逐项专门文献的点：核带宽、早停、集成权重、预训练配置"必须折内完成"无单篇专文，按公认实践表述并以 Vabalas et al. (2019) 的原则性结论概括即可。

## 6. 对无限测试循环的监管结论

旧仓库 RC-CKRR_semisynth 的 C1 线在 `eb09258` 终止；其循环特征是：无预先注册的停止规则、主端点摇摆（MSE_y 与 MSE_m 混用）、τ=0 退化被计为 win、单 seed 排名当机制证据、反复追加扫描而非决策。新项目 G0–G4 门、预注册网格、fold 级 15 值 t 区间（df=14）、Holm 对比族、负结果保留条款，已针对上述每一条设访。监管重点因此不是"再加约束"，而是**保证已写下的纪律在冻结与运行两刻都生效**：

1. 冻结先于运行：M1 首次正式运行前完成 G0 冻结标签（用户授权提交后）；SMOKE_ONLY 不得进 `docs/RESULTS.md`。
2. 网格即预算：冻结网格是计算预算上限；网格扩张= amendment，须书面回答"改变哪个已注册判断"。
3. "再来一轮"必须命名决策：任何新增运行请求须说明其结果会改变的已注册判断，否则不执行。
4. 停止规则：M1/M2 若无稳定差异，结论即负结果（判定标准 #6）；M4 仅在 M1/M2 支持时立项；不设第三轮架构搜索。
5. 主端点不可团队内部吸收：冻结后涉及主端点/主对比/判据变更的需求上交用户决定。
6. 本监管的审查门：冻结前协议文本、G1/G2 证据抽查、最终声明与证据一致性；审查强度与风险相称，不把每步变成审批。
7. 消息节奏：门边界与实质变化才通报，不做逐运行通知。

## 限制

- 本报告核验截至 2026-10-29；TabPFN 技术报告页面未能打开（验证墙），相关条目记未核实。
- 代码仓库核验基于 GitHub 当前 master 快照；若上游变动，以变动后复核为准。
- 本报告是事实与出处核验，不构成对团队科学判断的批准；科学结论由 G0–G4 门与用户目标共同决定。
