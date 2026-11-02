**状态：已完成；本配置 NO-GO，结果见 [RESULTS.md](RESULTS.md)。**

**实施确定的截距约定：** 历史 ICM 使用训练集标准化和 GLS 无惩罚截距，本实验按第 4.2 节的 GLS 投影公式进行条件 LOO，并保留该 ICM 预测约定。

**版本：2026-11-02**

**研究对象：** BGLR Wheat599；marker-only；四环境 CV1 预测

**主要基线：** 普通 ICM（折均宏平均 PCC **0.52019**，RMSE **0.85449**，MAE **0.65784**）
**候选名称：** BH-LMC——条件 LOO 校准的双核、环境特异正则化线性共区域化模型

## 1. 科学问题与本轮边界

研究问题：在未见材料的四环境预测（CV1）中，**让加性核与 RBF 核具有不同的环境共享结构，并使用训练集内部的预测可靠性调节核比例和正则化，是否能在现有普通 ICM 之上提供有效增量？**

完整候选采用成熟的 LMC 框架：

\[
\Omega=B_{\mathrm{add}}\otimes K_{\mathrm{add}}+B_{\mathrm{rbf}}\otimes K_{\mathrm{rbf}}+\Lambda\otimes I_n.
\]

其中，**双核 LMC、环境差异化正则化、解析 LOO 均有先例**；本轮检验的是它们在一套低自由度、可退化到 ICM 的具体校准规则下，能否改善 Wheat599 的预测。当前没有证据证明普通 ICM 的可分离结构是性能瓶颈，也不能预先把 RBF 成分解释为真实上位性。

只开展 **一个完整候选**；不增加数据集、神经网络、核函数、事后消融或临时扩张搜索网格。

## 2. 数据、划分与固定参照

| 项目 | 实验规范 |
|---|---|
| 输入 | `wheat.X`，599×1279；DArT 0/1 存在／缺失标记，保留固定列身份 |
| 预测目标 | `wheat.Y`，599×4，环境列按 `"1", "2", "4", "5"` 排列 |
| 排除信息 | 不使用谱系矩阵 `wheat.A`；不引入其他环境或外部数据 |
| 数据文件 | 项目已核验 `wheat.RData`，SHA-256 `8710523389007dd8b7c88a5deb74ee16aea0bae158496bc11f0626fb43b5ea21` |
| 数据加载 | 复用现有修复版加载器；R 一维列主序数据恢复使用 `reshape(shape, order='F')` |
| 外层测试 | 复用既有 `splits_manifest.json`：10 折；9 折各 60、1 折 59 个测试材料；环境共享同一材料划分 |
| 内层选择 | 复用 ICM 的外层开发集内三折索引；各内层训练约 359–360、验证约 179–180 个材料 |
| 最终拟合 | 每个外层折使用约 539–540 份开发材料重新拟合；测试材料的四环境标签完全不可见 |

所有 **X 均值、Y 均值、加性核归一化常数、Hamming 距离中位数、核参数、条件 LOO 统计量及环境相关矩阵** 均从当前拟合子集估计。内层验证只用于候选评分；外层测试标签只用于最终指标。

本方案延续已有 ICM 的**按当前训练材料求算术均值并中心化表型，再拟合核预测残差**的计算约定。它是 *plug-in mean*（训练均值代入）截距，不应误称为与核系数同时优化的 GLS 无惩罚截距。实施时必须与原 ICM 的实际中心化和预测反变换完全一致；否则不能报告“精确退化到原 ICM”。

## 3. 两个固定基础核

记当前拟合子集有 \(n\) 份材料，原始标记矩阵 \(X\in\{0,1\}^{n\times p}\)，\(p=1279\)；表型 \(Y\in\mathbb R^{n\times4}\)。令

\[
\mu_X=\frac1n\mathbf1^\top X,\quad Z=X-\mathbf1\mu_X,\quad
\mu_Y=\frac1n\mathbf1^\top Y,\quad Y_c=Y-\mathbf1\mu_Y.
\]

**加性核：**

\[
 c_A=\frac1n\operatorname{tr}(ZZ^\top),\qquad
 K_A=\frac{ZZ^\top}{c_A}.
\]

测试交叉核使用相同 \(\mu_X,c_A\)：\(K_{A,*}=Z_*Z^\top/c_A\)。要求 \(c_A>0\)。

**Hamming RBF 核：**始终使用**原始二值标记**计算距离，

\[
 D_{ij}=\sum_{k=1}^{p}|X_{ik}-X_{jk}|=\lVert X_i-X_j\rVert_2^2,\quad
 d_0=\operatorname{median}_{i<j}D_{ij},
\]
\[
 K_R(h)_{ij}=\exp\!\left(-hD_{ij}/d_0\right),\qquad h\in\{0.25,1,4\}.
\]

测试交叉距离与 RBF 核使用同一 \(d_0\)，要求 \(d_0>0\)。两个训练核均有平均对角线 1；**这只是核尺度规范化，不保证预测校准或遗传方差分解正确。**

## 4. 条件 LOO：明确截距与适用范围

### 4.1 固定基准参数

在**当前拟合子集内部**，按已有普通 ICM 的基础核选择流程确定 \((h,w_0,\lambda_0)\)，其中 \(\lambda_0>0\)，\(w_0\in[0,1]\)。定义

\[
K_0=w_0K_A+(1-w_0)K_R(h).
\]

以下分别对 \(K_m\in\{K_A,K_R(h),K_0\}\) 计算条件 LOO；三者使用同一个 \(\lambda_0\)，不额外为每个核调参。

### 4.2 正确的闭式公式

设 \(C_m=K_m+\lambda_0I_n\)，\(R_m=C_m^{-1}\)（实现中以线性求解取得，不必显式求逆）。令

\[
 a_m=R_mY_c,\qquad v_m=R_m\mathbf1,\qquad d_{m,i}=(R_m)_{ii}.
\]

如果**留出第 \(i\) 份材料时重新计算训练标签的算术均值**，而核矩阵及 \(\lambda_0\) 保持固定，则四环境 LOO 残差为

\[
\boxed{
 e_{m,i,t}^{\mathrm{cLOO}}=
 \frac{(a_m)_{it}+\dfrac{Y_{it}-\mu_{Y,t}}{n-1}(v_m)_i}{d_{m,i}}
 }
\]

并定义对应 LOO 预测

\[
\hat G_{m,it}^{\mathrm{cLOO}}=Y_{it}-e_{m,i,t}^{\mathrm{cLOO}}.
\]

**不能**对全训练集均值中心化之后，直接套用 \((Y_c-H_mY_c)_i/(1-(H_m)_{ii})\) 并称之为“重估截距的精确 LOO”；该式遗漏了留一均值变化。

公式对**固定的** \(K_m\)、\(\lambda_0\) 及核交叉项精确；它**不会**重新估计留一后的标记均值、核归一化、中位数带宽、其他超参数或所有数据驱动变换。因此必须称为 *conditional LOO*，而不是整个训练管线完全无偏的 LOO，更不能称为独立测试。

> 如果仓库中的 ICM 实际采用联合优化的 GLS 截距而非训练均值 plug-in，本节公式不能混用。必须先统一实现约定，并使用对应的 GLS LOO：\(P=R-R\mathbf1(\mathbf1^\top R\mathbf1)^{-1}\mathbf1^\top R\)，\(e_i=(PY)_i/P_{ii}\)。该情况不应作为运行后临时选择的配置。

## 5. 训练集内部的校准统计量

下述全部统计量仅对当前训练子集计算；\(\gamma=0\) 时应直接返回基准 ICM 路径，以保证退化关系不受任何 LOO 数值退化影响。

### 5.1 环境特异核混合权重

对于 \(m\in\{A,R\}\) 和环境 \(t\)，令

\[
 r_{m,t}=\max\bigl(0,\operatorname{Corr}(Y_{\cdot,t},\hat G^{\mathrm{cLOO}}_{m,\cdot,t})\bigr).
\]

定义

\[
\pi_t=\frac{r_{A,t}^2+10^{-4}}{r_{A,t}^2+r_{R,t}^2+2\times10^{-4}},\qquad
 w_t(\gamma)=(1-\gamma)w_0+\gamma\pi_t.
\]

\(w_t\in[0,1]\)；它是模型的**经验核混合比例**，不是加性遗传方差占比。

### 5.2 预测可靠性驱动的正则化

对基准核 \(K_0\) 的条件 LOO 残差，定义环境 \(t\) 的截断预测分数

\[
 q_t=\operatorname{clip}\left(1-\frac{\sum_i(e^{\mathrm{cLOO}}_{0,i,t})^2}{\sum_i(Y_{it}-\mu_{Y,t})^2},\;0.10,\;0.50\right).
\]

**\(q_t\) 是截断的条件 LOO 预测 \(R^2\)，不是遗传力。** 令

\[
\nu_t=\frac{1-q_t}{q_t},\quad
\bar\nu=\left(\prod_{t=1}^4\nu_t\right)^{1/4},\quad
\lambda_t(\gamma)=\lambda_0\left(\frac{\nu_t}{\bar\nu}\right)^\gamma.
\]

该规则保证 \(\lambda_t>0\) 且 \((\prod_t\lambda_t)^{1/4}=\lambda_0\)。\(\lambda_t\) 是**经验环境特异正则化强度**；可作为工作协方差中的对角噪声项使用，不能声称已估计真实残差方差。

### 5.3 两个任务相关靶标

令 \(C_Y=\operatorname{Corr}(Y)\)，以及 \(C_A,C_R\) 分别为四环境的条件 LOO 预测矩阵相关系数矩阵。定义

\[
\tilde C_A=(1-0.5\gamma)C_Y+0.5\gamma C_A,\quad
\tilde C_R=(1-0.5\gamma)C_Y+0.5\gamma C_R.
\]

所有相关矩阵须对称、半正定、对角为 1。若某环境的某组 LOO 预测近常数（标准差低于 \(10^{-12}\) 倍当前标签标准差），相关系数无定义：将该环境在对应 \(C_A\) 或 \(C_R\) 中的**非对角相关置 0、对角置 1**；其 \(r_{m,t}\) 置 0。该规则保持半正定，不删除材料、不使用验证标签修复。

## 6. 双核协方差与预测

设 \(D_A=\operatorname{diag}(\sqrt{w_t})\)、\(D_R=\operatorname{diag}(\sqrt{1-w_t})\)，并定义

\[
B_A=D_A[(1-\eta_A)I_4+\eta_A\tilde C_A]D_A,
\]
\[
B_R=D_R[(1-\eta_R)I_4+\eta_R\tilde C_R]D_R,
\]
\[
\Lambda=\operatorname{diag}(\lambda_1,\ldots,\lambda_4),
\qquad \Omega=B_A\otimes K_A+B_R\otimes K_R+\Lambda\otimes I_n.
\]

对四环境表型使用**列主序** \(\operatorname{vec}_F(Y_c)\)，求解

\[
\Omega\,\operatorname{vec}_F(\mathcal A)=\operatorname{vec}_F(Y_c).
\]

测试预测为

\[
\boxed{\hat Y_*=\mathbf1_{n_*}\mu_Y+K_{A,*}\mathcal A B_A^\top+K_{R,*}\mathcal A B_R^\top.}
\]

\(B_A,B_R\) 半正定，\(\Lambda\succ0\)，因此 \(\Omega\succ0\)，应使用 `float64` Cholesky 求解并记录异常；**不能通过过滤 NaN 或隐式更换参数掩盖失败。**

由于两核的平均训练对角线均为 1，\(\operatorname{diag}(B_A+B_R)=\mathbf1_4\)，每个环境的组合信号核保持平均对角尺度 1；**这不保证 RMSE、预测方差或真实遗传方差守恒。** 当某个 \(w_t\) 为 0/1 时，分量核可能奇异；只需求解完整的正定 \(\Omega\)，不计算 \(B_A^{-1}\) 或 \(B_R^{-1}\)。

### 与 ICM 的精确关系

\[
\gamma=0,\quad \eta_A=\eta_R=\eta
\quad\Longrightarrow\quad
\Omega=\bigl[(1-\eta)I_4+\eta C_Y\bigr]\otimes K_0+\lambda_0I_{4n}.
\]

这**严格恢复采用相同 \((h,w_0,\lambda_0,\eta)\)、中心化与截距约定的普通 ICM**。

若 \(\eta_A=\eta_R=0\)，四环境计算解耦为独立核岭，但共享基础 \(h\) 与 \(\lambda_0\)；**并不等同于四环境各自独立调参的历史单环境基线**。

## 7. 固定实验与超参数选择

### 7.1 训练与选模流程

对每个外层折：

1. **内部三折：** 对每个内部训练子集，按当前普通 ICM 的既有规则独立选定 \((h,w_0,\lambda_0)\)，计算该子集的基础核及条件 LOO 校准；不得使用其验证标签拟合这些参数。
2. **完整候选选择：** 在固定的 **48 组**配置上评价内层验证：
   \[
   \eta_A,\eta_R\in\{0,0.3,0.6,0.8\},\qquad\gamma\in\{0,0.5,1\}.
   \]
   每组的分数为 **3 个内层折 × 4 个环境**的 PCC 等权平均。使用相同内层分割与相同基础核选择规则。
3. **确定性平局规则：** 若宏平均 PCC 差的绝对值 \(\le10^{-12}\)，依次选择较小 \(\gamma\)、较小 \(|\eta_A-\eta_R|\)、较小 \(\eta_A+\eta_R\)，最后按 \((\eta_A,\eta_R,\gamma)\) 字典序。验证指标非有限则该候选无效，不能跳过失败折继续计算均值。
4. **完整开发集重拟合：** 在外层开发集重新选择基础核参数、重算全部训练统计量和条件 LOO，固定第 2 步的 \((\eta_A,\eta_R,\gamma)\) 拟合一次；随后只对外层测试材料输出一次四环境预测。

### 7.2 一个必要的公平比较：共享强度范围

原 ICM 已有结果在 \(\eta\in\{0,0.3,0.6\}\) 中 **10/10 折选中 0.6**；新候选进一步允许 0.8。因此，如果新方法提高 PCC，可能**只来自更大的 ICM 共享强度**，而不是双核或校准机制。

为消除这一混淆，不新增独立架构实验，而是直接取第 2 步网格中

\[
\gamma=0,\quad\eta_A=\eta_R=\eta\in\{0,0.3,0.6,0.8\}
\]

的 **ICM 子集**，使用完全相同的内层评分规则选择 \(\eta\)，然后在完整开发集拟合，形成 **匹配搜索范围 ICM**。该子集可重用同一计算过程；它与历史 ICM（PCC 0.52019）应**分别报告**。

这样可以区分“扩大共享强度搜索”与“BH-LMC 新增结构”的预测增量，不需要补做其他消融。

### 7.3 固定参照

| 方法 | 已有折均宏平均 PCC | 本轮用途 |
|---|---:|---|
| 普通 ICM（原搜索范围） | **0.52019** | 固定历史主要基线 |
| 单环境加性＋RBF | 0.50923 | 现有单环境参照 |
| MoEGP | 0.4592 | 已完成的深度学习参照，不重训 |
| **匹配范围 ICM** | 待测 | 消除 \(\eta=0.8\) 扩围造成的解释混淆 |
| **BH-LMC** | 待测 | 本轮唯一新架构 |

## 8. 运行前的最少正确性验收

只进行直接决定数值正确性的检查，不开展额外性能消融：

1. **条件 LOO：** 在小规模合成核矩阵上，以固定核和 \(\lambda\) 逐一删除训练标签、重新计算剩余标签均值并拟合；与第 4 节公式比较，`float64` 最大误差目标 \(<10^{-10}\)。不必对 Wheat599 逐个样本重训。
2. **ICM 退化：** 在**开发集或合成测试查询**上固定同一基础参数和 \(\eta\)，比较 \(\gamma=0,\eta_A=\eta_R\) 与现有 ICM 预测，误差目标 \(<10^{-10}\)。须检查核缩放、训练均值、向量化顺序及交叉核；不提前查看外层测试标签。
3. **正定与有限性：** 断言 \(\operatorname{diag}(B_A+B_R)\approx\mathbf1\)、两核及任务矩阵半正定、\(\Lambda>0\)、\(\Omega\) 可 Cholesky 分解。覆盖 \(w_0=0/1\)、相关矩阵退化及正常输入；失败必须记录、不可静默跳过。

求解复杂度为 \(O((4n)^3)\)，矩阵存储为 \(O((4n)^2)\)。当 \(n=540\) 时，\(2160\times2160\) 的 `float64` 原始矩阵约占 **37.3 MB**；分解与其他中间量需额外内存。**不预设训练时间或声称已经完成 Wheat599 运行。**

## 9. 结果报告与决策

**主指标：** 环境内每折 PCC → 十折均值 → 四环境宏平均。另报告十折样本 SD（`ddof=1`）、RMSE、MAE、每环境增量、40 个同折配对差及改善折数。每个环境拼接 599 条 OOF 后重新计算的指标另列，**不得与折均 PCC 混用**。

保存：所有外层 fold 的选中 \((h,w_0,\lambda_0,\eta_A,\eta_R,\gamma)\)、内层全候选验证分数、条件 LOO \(q_t,w_t,\lambda_t\) 及相关靶标摘要、完整开发集拟合记录、四环境 OOF、失败日志、运行耗时。

### 预先设定的实用 GO / NO-GO 门槛

保留原提案的工程判据；它们不是显著性检验，也不是理论定理：

| 判据 | GO 条件 |
|---|---|
| C1：主要相关性 | 相对历史 ICM 宏均 \(\Delta\mathrm{PCC}\ge+0.0030\)，且非平局折中 PCC 改善比例 \(>55\%\) |
| C2：误差 | 相对历史 ICM \(\Delta\mathrm{RMSE}\le0\)，\(\Delta\mathrm{MAE}\le+0.0005\) |
| C3：环境分布 | 至少 3/4 环境 \(\Delta\mathrm{PCC}\ge0\)；环境 4、5 平均 \(\Delta\mathrm{PCC}>+0.0030\)；任一环境下降不超过 0.0030 |
| 机制增量边界 | 相对匹配范围 ICM 的宏均 **ΔPCC ≥ +0.0030**，且 RMSE 不高于该对照；否则不能将增益归于双核异方差结构 |

比较使用保存的**未舍入**指标；表格显示值仅用于展示。只有同时满足 C1–C3 及匹配范围 ICM 的附加比较，才将 BH-LMC 列为本轮值得保留的新候选；否则保留普通 ICM。无论结果如何，不依据测试折事后扩大网格、调整阈值或补丁式修改公式。

该固定划分已经参加多轮模型开发，因此本轮结果属于**探索性同协议比较**，不是全新独立样本上的最终确认。若未取得增量，只能否定当前 BH-LMC 具体配置的实用收益，**不能**推出 ICM 已达到信息论上限、环境异方差不存在或 LMC 方法类别无效。

## 10. 理论与实现依据

- Gianola & Schön (2016), *Cross-Validation Without Doing Cross-Validation in Genome-Enabled Prediction*. G3. DOI: [10.1534/g3.116.033381](https://doi.org/10.1534/g3.116.033381) — 条件 CV / 核预测的代数基础；具体 LOO 公式仍须与截距和固定核约定匹配。
- Álvarez, Rosasco & Lawrence (2012), *Kernels for Vector-Valued Functions: A Review*. [DOI: 10.1561/2200000036](https://doi.org/10.1561/2200000036) — LMC 与向量值核的成熟理论。
- Rakitsch et al. (2013), *It is all in the noise: Efficient multi-task Gaussian process inference with structured residuals*. [NeurIPS 论文](https://papers.nips.cc/paper/5089-it-is-all-in-the-noise-efficient-multi-task-gaussian-process-inference-with-structured-residuals) — 任务信号与残差协方差的区分。
- Cuevas et al. (2018), *Genomic-Enabled Prediction Kernel Models with Random Intercepts for Multi-environment Trials*. DOI: [10.1534/g3.117.300454](https://doi.org/10.1534/g3.117.300454) — 多环境基因组核模型的相关先例。
- 项目现有 [RESULTS.md](https://github.com/Yhyb24P/Wheat599_model_bases/blob/main/docs/RESULTS.md) — 本实验的固定参照、划分与历史结果；仓库最新配置细节必须在实施时按实际文件核对，不以本文推测替代源码。
