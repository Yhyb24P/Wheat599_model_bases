## 第一部分：关键结论

### 原始证据范围与文件审阅说明

本报告严格建立在原始实验记录、数学推导与公开文献证据的基础之上。在项目材料审阅方面，已完整审阅最新上传的实验主记录文件 `docs/RESULTS.md`，并利用数值代数工具对多输出核方法在完全区组交叉验证（CV1）下的代数等价性及耦合系统求解复杂度进行了独立验算。由于外部代码仓库链接（`https://github.com/Yhyb24P/Wheat599_model_bases`）在当前网络环境下返回不可访问状态，故无法直接读取仓库内的 `docs/history/RESEARCH_PLAN.md`、`configs/*.yaml` 配置文件及 `src/wheat599/*.py` 源码实现；本报告涉及的所有历史基线超参数网格、数据加载修正（列主序 Fortran-order `reshape`）及嵌套评估协议，均严格以 `docs/RESULTS.md` 中经审计核对的正式实验记录为准，不对未直接读取的源码内部细节作无根据的推断。

### RQ1 核心解答：当前普通 ICM 的预测机制本质与实质性限制

综合现有十余种代表性模型在 BGLR Wheat599 数据集（$N=599$ 份历史小麦纯系、$P=1,279$ 个二值 0/1 DArT 标记、$T=4$ 个籽粒产量环境 `"1"`、`"2"`、`"4"`、`"5"`、无可靠物理染色体坐标、仅使用标记矩阵 $X$ 的固定外层十折 CV1 协议）上的完整实验证据，核心结论十分明确：**当前最强基座“普通内生共区域化模型（Ordinary ICM，四环境折均宏平均 PCC 0.52019，RMSE 0.85449，MAE 0.65784）”的预测增益完全源于基于跨环境表型协方差的多任务谱收缩正则化（Multi-task Spectral Shrinkage），而非捕获了任何新的遗传上位互作或环境特异基因型效应；与此同时，其一阶可分离克罗内克积结构（Separable Kronecker Structure）构成了阻碍预测性能进一步突破的实质性数学瓶颈**。

下表汇总了 Wheat599 在相同外层十折嵌套重拟合协议下各代表性架构的四环境分项 PCC 与宏平均表现，清晰展示了从特征空间深度神经网络、单环境核回归到多输出核模型的性能断层与环境间遗传异质性：

| **模型名称**                  | **模型类别与核心机制**                  | **环境 1 PCC** | **环境 2 PCC** | **环境 4 PCC** | **环境 5 PCC** | **宏均 PCC ↑** | **宏均 RMSE ↓** | **宏均 MAE ↓** |
| ------------------------- | ------------------------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------- | ------------ |
| **谱选择性 ICM**              | 同复合核谱滤波双分支多任务核岭                | 0.5659       | 0.5086       | 0.4633       | 0.5436       | 0.52035      | 0.85452       | 0.65793      |
| **普通 ICM（当前最强基座）**        | 可分离单复合核＋表型相关收缩多任务核岭            | 0.5657       | 0.5090       | 0.4634       | 0.5427       | **0.52019**  | **0.85449**   | **0.65784**  |
| **选择性分核＋OOF 可靠性**         | 环境独立选基础核＋OOF 可靠性乘积加权           | 0.5405       | 0.4984       | 0.4643       | 0.5395       | 0.51070      | 0.85930       | 0.66180      |
| **单环境加性＋RBF 核岭**          | 单环境中心化加性核与汉明 RBF 混合核岭          | 0.5408       | 0.4960       | 0.4598       | 0.5403       | 0.50923      | 0.86022       | 0.66246      |
| **Gaussian 核岭**           | 单环境全标记中位数缩放汉明距离 RBF 核岭         | 0.5475       | 0.4942       | 0.4428       | 0.5254       | 0.50250      | 0.86280       | 0.66530      |
| **冗余加权 RBF**              | 标记相关矩阵收缩加权汉明距离 RBF 核岭          | 0.5342       | 0.4728       | 0.4443       | 0.5174       | 0.49220      | 0.86780       | 0.66780      |
| **背景条件化＋RBF 三核**          | 加性＋PCA16 四门控背景核＋汉明 RBF 核       | 0.5348       | 0.4745       | 0.4428       | 0.5156       | 0.49190      | 0.86800       | 0.66760      |
| **背景条件化联合收缩**             | 全局加性与 PCA16 门控背景标记效应核岭         | 0.5456       | 0.4882       | 0.4120       | 0.4635       | 0.47740      | 0.87740       | 0.67980      |
| **BiMSGP**                | 双向 Mamba 选择性状态空间序列网络           | 0.5566       | 0.4300       | 0.4003       | 0.4963       | 0.47080      | 0.90500       | 0.69800      |
| **RR-BLUP**               | 单环境纯加性线性分子标记岭回归                | 0.4949       | 0.4938       | 0.4040       | 0.4468       | 0.45990      | 0.88700       | 0.68740      |
| **MoEGP**                 | 6 独立专家 Top-2 稀疏门控混合专家网络        | 0.5142       | 0.4352       | 0.3964       | 0.4910       | 0.45920      | 0.91120       | 0.70220      |
| **FT-Transformer (A 基线)** | 仿射标记 Tokenizer＋全注意力＋val-PCC 早停 | 0.4955       | 0.4300       | 0.3855       | 0.4258       | 0.43420      | 0.94540       | 0.73870      |
| **完整双向诱导注意力 FT-T**        | 共享低秩编码＋双向诱导注意力＋多槽读出            | 0.4487       | 0.2898       | 0.2328       | 0.3813       | 0.33820      | 0.97640       | 0.76560      |

为严格区分“已有数学或实验证据支持的限制”与“尚待检验的解释”，以下从代数定理与实证数据两个层面剖析当前普通 ICM 的内在机制与瓶颈。

#### 已有严格数学证明与实验证据支持的限制

第一项核心证据来自 **CV1 完全区组设计下的“典型特征任务差分收缩定理（Canonical Eigen-Task Differential Shrinkage）”**，该定理从数学上证明普通 ICM 并不学习任何新的遗传特征交互。在 Wheat599 的 CV1 协议中，训练集的 $N$ 份材料在全部 $T=4$ 个环境中均有完整表型观测 $Y \in \mathbb{R}^{N \times T}$（即平衡完全区组设计），而测试集的 $N_*$ 份材料在四个环境中均无表型观测。普通 ICM 采用单一复合遗传核 $K = w K_{\text{add}} + (1-w) K_{\text{rbf}}(h) \in \mathbb{R}^{N \times N}$、单一任务共区域化矩阵 $B = (1-\eta) I_T + \eta C_Y \in \mathbb{R}^{T \times T}$（其中 $C_Y$ 为训练集表型相关矩阵，外层十折均选中 $\eta=0.6$）以及各向同性噪声惩罚 $\lambda I_{NT}$。设 $B = V S V^\top$ 为 $B$ 的正交特征分解（特征值 $s_1 \ge s_2 \ge \dots \ge s_T > 0$，正交特征向量矩阵 $V = [v_1, \dots, v_T]$），则联合克罗内克线性系统 $(B \otimes K + \lambda I_{NT}) \operatorname{vec}(A) = \operatorname{vec}(Y)$ 对未见测试材料 $X_*$（交叉核 $K_* \in \mathbb{R}^{N_* \times N}$）的预测 $\hat{Y}_* = K_* A B$ 满足如下精确代数恒等式（数值验证最大绝对误差仅为 $8.88 \times 10^{-16}$）：

$$\hat{Y}_* = \sum_{j=1}^T \left[ K_* \left( K + \frac{\lambda}{s_j} I_N \right)^{-1} (Y v_j) \right] v_j^\top$$

这一恒等式证明，在平衡 CV1 设计下，普通 ICM 在数学上完全等价于先通过正交矩阵 $V$ 将四环境表型旋转为 $T=4$ 个正交的“典型特征环境（Canonical Eigen-Environments）” $\tilde{y}_j = Y v_j$，在每个特征环境上使用**完全相同的复合遗传核 $K$** 独立拟合单任务核岭回归，仅将第 $j$ 个特征任务的有效正则化强度设为 $\lambda_j^{\text{eff}} = \lambda / s_j$，最后通过 $V^\top$ 旋转回原环境空间。当 $\eta = 0$ 时，$s_j \equiv 1$，$V V^\top = I_T$ 彻底相消，模型退化为四个独立的单环境核岭回归（即多输出高斯过程在平衡设计下的自克里金化 Autokrigeability 退化性质）。当 $\eta = 0.6$ 时，代表四环境共性高产方向的第一主特征向量 $v_1$（对应最大特征值 $s_1 > 1$）获得较弱的惩罚 $\lambda / s_1 < \lambda$，从而保留更多跨环境共性遗传信号；而代表环境间差异对比方向的后三个特征向量 $v_2, v_3, v_4$（对应较小特征值 $s_j < 1$）受到显著增强的惩罚 $\lambda / s_j > \lambda$，从而强力抑制环境特异噪声的过拟合。因此，普通 ICM 相比单环境加性＋RBF 核岭的 `+0.0110` PCC 提升，本质上源于四环境联合选模对内层选模方差的降低，以及输出空间对对比特征方向的差分谱收缩。

第二项核心证据揭示了“谱选择性 ICM（0.52035）”必然失效的代数根源，即同特征基对易性退化。近期实验尝试在普通 ICM 的复合核 $K$ 上按特征值频率拆分为低频共享核 $K_s = K(K+\tau I)^{-1}K$ 与高频私有核 $K_p = K - K_s$，结果宏平均 PCC 仅微增 `+0.00016`（RMSE 反而微增 `+0.00003`，且 10 个外层折中有 5 折直接选中 $\tau=0$ 退化回普通 ICM）。其数学原因在于 $K_s$ 与 $K_p$ 均是同一复合核 $K = U \Lambda U^\top$ 的矩阵有理函数，二者在相同的特征向量基 $U$ 下完全对易（$K_s K_p = K_p K_s$）。因此，在同一复合核 $K$ 内部做谱滤波拆分，根本无法解耦原本互不对易的加性线性核 $K_{\text{add}}$ 与局部非线性汉明距离核 $K_{\text{rbf}}$（$K_{\text{add}} K_{\text{rbf}} \neq K_{\text{rbf}} K_{\text{add}}$），仅相当于对单一核谱 $\lambda_i$ 的收缩包络形状做了微小形变，故无法带来实质性的预测突破。

第三项核心证据表明，**加性效应与非线性上位效应的跨环境异质性被单一可分离核 $B \otimes K$ 强行抹平**。数量遗传学经典研究（Burgueño 等，2007，基于 CIMMYT 小麦历史多环境试验） 早已证明，小麦自交系的加性主效应（Additive $\times$ Environment，由 $K_{\text{add}}$ 建模）与加性×加性上位效应（Additive $\times$ Additive $\times$ Environment，由局部汉明 RBF 核 $K_{\text{rbf}}$ 捕获）具有截然不同的环境敏感性与跨环境相关模式。`docs/RESULTS.md` 的实验数据直接印证了 Wheat599 四环境在遗传机制上的剧烈分化：从纯线性加性模型 RR-BLUP 到单环境加性＋RBF 核岭，环境 2 的 PCC 仅从 `0.4938` 微增至 `0.4960`（非线性增量仅 `+0.0022`），而在背景＋RBF 三核模型中环境 2 的加性核权重（`1.036`）亦远高于 RBF 核权重（`0.327`），证明环境 2 属于典型的强加性主导环境；相反，环境 4 与环境 5 在引入 RBF 核后 PCC 分别从 `0.4040`、`0.4468` 跃升至 `0.4598`（`+0.0558`）与 `0.5403`（`+0.0935`），证明环境 4、5 高度依赖局部非线性上位效应。然而，当从单环境加性＋RBF 升级至普通 ICM 时，多任务增益几乎全部集中在环境 1（`+0.0249`）与环境 2（`+0.0130`），而在强非线性/低信噪比的环境 4（`+0.0036`）与环境 5（`+0.0024`）几乎停滞。这正是因为普通 ICM 的可分离结构 $B \otimes (w K_{\text{add}} + (1-w) K_{\text{rbf}}) + \lambda I_{NT}$ 强行施加了三重同质化约束：迫使四环境共享同一个加性比例 $w$；迫使跨环境高度稳定的加性核 $K_{\text{add}}$ 与环境特异性较强的非线性核 $K_{\text{rbf}}$ 采用完全相同的共享强度 $\eta$ 和混杂了非遗传残差相关的表型相关矩阵 $C_Y$；同时对可预测性最低的高噪声环境 4（PCC 约 `0.46`）与高可预测性环境 1、5（PCC 约 `0.54–0.57`）施加完全相同的同方差正则化参数 $\lambda$，导致环境 4 的高方差噪声在联合求解中以同等精度权重污染跨环境共享表征。

第四项核心证据解释了**此前“选择性分核＋OOF 可靠性（0.5107）”为何未能击败普通 ICM：内层独立选核导致的 RKHS 空间失配与非对角共享强度的“双重衰减陷阱”**。实验记录显示，该模型在环境 4（`0.4643` vs 普通 ICM `0.4634`）虽展现出分核建模对难点环境的潜在益处，但在环境 1（`0.5405` vs `0.5657`）与环境 2（`0.4984` vs `0.5090`）大幅回落，整体宏均 PCC（`0.5107`）反而不如普通 ICM（`0.5202`）。剖析其实现规则可知两处数学缺陷：其一，该方案在仅约 320 个样本的内层子集上为每个环境独立选择各自的基础核参数（包括不同的 RBF 带宽 $h_t$），不仅引入了高昂的内层选模方差，更导致不同环境的 $K_{\text{rbf}}(h_t)$ 落在不兼容的再生核希尔伯特空间（RKHS）中，破坏了跨环境交叉协方差的物理意义；其二，该方案以单核三折 OOF 预测相关系数乘积 $r_t r_{t'}$ 加权任务相关矩阵，由于小麦产量单核 OOF 相关系数 $r_t \approx 0.45\text{–}0.54$，乘积 $r_t r_{t'} \approx 0.20\text{–}0.28$，再乘以共享强度 $\eta \in \{0, 0.3, 0.6\}$ 后，非对角线真实有效共享强度被双重衰减至 $0.06\text{–}0.15$，使共区域化矩阵退化为近似对角阵 $I_T$，从而丧失了普通 ICM 在环境 1 和环境 2 上的多任务收缩收益。

#### 尚待实验检验的假设与设计新模型的充分理由

基于上述数学证明与负结果归因，**存在充分且严密的机制理由设计新模型，但突破方向绝不是继续堆叠神经网络层数、注意力门控或样本级 PCA 门控，而是构建保持跨环境物理核空间对齐（共享带宽 $h$）、通过闭式解析留一法（Analytical LOO）零泄漏解耦“加性共区域化 $B_{\text{add}}$”与“非线性共区域化 $B_{\text{rbf}}$”、并引入“环境异方差残差精度校准 $\Lambda_\varepsilon$”的双核异方差内生共区域化模型（BH-LMC）**。这一新机制能否在 $N=599$ 的样本规模下稳定转化为超越 `0.52019` PCC 的泛化增益，正是本研究待通过集中实验检验的核心科学假说。

## 第二部分：最近邻 SOTA 与理论研究

### RQ2 核心解答：统计核方法与神经网络机制的系统对比

针对 Wheat599 的数据本质（$N=599 < P=1,279$、无物理染色体坐标的二值 0/1 DArT 标记、由大量微效基因与复杂基因网络控制的籽粒产量性状），实验记录与文献证据一致表明：**在对偶样本空间（$N \times N$）建模的再生核希尔伯特空间（RKHS）与多输出高斯过程（Multi-output GP / LMC）在统计自由度控制与归纳偏置上全面优于直接在特征空间（$P=1,279$）学习的深度学习架构（FT-Transformer、双向 Mamba、混合专家网络 MoE）**。

具体而言，仓库已完成的神经网络与门控核实验构成了完整的负结果证据链。首先，FT-Transformer 及其重构变体（A 基线 PCC `0.4342`、D 变体 `0.4324`、Direct 槽位读出 `0.4193`、完整双向诱导注意力 `0.3382`）之所以全面落后于简单线性模型 RR-BLUP（`0.4528` / `0.4599`），是因为 DArT 标记缺乏 DNA 序列的连续局部语义且单标记信噪比极低，拥有 21 万至 30 万参数的逐标记注意力网络在约 485 个训练样本上极易记忆训练集噪声（训练集宏均 PCC 高达 `0.7410`，而验证集与测试集骤降至 `0.4462` 与 `0.4325`），并在 4/40 至 5/40 折出现早期近常数预测退化。其次，在“固定 RR-BLUP + FT-T 残差修正（PCC `0.4517` vs 同折 RR-BLUP `0.4528`）”实验中，均方误差的精确分解恒等式 $\mathrm{MSE} = b^2 + \sigma_y^2(1-r^2) + \sigma_y^2(k-r)^2$（其中 $b$ 为均值偏移，$r$ 为预测相关系数，$k = s_{\hat{y}}/s_y$ 为相对预测幅度比）揭示了深刻的统计规律：对于固定的预测方向 $r$，最小化 MSE 的最优预测尺度严格满足 $k^* = r$。神经网络残差分支虽然在 40/40 个验证折上虚高了验证集 PCC（平均 `+0.0338`），但在测试集上完全未能改善方向项 $\sigma_y^2(1-r^2)$（从 `0.7671` 微增至 `0.7676`），反而将相对尺度 $k$ 从接近最优的 `0.4372` 推高至 `0.6119`，导致尺度失配误差从 `0.0243` 恶化至 `0.0612`。最后，2026 年最新发表的植物基因组预测神经架构 BiMSGP（Liu 等，2026 年 9 月，评测宏均 PCC `0.4708`，RMSE `0.9050`） 依赖源文件中的标记列顺序进行双向状态空间扫描，在无物理坐标的 DArT 矩阵上缺乏生物学序列基础，在加性主导的环境 2（`0.4300`）与高噪环境 4（`0.4003`）表现低迷；而 MoEGP（Pan 等，2026 年 2 月，评测宏均 PCC `0.4592`，RMSE `0.9112`） 的样本级稀疏专家路由在 540 个训练材料上面临严重的高方差门控过拟合，这一现象与局部多核学习 `背景条件化三核`（`0.4919`）落后于无门控 `单环境加性＋RBF 核岭`（`0.50923`）达 `-0.0173` PCC 的统计机制完全一致。

### 决定研究方向的核心文献精炼比较

下表系统梳理了截至 2026 年 11 月决定本课题研究边界的 10 篇核心文献（涵盖经典统计遗传基石、多输出核理论及 2025–2026 年最新模型），严格核查其数学机制、适用边界及与当前 ICM 和拟议新架构的关系：

|**文献题名、作者与年份**|**DOI、论文地址与源码地址**|**实际研究问题与核心数学机制**|**数据、评价协议、主要结果与适用边界**|**与当前普通 ICM 的具体关系**|**是否已覆盖本研究拟议创新（BH-LMC）**|
|---|---|---|---|---|---|
|**1.** _Prediction of Genetic Values of Quantitative Traits in Plant Breeding Using Pedigree and Molecular Markers_<br><br>  <br><br>Crossa, de los Campos, Pérez, Gianola, Burgueño 等 (2010) _Genetics_|**DOI:** `10.1534/genetics.110.118521`<br><br>  <br><br>**URL:** `https://pmc.ncbi.nlm.nih.gov/articles/PMC2954475/`<br><br>  <br><br>**Code:** CRAN `BGLR` 包 (`wheat.RData`)|针对植物育种复杂性状，提出单环境贝叶斯 RKHS 高斯核回归：$y = \mu \mathbf{1} + u + \varepsilon$，$u \sim \mathcal{N}(0, \sigma_u^2 K_{\text{rbf}})$，其中 $K_{\text{rbf}}(x_i, x_{i'}) = \exp(-h d_{ii'})$ 捕获标记间高阶上位互作。|**数据：** 首发本研究的 Wheat599（599 份小麦、1279 个 DArT、4 环境）及玉米数据。<br><br>  <br><br>**协议与结果：** 单环境随机十折交叉验证；证明在小麦自交系中非线性高斯核一致优于线性 Bayesian LASSO 和谱系模型。<br><br>  <br><br>**边界：** 仅限单环境独立建模（不同划分下数字不与本表直接排名）。|构成项目中单环境 Gaussian 核岭基线（`0.5025` PCC）及普通 ICM 中非线性 RBF 分量的直接物理原型。|**否。** 未涉及多环境共区域化矩阵 $B$、加性与 RBF 双核分路共享及异方差校准。|
|**2.** _Modeling Additive $\times$ Environment and Additive $\times$ Additive $\times$ Environment Using Genetic Covariances of Relatives of Wheat Genotypes_<br><br>  <br><br>Burgueño, Crossa, Cornelius 等 (2007) _Crop Science_|**DOI:** `10.2135/cropsci2006.09.0564`<br><br>  <br><br>**URL:** `https://hdl.handle.net/10568/166515`<br><br>  <br><br>**Code:** SAS / ASReml 脚本（无公开 GitHub 仓库）|研究小麦多环境试验中加性效应与加性×加性上位效应的不同环境互作模式。构建线性混合模型：$\operatorname{Var}(g) = \Sigma_{AE} \otimes A + \Sigma_{AAE} \otimes (A \circ A)$，分别以两个独立的跨环境协方差矩阵建模加性与上位 $G \times E$。|**数据：** CIMMYT 历史小麦多环境产量试验（与 Wheat599 同源育种群体）。<br><br>  <br><br>**结果与边界：** 证实小麦品种的加性效应跨环境相关结构 $\Sigma_{AE}$ 与上位效应跨环境相关结构 $\Sigma_{AAE}$ 显著不同；但仅使用谱系矩阵 $A$，未引入分子标记核或 CV1 预测评估。|直接揭示了普通 ICM 强行令加性核与非线性核共享同一环境相关矩阵 $B$（即假设 $\Sigma_{AE} \propto \Sigma_{AAE}$）在小麦遗传学上的不合理性。|**否。** 仅限谱系矩阵 $A$ 与阿达玛积 $A \circ A$ 的 REML 因子分析，未涉及分子标记 $K_{\text{add}} + K_{\text{rbf}}$ 及小样本 CV1 解析留一收缩估计。|
|**3.** _Genomic Prediction of Breeding Values when Modeling Genotype $\times$ Environment Interaction Using Pedigree and Dense Molecular Markers_<br><br>  <br><br>Burgueño, de los Campos, Weigel, Crossa (2012) _Crop Science_|**DOI:** `10.2135/cropsci2011.06.0299`<br><br>  <br><br>**URL:** `https://doi.org/10.2135/cropsci2011.06.0299`<br><br>  <br><br>**Code:** ASReml / BGLR 脚本|正式定义多环境基因组预测的 **CV1（完全未测试材料预测）** 与 **CV2（稀疏测试预测）** 协议；提出多环境混合模型 $\operatorname{Var}(y) = G_0 \otimes K_{\text{add}} + R_0 \otimes I_N$，采用因子分析（FA）或非结构化矩阵估计 $G_0, R_0$。|**数据：** Wheat599 四环境数据。<br><br>  <br><br>**协议与结果：** 5 折 CV1 与 CV2；指出 CV1 预测仅依赖测试与训练材料的遗传协方差，多环境模型在 CV1 下的收益取决于跨环境遗传相关与异方差残差控制。<br><br>  <br><br>**边界：** 仅使用单一线性标记矩阵（GBLUP），未包含非线性核。|奠定了本项目 CV1 评估协议与线性多环境协方差框架；普通 ICM 可视为其将线性核替换为“加性＋RBF 复合核”并固定 $B=(1-\eta)I+\eta C_Y$ 的核岭版本。|**否。** 未建模双核（加性＋RBF）异构共区域化，且传统 FA/REML 在双非对易核下易出现方差组分不收敛。|
|**4.** _Multi-task Gaussian Process Prediction_<br><br>  <br><br>Bonilla, Chai, Williams (2007) _NeurIPS_<br><br>  <br><br>及 _Kernels for Vector-Valued Functions: A Review_<br><br>  <br><br>Álvarez, Rosasco, Lawrence (2012) _Found. Trends Mach. Learn._|**DOI:** `10.1561/2200000036` (Álvarez 2012)<br><br>  <br><br>**URL:** `https://papers.nips.cc/paper/3189-multi-task-gaussian-process-prediction`<br><br>  <br><br>**Code:** GPy / GPflow 标准库|系统建立向量值 RKHS 核函数理论：定义内生共区域化模型（ICM，$K_{\text{multi}} = B \otimes k$）与线性共区域化模型（LMC / SoS，$K_{\text{multi}} = \sum_{q=1}^Q B_q \otimes k_q$）。证明在完全区组设计与同构噪声下，单核 ICM 存在“块相消（Autokrigeability）”退化性质。|**数据与边界：** 机器学习多任务基准数据；理论指出 ICM 强制所有输出任务共享同一输入长度尺度（Length-scale），而 LMC 允许不同输出任务以不同权重组合多个具有不同平滑尺度的基础核 $k_q$。|当前最强基座“普通 ICM”正是 $Q=1$ 的单核 ICM；由于它将 $K_{\text{add}}$ 与 $K_{\text{rbf}}$ 预先合并为单一标量核 $k$，导致所有环境及其特征方向被迫共享同一加性/非线性长度尺度。|**部分覆盖理论框架，未覆盖具体估计机制。** LMC 给出了 $\sum_q B_q \otimes k_q$ 的通用泛函形式，但未给出针对高维遗传标记 $N=599$ 的解析 LOO 遗传信号收缩与异方差校准机制。|
|**5.** _It is all in the noise: Efficient multi-task Gaussian process inference with structured residuals_<br><br>  <br><br>Rakitsch, Lippert, Borgwardt, Stegle (2013) _NeurIPS_|**DOI:** `10.5555/2999611.2999775`<br><br>  <br><br>**URL:** `https://proceedings.neurips.cc/paper/2013/file/59c33016884a62116be975a9bb8257e3-Paper.pdf`<br><br>  <br><br>**Code:** `https://github.com/limix/limix`|针对遗传表型多任务预测，研究双克罗内克和模型 $\operatorname{Cov}(\operatorname{vec}(Y)) = C \otimes R + \Sigma \otimes I_N$；严格证明当信号任务协方差 $C$ 与残差噪声协方差 $\Sigma$ 成比例时，多任务联合预测完全退化为单任务独立预测（$\operatorname{vec}(M_*) = \operatorname{vec}(R_*)$）。|**数据：** 酵母（$N=109, P=2956$）与拟南芥（$N=147, P=216,130$ 二值 SNP，4 个开花/成熟期性状）。<br><br>  <br><br>**结果与边界：** 证明显式区分遗传信号协方差 $C$ 与非遗传残差协方差 $\Sigma$ 是多任务 GP 在遗传学中奏效的关键；但利用双矩阵特征分解要求仅含**单个**遗传核 $R$。|直接解释了普通 ICM 用原始表型相关 $C_Y$（混杂 $C$ 与 $\Sigma$）及同方差 $\lambda I_T$ 建模的理论缺陷。|**否。** Rakitsch 等仅支持单一线性核 $R$（以便与 $I_N$ 同步对角化），无法处理 $K_{\text{add}}$ 与 $K_{\text{rbf}}$ 两个非对易遗传核的联合共区域化。|
|**6.** _Predicting Multivariate Responses in Multiple Linear Regression (Curds and Whey)_<br><br>  <br><br>Breiman, Friedman (1997) _J. R. Stat. Soc. B_|**DOI:** `10.1111/1467-9868.00054`<br><br>  <br><br>**URL:** `https://www.stat.berkeley.edu/~breiman/curds-whey-justtext.pdf`<br><br>  <br><br>**Code:** 无官方源码仓库（闭式矩阵算法）|研究完全区组多响应回归的最优线性收缩：将响应变量旋转至典型相关坐标系 $Y' = Y T$，基于留一交叉验证（LOO/GCV）广义判定系数对各典型方向施加差分收缩矩阵 $D = \operatorname{diag}(d_1, \dots, d_T)$，再反变换回原空间 $\tilde{Y} = \hat{Y} T D T^{-1}$。|**数据与结果：** 多响应模拟与实测回归基准；证明利用跨任务相关与 LOO 信噪比在典型坐标下进行差分收缩，可严格降低各响应变量的期望均方误差（MSE）。<br><br>  <br><br>**边界：** 仅对单任务预测结果做后乘线性组合，不改变输入特征核空间。|从频率派角度完美诠释了本研究在 RQ1 中推导的普通 ICM 代数等价性：普通 ICM 即是在固定核 $K$ 下对典型任务施加 $\lambda / s_j$ 惩罚的核岭版本。|**否。** C&W 仅在单一预测器输出空间做后处理组合，无法让不同环境或不同典型方向动态调整加性核 $K_{\text{add}}$ 与非线性核 $K_{\text{rbf}}$ 的相对权重。|
|**7.** _Genomic Prediction of Genotype $\times$ Environment Interaction Kernel Regression Models_<br><br>  <br><br>Cuevas, Crossa, Montesinos-López, Pérez-Rodríguez, de los Campos 等 (2017/2018) _G3_|**DOI:** `10.1534/g3.117.300454`<br><br>  <br><br>**URL:** `https://pmc.ncbi.nlm.nih.gov/articles/PMC5873923/`<br><br>  <br><br>**Code:** CRAN `BGGE` 与 `BGLR` 包|提出多环境非结构化协方差核模型（`MUC` 与 `MUC_f`）：$y = \mu \mathbf{1} + Z_E \beta_E + u + f + \varepsilon$，其中 $\operatorname{Cov}(u) = \Sigma_E \otimes K$（$K$ 取线性核 `GB` 或高斯核 `GK`），且允许环境异方差残差 $\operatorname{Cov}(\varepsilon) = \operatorname{diag}(\sigma_{\varepsilon, 1}^2, \dots, \sigma_{\varepsilon, m}^2) \otimes I_N$。|**数据与协议：** 小麦与玉米多环境试验，CV1 与 CV2 随机划分。<br><br>  <br><br>**结果与边界：** 证明在环境间存在正遗传相关时，`MUC-GK` 结合环境异方差残差 $\operatorname{diag}(\sigma_{\varepsilon, m}^2)$ 显著优于同方差单环境或主效应模型；但单独使用 `GB` 或单独使用 `GK`。|`MUC` 模型验证了在多环境核回归中引入环境异方差残差 $\operatorname{diag}(\sigma_{\varepsilon, t}^2)$ 的必要性（直接针对普通 ICM 的同方差 $\lambda I_T$ 缺陷）。|**否。** `MUC` 在单次建模中仅采用单一核（`GB` 或 `GK` 二选一），未将加性核与高斯核结合并赋予各自独立的共区域化矩阵。|
|**8.** _Cross-Validation Without Doing Cross-Validation in Genome-Enabled Prediction_<br><br>  <br><br>Gianola, Schön (2016) _Genetics_<br><br>  <br><br>及 _MegaLMM_<br><br>  <br><br>Runcie, Qu, Cheng, Crawford (2021) _Genome Biology_|**DOI:** `10.1534/genetics.115.185850` (Gianola 2016); `10.1186/s13059-021-02416-w` (Runcie 2021)<br><br>  <br><br>**URL:** `https://pmc.ncbi.nlm.nih.gov/articles/PMC5068934/`<br><br>  <br><br>`https://pmc.ncbi.nlm.nih.gov/articles/PMC8299638/`<br><br>  <br><br>**Code:** `https://github.com/deruncie/MegaLMM`|Gianola & Schön (2016) 推导出基于帽子矩阵 $H = K(K+\lambda I)^{-1}$ 的精确闭式解析留一误差 $e_{-i} = (y_i - \hat{y}_i)/(1 - H_{ii})$；Runcie 等 (2021) 证明多性状预测必须分离遗传相关与表型相关。|**数据：** Gianola & Schön 直接在 **Wheat599** 上验证了解析 LOO 与多核估计的高效性与准确性；MegaLMM 主要面向成百上千个高通量表型（CV2 场景）。|为解决“选择性分核＋OOF 可靠性”中内层三折切分样本过少、方差过大的问题提供了精确的**零额外切分闭式解析 LOO 工具**。|**否。** Gianola & Schön 仅研究单环境解析 LOO 与多核权重，未构建多环境双核共区域化系统。|
|**9.** _MoEGP: an efficient crop genomic prediction approach based on the mixture of experts network_<br><br>  <br><br>Pan, Yang, Zhang, Wei 等 (2026) _Plant Methods_|**DOI:** `10.1186/s13007-026-01500-1`<br><br>  <br><br>**URL:** `https://pmc.ncbi.nlm.nih.gov/articles/PMC12958669/`<br><br>  <br><br>**Code:** `https://github.com/CNRRI-RGRT/MoEGP` (`d3a3c44f`)|将稀疏门控混合专家网络（6 个独立 MLP 专家，Top-$k=2$ 门控路由＋负载均衡辅助损失）应用于作物基因组预测，试图按样本特征动态分配专家权重。|**数据与结果：** 原文在 5 个数据集 36 个性状案例上报告提升；但在本项目 Wheat599 严格嵌套十折 CV1 协议（修正官方共享专家引用缺陷后）下，宏均 PCC 仅为 **`0.4592`**（RMSE `0.9112`），落后普通 ICM 达 `-0.0610` PCC。|证实了在 $N=599$ 小样本标记矩阵上，样本级输入门控与高容量 MLP 专家网络会产生严重过拟合，远逊于全局多任务核回归。|**否。** 属于单环境特征空间深度专家网络，在本项目基准测试中已证实为负结果。|
|**10.** _BiMSGP: A Bidirectional Mamba-Based Model for Scalable and Accurate Genomic Prediction in Plants_<br><br>  <br><br>Liu, Yao, Ma, Cheng, Liu 等 (2026) _Genes_|**DOI:** `10.3390/genes17091156`<br><br>  <br><br>**URL:** `https://www.mdpi.com/2073-4425/17/9/1156`<br><br>  <br><br>**Code:** 官方双向 Mamba 源码（已集成至本项目评测）|将标记序列映射至嵌入维度 $d_{\text{model}}$，通过正向与反向翻转的双向 Mamba 选择性状态空间层（Selective SSM，$O(P)$ 线性复杂度扫描）提取长程标记依赖，接轻量 MLP 解码器预测表型。|**数据与结果：** 原文包含 Wheat599、Wheat2000 与 SoyBase 10 折 CV；在本项目严格嵌套十折 CV1 重拟合评测中，Wheat599 宏均 PCC 为 **`0.4708`**（环境 1、2、4、5 依次为 `0.5566`、`0.4300`、`0.4003`、`0.4963`，RMSE `0.9050`）。|虽在强非线性环境 1、5 优于线性 RR-BLUP（`0.4599`），但在环境 2、4 严重退化，整体大幅落后普通 ICM（`0.5202`）达 `-0.0494` PCC。|**否。** 单任务序列状态空间模型，不具备跨环境正则化能力，且受限于无物理坐标标记的任意排列顺序。|

## 第三部分：最优候选架构

经过对上述实证记录、代数性质与前沿文献的系统筛选，本研究明确摒弃缺乏小样本统计依据的深度神经模块、注意力机制、局部样本门控或启发式特征筛选，提出一套具有严密统计遗传学与多输出核学习理论支撑的新架构：**基于解析留一信号分解与环境异方差校准的双核内生共区域化模型（Bi-Kernel Heteroscedastic Coregionalization Model with Analytical LOO Signal Calibration，简称 BH-LMC）**。

### 1. 核心机制：解决普通 ICM 的哪三个具体问题？

BH-LMC 直接针对第一部分中经数学与实验证实的普通 ICM 三大结构性瓶颈，并同时消除了“选择性分核＋OOF 可靠性”的两阶段崩溃缺陷。

首先，BH-LMC **解耦了加性主效应（$K_{\text{add}}$）与局部非线性上位效应（$K_{\text{rbf}}$）的跨环境共区域化，打破了普通 ICM 单一可分离核 $B \otimes K$ 的刚性限制**。该架构不再将 $K_{\text{add}}$ 与 $K_{\text{rbf}}$ 预先合并为单一标量核，而是构建二阶线性共区域化结构 $B_{\text{add}} \otimes K_{\text{add}} + B_{\text{rbf}} \otimes K_{\text{rbf}}(h)$。这不仅允许加性遗传效应（由 $\eta_{\text{add}}$ 控制）与非线性上位效应（由 $\eta_{\text{rbf}}$ 控制）拥有各自独立的跨环境共享强度与遗传相关靶标，更在代数上使得不同典型特征环境（Canonical Eigen-Environments）自动获得不同的加性/非线性核组合比例，从而能够同时表达跨环境高度保守的加性主效应与环境特异的局部上位效应。

其次，BH-LMC 引入了**基于闭式解析留一法（Analytical LOO）的零泄漏环境结构与异方差精度校准，直接解决同方差惩罚 $\lambda I_T$ 及强非线性环境（环境 4、5）多任务增益停滞的问题**。利用帽子矩阵恒等式 $e_{\text{LOO}, i, t} = (y_{c,i,t} - \hat{y}_{i,t}) / (1 - H_{ii})$，模型无需额外切分内层样本即可在完整训练折上一次性闭式算出各环境的留一可预测方差（经验留一遗传力 $h_{\text{LOO}, t}^2$）以及各环境对加性核与 RBF 核的留一信号亲和度（$r_{\text{add}, t}^2, r_{\text{rbf}, t}^2$）。通过单一自适应校准超参数 $\gamma \in [0, 1]$，模型一方面让强非线性环境（如环境 4、5）温和上调 RBF 核比重 $1 - w_t(\gamma)$，让强加性环境（如环境 2）保持高加性比重 $w_t(\gamma)$；另一方面按环境噪信比相对倍数构建异方差残差惩罚矩阵 $\Lambda_\varepsilon(\gamma) = \operatorname{diag}(\lambda_1(\gamma), \dots, \lambda_T(\gamma))$，使高噪声环境（如环境 4）在联合求解中自动获得更强的正则化保护，防止其高方差残差反向污染高信噪比环境（环境 1、5），同时实现从高遗传力环境向低遗传力环境的非对称信息借力。

最后，BH-LMC 严格遵循**跨环境物理核空间对齐与边际方差守恒原则，彻底修正了“选择性分核＋OOF 可靠性”的失败根源**。模型强制四个环境共享同一物理 RBF 带宽 $h$，确保所有环境的非线性交叉协方差严格定义在同一再生核希尔伯特空间内；同时，模型绝不将任务相关矩阵乘以 $r_t r_{t'} \approx 0.25$ 进行人为双重衰减，而是以单位对角线的解析 LOO 遗传预测相关矩阵 $C_{\text{add}}, C_{\text{rbf}}$ 与表型相关矩阵 $C_Y$ 的凸组合作为良态共区域化靶标，严格保证各环境边际先验遗传方差恒等于 $1$。

### 2. 完整架构与文字架构图

从 $1,279$ 个二值 0/1 DArT 标记输入到四个环境（`"1"`、`"2"`、`"4"`、`"5"`）籽粒产量预测输出的完整信息流由五个严格顺序衔接的计算阶段构成：数据流首先进入**阶段 I（仅训练集统计量提取与无泄漏中心化）**，在当前拟合的训练材料（$N \times 1,279$）上独立计算标记列均值向量 $\mu_X$ 与四环境表型均值向量 $\mu_Y$，生成中心化标记矩阵 $X_c, X_{*,c}$ 与中心化训练表型 $Y_c$，并将 $\mu_Y$ 作为无惩罚环境截距单独保存。随后，数据流在阶段 II（双分支物理基础核矩阵构造）__中分流为两个并行且跨环境对齐的核分支：第一分支对中心化标记 $X_c$ 计算内积并按训练集平均对角线迹 $c_{\text{add}}$ 归一化，输出全局加性关系核 $K_{\text{add}} \in \mathbb{R}^{N \times N}$ 与测试交叉核 $K_{\text{add}}^_ \in \mathbb{R}^{N__ \times N}$；第二分支直接在原始 0/1 标记上计算两两汉明距离矩阵 $D$ 与 $D_*$，提取训练集非对角距离中位数 $d_0$，并以四环境共享带宽 $h$ 生成局部非线性上位核 $K_{\text{rbf}}(h) = \exp(-h D / d_0)$ 与测试交叉核 $K_{\text{rbf}}^*(h)$。

接着，训练核 $K_{\text{add}}, K_{\text{rbf}}(h)$ 与中心化表型 $Y_c$ 汇入**阶段 III（基础核基准参数锚定与闭式解析 LOO 信号分解模块）**：该模块先按既有普通 ICM 规则确定基准参数 $(h, w_0, \lambda_0)$，再利用预特征分解快速构造加性核、RBF 核及基准复合核 $K_0 = w_0 K_{\text{add}} + (1-w_0) K_{\text{rbf}}(h)$ 的帽子矩阵 $H_{\text{add}}, H_{\text{rbf}}, H_0$，通过闭式留一除数 $1 - (H_m)_{ii}$ 提取零样本内偏差的留一遗传预测矩阵 $\hat{G}_{\text{LOO}}^{(\text{add})}, \hat{G}_{\text{LOO}}^{(\text{rbf})}, \hat{G}_{\text{LOO}}^{(0)} \in \mathbb{R}^{N \times 4}$，进而闭式輸出环境特异加性权重 $w_t(\gamma)$、几何均值锚定的异方差惩罚系数 $\lambda_t(\gamma)$ 以及收缩遗传相关矩阵 $\tilde{C}_{\text{add}}(\gamma), \tilde{C}_{\text{rbf}}(\gamma)$。这些校准量随即传入**阶段 IV（双核异方差共区域化组装与联合系统求解器）**，分别组装 $4 \times 4$ 加性共区域化矩阵 $B_{\text{add}}(\eta_{\text{add}}, \gamma)$、非线性共区域化矩阵 $B_{\text{rbf}}(\eta_{\text{rbf}}, \gamma)$ 与异方差噪声对角阵 $\Lambda_\varepsilon(\gamma)$，构建 $4N \times 4N$ 对称正定耦合矩阵 $\Omega = B_{\text{add}} \otimes K_{\text{add}} + B_{\text{rbf}} \otimes K_{\text{rbf}}(h) + \Lambda_\varepsilon \otimes I_N$，并通过块矩阵 Cholesky 分解一次性求解对偶系数矩阵 $A \in \mathbb{R}^{N \times 4}$。最后，在阶段 V（CV1 跨环境联合推理与均值还原）__中，测试交叉核 $K_{\text{add}}^*, K_{\text{rbf}}^*(h)$ 与对偶系数矩阵 $A$ 及两个共区域化矩阵做双通道张量缩并并加上环境截距 $\mu_Y$，直接输出未见测试材料的四环境预测矩阵 $\hat{Y}__ \in \mathbb{R}^{N__ \times 4}$。

下表为上述文字架构图的模块化结构对应表，精确列出了每一阶段的输入张量、核心算子与输出张量维度：

|**架构阶段**|**模块名称**|**输入张量与维度**|**核心算子与转换逻辑**|**输出张量与维度**|
|---|---|---|---|---|
|**阶段 I**|**仅训练集统计量提取与无泄漏中心化**|训练标记 $X \in \{0,1\}^{N \times 1279}$，测试标记 $X_* \in \{0,1\}^{N_* \times 1279}$，训练表型 $Y \in \mathbb{R}^{N \times 4}$|仅在当前训练集计算标记列均值 $\mu_X \in \mathbb{R}^{1 \times 1279}$ 与表型列均值 $\mu_Y \in \mathbb{R}^{1 \times 4}$；执行中心化 $X_c = X - \mathbf{1}_N \mu_X$，$X_{*,c} = X_* - \mathbf{1}_{N_*} \mu_X$，$Y_c = Y - \mathbf{1}_N \mu_Y$（保留无惩罚环境截距 $\mu_Y$）。|$X_c \in \mathbb{R}^{N \times 1279}$，$X_{*,c} \in \mathbb{R}^{N_* \times 1279}$，$Y_c \in \mathbb{R}^{N \times 4}$，截距 $\mu_Y \in \mathbb{R}^{1 \times 4}$|
|**阶段 II**|**双分支物理基础核矩阵构造（跨环境对齐）**|原始 0/1 标记 $X, X_*$ 及中心化标记 $X_c, X_{*,c}$|**分支 1（全局加性核）：** 按训练集平均对角线迹 $c_{\text{add}} = \frac{1}{N}\operatorname{tr}(X_c X_c^\top)$ 归一化构造 $K_{\text{add}}$ 与 $K_{\text{add}}^*$。<br><br>  <br><br>**分支 2（局部上位汉明 RBF 核）：** 计算原始 0/1 标记两两汉明距离矩阵 $D$ 与 $D_*$，取训练集非对角距离中位数 $d_0$，以四环境共享带宽 $h$ 构造 $K_{\text{rbf}}(h) = \exp(-h D / d_0)$ 与 $K_{\text{rbf}}^*(h)$。|训练核 $K_{\text{add}}, K_{\text{rbf}}(h) \in \mathbb{R}^{N \times N}$；测试交叉核 $K_{\text{add}}^*, K_{\text{rbf}}^*(h) \in \mathbb{R}^{N_* \times N}$|
|**阶段 III**|**基础核基准参数锚定与闭式解析 LOO 信号分解**|$K_{\text{add}}, K_{\text{rbf}}(h), Y_c$|按既有普通 ICM 规则确定基准参数 $(h, w_0, \lambda_0)$；对 $K_{\text{add}}$、$K_{\text{rbf}}(h)$ 及基准复合核 $K_0 = w_0 K_{\text{add}} + (1-w_0) K_{\text{rbf}}(h)$ 构造平滑算子 $H_m = K_m(K_m + \lambda_0 I_N)^{-1}$，利用闭式公式 $\hat{g}_{\text{LOO}, i, t}^{(m)} = y_{c,i,t} - \frac{y_{c,i,t} - [H_m y_{c,\cdot,t}]_i}{1 - (H_m)_{ii}}$ 提取留一遗传预测矩阵 $\hat{G}_{\text{LOO}}^{(m)} \in \mathbb{R}^{N \times 4}$。|环境亲和权重 $w_t(\gamma) \in (0,1)^4$，异方差惩罚 $\lambda_t(\gamma) \in \mathbb{R}_+^4$，收缩遗传相关矩阵 $\tilde{C}_{\text{add}}, \tilde{C}_{\text{rbf}} \in \mathbb{R}^{4 \times 4}$|
|**阶段 IV**|**双核异方差共区域化组装与联合系统求解**|$K_{\text{add}}, K_{\text{rbf}}(h)$，$B_{\text{add}}, B_{\text{rbf}}, \Lambda_\varepsilon$，$Y_c$|组装 $4 \times 4$ 加性共区域化矩阵 $B_{\text{add}}(\eta_{\text{add}}, \gamma)$、非线性共区域化矩阵 $B_{\text{rbf}}(\eta_{\text{rbf}}, \gamma)$ 与异方差噪声对角阵 $\Lambda_\varepsilon(\gamma) = \operatorname{diag}(\lambda_1(\gamma), \dots, \lambda_4(\gamma))$；构建 $4N \times 4N$ 对称正定矩阵 $\Omega = B_{\text{add}} \otimes K_{\text{add}} + B_{\text{rbf}} \otimes K_{\text{rbf}}(h) + \Lambda_\varepsilon \otimes I_N$，通过 Cholesky 分解求解 $\operatorname{vec}(A) = \Omega^{-1} \operatorname{vec}(Y_c)$。|对偶权重矩阵 $A \in \mathbb{R}^{N \times 4}$|
|**阶段 V**|**CV1 跨环境联合推理与均值还原**|测试交叉核 $K_{\text{add}}^*, K_{\text{rbf}}^*(h)$，权重 $A$，$B_{\text{add}}, B_{\text{rbf}}$，截距 $\mu_Y$|通过矩阵乘积同时计算加性跨环境迁移项与非线性跨环境迁移项并加上环境截距：$\hat{Y}_* = \mathbf{1}_{N_*} \mu_Y + K_{\text{add}}^* A B_{\text{add}}^\top + K_{\text{rbf}}^*(h) A B_{\text{rbf}}^\top$。|最终四环境预测矩阵 $\hat{Y}_* \in \mathbb{R}^{N_* \times 4}$|

### 3. 数学定义：核函数、概率模型、目标函数与闭式估计公式

#### 基础核函数定义

设当前训练集样本数为 $N$（外层完整开发集 $N \approx 539\text{–}540$），标记数为 $P = 1,279$，环境数为 $T = 4$。原始标记矩阵记为 $X \in \{0,1\}^{N \times P}$，中心化标记矩阵记为 $X_c = X - \mathbf{1}_N \mu_X$，其中 $\mu_X = \frac{1}{N}\mathbf{1}_N^\top X$。归一化加性遗传关系核（Normalized Additive Kernel）定义为：

$$K_{\text{add}} = \frac{X_c X_c^\top}{c_{\text{add}}}, \qquad c_{\text{add}} = \frac{1}{N} \operatorname{tr}(X_c X_c^\top) = \frac{1}{N} \|X_c\|_F^2$$

对于全标记汉明距离局部 RBF 核（Hamming RBF Kernel），由于 DArT 标记严格为二值 0/1 输入，任意两份材料 $x_i, x_{i'} \in \{0,1\}^P$ 之间的汉明距离恒等于平方欧氏距离：

$$D_{ii'} = \sum_{p=1}^P |x_{ip} - x_{i'p}| = \|x_i - x_{i'}\|_2^2$$

记 $d_0 = \operatorname{median}_{1 \le i < i' \le N}(D_{ii'})$ 为训练集上三角两两汉明距离的中位数，四环境共享带宽参数 $h \in \{0.25, 1, 4\}$，则局部非线性 RBF 核定义为：

$$K_{\text{rbf}}(h)_{ii'} = \exp\left( -h \frac{D_{ii'}}{d_0} \right)$$

在上述定义下，$\frac{1}{N}\operatorname{tr}(K_{\text{add}}) = 1$ 且 $\frac{1}{N}\operatorname{tr}(K_{\text{rbf}}(h)) = 1$，两个基础核在训练集上严格处于相同的单位平均对角线方差尺度。

#### 闭式解析留一（Analytical LOO）信号提取算子

给定当前训练集按普通 ICM 规则选定的基准参数组 $(h, w_0, \lambda_0)$，定义基准复合核为 $K_0 = w_0 K_{\text{add}} + (1 - w_0) K_{\text{rbf}}(h)$。对 $m \in \{\text{add}, \text{rbf}, 0\}$，对应单核或复合核在训练集上的核岭平滑算子（Hat Matrix）为： $$H_m = K_m \left( K_m + \lambda_0 I_N \right)^{-1} \in \mathbb{R}^{N \times N}

$$根据 Sherman-Morrison-Woodbury 秩一更新留一恒等式（Gianola & Schön，2016）[cite: 1]，在无需逐个样本重拟合模型的前提下，第 $i$ 个训练材料在第 $t$ 个环境下的精确闭式留一预测残差 $e_{\text{LOO}, i, t}^{(m)}$ 与留一遗传预测值 $\hat{g}_{\text{LOO}, i, t}^{(m)}$ 由下式给出：$$e_{\text{LOO}, i, t}^{(m)} = \frac{y_{c, i, t} - \left[ H_m y_{c, \cdot, t} \right]_i}{1 - (H_m)_{ii}}, \qquad \hat{g}_{\text{LOO}, i, t}^{(m)} = y_{c, i, t} - e_{\text{LOO}, i, t}^{(m)}$$利用该闭式留一预测，可在训练集内部一次性导出三项无样本内过拟合偏差（In-sample bias-free）的校准统计量。第一项是**环境特异加性/非线性信号亲和比 $\pi_{\text{add}, t}$**：记 $r_{m, t} = \max\left(\operatorname{Corr}\left(y_{c, \cdot, t}, \hat{g}_{\text{LOO}, \cdot, t}^{(m)}\right), \, 0\right)$（$m \in \{\text{add}, \text{rbf}\}$），定义环境 $t$ 的留一加性信号相对亲和度与受控于校准超参数 $\gamma \in [0, 1]$ 的环境自适应加性核权重 $w_t(\gamma)$ 为：$$\pi_{\text{add}, t} = \frac{r_{\text{add}, t}^2 + \epsilon_0}{r_{\text{add}, t}^2 + r_{\text{rbf}, t}^2 + 2\epsilon_0} \quad (\epsilon_0 = 10^{-4}), \qquad w_t(\gamma) = (1 - \gamma) w_0 + \gamma \pi_{\text{add}, t} \in (0, 1)$$第二项是**环境特异留一遗传力与异方差残差惩罚 $\lambda_t(\gamma)$**：记环境 $t$ 在基准复合核 $K_0$ 下的留一可解释方差比（经验留一遗传力）为：$$h_{\text{LOO}, t}^2 = \operatorname{clip}\left( 1 - \frac{\|e_{\text{LOO}, \cdot, t}^{(0)}\|_2^2}{\|y_{c, \cdot, t}\|_2^2}, \; 0.10, \; 0.50 \right)$$设环境 $t$ 的原始留一噪信比为 $\nu_t = \frac{1 - h_{\text{LOO}, t}^2}{h_{\text{LOO}, t}^2}$，四环境噪信比的几何均值为 $\bar{\nu}_{\text{geom}} = \left(\prod_{t=1}^T \nu_t\right)^{1/T}$，则环境 $t$ 的异方差正则化参数定义为：$$\lambda_t(\gamma) = \lambda_0 \left( \frac{\nu_t}{\bar{\nu}_{\text{geom}}} \right)^\gamma$$该幂次归一化形式严格保证四环境惩罚系数的几何均值恒等于基准惩罚 $\lambda_0$，即 $\left(\prod_{t=1}^T \lambda_t(\gamma)\right)^{1/T} \equiv \lambda_0$，从而在不改变全局平均正则化尺度的前提下，按环境间相对信噪比进行精确的异方差重分配 [cite: 6, 10]。第三项是**分核遗传相关收缩靶标 $\tilde{C}_{\text{add}}(\gamma), \tilde{C}_{\text{rbf}}(\gamma)$**：记原始中心化训练表型相关矩阵为 $C_Y = \operatorname{Corr}(Y_c) \in \mathbb{R}^{T \times T}$，加性留一预测与 RBF 留一预测的跨环境相关矩阵分别为 $C_{\text{add}} = \operatorname{Corr}\left(\hat{G}_{\text{LOO}}^{(\text{add})}\right)$ 与 $C_{\text{rbf}} = \operatorname{Corr}\left(\hat{G}_{\text{LOO}}^{(\text{rbf})}\right)$。为抑制有限样本下相关矩阵的估计方差，构造如下凸收缩相关矩阵 [cite: 6, 28]：$$\tilde{C}_{\text{add}}(\gamma) = (1 - 0.5\gamma) C_Y + 0.5\gamma C_{\text{add}}, \qquad \tilde{C}_{\text{rbf}}(\gamma) = (1 - 0.5\gamma) C_Y + 0.5\gamma C_{\text{rbf}}$$

由于 $C_Y, C_{\text{add}}, C_{\text{rbf}}$ 均为对称半正定且对角元全为 $1$ 的相关矩阵，其凸组合 $\tilde{C}_{\text{add}}(\gamma)$ 与 $\tilde{C}_{\text{rbf}}(\gamma)$ 恒保持对称半正定且对角元严格等于 $1$。

#### 双核共区域化概率模型、RKHS 目标函数与推理公式

定义对角尺度矩阵 $D_{\text{add}}(\gamma) = \operatorname{diag}\left(\sqrt{w_1(\gamma)}, \dots, \sqrt{w_T(\gamma)}\right)$ 与 $D_{\text{rbf}}(\gamma) = \operatorname{diag}\left(\sqrt{1 - w_1(\gamma)}, \dots, \sqrt{1 - w_T(\gamma)}\right)$。给定加性跨环境共享强度 $\eta_{\text{add}} \in [0, 1)$ 与非线性跨环境共享强度 $\eta_{\text{rbf}} \in [0, 1)$，加性共区域化矩阵 $B_{\text{add}}$ 与非线性共区域化矩阵 $B_{\text{rbf}}$ 分别定义为：

$$B_{\text{add}}(\eta_{\text{add}}, \gamma) = D_{\text{add}}(\gamma) \left[ (1 - \eta_{\text{add}}) I_T + \eta_{\text{add}} \tilde{C}_{\text{add}}(\gamma) \right] D_{\text{add}}(\gamma)$$

$$B_{\text{rbf}}(\eta_{\text{rbf}}, \gamma) = D_{\text{rbf}}(\gamma) \left[ (1 - \eta_{\text{rbf}}) I_T + \eta_{\text{rbf}} \tilde{C}_{\text{rbf}}(\gamma) \right] D_{\text{rbf}}(\gamma)$$

在多输出高斯过程概率生成模型视角下（Bonilla 等，2007；Rakitsch 等，2013），观测表型向量化后的联合分布为：

$$\operatorname{vec}(Y) = \operatorname{vec}(\mathbf{1}_N \mu_Y) + \operatorname{vec}(F_{\text{add}}) + \operatorname{vec}(F_{\text{rbf}}) + \operatorname{vec}(\mathcal{E})$$

$$\operatorname{vec}(F_{\text{add}}) \sim \mathcal{N}\left(\mathbf{0}, \, \sigma_g^2 B_{\text{add}} \otimes K_{\text{add}}\right), \quad \operatorname{vec}(F_{\text{rbf}}) \sim \mathcal{N}\left(\mathbf{0}, \, \sigma_g^2 B_{\text{rbf}} \otimes K_{\text{rbf}}(h)\right), \quad \operatorname{vec}(\mathcal{E}) \sim \mathcal{N}\left(\mathbf{0}, \, \sigma_g^2 \Lambda_\varepsilon(\gamma) \otimes I_N\right)$$其中异方差残差协方差矩阵为 $\Lambda_\varepsilon(\gamma) = \operatorname{diag}\left(\lambda_1(\gamma), \dots, \lambda_T(\gamma)\right)$。在频率派向量值再生核希尔伯特空间（RKHS）变分正则化视角下（Álvarez 等，2012）[cite: 4]，上述后验均值估计严格等价于最小化如下双核异方差正则化泛函：$$\min_{F_{\text{add}}, F_{\text{rbf}} \in \mathbb{R}^{N \times T}} \sum_{t=1}^T \frac{1}{\lambda_t(\gamma)} \left\| y_{c, \cdot, t} - f_{\text{add}, \cdot, t} - f_{\text{rbf}, \cdot, t} \right\|_2^2 + \operatorname{tr}\left( F_{\text{add}}^\top K_{\text{add}}^{-1} F_{\text{add}} B_{\text{add}}^{-1} \right) + \operatorname{tr}\left( F_{\text{rbf}}^\top K_{\text{rbf}}(h)^{-1} F_{\text{rbf}} B_{\text{rbf}}^{-1} \right)$$令总协方差矩阵 $\Omega \in \mathbb{R}^{NT \times NT}$ 为 [cite: 2, 4]：$$\Omega = B_{\text{add}}(\eta_{\text{add}}, \gamma) \otimes K_{\text{add}} + B_{\text{rbf}}(\eta_{\text{rbf}}, \gamma) \otimes K_{\text{rbf}}(h) + \Lambda_\varepsilon(\gamma) \otimes I_N$$

通过对称正定 Cholesky 分解 $\Omega = L L^\top$ 求解线性方程组 $\Omega \operatorname{vec}(A) = \operatorname{vec}(Y_c)$，即可获得对偶系数矩阵 $A \in \mathbb{R}^{N \times T}$（按列主序重排）。对于 CV1 设定下完全未见的测试材料 $X_*$，利用克罗内克积矩阵乘法恒等式 $(B \otimes K_*) \operatorname{vec}(A) = \operatorname{vec}(K_* A B^\top)$，其四环境预测矩阵 $\hat{Y}_* \in \mathbb{R}^{N_* \times T}$ 由如下闭式矩阵方程给出：

$$\hat{Y}_* = \mathbf{1}_{N_*} \mu_Y + K_{\text{add}}^* A B_{\text{add}}(\eta_{\text{add}}, \gamma)^\top + K_{\text{rbf}}^*(h) A B_{\text{rbf}}(\eta_{\text{rbf}}, \gamma)^\top$$

### 4. 统计合理性：在 599 个材料的条件下如何限制自由度？

在 $N \approx 540$（内层三折训练子集仅 $N_{\text{in}} \approx 360$）、$T=4$ 的高维小样本条件下，历史模型失败的首要统计原因在于有效自由度膨胀导致内层选模方差超过了模型偏差的降低量（例如 `背景条件化联合收缩` 引入了 $16 \times 4 = 64$ 个 PCA 门控参数，`选择性分核＋OOF 可靠性` 对 4 个环境分别独立搜索基础核参数导致超参数搜索自由度翻了 4 倍）。BH-LMC 通过四道统计防线严格限制模型自由度：首先，模型将基础核超参数 $(h, w_0, \lambda_0)$ 与普通 ICM 在同一训练子集上的选模结果完全锁定，不允许各环境独立选择各自的 RBF 带宽 $h_t$，从根源上消除了跨环境核空间失配与内层选模方差膨胀；其次，四个环境的核比重 $w_1, \dots, w_4$ 与四个环境的异方差惩罚 $\lambda_1, \dots, \lambda_4$ 并非作为 8 个自由超参数进行盲目网格搜索，而是由闭式解析留一统计量完全确定相对方向、仅由**单一标量收缩参数 $\gamma \in \{0.0, 0.5, 1.0\}$** 控制偏离普通 ICM 的步长；再次，对任意 $\gamma \in [0, 1]$ 和任意 $(\eta_{\text{add}}, \eta_{\text{rbf}})$，第 $t$ 个环境的对角先验方差块恒为 $(B_{\text{add}})_{tt} K_{\text{add}} + (B_{\text{rbf}})_{tt} K_{\text{rbf}}(h) = w_t(\gamma) K_{\text{add}} + (1 - w_t(\gamma)) K_{\text{rbf}}(h)$，其对角线平均迹严格恒等于 $w_t(\gamma) + 1 - w_t(\gamma) \equiv 1$，从而保证了边际遗传方差严格守恒，避免了神经网络残差模型中因预测尺度 $k$ 偏离最优 $k^* = r$ 而引发的 RMSE 退化；最后，在固定基础核后，内层三折仅需在 $\eta_{\text{add}} \in \{0.0, 0.3, 0.6, 0.8\}$、$\eta_{\text{rbf}} \in \{0.0, 0.3, 0.6, 0.8\}$ 与 $\gamma \in \{0.0, 0.5, 1.0\}$ 构成的紧凑网格上按四环境平均验证 PCC 选模，有效新增超参数仅 2 个（分核共享差 $\eta_{\text{add}} - \eta_{\text{rbf}}$ 与校准强度 $\gamma$）。

### 5. 模型关系：与现有基线及最近邻文献的精确区别与退化性质

下表精确界定了 BH-LMC 与项目中已评测基线及最近邻文献模型的数学退化条件与本质差异：

|**对比模型**|**协方差 / 预测算子数学形式**|**BH-LMC 退化至该模型的精确条件**|**BH-LMC 相比该模型的本质突破**|
|---|---|---|---|
|**普通 ICM（当前最强基座，0.52019）**|$B \otimes \left(w_0 K_{\text{add}} + (1-w_0) K_{\text{rbf}}(h)\right) + \lambda_0 I_T \otimes I_N$，其中 $B = (1-\eta)I_T + \eta C_Y$|令 $\eta_{\text{add}} = \eta_{\text{rbf}} = \eta$ 且 $\gamma = 0$，则 $B_{\text{add}} = w_0 B$，$B_{\text{rbf}} = (1-w_0) B$，$\Lambda_\varepsilon = \lambda_0 I_T$，**100% 精确退化为普通 ICM（机器精度 $10^{-15}$）**。|打破单核可分离限制，允许加性与上位效应以不同强度（$\eta_{\text{add}} \neq \eta_{\text{rbf}}$）及不同环境比重（$w_t(\gamma)$）跨环境迁移，并对高噪声环境施加异方差惩罚 $\lambda_t(\gamma)$。|
|**单环境加性＋RBF 核岭（0.50923）**|各环境独立求解 $\left(w_t K_{\text{add}} + (1-w_t) K_{\text{rbf}}(h_t) + \lambda_t I_N\right) \alpha_t = y_{c,\cdot,t}$|令 $\eta_{\text{add}} = 0, \eta_{\text{rbf}} = 0$，此时 $B_{\text{add}}$ 与 $B_{\text{rbf}}$ 均为对角阵，系统完全解耦为 4 个独立的单环境加性＋RBF 核岭模型。|引入跨环境加性与上位双通道协同正则化，利用环境间遗传协方差对单环境对比噪声方向进行收缩降噪。|
|**选择性分核＋OOF 可靠性（0.5107）**|各环境独立选基础核参数 $(h_t, w_t, \lambda_t)$，并以单核三折 OOF 预测相关系数乘积 $r_t r_{t'}$ 加权任务相关矩阵|不退化至其带缺陷的 $r_t r_{t'}$ 双重衰减形式；当 $\gamma=1$ 且固定共享基础核时，BH-LMC 保留其分核思想但彻底修复了其两项数学缺陷。|保持全环境共享同一 RBF 带宽 $h$ 以确保 RKHS 空间对齐；用闭式解析 LOO 替代高方差内层三折 OOF；消除 $r_t r_{t'} \approx 0.25$ 造成的非对角共享强度崩溃。|
|**谱选择性 ICM（0.52035）**|$B_s \otimes K(K+\tau I)^{-1}K + I_T \otimes (K - K_s) + \lambda_0 I_{NT}$|令 $\tau=0$ 且 $\eta_{\text{add}} = \eta_{\text{rbf}}, \gamma=0$ 时二者精确交汇于普通 ICM。|谱选择性 ICM 在同一复合核 $K$ 的对易特征基 $U$ 上按特征值大小拆分；BH-LMC 则按**非对易的生物学遗传机制（加性 $K_{\text{add}}$ vs 上位 $K_{\text{rbf}}$）**拆分。|
|**Cuevas 等 (2017/2018) `MUC` 模型**|$\Sigma_E \otimes K + \operatorname{diag}(\sigma_{\varepsilon, 1}^2, \dots, \sigma_{\varepsilon, T}^2) \otimes I_N$（$K$ 仅取 `GB` 或仅取 `GK`）|当 $w_0 = 1$（纯加性）或 $w_0 = 0$（纯高斯）时，BH-LMC 对应 `MUC-GB` 或 `MUC-GK` 的经验贝叶斯核岭形式。|同时融合 `GB`（$K_{\text{add}}$）与 `GK`（$K_{\text{rbf}}$）两个核并赋予各自独立的环境协方差结构 $B_{\text{add}}, B_{\text{rbf}}$，无需缓慢的 MCMC 采样。|

### 6. 计算可行性与复杂度分析

针对 Wheat599 的数据规模（外层开发集 $N \approx 540$，内层三折训练子集 $N_{\text{in}} \approx 360$，标记数 $P = 1,279$，环境数 $T = 4$），BH-LMC 在时间和空间复杂度上均具备极高的计算效率。在阶段 I 至阶段 III 的基础核构造与闭式解析 LOO 提取中，计算 $K_{\text{add}}$ 与汉明距离矩阵 $D$ 的时间复杂度为 $O(N^2 P)$，当 $N=540, P=1279$ 时仅涉及约 $3.7 \times 10^8$ 次基础浮点运算，耗时不足 $0.02$ 秒；对 $K_{\text{add}}$ 与 $K_{\text{rbf}}(h)$ 执行 $N \times N$ 对称特征分解的时间复杂度为 $O(N^3)$（$540^3 \approx 1.57 \times 10^8$），耗时不足 $0.05$ 秒，且特征分解完成后，任意 $\lambda_0$ 下的帽子矩阵对角线 $(H_m)_{ii} = \sum_{k=1}^N U_{m,ik}^2 \frac{\lambda_{m,k}}{\lambda_{m,k} + \lambda_0}$ 与闭式解析 LOO 预测均可在 $O(N^2)$ 时间内（$< 0.002$ 秒）瞬时完成。在阶段 IV 与阶段 V 的耦合系统求解中，由于 $K_{\text{add}}$ 与 $K_{\text{rbf}}(h)$ 互不对易，当 $\eta_{\text{add}} \neq \eta_{\text{rbf}}$ 或 $\gamma > 0$ 时，总协方差矩阵 $\Omega \in \mathbb{R}^{NT \times NT}$ 为三项克罗内克积之和，直接采用对称正定 Cholesky 分解求解的时间复杂度为 $O((NT)^3)$，空间复杂度为 $O((NT)^2)$。在 Python/NumPy 实测基准中，当外层重拟合 $N=540, T=4$（即 $NT = 2,160$）时，$\Omega$ 矩阵内存占用仅为 $2160^2 \times 8 \text{ Bytes} \approx 37.3 \text{ MB}$，单次完整组装加线性求解仅耗时 **$0.4120$ 秒**（内层三折 $N_{\text{in}}=360, NT=1,440$ 单次求解耗时仅约 **$0.08$ 秒**）。因此，完成全部 10 个外层折的内层三折网格搜索与外层重拟合，在单台标准 CPU 上总运行时间仅需 **2 至 4 分钟**，远快于需要训练 100 个 epoch 的 BiMSGP、MoEGP 或 FT-Transformer。

### 7. 创新边界声明

为恪守科学严谨性，必须明确界定本方案中已有机制与新增设计的边界：线性共区域化模型（LMC，$\sum_q B_q \otimes K_q$）的泛函形式属于 Álvarez 等 (2012) 的经典框架；加性与上位效应具有不同环境协方差的遗传学认识源于 Burgueño 等 (2007)；利用帽子矩阵对角线计算核岭回归闭式解析留一误差源于 Gianola & Schön (2016)；异方差多环境残差源于 Cuevas 等 (2017) 与 Rakitsch 等 (2013)。在此基础上，本研究真正新增的建模机制是：**针对高维小样本多环境 CV1 基因组预测，提出将闭式解析留一（Analytical LOO）算子嵌入二阶双核（加性＋汉明 RBF）LMC 的单参数经验贝叶斯收缩校准机制**——通过单一标量塑性参数 $\gamma \in [0, 1]$ 同步驱动“边际方差守恒下的环境特异加性/上位核比重分配”、“几何均值锚定下的环境异方差残差惩罚校准”以及“无双重衰减的分核遗传相关靶标收缩”，并严格保证在 $\gamma=0, \eta_{\text{add}}=\eta_{\text{rbf}}$ 时精确退化至普通 ICM。该设计相对于普通 ICM 的实际预测增益目前属于**有待集中实验验证的科学假说**，在实验执行前绝不宣称为已证实事实。

## 第四部分：创新性判断

### RQ4 核心解答：数学等价性核查与先例边界界定

为防止将已有方法的简单改写或平凡组合误判为算法创新，本节对 BH-LMC 与统计遗传学及机器学习中的最近邻方法进行严格的数学等价性核查。

首先必须检验的核心问题是：**BH-LMC 是否会像普通 ICM 或谱选择性 ICM 一样，在平衡 CV1 设计下代数退化为某个单一复合核上的典型特征任务岭回归？数学反证表明：当且仅当 $\eta_{\text{add}} \neq \eta_{\text{rbf}}$ 或 $\gamma > 0$ 时，BH-LMC 严格不可约化为单核模型**。在第一部分中已证明，普通 ICM 的协方差矩阵 $B \otimes K + \lambda I_{NT}$ 之所以能通过 $B = V S V^\top$ 解耦为 $T=4$ 个在同一核矩阵 $K$ 上独立的典型任务岭回归，其充要条件是环境空间与样本空间的耦合仅含单一克罗内克积项且噪声矩阵与 $I_T$ 成比例。考察 BH-LMC 的总协方差算子：

$$\Omega = B_{\text{add}}(\eta_{\text{add}}, \gamma) \otimes K_{\text{add}} + B_{\text{rbf}}(\eta_{\text{rbf}}, \gamma) \otimes K_{\text{rbf}}(h) + \Lambda_\varepsilon(\gamma) \otimes I_N$$

根据矩阵代数中“多个对称矩阵可通过正交或相合变换同时对角化当且仅当它们两两广义对易”的充要条件：在样本维度（$N \times N$）上，由于 $K_{\text{add}}$（中心化标记线性内积矩阵）与 $K_{\text{rbf}}(h)$（非线性指数汉明距离矩阵）满足 $K_{\text{add}} K_{\text{rbf}}(h) \neq K_{\text{rbf}}(h) K_{\text{add}}$，不存在任何正交矩阵 $U \in \mathbb{R}^{N \times N}$ 能同时对角化 $K_{\text{add}}$、$K_{\text{rbf}}(h)$ 与 $I_N$；在环境维度（$T \times T$）上，仅当 $B_{\text{add}}$ 与 $B_{\text{rbf}}$ 严格线性相关（即 $B_{\text{add}} = w_0 B$ 且 $B_{\text{rbf}} = (1-w_0) B$，对应 $\eta_{\text{add}} = \eta_{\text{rbf}}$ 且 $\gamma = 0$）时，前两项才能合并为单一克罗内克积 $B \otimes (w_0 K_{\text{add}} + (1-w_0) K_{\text{rbf}})$。因此，只要 $\eta_{\text{add}} \neq \eta_{\text{rbf}}$（加性效应与上位效应具有不同的跨环境共享强度）或 $\gamma > 0$（不同环境具有不同的加性/非线性比重 $w_t(\gamma)$ 或不同的噪声方差 $\lambda_t(\gamma)$），BH-LMC 的求解算子在数学上就严格无法退化为任何单一复合核上的独立或典型任务核岭回归，而是真正实现了不同环境特征方向对加性核与非线性核的动态差异化借力。

下表将 BH-LMC 的各个组成部分严格归类到“成熟方法的实现”、“已有方法的组合”、“工程计算优化”与“真正新增的建模机制”四个层级，以杜绝夸大创新：

|**维度分类**|**在本方案中的具体对应部分**|**创新性定性判断**|
|---|---|---|
|**1. 成熟方法的实现**|归一化加性核 $K_{\text{add}}$、中位数缩放汉明距离 RBF 核 $K_{\text{rbf}}(h)$、基于帽子矩阵 $H = K(K+\lambda I)^{-1}$ 的闭式留一误差公式 $e_{\text{LOO}, i} = (y_i - \hat{y}_i)/(1 - H_{ii})$，以及 $NT \times NT$ 块矩阵 Cholesky 求解。|属于统计遗传学（Crossa 2010；Gianola & Schön 2016）与向量值核方法（Álvarez 2012）的标准成熟组件，**不单独宣称创新**。|
|**2. 已有方法的组合**|将多输出线性共区域化模型（LMC：$B_{\text{add}} \otimes K_{\text{add}} + B_{\text{rbf}} \otimes K_{\text{rbf}}$）与多环境异方差残差（Cuevas 2017：$\Lambda_\varepsilon \otimes I_N$）结合用于小麦多环境预测。|属于机器学习 LMC 理论与数量遗传学 $A \times E + A \times A \times E$ 混合模型的交叉整合，**属于有明确机制依据的跨领域方法组合**。|
|**3. 工程计算优化**|利用预先计算的基础核特征分解 $K_m = U_m \Lambda_m U_m^\top$ 在 $O(N^2)$ 时间内完成任意 $\lambda_0$ 的闭式 LOO 信号提取，并将固定基础核后的 $2160 \times 2160$ 耦合系统求解控制在 0.41 秒内。|属于保障嵌套交叉验证零泄漏与高吞吐率的数值计算工程优化，**不作为建模原理创新**。|
|**4. 真正新增的建模机制**|**单参数约束下的闭式留一信号分解与边际方差守恒共区域化机制**：针对小样本 CV1 中自由估计双核 LMC 协方差矩阵（$B_{\text{add}}, B_{\text{rbf}}, \Lambda_\varepsilon$ 共需 $\frac{4\times 5}{2} \times 2 + 4 = 24$ 个方差组分参数）必然过拟合或导致非对角共享衰减的难题，设计了以普通 ICM 为严密退化锚点（$\gamma=0$）、由闭式解析留一统计量完全确定各环境核亲和度 $w_t(\gamma)$、相对信噪比惩罚 $\lambda_t(\gamma)$ 及遗传相关矩阵 $\tilde{C}_m(\gamma)$ 的低自由度耦合算子。|**具有独立的方法学价值。** 解决了“如何在不增加内层切分方差、不破坏跨环境核对齐、不破坏边际方差尺度的前提下，实现多核异构共区域化与异方差收缩”这一具体统计学习问题。|

在明确独立价值的同时，还必须确立**最重要的潜在反证（Falsification Condition）**。如果该架构在 Wheat599 实验中出现以下两种情形之一，则其新增机制将被直接证伪：第一种反证情形是**内层选模完全退化**，即在 10 个外层开发折的内层三折选模中，有 8 折或以上同时选中 $\gamma = 0$ 且 $\eta_{\text{add}} = \eta_{\text{rbf}}$，这将直接证明在 Wheat599 的样本规模下，区分加性/上位共享强度与环境异方差未能提供超越普通 ICM 的内部交叉验证信号；第二种反证情形是**验证增益无法迁移至外层测试集**，即内层三折选模虽频繁选中 $\gamma > 0$ 或 $\eta_{\text{add}} \neq \eta_{\text{rbf}}$，但完整开发集重拟合后在外层测试集上的宏均 PCC 相比普通 ICM（`0.52019`）提升不足 `+0.0030`（如同谱选择性 ICM 的 `+0.00016`），或者因环境 1、2 出现明显负迁移而抵消了环境 4、5 的增益，这将证实在 $N=599, T=4$ 的完全区组 CV1 设计下，单核普通 ICM 的典型特征方向收缩已经穷尽了可泛化的跨环境线性借力上限。

## 第五部分：Wheat599 实验方案

本实验方案专为科研工程 Agent 制定，遵循“集中验证、预算明确、一次完整候选评价、不做无意义的事后消融或扩网格刷分”的科研准则，确保在严格统一的 CV1 协议下给出明确的 GO / NO-GO 结论。

### 1. 数据、划分与防泄漏硬性规范

实验唯一允许使用的数据集为 BGLR `wheat.RData`（文件 SHA-256 校验码为 `8710523389007dd8b7c88a5deb74ee16aea0bae158496bc11f0626fb43b5ea21`），工程实现必须使用列主序 `reshape(shape, order='F')` 加载并校验缓存布局印记，输入严格限定为 $599 \times 1,279$ 的二值 0/1 DArT 标记矩阵 $X$（严禁引入谱系矩阵 $A$），输出为原始环境列 `"1"`、`"2"`、`"4"`、`"5"` 构成的籽粒产量表型矩阵 $Y \in \mathbb{R}^{599 \times 4}$。外层与内层数据划分必须严格复用仓库现有的固定外层十折划分清单（`splits_manifest.json`，由 seed 42 生成的材料索引随机排列，包含 9 折 60 个测试材料与 1 折 59 个测试材料）以及与 `configs/multienv_kernel_nested.yaml`（普通 ICM）和 `configs/spectral_icm_nested.yaml` 完全一致的开发集内层三折切分索引。在防泄漏规范上，外层测试材料的全部四环境表型在标记中心化、表型中心化、迹归一化常数 $c_{\text{add}}$ 计算、汉明距离中位数 $d_0$ 估计、闭式解析留一（LOO）统计量计算、内层超参数选择及外层重拟合的任何环节均完全不可见；所有统计量必须仅在当前拟合子集（内层训练折或完整外层开发集）内部独立重新计算。

### 2. 模型配置与两阶段嵌套选模流程

为彻底消除基础核参数差异带来的归因混淆（汲取此前 `选择性分核＋OOF 可靠性` 因独立重选基础核导致性能下降的教训），BH-LMC 严格采用与普通 ICM 同构的嵌套选模与重拟合流程，具体规格如下表所示：

|**执行步骤**|**作用数据集与样本规模**|**核心计算与参数选择规则**|**冻结与传递输出**|
|---|---|---|---|
|**步骤 1：基础核基准参数锚定**|各内层训练子集（$N_{\text{in}} \approx 360$）及完整外层开发集（$N \approx 540$）|完全复用既有普通 ICM 在对应训练子集上的基础核参数选定规则（带宽 $h \in \{0.25, 1, 4\}$、基准加性比例 $w_0 \in \{0, 0.25, 0.5, 0.75, 1\}$、基准岭参数 $\lambda_0$）；若既有实验已落盘各折选定的 $(h, w_0, \lambda_0)$，可直接读取并通过退化断言核对一致。|当前训练子集的基准参数 $(h, w_0, \lambda_0)$|
|**步骤 2：闭式解析 LOO 统计量提取**|与步骤 1 相同的当前训练子集内部|基于当前子集选定的 $(h, w_0, \lambda_0)$，按第三部分公式计算平滑矩阵 $H_{\text{add}}, H_{\text{rbf}}, H_0$，闭式提取留一遗传预测 $\hat{G}_{\text{LOO}}^{(\text{add})}, \hat{G}_{\text{LOO}}^{(\text{rbf})}, \hat{G}_{\text{LOO}}^{(0)}$，算出环境亲和比 $\pi_{\text{add}, t}$、相对噪信比 $\nu_t / \bar{\nu}_{\text{geom}}$ 及相关矩阵 $C_Y, C_{\text{add}}, C_{\text{rbf}}$。|当前训练子集的闭式校准统计量元组|
|**步骤 3：开发集内层三折联合选模**|外层开发集的内层三折（每折训练 $\approx 360$，验证 $\approx 180$）|在紧凑网格 $\eta_{\text{add}} \in \{0.0, 0.3, 0.6, 0.8\} \times \eta_{\text{rbf}} \in \{0.0, 0.3, 0.6, 0.8\} \times \gamma \in \{0.0, 0.5, 1.0\}$（共 $4 \times 4 \times 3 = 48$ 组配置）上求解 $\Omega$ 并预测内层验证折；按**内层三折四环境平均验证 PCC 最高**选出每个外层折的最优超参数 $(\eta_{\text{add}}^*, \eta_{\text{rbf}}^*, \gamma^*)$（平局优先选 $\gamma$ 更小、$|\eta_{\text{add}} - \eta_{\text{rbf}}|
|**步骤 4：完整开发集重拟合与单次测试评价**|完整外层开发集（$N \approx 540$）拟合，外层测试集（$N_* = 59\text{ 或 }60$）单次预测|使用完整外层开发集重新计算中心化统计量、基础核、闭式解析 LOO 统计量，代入步骤 3 选定的 $(\eta_{\text{add}}^*, \eta_{\text{rbf}}^*, \gamma^*)$ 组装 $4N \times 4N$ 矩阵 $\Omega$，求解对偶系数 $A$ 并对未见测试材料执行单次预测，生成全部 $40/40$ 个环境折与 $2,396$ 条 OOF 预测记录。|完整逐折指标表、OOF 预测表与超参数选择审计日志|

### 3. 实施前强制正确性检查（Sanity & Degeneration Checks）

在正式运行完整十折评测或读取任何外层测试集指标之前，科研工程 Agent 必须先执行并记录三项无需查看测试集标签的代码正确性断言：第一项是**普通 ICM 退化恒等断言（ICM Degeneration Identity Check）**，即固定 $\gamma = 0.0$ 且 $\eta_{\text{add}} = \eta_{\text{rbf}} = 0.6$，在全部 10 个外层开发集上运行 BH-LMC 重拟合，其输出的测试集预测值 $\hat{Y}_*$ 与既有普通 ICM（`results/multienv_kernel_final_20261102/`）预测值的最大绝对差必须满足 $\|\hat{Y}_*^{\text{BH-LMC}} - \hat{Y}_*^{\text{ICM}}\|_\infty < 10^{-12}$；第二项是**零共享单环境退化断言（Zero-Sharing Degeneration Check）**，即固定 $\gamma = 0.0$ 且 $\eta_{\text{add}} = \eta_{\text{rbf}} = 0.0$，其预测结果与使用相同 $(h, w_0, \lambda_0)$ 的单环境加性＋RBF 核岭回归预测值的最大绝对差必须小于 $10^{-12}$；第三项是**矩阵正定性与边际迹守恒断言（PSD & Trace Conservation Check）**，即对网格内全部 48 组 $(\eta_{\text{add}}, \eta_{\text{rbf}}, \gamma)$，断言 $\operatorname{diag}(B_{\text{add}} + B_{\text{rbf}}) = \mathbf{1}_4$（最大绝对误差 $< 10^{-15}$），且 $\Omega$ 的 Cholesky 分解在全部 40 个环境折中 100% 成功、0 失败。

### 4. 对照基线、评估指标与结果判定准则（Pre-registered GO / NO-GO Criteria）

实验以普通 ICM（宏均 PCC `0.52019`，RMSE `0.85449`，MAE `0.65784`）作为首要性能对比基线，同时保留谱选择性 ICM（`0.52035`）、选择性分核＋OOF 可靠性（`0.5107`）、单环境加性＋RBF 核岭（`0.50923`）、Gaussian 核岭（`0.5025`）、BiMSGP（`0.4708`）、RR-BLUP（`0.4599`）及 MoEGP（`0.4592`）作为完整参照系。结果报告必须同时给出四环境折均宏平均 PCC、RMSE、MAE（主表口径，先算每折指标再对十折取均值 ± 样本标准差 `ddof=1`）以及合并 OOF（2,396 条记录按环境 pooled 计算）的宏均指标，并详细列出相对普通 ICM、选择性分核＋OOF 可靠性及单环境加性＋RBF 的 40 个同折配对差（$\Delta\text{PCC}, \Delta\text{RMSE}, \Delta\text{MAE}$）与更优折数占比（`X/40`），同时披露十个外层折选中的 $(\eta_{\text{add}}^*, \eta_{\text{rbf}}^*, \gamma^*)$ 频次分布及完整开发集上闭式解析 LOO 估计出的四环境 $h_{\text{LOO}, t}^2$ 与 $w_t(\gamma^*)$ 均值 ± 标准差。

下表定义了在运行测试集评价前预注册的完整结果判定规则（GO / NO-GO Criteria）：

|**判据编号**|**判定维度**|**预设量化标准**|**科学含义与统计目的**|
|---|---|---|---|
|**C1（主精度突破）**|宏平均 PCC 增益与折间一致性|$\text{PCC}_{\text{BH-LMC}} - 0.52019 \ge +0.0030$（即宏均 PCC $\ge 0.5232$），且在非平局折中 PCC 更优折数占比 $> 55\%$|确证新架构产生了超越谱选择性 ICM（`+0.00016`）数值噪音波动的实质性跨材料排序增益，且增益并非仅由 1–2 个异常折驱动。|
|**C2（误差校准维持）**|宏平均 RMSE 与 MAE 约束|相对普通 ICM 的 $\Delta\text{RMSE} \le 0.0000$（即 RMSE $\le 0.85449$）且 $\Delta\text{MAE} \le +0.0005$|确保相关性提升未以破坏预测尺度校准（如残差 FT-T 放大尺度失配项 $\sigma_y^2(k-r)^2$）为代价。|
|**C3（无严重负迁移）**|四环境表现与难点环境突破|至少 3/4 个环境 $\Delta\text{PCC} \ge 0$，环境 4 与环境 5 的平均 $\Delta\text{PCC} > +0.0030$，且任一单环境最大回撤不劣于 $-0.0030$ PCC|验证双核分路与异方差校准确实改善了此前普通 ICM 停滞的强非线性/低信噪比环境（环境 4、5），且未重蹈“选择性分核＋OOF”损害环境 1、2 的覆辙。|

最终决策规则如下：若完整候选模型一次性同时满足 **C1、C2 与 C3**，则判定 **GO = True**，确立 BH-LMC 为 Wheat599 新的最强基座模型；若任一判据未通过，则判定 **GO = False**，保留普通 ICM（`0.52019`）为主基线，**严禁事后扩大超参数网格、拆分消融子模块或根据测试集表现微调公式重新运行**。

## 第六部分：研究风险与最终取舍

本节仅保留四项真正可能改变本研究科学结论的实质性限制与最终取舍依据。

第一项实质性限制是**同一套固定划分多轮迭代导致的“研究选择偏差（Researcher Degrees-of-Freedom / Adaptive Overfitting）”无法通过重新随机划分现有材料消除**。当前 `seed-42` 外层十折划分已先后用于 FT-Transformer 多个变体、残差修正、双向诱导注意力、BiMSGP、MoEGP、背景条件化核、单环境核岭、普通 ICM、分核 OOF 及谱选择性 ICM 等十余轮完整架构研发，且部分早期冒烟测试曾观测过折 0 的测试集指标。因此，即便 BH-LMC 在本次实验中通过全部预设判据并超越 `0.52019` PCC，该结果在统计学认识论上仍属于**同一固定划分上的探索性协议内比较（Exploratory Comparison）**，绝不能宣称为全新独立确认结果。更为关键的是，由于全部 599 份小麦材料在过去多轮模型筛选中均已轮流作为外层测试集参与了建模方向的自适应淘汰，**仅仅更换随机种子对现有 599 份材料重新进行随机十折划分，并不能消除针对这 599 份材料整体经验分布的研究选择偏差**；未来的真正独立确认（Independent Confirmation）必须冻结最终模型架构与选模程序，并在此前从未参与本课题开发的全新小麦育种群体或新环境试验数据上进行盲测。

第二项实质性限制是 **CV1 平衡完全区组设计的理论信息论天花板**。在 CV1 设定下，测试材料在全部四个环境中均无表型观测，模型无法像 CV2（稀疏测试）那样利用同一测试材料在已测环境中的表型残差直接插补未测环境。因此，CV1 下多环境模型相比单环境模型的理论最大收益，严格受限于“跨环境遗传协方差对单环境参数估计方差的缩减量”以及“加性核与非线性核在环境间的真实相关结构差异度”。普通 ICM 已实现了 `+0.0110` PCC 的方差缩减；若 BH-LMC 的双核异方差机制最终未能带来 $\ge +0.0030$ PCC 的进一步突破，则将提供强有力的实证证据表明：在 $N=599, T=4$ 的完全区组 CV1 条件下，单核典型任务收缩已逼近多任务核回归的信息论上限，此时应果断终止在 $4 \times 4$ 任务矩阵上的进一步复杂化改造。

第三项实质性限制是**小样本下留一遗传相关矩阵 $C_{\text{add}}, C_{\text{rbf}}$ 与留一遗传力 $h_{\text{LOO}, t}^2$ 的有限样本估计方差**。在约 360 个样本的内层训练子集上，单核闭式留一预测虽然消除了样本内拟合偏差，但其跨环境相关矩阵仍存在有限样本波动。BH-LMC 已通过将其与表型相关矩阵 $C_Y$ 进行 $0.5\gamma$ 凸收缩、对留一遗传力施加 $[0.10, 0.50]$ 截断保护、并在内层网格中纳入 $\gamma=0$ 退化锚点来系统对冲该风险；若个别折因验证集与测试集的分布波动（如历史诊断显示的折间均值/尺度异质性） 选偏 $\gamma$，仍可能在少数折产生轻微的方差损耗。

第四项是关于**彻底舍弃深度神经网络与样本级局部核门控的最终取舍**。综合本项目全部历史实验证据，在无可靠物理染色体坐标的 $599 \times 1,279$ 二值 0/1 DArT 标记矩阵上，任何试图在特征空间学习 $P \times P$ 自由标记交互（FT-Transformer、诱导注意力、BiMSGP）或引入样本级非线性路由门控（MoEGP、PCA16 四门控背景核）的方案，均因参数自由度与样本量严重失配而显著劣于无门控核岭回归。因此，本研究明确舍弃一切神经混合架构与局部样本门控，将全部验证预算集中于物理意义清晰、自由度严格受控且包含普通 ICM 为精确退化特例的 **BH-LMC** 架构。