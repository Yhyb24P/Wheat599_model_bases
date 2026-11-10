# 密集多基因背景＋多环境稀疏贝叶斯树

本轮只开发一个完整候选：所有标记的高斯线性背景、各环境专属的稀疏森林、四环境相关残差联合采样。仅做既有30个开发单元，候选未通过前不进入外层。现役固定融合外层PCC仍为0.526506。

## 模型与先验

拟合集内按列标准化表型和线性背景标记；常量标记中心化后为零，仍保留输入位置。树使用原二值标记及阈值0.5，不赋予物理列序意义。查询只输入标记。

\[
Y_i=\mu+B^\top z_i+(f_1(x_i),\ldots,f_4(x_i))+\epsilon_i,
\quad \epsilon_i\sim N_4(0,\Sigma).
\]

- 所有\(B_{je}\sim N(0,1/p)\)，与\(\Sigma\)独立；\(\mu\sim N_4(0,100I)\)。不把线性和树分量命名为已识别的生物学效应。
- 各环境有M棵树。叶参数先验标准差为拟合集标准化表型范围除以\(2\kappa\sqrt M\)，与suBART的范围缩放配方等价。配置仅为(M,kappa)=(50,2)、(100,2)、(100,3)。
- 合法变量集合A由当前节点拟合行决定：两个子叶至少各5行。A非空时分裂概率为\(.95/(1+d)^2\)，变量概率为\(\pi_j/\sum_{k\in A}\pi_k\)；A为空时必为终端。
- 各环境有独立\(\pi_e\sim Dirichlet(\theta_e/p,\ldots,\theta_e/p)\)，共享于该环境的全部树；\(\theta_e/(\theta_e+p)\sim Beta(1/2,1)\)。这是DART式稀疏分裂先验，不是训练前硬筛变量。
- 沿用suBART的Huang–Wand协方差层次先验，nu=3，A由拟合集表型标准差与half-t的90%分位数校准；每轮更新辅助a和逆Wishart条件分布。它是残差协方差，不是遗传相关矩阵。

## 推断实现

成熟来源：[suBART](https://github.com/MateusMaiaDS/subart/tree/deddf4ea1290513cd6bce4e7867349d6c32fc2fa)，MIT。复用节点管理、条件残差充分统计量、叶高斯后验及层次逆Wishart更新；仅保留回归必需路径，不引入分类、缺失表型、类别重编码等未使用分支。源码及原许可在`src/wheat599/bayesian_trees/`。

扩展后的联合链依次更新：密集B、截距、四个森林、稀疏概率及其浓度、协方差。B采用观测空间精确高斯抽样；对Sigma旋转后，各通道使用固定的\(ZZ^T\)谱分解，避免p×p分解。

树只用生长/剪枝，提议概率始终各1/2，包括stump处被拒绝的剪枝；选择终端/nog的数量使用真实状态。合法变量提议与先验相同，接受率中两者抵消；叶积分似然按高斯积分计算，去掉上游每叶多余的\(-\log(2\pi)/2\)。这些是推断适配修正，不是改变预测目标。

合法变量集合的归一化会影响DART全条件：

\[
p(\pi\mid T,\theta)\propto Dir(\pi;\theta/p+N_j)
\prod_{v\in\text{internal nodes}}\left(\sum_{j\in A_v}\pi_j\right)^{-1}.
\]

因此不能直接套用Dirichlet计数更新。使用Dirichlet独立提议，并以合法集合归一化乘积作精确MH校正；概率在log域计算。另对(theta,pi)作联合提议，接受率包含Dirichlet–多项式正规化比、浓度超先验和合法集合校正，缓解高维逐块Gibbs锁定。不使用期望伪计数近似。

## 采样与评价边界

四条独立种子链，所有预热后的完整函数样本等权平均；不按验证PCC选迭代。初始预热1000、保留2000；仅依据训练侧诊断按1、2、4、8倍重跑同一种子链，最大预热8000、保留16000。

监测全部拟合材料的四环境预测、Sigma上三角、各环境浓度/分裂数及线性/森林RMS。要求rank-normalized split Rhat<=1.05、bulk ESS>=100；训练预测的最大MCSE<=拟合集表型SD的0.025倍。使用ArviZ实现；诊断未达标不能计作可用的性能结果，不删除失败单元或挑取链。采样长度不看验证成绩；查询值与查询标签均不用于混合验收。

各配置完成全部3个内部单元后，每个开发折按四环境平均PCC选择一个完整配置，平分取列表先者。所有配置与选中配置的分数分别保存。参照沿用同材料原分位数TabM（0.494565432）、Huber（0.481082622）和固定融合（0.494512272，RMSE0.870234699），不新训参照。

正式入口：

```bash
WHEAT599_R_PREFIX=/home/hsy/venvs/wheat599-r \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/home/hsy/venvs/wheat599/bin/python scripts/run_background_nested.py \
 --config configs/sparse_polygenic_bart.yaml --out results/sparse_polygenic_bart_development_20261110
```

Rcpp只编译一次，工作进程只加载已编译共享库，不并行写sourceCpp缓存。R环境与Python ArviZ版本记入运行清单。原始输入、逐链样本、完整系数和大预测仅存运行机；小型诊断、摘要、文件SHA及协议提交Git。

## 验证与当前状态

原生R测试`tests/test_bayesian_trees.R`对照完整高斯后验矩、两状态树精确后验概率、合法集合下DART先验矩、浓度超先验矩及逆Wishart全条件矩，并检查重现与查询行置换隔离。

当前为实现及训练侧采样验收阶段，尚无正式开发分数。短链试运行、并发缓存错误和未达到混合验收的样本不作性能证据。

方法依据：[suBART论文](https://arxiv.org/html/2404.02228v3)、[DART论文](https://doi.org/10.1080/01621459.2016.1264957)。稀疏先验、半参数树与相关残差分别已有研究；本轮检验的是完整预测器在相同材料留出条件下是否增加实际预测能力。
