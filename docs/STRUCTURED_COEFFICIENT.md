# 结构化系数先验的稳健多环境回归

本轮只开发完整候选：`Y*=1 b^T + R* A`，以现有材料划分比较结构化系数先验、多环境共享和Huber损失组成的整体系统。现役冻结分位数TabM–Huber外层PCC 0.526506不变；本轮不自动进入外层、扩展数据集或追加其他基线。

依据：[GWKBR论文](https://doi.org/10.1093/bib/bbag133)及[作者代码d9019e0](https://github.com/Wangxuer521/GWKBR/blob/d9019e012cae4d7207eab8c318942450fded67d3/GWKBR.py)。原文Wheat四环境最终均选择unweighted，因此不以该结果证明标记加权有效。作者K_Model后验解为`(R^T R + lambda_eff G)^-1 R^T y`，G约束系数精度；其对应的有效函数核为`R G^-1 R^T`，与加性/RBF相加有实质区别。文献成绩不与项目结果直接排名。

## 固定模型定义

- 0/1 DArT保持原始编码。G沿用项目`kernels()`的拟合集中心化二值标记Gram矩阵，并归一化为平均对角1；不使用作者的二倍体剂量频率解释。
- `P=G+epsilon I`，epsilon固定候选0.001、0.01、0.1（相对G平均对角1的明确单位），作为统计模型的一部分选择，不作为隐藏jitter。P正定。
- RBF为`exp(-h*d^2/d0)`，d0仅用拟合集两两平方距离中位数。h为0.25、1、4。查询行始终对同一拟合集参考列计算R*，不重新选择基函数。
- 每环境表型仅用拟合集均值、总体标准差归一化。环境矩阵`B=(1-eta)I+eta Corr(Y_fit)`，eta候选0、0.3、0.6、0.8。
- 在标准化表型尺度，优化`sum Huber/(4n) + lambda/2 trace(B^-1 A^T P A)`，b为不惩罚的截距。lambda候选1e-6、1e-5、1e-4、0.001、0.01、0.1、1。这里lambda是归一化目标系数；调用原Huber求解器的ridge严格取`lambda*(4n)`。
- Huber delta为1.345、2.5乘以当前配置拟合集初始岭解残差MAD（沿用现有Huber实现与1e-6尺度下限），阈值在优化前固定，不读取验证标签。

P的Cholesky为L时，以`D=R L^-T`白化，`theta=L^T A`。任务协方差再因子化为B，使用现有`robust_icm.fit_predict()`，复用原损失/梯度；不满足最终证书时用SciPy trust-ncg及同一目标的精确Hessian-vector积进一步收敛。D的拟合集列均值可吸收到自由截距，不改变模型。SVD仅去掉相对最大奇异值<=1e-12的数值零方向，查询使用同一旋转和均值。保存时转换回原A及b，逐配置核对`R* A+b`预测。

## 作者基础方法参照

参照为作者未加权K_Model后验公式，显式提供同一二值P；fit表型中心化/标准化后求解，再反标准化，不另拟合截距。有效ridge候选0.001、0.01、0.1、1、10、100、1000，吸收sigma_e2/sigma_a2。各环境分别按三内折PCC选择h/epsilon/ridge。

该参照不是作者完整GWKBR流水线的原样复现：不运行GWAS、REML或100次贝叶斯搜索，不使用其二倍体G构造。它用于核对基础结构机制与整体预测比较，不据此声称已超过论文完整方法。作者仓库该提交无LICENSE文件，未vendoring/分发作者代码；独立公式实现已与指定提交的`K_Model._compute_posterior()`实际方法数值核对（显式传入P、方差比为1），最大预测差9.47e-14，源SHA及说明见`results/structured_coefficient_upstream_20261110/verification.json`。

## 开发预算和验收

30单元（10外层开发集×3内折），候选每单元3带宽×3epsilon×7lambda×4eta×2delta=504次四环境凸拟合，总15,120次；参照每单元63次四目标解析计算，总1,890次（等价7,560单环境计算）。参照和候选均保存全部预测及系数；候选按12个“内折×环境”PCC平均选一个完整配置，不拆环境拼接最优。参照允许各环境独立选配置，其额外选择自由度公开记录。

主参照复用同材料原分位数TabM、Huber及固定等权融合预测，目标为开发融合PCC 0.494512且误差竞争，不把大网格择优增益直接称为泛化改进。只评价完整模型，不增加消融或新一轮搜索。

每次候选拟合必须通过原始梯度阈值1e-5，以及Huber Fenchel对偶证书：对偶score满足每环境零和与Huber盒约束（残差<=1e-12）；相对间隙`(P-D)/max(1e-12,abs(P),abs(D))`必须在[-1e-10,1e-6]。初始L-BFGS-B最多1000次、ftol1e-15、gtol1e-12；若证书未达标，固定同一阈值、lambda及起点后使用trust-ncg最多1000次（gtol1e-14）。初始阶段仅提供数值起点，最终仍严格检查原梯度、可行性及对偶间隙，达到迭代上限不自动视为收敛。非有限值、无定义PCC或精度不达标显式失败，不过滤配置。

必要测试：白化与直接结构化后验/有效核等价、与直接多环境原始问题对照、奇异G加epsilon、查询基函数不依赖其他查询、系数重载及对偶可行性。正式运行前可仅在训练侧检查数值精度/耗时，不用验证成绩调求解容差。

正式入口`python scripts/run_background_nested.py --config configs/structured_coefficient.yaml --out results/<新目录>`，omen两个单线程CPU进程，不占GPU。数据/划分/源代码/配置/引用预测SHA写入provenance。实现与数值验收已完成，正式开发配置按本文件冻结；尚无新性能结果。

训练侧验收（不读取验证标签）：真实首个拟合集，epsilon=0.001/0.1、lambda=1e-6/1、eta=.8四个边界配置通过原始梯度与相对对偶间隙检查。相对间隙最大2.26e-9；进一步收紧ftol至1e-16、gtol至1e-13后，拟合集预测最大变化1.76e-6。该检查仅用于数值精度与耗时，记录位于omen /tmp/structured_training_check.json，不作为性能证据。本机/omen各4项相关测试通过。

正式执行冻结`1f77591`、标签`structured-coefficient-dev`，omen工作树`/home/hsy/worktrees/structured-coefficient`，输出`results/structured_coefficient_development_20261110`。两个CPU工作进程各单线程，正式预算已启动；完成后重算全部候选选择、重载原始系数并复核目标与对偶证书，再交付性能结果。

首轮正式执行`1f77591`在一个配置相对对偶间隙2.617e-6时按预定阈值1e-6停止，完整单元0/30，不作性能证据；原目录保留。后续仅补充基于原Huber目标及Hessian-vector积的trust-ncg数值精化，不改变模型、候选集或验收阈值；从独立新目录执行。另明确n×E表示总观测数4n，ridge=lambda*y.size是正确归一化，新增字段记录ridge/normalized_lambda/number_observations并测试该等式。
