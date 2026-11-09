# 异构教师指导的分位数 TabM

本轮只检验一种训练方法：把原分位数TabM与Huber ICM的预固定0.5融合教师预测，蒸馏进完整的原分位数TabM。学生保持四环境独立、原多成员机制、99分位点与全标记预测路径；推理不调用教师。现役冻结融合外层PCC 0.526506不变，教师并不会因被蒸馏而自动被学生超越。

依据为知识蒸馏及交叉拟合，而非新的神经骨干。[Knowledge Distillation as Semiparametric Inference](https://www.microsoft.com/en-us/research/publication/knowledge-distillation-as-semiparametric-inference-paper/)（2021）讨论教师过拟合与交叉拟合；[Distillation of Tabular Foundation Models into Efficient Predictors](https://arxiv.org/abs/2610.01435)（2026）给出TabM等学生的蒸馏证据，但其全训练上下文、合成查询条件不同，不作为本项目全上下文或CutMix的依据。本轮不使用外部模型权重、标记拼接增强或新数据集。

## 教师边界

每个既有开发单元有学生fit和val。只在学生fit内部做三折交叉拟合；每个材料的四环境标签全部从为其生成目标的teacher train中排除。API `fit_teacher(x_teacher_train, y_teacher_train, query_x)`不接收query标签，当前学生val及外层test也不进入该API。

每个teacher train内部独立生成三个调参划分，QTabM沿用原八配置、双种子、逐环境配置选择和内折最佳轮数中位数，重新拟合teacher train再预测query。Huber沿用既有材料核、eta、delta和ridge multiplier候选；每个教师内折的材料核基础参数只在其训练子集再三折选择，Huber24候选以教师内验证的四环境宏PCC选择；最后在整个teacher train内重选基础核、重估标准化/环境关系/MAD并重拟合。所有核参数、停止轮数和配置都不能从现成外层OOF或学生val选模记录复制。

教师输出先各自回到原表型尺度，再固定等权平均。交叉目标覆盖当前studentfit每个材料恰好一次，保存teacher QTabM、Huber、融合三种输出、全套嵌套ID、教师配置/轮数与权重哈希。交叉目标的诊断指标只作记录，不参与教师或学生搜索规则的修改。交叉教师拟合材料更少，它们的实际预测质量不等同于全开发集教师0.526506。

## 学生与唯一搜索

学生结构不变，配置固定为原开发比较每外折每环境已选配置；对应每个内折/种子重建原最佳checkpoint，必须与保存的原验证预测最大绝对差≤1e-6。教师目标的生成不使用这些学生选择记录。模型初始化和数据顺序采用原种子/配置索引，不能从外层权重初始化。

真实表型与教师值都使用当前studentfit的同一均值、标准差。损失为逐成员 `sum_99(pinball) + 99*lambda*(mean_99(q)-teacher_z)^2`，再在材料和成员上平均。蒸馏项只作用于每个成员的分位数平均值，不把99个头都训练成同一个值。lambda候选预固定为0、0.01、0.1、1；lambda=0是相同微调流程下的额外训练对照。每臂从同一个原模型权重重新开始，重置AdamW状态，学习率固定为原配置的0.1倍，weight_decay、batch128、全梯度范数裁剪5不变；所有lambda使用相同微调RNG和batch顺序。100 epoch上限，patience20，按单模型验证PCC早停，保留第0轮原模型候选。

每外折每环境按三内折、双种子平均预测PCC选择lambda，平局优先较小lambda。同时报告原模型、lambda0继续训练、选中学生、原固定融合。新增搜索带来的开发选择优势不作为泛化证据；不能仅以“学生比原模型好”归因蒸馏，必须同时超过lambda0。

## 固定预算与停止条件

30个学生开发单元（10折×3内折），每个单元三个teacher crossfit块：

| 工作 | 数量 |
|---|---:|
| 神经教师内层选模 | 17,280 |
| 神经教师固定轮数重拟合 | 720 |
| Huber候选求解 | 6,480 |
| Huber固定重拟合 | 90 |
| 学生原checkpoint重建 | 240 |
| 学生微调（含lambda0） | 960 |

此外原材料核`choose`在教师各内层fit及teacher train重做，不增加其原候选集合。总计19,200次神经拟合；大部分预算用于未见标签的教师目标生成，而非新增学生架构搜索。两个学生种子和全部lambda复用同一份双种子融合教师目标。输入数据与已生成目标留内存使用，教师权重和结果落盘供核验，不反复重算。

在看到结果前固定外层条件：选中学生的开发PCC严格高于原分位数、原固定融合和lambda0继续训练三者，同时RMSE、MAE不劣于原分位数。未满足则结束本配置，不追加lambda/LR搜索。满足才生成仅依赖开发记录的外层两阶段重拟合方案；不直接使用旧外层OOF作为训练目标。外层替代目标仍为冻结融合0.526506；已有划分反复用于研究，应限定结论范围。

执行入口：`python scripts/run_background_nested.py --config configs/distill_quantile.yaml --out results/<新目录>`。只保留这一正式入口，无外部教师数据或权重。当前教师、学生配方的代码与数据/划分来源SHA随provenance记录；失败显式保存，不过滤。

执行记录：初版`9ad8424`首次启动在第一条教师轨迹写记录时，教师内折与学生内折同名`inner`触发异常；产物保留omen `/home/hsy/worktrees/distill-9ad8424/results/distill_quantile_development_20261109/`，为实现失败，不作模型效果证据。修正仅将教师日志字段命名为`teacher_inner`，学生划分仍为`inner`，目标函数与预算未变；新增真实小问题的两层上下文记录回归测试。正式运行使用独立的新目录，不覆盖该失败记录。

修正版本`7a1e989`已在omen独立工作树`/home/hsy/worktrees/distill-7a1e989`启动，正式结果目录`results/distill_quantile_development_20261109_v2`。本机和omen各7项测试通过；真实开发checkpoint重建与旧验证预测最大差为0。ds独立复核未发现信息边界或损失实现错误。当前仍在运行，尚无蒸馏效果结论。
