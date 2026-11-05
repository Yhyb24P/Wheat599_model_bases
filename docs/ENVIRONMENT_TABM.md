# 环境共享方式的 TabM 开发比较

2026-11-05 用户授权。冻结代码 `d10c9cb`、标签 `environment-tabm-protocol-20261105`，配置 `configs/environment_tabm.yaml`。omen正式新增2400fit与shared480条复用已完成、0失败，墙钟2232.25秒；结果见 [RESULTS.md](RESULTS.md)。独立系统开发PCC提高，条件化未建立收益；本轮配置预算结束，不追加训练／消融／外层评价。以下为实际执行协议。

## 比较与边界

三种完整系统：shared为原有四输出全共享TabM；independent为四个独立单环境TabM；conditioned为保留TabM成员机制并在每隐藏线性层加入环境输入／输出乘性缩放u/v的网络。u/v全1时应恢复shared的eval预测，保留对应环境的原输出head。新增参数以全1初始化，权重偏离1的均方惩罚固定为0.001，普通weight decay不作用于u/v，避免将其拉向0。该正则不进入额外搜索。

机制沿用 [BatchEnsemble](https://research.google/pubs/batchensemble-an-alternative-approach-to-efficient-ensemble-and-lifelong-learning/) 的共享矩阵乘性调制，不作为新集成原理。[Mul-PheG2P](https://pmc.ncbi.nlm.nih.gov/articles/PMC13539974/) 提供性状特异学习与预测空间融合的领域依据，但其机制并非本候选隐藏层调制，不能替代实际验证。全体表型相关仅为提出问题的历史背景，不进入模型初始化、任务分组、损失权重或任何先验；不预先认定存在负迁移。

对照精度锚为**固定**0.5原始TabM＋0.5 Huber ICM，继续使用前一轮原始TabM，不改成候选的新融合。已有 `grid_scores[2]` 的开发PCC为0.490091761562，10/10开发集高于TabM、9/10高于ICM；逐集选权重仅额外增加0.000351240573。这是无新增训练的开发描述，不是独立确认。

## 预算与选择

延用已冻结的8组深度／宽度／成员数／dropout／学习率／weight decay配置、种子42/2026、10开发集×3内层、max500epoch、patience50、逐成员MSE。shared完整480条已有曲线与预测，验证数据／划分／源码／配置后复用，不重复训练。conditioned新增480次fit，independent每环境480次共1920次single-env fit，新增总计2400次；四个single-env fit算一个完整预测系统单元时，必须另外报告实际训练次数与成本，不隐去四倍计算。

shared和conditioned按三内层种子预测平均后的macro PCC选择每开发集共同配置。independent按各环境相同8配置预算独立早停和选模，再拼成四环境系统；具有更灵活的环境独立选择空间，结果中须明确。逐seed分数均值和预测集成指标分开。保存逐环境与宏PCC/RMSE/MAE及预测ID，不以平均改善替代逐环境发现。判定依据是完整系统的同协议预测表现，不把参数节省当收益。

预测CSV采用统一long格式，每行一个材料／环境／seed／配置的标量预测。环境列保留数据原标号`1,2,4,5`，对应四个输出列的顺序；报告使用位置命名Env1–Env4时须给出此对应，不能把原标号4误认为输出第四列。

仅使用每内层fit的X/Y统计，val用于早停/选模，outer test不参与。epoch0规则与旧shared一致；conditioned全1初始与shared的可比性须测试。best state内存保存、流式SHA、不持久化大权重。失败显式保留并中止，不吞掉选模异常。所有正式执行继续SSH omen，先贯通与成本测量，再本地提交/标签冻结；共享GPU不超配置显存份额。

本预算结束后按结果判断当前共享假设；不自动扩搜索、做检索追加、RAGP校准或进入外层测试。总体论文目标是否完成须由完整证据判断。

## 配置选择自由度重汇总（后续用户授权，无新训练）

四独立单环境TabM提升为新的神经参照，当前环境乘性调制方案结束。为区分配置选择与训练方式，使用原有全配置验证预测交叉比较shared／independent训练及common／per_environment配置选择，原有每条轨迹checkpoint完全固定，不重新早停。每开发集先平均两个seed预测，求各环境PCC并平均三内层；共同选择最大宏平均配置，分别选择各环境最大配置，接近并列（1e-12）取配置列表靠前者。不逐inner独立取优。

代码 `src/wheat599/environment_selection.py`、配置 `configs/environment_selection.yaml`，原始预测SHA锁定，严格核对全部配置/seed/material/environment键、数据与划分。两已知corner必须复现shared共同0.485389和independent分别0.491874（1e-8内）才落盘。输出 `results/environment_selection_development_20261105/`，继续只在omen CPU执行，不启动GPU或训练。保留两种训练方式下原早停差异；即使选模自由度对齐，剩余差异仍包括容量、梯度目标和checkpoint选择，不能自动解释为负迁移。后续架构训练待此证据判断，本次未授权新后期共享模块训练。

重汇总已在e3708ca完成：shared共同／分别0.485389／0.487909，independent共同／分别0.490810／0.491874。共同配置下独立高0.005421、逐环境配置下高0.003965。配置选择的代数优势在两臂都成立，但没有消除独立系统优势；仍不将残余全部归因为去共享。无新训练，无重新早停，无外层评价。模型与参照地位以最新RESULTS/RESEARCH_PLAN为准。

## 写入分工

cn唯一写 `src/wheat599/environment_tabm.py`、`tests/test_environment_tabm.py`；gb唯一写 `src/wheat599/environment_run.py`、`tests/test_environment_run.py`并负责omen执行；gpt66统筹、配置、已有CLI分派、文档、Git与结果判断。常规进度只写文件，仅实质问题或完整结果通知用户。
