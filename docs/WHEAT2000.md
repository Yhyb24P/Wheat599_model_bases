# Wheat2000：原独立TabM的跨数据集评价

2026-11-05新附件要求优先检验原候选的跨数据集表现，全部八性状保留，比较GBLUP、BiMSGP、DeepGS，并报告完整选择与训练成本。四方向旋转先在Wheat599完成预定开发比较，未通过则结束该变换；原TabM不依赖旋转成功才推进。

**当前执行：** 协议源码/数据/划分于`48dc908`提前冻结。GBLUP的80个trait-fold内层选择于`cb6d142`冻结后已完成外层，80性状折宏PCC0.602837、RMSE0.782494、MAE0.606879，0失败；原TabM的3840次开发拟合正在omen执行，尚不能与GBLUP外层排名。Wheat599旋转未通过，已结束，不向本数据集添加旋转臂。BiMSGP/DeepGS网络、统一两阶段runner及隔离依赖已准备，尚未冻结其正式训练预算或执行真实数据训练，不将论文数值列为本轮实测。

GBLUP开发结果位于`results/wheat2000_gblup_development_20261105/`，80项选择完整、0失败，三内折按PCC选择的lambda均在预定19点网格内，数据/配置/核心源码SHA已逐项复核。内层PCC按性状分别为TKW 0.649510、TW 0.611970、GL 0.730294、GW 0.727540、GH 0.681616、GP 0.537472、SDS 0.519290、PHT 0.319667。CPU开发墙钟7.32秒；它是共享核分解、各性状独立系数/选择的完整八性状成本。尚未完成的TabM开发及任何外层结果不与这一条件选模分数混排。

GBLUP外层证据位于`results/wheat2000_gblup_final_20261105/`，完整重拟合/预测CPU墙钟5.03秒；16000条保存OOF的全部材料、双ID、真值和十折覆盖已独立逐项核对，80项指标与summary重算一致。合并OOF PCC另为0.615484、RMSE0.788406，不与上述折均混称；逐性状合并指标另存`pooled_oof_metrics.json`。主候选尚在开发，此处没有神经架构收益或领先结论。

## 明确的数据版本

采用[官方DeepGS仓库](https://github.com/cma2015/DeepGS/tree/33ae2dc1ed7bda4e38b76460b18756649d4478a5)发布的`wheat_GS.RData`，SHA256为`e9ebecc0c73b39f4747bf0eda036da92640f694938046de557f0c9e68f926696`。它是2000×33709二值标记和2000×8表型矩阵，顺序固定为TKW、TW、GL、GW、GH、GP、SDS、PHT，八性状均2000条完整记录，不作完整表型交集筛选。原文件已是发布方处理后的矩阵，本项目不从中声称重建了原始测量或标记质控。

不能确认该文件与BiMSGP README所称未公开`wheat_GS_v2.RData`字节相同，故明确使用上述公开版本，对全部模型一致。[官方BiMSGP读取函数](https://github.com/liuxumail-create/BiMSGP/blob/71084c8fa5444fbc747b0291ace627b1f2d6ef33/bimsgp/data.py)对RData按行位置取X和pheno；两矩阵的原rownames不是同一套标识，不能按行名join。保留sample_id0..1999、原基因型line ID和原表型row ID三套索引。表型行名是非连续、非严格排序的原索引，范围1..2184，不重建成1..2000。位置对齐的依据是公开数据对象及官方读取语义，不能用一次预测正相关代替ID证据。

数据准备时gb曾在本地R随机构造十折、固定lambda=1和未中心化XXᵀ/p核上做GL闭式对齐诊断，确实拟合并计算过留出PCC；先前“no training”仅指无神经/正式协议训练，现更正。该R划分与本项目sklearn外折不同，命令和原日志未持久保存，诊断数值不进入正式主表、不据此选网络/配置/报告性状，也不声称GL留出信息完全首次接触。没有重跑该诊断。

本地R分别导出列主序和行主序X/Y，Python逐元素核对，文件MD5/SHA一致，float32二值X转换精确。缓存NPZ SHA为`6f729cec905abb4d5a1ebc59d634ef291e37f7a1605630a555897698a1123733`，omen只读取已核验缓存和原RData，无需安装R。原件和缓存留`data/raw/wheat2000/`，不入Git；小型证据见`results/wheat2000_protocol_20261105/data_provenance.json`。

## 材料与选模边界

固定KFold10、shuffle、seed42，每折test200、dev1800；dev内KFold3、shuffle、seed42+fold，每内折fit1200/val600。八性状共用同一份已核验完整材料列表，十折test恰覆盖全部材料一次。划分源码不依赖标签数值。清单`results/wheat2000_protocol_20261105/splits_manifest.json`的SHA为`4fb82c0c875c1a6639e8547a5fe98e2c30433acca1c0c3ac74b8f6f8f4b2f08c`。

每次拟合的表型与基因型预处理仅用当前fit；配置、checkpoint和停止轮数只从内层选择，最终选择记录提交/打标签后再完整dev重拟合，test不用于调整学习率或选checkpoint。原BiMSGP论文的留出折选checkpoint报告值单独列为文献结果，不能作为本协议下已超过的对手。主要比较必须来自这里相同版本、相同测试材料的新预测。

## 先冻结原TabM与GBLUP

`configs/wheat2000_tabm.yaml`原样保留现有八组官方TabM深度/宽度/成员数/正则与AdamW配置，全部33709标记、每次单输出、逐成员MSE、batch128、clip5、max500epoch/patience50，seed42/2026。八性状×十外折×三内折×八配置×两seed=3840次开发拟合，不按早期性状成绩缩减报告。每trait/fold按三内折的两seed向量平均PCC选配置；所选配置每seed三个best_epoch的整数中位数用于完整dev重拟合，保留epoch数，0合法。冻结160行重拟合清单后执行160fit，先平均成员、再平均两seed；保存16000条集成OOF、32000条seed预测、时间、显存、参数与权重SHA，权重仅留omen。开发checkpoint只记录state SHA，不存成大量权重文件。

`configs/wheat2000_gblup.yaml`使用线性标记核K=ZZᵀ/p（Z仅按当前fit均值中心化）、fit表型均值截距，各性状独立选择lambda，候选为10的−6到3次方间19个对数等距值。三内折平均PCC选lambda。固定原始marker点积可共享缓存，但中心化和截距仅依赖当前fit；共用特征核的谱分解不等于共用表型系数。选择记录冻结后再外层重拟合。NumPy实现已在合成数据上与原始标记岭回归解析解逐元素对齐，查询表型改成NaN不影响预测。

BiMSGP保持官方Wheat2000的64维/64state/L1/wide头和AMP机制，DeepGS需依据原网络作明确同协议适配；依赖/实现与可用显存核对后分别固定预算，不把未运行的论文数值冒充同协议成绩。依赖使用隔离overlay，不改现有共享环境；目前没有承诺这两个对手已执行。

实际准备的DeepGS为原R/mxnet网络的独立PyTorch结构适配（Conv8/kernel18/pool4、FC32/Sigmoid、三处dropout），33709标记时2156505参数；它不等于BiMSGP仓库仅公布结果的`deepgs_fixed`（其记录为8627649参数），不冒充后者逐位复现。原DeepGS默认6000轮/SGD与该`fixed`记录的100轮不同，正式预算需要明确区分。BiMSGP按公开L1源码实际构造1104713089参数，与论文记录1105020289相差307200；原因未确定，不归因于mamba版本。本项目保持声明和配置中的L1，不为凑计数调整结构。

Mamba隔离环境准备：torch2.14.1+cu130保持原安装，overlay为`/home/hsy/overlays/wheat2000_mamba`，使用mamba-ssm2.3.2.post1、causal-conv1d1.7.0；仅将Mamba构建标准由C++17改为C++20以匹配Torch头文件，arch8.6、MAX_JOBS2，无Torch/nvidia包副本。构建wheel SHA为`9ccd91229426bb2b2680b993af91674c71d658b5e064d70b91da9458a88d0cf6`。已通过真实依赖导入和CPU构造，尚未在占用GPU的TabM运行期间做大模型GPU探针，不能据此保证训练峰值显存或耗时。

## 报告口径

主指标为80个trait×outer-fold的等权PCC，另报逐性状十折均PCC/RMSE/MAE及合并OOF，合并和折均明确区分；原表型单位随性状不同，误差同时逐性状报告。保留全部失败，不nanmean，不以十折作独立重复构造p/CI。开发搜索、完整dev重拟合和推理成本分开计，完整八性状、双seed系统总成本与单trait单seed参数/峰值分开。成熟架构适配可以成为论文贡献，是否领先由实际同协议结果决定。
