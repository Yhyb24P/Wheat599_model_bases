# 四独立TabM：Wheat599同协议外层评价

用户附件明确授权实施。主候选为四独立单环境全标记TabM，成熟架构适配可作为论文候选，不以新增模块为准入条件。固定已有40个外层折×环境配置与两个种子；不追加超参数搜索。

**执行结果（2026-11-05，已完成）：** 清单`238fe14`、模型训练源码`4111e58`已提前提交、打标签并推送。omen完成80/80模型拟合、拟合失败0次；折均宏PCC0.508190，高于同材料BiMSGP0.470781，低于Huber ICM0.521745。80个权重和4792条逐seed预测已保存。首次基线汇总因omen未同步历史OOF而失败；补传本地原件后，`8b8dd42`的`--resume-summary`已成功接续汇总，没有重拟合或修改预测，原失败记录保留。准确状态、指标与成本统一见 [RESULTS.md](RESULTS.md)。

## 清单先冻结

从`results/environment_selection_development_20261105/selection.json`读取independent/per_environment的40组配置，逐项核对`results/environment_tabm_development_20261105/selection.json`一致。原`trajectories.json`先校验已发布SHA，再对每fold/env/seed取该配置三个inner的`best_epoch`，取整数中位数为完整开发集训练epoch数。使用best_epoch而非含等待期的epochs_run；0表示不更新、合法。**保留轮数**，不将内层3batch/epoch换算成完整开发集5batch/epoch的更新次数，不根据外层结果更改约定。

清单80行，保留3个原轮数、中位数、配置、seed、原数据环境标号1/2/4/5、完整dev/test材料ID、原数据／划分／轨迹／选择摘要。当前提取轮数min/median/max为5/13/143、无0。清单和执行代码本地提交并打标签、推送之后才允许运行外层。历史BiMSGP划分文件可不同，但各折test顺序和dev材料集合必须完全一致。

## 固定执行与输出

仅10fold×4env×2seed=80次fit，SSH omen执行。每次在完整539/540开发材料上重新拟合X/Y scaler，构建官方TabM单输出头、全部1279标记；构建前设seed，AdamW／学习率／weight decay／深度／宽度／成员数／dropout取该行配置，逐成员MSE、batch128、clip5；shuffle generator沿用seed*1000+config_index。按冻结轮数训练，不接收test标签作训练决策。每个模型训练完成只预测对应test材料。

先平均成员预测，再平均两个seed。保存80个权重于omen并记录文件SHA；保存4792条逐seed预测及2396条材料×环境OOF、80次训练轮数／时间／参数／显存和失败；权重不入Git。主指标为40环境折等权平均PCC，RMSE/MAE同口径；另报告每环境10折平均、合并599OOF的PCC与其他指标，不混称。失败明确记录并中止，不过滤或nanmean。

同折参照复用已完成结果：Huber ICM宏PCC0.521745，匹配范围GLS ICM宏PCC0.520586／RMSE0.854215，BiMSGP同协议适配0.4708。Huber、BiMSGP有保留OOF，可逐材料ID和顺序核对；精确匹配GLS40指标可由Huber的per_fold减其paired_differences（Huber−GLS）恢复，明确记录两个来源及SHA，不能误用旧multienv ICM或GET结果。比较报告40环境折与10宏折的描述性差，不作p/CI，不把历史多轮开发划分称独立最终确认。

## 融合与后续范围

本轮优先交付80fit直接预测。暂不纳入强专家标量融合，避免用已凭验证标签选过checkpoint的旧inner预测作为二层干净OOF。若后续评价融合，生成每行二层训练输入时，该行须排除在一级拟合／配置选择／停止选择之外，目标外层材料仅输入基因型；另冻结相应预算，不冒充本轮已有比较。

Wheat2000是后续跨数据集确认候选，全部八性状预先纳入统一报告；本轮不自动取数或启动其训练。Wheat599四输出是同一产量性状的四环境，Wheat2000八输出是不同性状，两者不可混称。最终SOTA主张仍需多数据集同输入、同评价层的强方法证据。

写者：gb维护`independent_freeze.py`及其测试，生成清单并负责omen执行；cn维护`independent_final.py`及测试；gpt66负责配置、CLI、清单审核冻结、云端交接与文档／Git／结果整合。无并行重复重拟合，常规折进度内部记录。
