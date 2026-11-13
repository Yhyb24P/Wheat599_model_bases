# 查询条件化的标记证据读取网络

状态：完整候选及开发比较入口已实现；完成软件测试和合成输入训练链路验证，尚无 Wheat599 正式开发或外层结果。当前最优固定融合不变。本候选从头初始化，不保留 TabM 或 Huber 的预测函数。

## 完整计算路径

四环境独立建模，每个环境每个种子一套网络。全部 1,279 个标记进入模型；0 是观测状态，不作 padding，也不把列顺序解释为物理距离。

1. 标记按当前拟合集均值/总体标准差处理，拟合集常量列归零。标记 j 的 token 为共享两层 GELU MLP 作用于 `b_j + z_j w_j`；b、w 为每个标记的可学习向量。没有跨标记自注意力或全局预池化。
2. 连续分位水平 τ 编码为 `[cos(π r τ)]`，r=0,…,32，再经过两层共享 GELU MLP。τ 必须在 (0,1) 内，支持任意非空一维查询列表。
3. 两层 Pre-LN 交叉注意力：query 逐层更新，K/V 来自同一份材料标记记忆；每层后接残差 ReGLU FFN。直接复用 `induced_ftt.PreLNCrossAttention`，隐藏宽度 d，4 头，FFN 隐层 2d。无查询间自注意力、查询集合池化或跨材料注意力。
4. 共享读出 `LayerNorm → Linear(d,d) → GELU → Linear(d,1)` 输出每个查询的标准化分位值。标准接口为 `[batch, 1, queries]`，中间的 1 是单成员兼容轴，不是假称保留了 TabM 内部集成。

在 eval 模式下，同一个 τ 的输出不随其他查询、查询顺序、重复查询或无关批次材料变化（浮点误差容差内）。训练时 Dropout 有随机性，不要求不同调用的随机掩码逐元素一致。查询处理独立并不保证分位数单调，也未验证区间覆盖率。

## 训练与预设开发预算

复用 `quantile_tabm.train/run` 的损失、逐种子早停、三内折选模、预测存储、失败记录和汇总，不复制训练器。损失对 99 个分位点求和，对材料和成员轴平均；点预测对分位点平均、反标准化后再平均种子 42/2026。固定网格仍为 0.001–0.999 的 99 点。

预设八配置：宽度 64/128 × Dropout 0/0.1 × 学习率 0.0005/0.002；均为两次读取、4 头、AdamW weight decay 0.01。每次最多 500 epoch、patience 50、batch 128、梯度裁剪 5。沿用既有十外折的三个内折；开发预算上限为 30 × 4 × 8 × 2 = 1,920 次拟合，不追加配置。当前实现完成不代表这批拟合已经运行。

参考预测复用已核对的 Huber、原分位数 TabM 和原固定分位数–Huber 融合。汇总中 `quantile` 是本候选、`original_quantile` 是原分位数、`original_fusion` 是原分位数固定融合；历史 `fixed_fusion` 键仍是 MSE TabM 的固定融合，不能混称。不自动增加候选–Huber 融合或进入外层。

候选只有一个成员，原 TabM 有 16/32 个成员，骨干和参数共享方式也不同；首批只能比较完整系统，不能把任何差异单独归因于查询参与的时间位置。旧诱导注意力配置及层级条件网络的负结果不等于这个新候选的结果。

```bash
python scripts/run_background_nested.py --config configs/query_quantile.yaml --out results/query_quantile_development_<日期>
```

配置中的 `model: quantile_tabm` 选择现有分位开发执行器，`optimizer_variant: query` 选择新架构并使用 AdamW；模型本身不是 TabM。必须准备原始数据、既有划分与基线预测；输出新目录，不能覆盖历史产物。生成 provenance 同时记录新模型和复用交叉注意力模块的 SHA。

## 实现检查

测试覆盖两层读取与编码/读出的梯度、查询单独/联合/置换/重复推理一致性、材料批次隔离、保留全部标记、拟合统计量固定、常量列安全、checkpoint 恢复、初始化种子重置，以及完整十折三内折开发执行器的合成产物保存与双种子预测聚合。合成测试不作为 Wheat599 性能证据。

## 方法来源与边界

[IQN](https://proceedings.mlr.press/v80/dabney18a.html)提供连续分位编码的已有依据，本候选不使用其 Bellman 更新；[Perceiver IO](https://arxiv.org/abs/2107.14795)已有查询驱动输出机制。[Jeva 决策块](https://github.com/LiQiiiii/Jeva/blob/e2104e0720100f979f2cdd456a1cbbee96da0abd/src/jeva/_runtime/matrix_decision.py)与[Clef 源码](https://huggingface.co/Cloudflare/clef/raw/main/joint_schema_model.py)是读取方向的参考，不导入语言模型权重，也不声称复现它们。这里不实现候选集合反向改写基因型表示。

可检验的差异是：τ 在标记证据汇总之前进入计算。交叉注意力、连续查询编码或两层读取本身不是原创性证据，注意力也不直接构成因果位点解释。
