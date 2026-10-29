# Stage 1 Plan — TabPFN-3.5 vs 小型 FT-Transformer（first glance）

**状态：** 2026-10-29 用户确认两阶段；本文件是 Stage 1 冻结对象。Stage 2 完整六方法协议见 `docs/STAGE2_PLAN.md`，未冻结。

## 1. 目的

在 BGLR Wheat599 上做一次无泄漏、可复现的 first-glance 比较：冻结的 TabPFN-3.5 ICL 与小型 FT-Transformer 相对 Ridge 参照是否有值得进入 Stage 2 的信号。**Stage 1 是门控，不是发表推断；结果不并入 Stage 2 推断，也不作稳定改进声明。**

## 2. 方法

| 方法 | 配置 | 选择 |
|------|------|------|
| Ridge 参照 | `sklearn.linear_model.Ridge`，`fit_intercept=True` | α 对数网格，内层 CV 选择 |
| 小型 FT-Transformer | d∈{16,32}、层数 {1,2}、dropout {0,0.1}、学习率 {1e-3,3e-3}、权重衰减 {1e-5,1e-4}；无位置编码；二值 token 编码 $t_j(0)=b_j$、$t_j(1)=b_j+w_j$ | 32 配置，内层 CV 选择；AdamW/batch 64/patience 20/max 200；refit epochs = 内层 best epoch 中位数 |
| TabPFN-3.5 冻结 ICL | `tabpfn==9.2.0`，`ModelVersion.V3_5`，checkpoint `tabpfn-v3.5-20260909.safetensors`，不微调 | 默认配置，不调参；记录 device、成员数、子采样配置 |

FT-Transformer 的 d=16/32、1–2 层是**本项目针对 $n\approx480$ 的自主小模型选择**，不是论文/rtdl 默认；论文默认 3 层、embedding 192、8 头，调参空间层数 $[1,4]$、embedding $[64,512]$。rtdl 的 `LinearEmbeddings` 是逐特征 weight/bias，`token_j = x_j*w_j + b_j`，对 0/1 即 `t_j(0)=b_j`、`t_j(1)=b_j+w_j`；实现无需自定义 tokenizer。

## 3. 数据与划分

- 数据：BGLR `wheat.RData`，SHA-256 `8710523389007dd8b7c88a5deb74ee16aea0bae158496bc11f0626fb43b5ea21`；四环境分别报告，列名 `"1"`、`"2"`、`"4"`、`"5"`；marker-only。
- 划分：random 外层 5 折 × 3 次重复（种子 20261029、20261030、20261031）；内层 3 折。分层键 = 4 环境标准化 Y 的行均值 5 分位；每层随机 offset。
- 无亲缘臂、无 A 臂、无 `wheat.sets`（这些属于 Stage 2）。

## 4. 无泄漏底线

- α、FT 配置、早停、refit epochs 全部只在外层训练折内完成；内层选择指标 = RMSE（逐环境、3 折均值）。
- 泄漏注入测试仍须通过：故意全样本标准化 Ridge 必须被拒绝；guarded slice/closure leak 与 stage-specific 测试通过。
- 每个材料在每个重复中恰好一个 OOF 预测；外层留出折不参与任何选择。

## 5. 指标与 first-glance 报告

- 每个环境、方法、重复计算 PCC、RMSE、MAE、R²；主看 PCC 与 RMSE。
- 外层折按 4 环境标准化 Y 的行均值 5 分位（`row_mean_y_5bin`）分层，这是降低划分方差的常规做法；first-glance 区间条件于该划分构造，不把分层当纯随机。
- 每个对比（T1 = TabPFN − Ridge；T2 = FT-Transformer − Ridge）在每个环境报告：
  - 15 个 fold 级配对差逐值落盘（3 重复 × 5 折）；
  - 15 值的均值与 [最小, 最大]；
  - df=14 近似 $t$ 区间，固定标注：材料复用/折间相关未修正、方向偏乐观、只作幅度指示、不构成稳定改进声明。
- 改善方向统一为“正 = 更好”：$\Delta$PCC = PCC(model) − PCC(Ridge)；$\Delta$RMSE = RMSE(Ridge) − RMSE(model)。
- TabPFN 与 FT-Transformer 之间也做同材料配对差，描述性。

## 6. Stage 1 信号门（gb 口径，先于正式运行写死）

1. **δ 估计（冒烟阶段）：** 用冒烟运行估计 fold 级配对差的标准差 SD；$\delta_{\mathrm{PCC}} = t_{0.975,14}\times \mathrm{SD}/\sqrt{15}$，取整入文本；兜底 $\delta_{\mathrm{PCC}}=0.03$，$\delta_{\mathrm{RMSE}}=0.05$。δ 与出处写入 `results/<stage1_run>/trigger.md`，正式运行前固定，禁止看结果后调整。
2. **判定规则：** T1、T2 各自按环境计算 15 个 fold 级配对差均值；若某对比在 ≥3/4 环境均值 >0 且跨环境均值 ≥ $1.2\delta$，记为正向信号。$[0.8\delta,1.2\delta)$ 为留痕带，不触发。
3. **指标规则（gb 裁决，OR）：** T1/T2 的 PCC 与 RMSE 各自判定；任一指标满足“≥3/4 环境均值 >0 且跨环境均值 ≥ $1.2\delta$”，即记为正向信号。每指标用自己的 δ（PCC 兜底 0.03、RMSE 0.05，先验化，禁看结果后改）。触发时把**触发指标来源**（PCC、RMSE 或双）写入 `trigger.md`；若一指标过 $1.2\delta$ 而另一指标在 ≥3/4 环境 ≤ $-\delta$（判别与校准对立），触发成立，负向指标入记录，Stage 2 判读必须处理该张力，不得只报正面。
4. **弱信号：** 两指标均未达触发带（$<1.2\delta$），但任一指标跨环境均值落在 $[0.8\delta,1.2\delta)$ 且 ≥3/4 环境同向 → 记“弱信号留痕、未触发”；留痕不等于放行。
5. **动作映射（穷举）：**
   - 任一对比正向信号（任一指标）→ 建议冻结 Stage 2，并在冻结申请中命名该信号对应的已注册判断与触发指标；
   - 均不成立 → 记录负结果；是否投入完整 Stage 2 由用户决定；
   - 灰区/留痕带 $[0.8\delta,1.2\delta)$ → 视为不成立但留痕；
   - ≥3/4 环境 < $-\delta$ → 记录负向信号，供用户再裁。
   - 第一眼的任何触发只建立“进入第二阶段”，不建立任何正向声称；“稳定改进”字样保留给 Stage 2 判据。
6. **落盘：** `trigger.md` 包含 δ 与出处、T1/T2 判定表、触发指标来源、对立形态记录、弱信号/灰区/负向记录、触发结论；它是 Stage 2 冻结申请的必备附件。G1 覆盖检查包含 trigger.md。

## 7. 预算

- FT-Transformer：32 配置 × 4 环境 × 15 外层拟合（内层 3 折）；TabPFN：15×4 次前向。
- GPU 上限约 **2 GPU-hours**；CPU 优先；超过上限停止并报告，不静默延长。
- 冒烟先跑：数据 SHA、1 折/小网格、device 记录、泄漏注入测试。

## 8. 结果定位与非声称

- **探索性 first glance**；不作稳定改进声明；不并入 Stage 2 推断。
- 不声称生物学上位互作、因果基因或标记效应；不声称对更大小麦群体/其它物种的泛化；不声称育种选择响应或经济收益。
- TabPFN 结论仅针对 checkpoint `tabpfn-v3.5-20260909`；权重非商用。

## 9. 与 Stage 2 的关系

Stage 2 完整协议（六方法、8 主对比、10/5 重复、random + IBS 亲缘臂、LightGBM secondary）保留在 `docs/STAGE2_PLAN.md`，**未冻结**。只有 Stage 1 触发正向信号后，才启动 Stage 2 G0（gb/iq 签核 + 用户授权提交）；Stage 2 的 §6.1 判据改为重复级 df=9 口径。

## 10. 文件与版本

- 配置：`configs/stage1.yaml`。
- Stage 2 配置：`configs/stage2.yaml`（保留，不执行）。
- 版本锚点：`docs/G0_VERSION.md`（正式运行前更新）。