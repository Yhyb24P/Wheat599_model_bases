# Review Protocol — Wheat599 模型基座比较

本文件规定协议冻结、结果审计与报告评审的做法。当前 active 阶段是 **Stage 1**；Stage 2 的完整协议见 `docs/STAGE2_PLAN.md`，未冻结，触发后重新走 G0。

## 角色

| 角色 | 责任 |
|------|------|
| 主导者 | 协议所有权、最终裁决、答复评审意见、维护冻结记录 |
| gb | 统计与协议评审；**关键实现与验证**（用户 2026-10-29 调整）：Stage 1 适配器、runner、泄漏合同、测试与 smoke |
| iq | 数据/环境支持与辅助实现；暂停关键文件写入，向 gb 交接 |
| law | 独立监管（用户指定）：冻结前协议文本、G1/G2 证据抽查、最终声明与证据一致性；不接管日常技术裁决 |
| ARS 评审面板 | 外部风格评审：Journal-Fit + 方法两席（methodology-focus）；用于 G0 与报告阶段 |
| 用户 | 目标、范围与资源授权；冻结后主端点/主对比/判据变更由用户决定 |

## 门（Gates）

### G0 Stage 1 协议冻结评审

在首次正式运行前执行。输入为冻结提交的 `docs/STAGE1_PLAN.md`、`configs/stage1.yaml`、`docs/DATA_FACTS.md`、`docs/LICENSE_NOTES.md`、`docs/REFERENCES.md`、`docs/G0_VERSION.md` 与 stage1 测试。

检查项：

- Stage 1 的目标与定位：门控、first glance、不作稳定改进声明、不并入 Stage 2 推断；
- 方法：TabPFN-3.5 冻结 ICL（默认配置、不微调）、小型 FT-Transformer、Ridge 参照；FT-Transformer 的 d=16/32、1–2 层标注为本项目自主小模型选择，不是论文/rtdl 默认；
- 数据 SHA、四环境列名 `"1"`、`"2"`、`"4"`、`"5"`、marker-only；
- random 外层 5 折 × 3 重复；无亲缘臂、无 A 臂、无 `wheat.sets`；分层键/随机 offset/折大小/inner 3 折写明；
- 指标 PCC/RMSE、配对改善方向（$\Delta$PCC、$\Delta$RMSE）；15 个 fold 级配对差逐值、均值 [min,max]、df=14 近似 $t$ 区间与固定 caveat；
- 信号门：δ 在冒烟阶段估计并在正式运行前固定；兜底 $\delta_{\mathrm{PCC}}=0.03$、$\delta_{\mathrm{RMSE}}=0.05$；T1/T2、≥3/4 环境规则、触发带 ≥1.2δ、留痕带 $[0.8\delta,1.2\delta)$、负向规则、动作映射写明；
- 指标规则：PCC 与 RMSE 任一满足 ≥1.2δ 即触发（OR，gb 裁决）；每指标 own δ；触发指标来源、对立形态与弱信号定义写入 `trigger.md`；
- 内层选择指标 = RMSE、选择单位 = 环境；refit epochs = 内层 best epoch 中位数；AdamW/batch 64/patience 20/max 200；
- `trigger.md` 模板与落盘位置；
- `tests/` 的 fold-local 合同测试与泄漏注入用例存在，且泄漏注入必须失败；guarded slice/closure leak 与 stage-specific 测试通过；
- 2 GPU-hours 预算与耗尽规则；
- `AGENTS.md`、`STAGE1_PLAN.md`、`STAGE2_PLAN.md` 的切分清楚；gb/iq/law 签核；用户授权提交。

通过条件：gb、iq、law 各自给出书面意见；主导者逐条回应；ARS `methodology-focus` 面板评审 Stage 1；泄漏注入测试通过；所有阻断项解决后打冻结标签。冻结后任何偏离记录为 amendment；amendment 不得修改主端点、主对比、信号门或判据。

### G1 Stage 1 数据与环境审计

- `scripts/fetch_data.py` 重跑，SHA-256 与 `docs/DATA_FACTS.md` 一致；
- `data/raw/` 文件大小、对象名、维度、取值与文档一致；
- `env/` 锁文件可复现安装；GPU/CPU 运行方式与 device 记录；
- 划分文件覆盖全部 599 个材料，无重复、无遗漏；
- 3 重复 × 5 折；每折 $n$ 在 115–125；分层单元计数表；每折 4 环境 Y 均值与行标记均值检查表；
- inner 3 折互斥且覆盖 outer_train；
- 冒烟估计的 δ 与出处、`trigger.md` 模板已落盘；δ 在正式运行前固定；
- 泄漏注入测试与 stage-specific 测试通过；
- GPU 预算 ≤2 GPU-hours，超出停止并报告。

### G2 Stage 1 结果审计

- 从 OOF 预测独立重算 PCC、RMSE、MAE、R²，与结果表一致；
- 检查每个材料在每个重复、环境、方法中恰好一个 OOF 预测；
- 检查外层留出折未参与任何选择（α、FT 配置、早停、refit epochs）；
- 检查 15 个 fold 级配对差逐值落盘，均值 [min,max] 与 df=14 近似 $t$ 区间可复算；
- 检查固定 caveat 已写入：材料复用/折间相关未修正、方向偏乐观、只作幅度指示、不构成稳定改进声明；
- 检查 `trigger.md`：δ 与出处、T1/T2 判定表、灰区/负向规则、触发结论；
- 检查指标规则与冻结文本一致；
- 检查泄漏注入测试、guarded slice/closure leak 与 stage-specific 测试通过；
- 检查配置快照、种子、代码提交、数据 SHA、依赖版本完整；
- 检查负结果与不确定性如实报告；不把 Stage 1 结果写成稳定改进或 Stage 2 推断。

### G3 报告评审

- Stage 1 报告是 first glance，可由 ARS `academic-paper-reviewer` 以 `methodology-focus` 模式复核；
- 若进入 Stage 2，报告/论文按 Stage 2 G3 的 `full` 模式评审；
- 评审意见逐条回应，修改后以 `re-review` 模式复核；
- 结论必须能追溯到具体结果表与区间；明确区分"本地完成""已复核""已发布"。

### G4 发布

- `docs/RESULTS.md` 只汇总已通过 G2 的 Stage 1 first-glance 结果；
- 结果目录包含 manifest、配置、指标、OOF、`trigger.md`；
- 提交与标签在授权范围内；对外发布前确认许可边界（TabPFN 权重与 `rtdl-revisiting-models`）。

## Stage 2 门（条件执行，未激活）

Stage 2 完整六方法协议在 `docs/STAGE2_PLAN.md`。只有 Stage 1 触发正向信号后，才：
1. 由用户决定是否投入 Stage 2；
2. 重新执行 Stage 2 G0（8 主对比、10/5 重复、repeat-level df=9、random + IBS 亲缘臂、LightGBM secondary、gb/iq/law 签核、用户授权提交）；
3. Stage 2 的 §6.1 判据改为“主区间（重复级，df=9）”。

## 证据清单

| 证据 | 位置 | 检查 |
|------|------|------|
| 数据版本 | `data/raw/SHA256SUMS` | 与 `docs/DATA_FACTS.md` 一致 |
| 划分 | `results/<run>/splits/*.csv` | 覆盖、无重复、协议字段完整 |
| OOF | `results/<run>/oof/*.csv` | 每材料每重复一条 |
| 配置 | `results/<run>/config.yaml` | 与 `configs/stage1.yaml` 一致；偏离有 amendment |
| 指标 | `results/<run>/metrics.csv` | 可由 OOF 重算 |
| 信号门 | `results/<run>/trigger.md` | δ、T1/T2 判定表、触发结论 |
| 环境 | `env/lock.txt` | 精确版本 |
| 代码 | `git rev-parse HEAD` | 与 manifest 一致 |
| 数据 SHA | manifest | 与 fetch 脚本一致 |

## 独立性与披露

- ARS 评审面板是 AI 模拟评审，角色分离不等于独立误差过程，不替代期刊同行评审；报告中披露 AI 辅助评审。
- law 是用户指定的独立监管，审查门边界与证据，不接管日常技术裁决；其意见与 gb/iq 意见并列记录。
- 作者不评审自己撰写的组件；无法避免时在评审记录中注明。
- 同一模型的多角色评审可能共享盲点；对外投稿前建议补充人类同行评审。
- 评审记录保存在 `docs/reviews/`，注明日期、门、版本与结论。

## 停止与回退

- G0 未通过不进入正式运行；允许继续做数据/实现冒烟测试，但结果标为 `SMOKE_ONLY`。
- G1/G2 发现泄漏或复现失败：停止当前结果的使用，修复后重跑；旧结果保留但标注失效原因。
- G3 出现根本性方法质疑：回到 G0 修订协议，不通过追加分析掩盖。
- Stage 1 无正向信号：记录负结果；是否投入 Stage 2 由用户决定；不设第三轮架构搜索。

## 监管与冻结纪律

1. **冻结先于运行。** Stage 1 首次正式运行前完成 G0 冻结标签；`SMOKE_ONLY` 不得进入 `docs/RESULTS.md`。
2. **网格即预算。** 冻结网格是计算预算上限；扩张网格 = amendment，须书面说明会改变哪个已注册判断。
3. **“再来一轮”必须命名决策。** 任何新增运行须说明其结果会改变的已注册判断，否则不执行。
4. **停止规则。** Stage 1 无正向信号即记录负结果；Stage 2 无稳定差异即负结论；不设第三轮架构搜索。
5. **主端点不可团队内部吸收。** 冻结后涉及主端点/主对比/信号门/判据的变更上交用户决定。
6. **监管门。** 冻结前协议文本、G1/G2 证据抽查、最终声明与证据一致性；审查强度与风险相称，不把每步变成审批。
7. **消息节奏。** 门边界与实质变化才通报，不做逐运行通知。