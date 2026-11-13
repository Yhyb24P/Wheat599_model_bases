> **已被取代：** 本草案由 `docs/STAGE1_PLAN.md` 取代（2026-10-29 用户确认 stage1 收敛）。

# Stage 1 Scope Draft（待用户直接确认）

**来源：** law 转达的“用户 2026-10-29 范围收敛”请求。按 AIA 规则，范围变更需用户直接确认；本文件是草案，不是冻结文本。

## 阶段一（拟冻结对象）

- **方法：**
  - TabPFN-3.5 冻结 ICL（默认配置，不微调；checkpoint `tabpfn-v3.5-20260909.safetensors`，`tabpfn==9.2.0`）。
  - 小型 FT-Transformer：自主小模型网格 d∈{16,32}、层数 {1,2}、无位置编码；论文默认 3 层/embedding 192/8 头不是本网格的出处；选择在 inner CV 内。
  - Ridge 参照：α 在 inner CV 内选；`fit_intercept=True`。
- **数据：** BGLR wheat.RData；四环境分别报告，列名 `"1"`、`"2"`、`"4"`、`"5"`；marker-only。
- **划分：** random 外层 5 折 × 3 重复（固定种子）；无亲缘臂、无 A 臂、无 `wheat.sets`。
- **指标：** PCC 与 RMSE 同时报告；同划分配对差（以 Ridge 为参照；TabPFN 与 FT-Transformer 之间也配对）。
- **无泄漏底线：** α、FT 配置、早停全部只在外层训练折内完成；泄漏注入测试仍须通过。
- **结果定位：** 探索性第一眼；不作稳定改进声明。触发第二阶段的“信号”定义由 gb 给出（待补）。
- **预算：** FT-Transformer 32 配置 × 4 环境 × 15 外层拟合；TabPFN 15×4 次前向；GPU 上限约 2 GPU-hours；CPU 优先。

## 阶段二（暂不冻结）

- 现行六方法完整协议（线性收缩、RKHS、RBF-SVR、TabM、FT-Transformer、TabPFN；random + IBS 亲缘臂；10/5 重复；8 主对比；LightGBM secondary）保留为 stage2 文档。
- 触发条件：第一阶段有信号；冻结时再走 G0（gb/iq 签核 + 用户授权提交）。
- §6 判定标准随第二阶段走；§6.1 的“fold 级”旧措辞改为“主区间（重复级，df=9）”列入 stage2 修订清单。

## 待确认

1. 用户是否确认上述阶段一切分。
2. 第一阶段“信号”定义（gb）：例如 TabPFN 或 FT-Transformer 相对 Ridge 的配对差区间是否排除 0、效应量阈值、跨环境一致性要求。
3. 第一阶段随机种子、FT-Transformer 网格是否保持 32 配置、TabPFN checkpoint/许可是否沿用。
4. 冻结方式：stage1 文本单独 G0，还是把现行六方法协议标为 stage2 后统一冻结 stage1。