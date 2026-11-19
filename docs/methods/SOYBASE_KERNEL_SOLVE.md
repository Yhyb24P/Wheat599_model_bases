# SoyBase 核回归求解路径判据（2026-11-20，sol-qo-cholesky-branch-ruling 执行记录）

## 分支判定的那一个事实

失败候选（inner0, bandwidth=0.25, trait=Hgt, model=gaussian）在代码中的对角正则项实际数值：
`k = 0.0*a + 1.0*exp(-0.25*d/d0)`，求解系统为 **`(k + 0.1*I) α = b`** —— ridge=0.1 真实加在对角上，frac=0.0 只是混合分数（gaussian 语义），**不是**"该候选不加正则"。
→ 落入判据分支 ②：求解精度问题，改共享求解处一次（eigh 等价路径）。

## 实测证据（tim，kernel_repro.log 05:41 / cholesky_verdict.log）

- 生产与定点复现均确定性命中同一网格语义（inner0 h=0.25 Hgt gaussian ridge=0.1）；
- 同一 `K = r + 0.1*I` 张量重复 `torch.linalg.cholesky`：threads=8 五次 4 过 1 抛（minor 2927），threads=1 三次 1 抛 —— **torch potrf 在该矩阵上非确定性**；
- numpy 串行参考 potrf 前 3300 阶全部主元 s>0，无失败；lead3104+ridge 的 eigvalsh min_eig=1.000e-01、frac_neg=0 —— 矩阵数值正定，失败属求解器层。

## 处置（按裁定 ②）

- `configs/soybase/regression_classical.yaml` 设 `kernel_solve: eigh`；`nested_classical` 记录 `kernel_solve` 入 provenance。
- 共享求解处 `classical.py`：内层 labeled 分支走 `_eigh_predict`（每混合分数一次特征分解，ridge 复用）；外层逐候选传 `solve`。默认 `cholesky` 保持原配方逐字不变，Wheat2000 不受影响。
- 一致性比对（tests/test_soybase_regression.py）：良态矩阵两法预测差 <1e-9；端到端 nested_classical 两法选择一致、OOF 预测差 <1e-8。
- 与 cholesky 原配方**不同源**：正式结果须注明求解路径为 eigh 谱分解；非确定性 potrf 候选不再有 cholesky 参照值，比对以健康候选为准。

## 重启

- r6 失败证据（provenance/execution_status/failures）保留于 `soybase_regression_dev_tim_20261119_r6/` 并同步 `~/soybase-r6-backup/r6/`。
- r7 以同网格、同种子（42/2026）、同冻结划分、jobs=2、kernel_threads=8 重启，仅求解路径变更。
