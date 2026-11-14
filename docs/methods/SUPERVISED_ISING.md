# 监督条件 Ising 隐状态：本机开发比较

已完成：冻结1011915的三臂各1,920次开发拟合、0失败。耦合/J=0/MLP开发PCC分别为0.458179/0.457658/0.482152，未超过原分位数0.494565；不进入外层或扩网格。详见 [RESULTS](../RESULTS.md)。本轮只在本机运行，未前往omen。

## 模型及三项比较

每个环境独立、全标记输入；拟合集内按列中心化和标准化，常量标记贡献归零。标记是观测值，不让求解器修改标记。

- **耦合隐状态**：a=Wz+b，E(h|x)=-aᵀh−Σ(k<l)J_kl h_k h_l，h∈{−1,+1}^r。固定 T=1，枚举全部 2^r 状态，以稳定 softmax(-E) 精确计算 m=E[h|x]。J 只存严格上三角，不含对角、不重复计数。预测为 μ+βᵀz+cᵀm，再反标准化。r=12 时每环境每种子16,718个可训练参数、4,096种状态。
- **J=0 对照**：同样的 r、初始化、线性直连、优化器和搜索范围；J 固定为零，以解析 tanh(a) 计算条件均值，无需枚举。相同种子下 W、b、β、c、μ 的初值一致。
- **普通网络**：两层 ReLU MLP、共享标量输出和同样的全标记线性直连。宽度64/128；末层不额外加偏置，使用公共 μ。它是合理容量参照，不宣称与 Ising 参数量相同。

β、μ 初始为零，W、b 用 PyTorch 线性层初始化，c 用标准差0.02的正态初始化，J 初始为零。非零 c 避免隐分支在首步完全没有监督梯度。所有可训练参数由同一个 MSE 目标更新；AdamW 对它们应用配置中的 weight decay。这里没有无监督标记重构、负相采样或最小能量 MAP 替代。

J 连接的是隐表示，不是已知基因；隐状态概率不等于已校准的表型不确定性。模型不保留原 TabM 或固定融合的预测函数。

## 运行前固定的预算

每臂八配置：隐维8/12（普通网络对应宽度64/128）×学习率0.0005/0.002×weight decay 0.0003/0.01。统一使用 AdamW、batch128、最多500 epoch、patience50、梯度裁剪5。均按对应环境验证PCC早停，按三内折的双种子平均预测PCC选配置；不以测试标签训练或选模。

每臂沿用既有10外折的3个内折、4环境、种子42/2026，共1,920次拟合；三臂合计**5,760次开发拟合**，不自动扩大网格。完整系统分别选模；耦合与J=0的逐配置预测均保存，可识别容量和选模差异，不能只凭最优分数把所有增量归因于J。

复用同材料的原分位数TabM、Huber和原固定融合。沿用现有开发执行器，`quantile` 是执行器沿用的候选指标键，在本轮三个目录里对应各自的**标量MSE预测器**，不表示分位数监督；`original_quantile`/`original_fusion` 才是原分位数参照。历史 `fixed_fusion` 键仍是 MSE TabM 的融合，不能混称。

```bash
python scripts/run_background_nested.py --config configs/supervised_ising_coupled.yaml --out results/supervised_ising_coupled_development_20261114
python scripts/run_background_nested.py --config configs/supervised_ising_independent.yaml --out results/supervised_ising_independent_development_20261114
python scripts/run_background_nested.py --config configs/supervised_ising_mlp.yaml --out results/supervised_ising_mlp_development_20261114
```

首批冻结 `1011915` 使用一个进程顺序运行三臂。根据用户的效率要求，当前配置明确设置 `workers: 4`，以后通过4个独立进程执行同一内折内的独立拟合；不改变每次拟合的种子、batch、更新、早停或选模规则。作业 CUDA 显存比例0.3由worker均分（每worker 0.075）。自定义 trainer / paired trainer 默认仍为1个worker，未改变其他任务。首批结果保留原串行来源，不为展示加速重跑或冒充并行结果。不安装或调用Kaiwu SDK。数据与基线预测沿用现有只读文件，原始预测不入Git。若发生失败，保留失败和已完成产物，不以默认结果替换。

本机 RTX 5090 D 的合成吞吐检查（540材料、1,279标记、12隐状态、128任务、每任务10轮）中，串行6.236秒、四进程稳定执行2.723秒，吞吐为2.29倍；进程池启动及16任务预热另计3.855秒，短任务总耗时未必更短。两种执行的预测逐元素一致。该结果只衡量合成任务计算效率，不代表正式整批加速倍数。复查入口：`PYTHONPATH=src python tests/benchmark_fit_workers.py`；记录写入 `.state/ising_parallel_timing.json`。

## 方法与实现依据

附件提出的条件能量模型在本轮采用精确有限温度计算。[Kaiwu固定源码](https://github.com/qboson/kaiwu-pytorch-plugin/tree/cbef8d86b297963326b92eeabd9d32be2a640c21)用于辨别BM采样、MAIFS掩码选择与QDEQ固定点求解的不同职责；本模型不复制该插件代码、不加载量子求解器。其求解器最大化矩阵二次型、辅助自旋和能量缩放约定，不能直接当作此处温度1的概率采样。

成熟背景：[监督式RBM](https://proceedings.mlr.press/r15/nguyen17b.html)已有监督表示与回归研究；[Ising/QUBO定义](https://docs.dwavequantum.com/en/latest/quantum_research/qubo_ising.html)说明两种变量表示的转换。本轮检验监督隐状态耦合的完整预测系统，不主张量子优势。

必要软件验证包括J=0精确退化、协方差导数恒等式、两自旋耦合符号/计数、极端输入场稳定性、初始化一致性、完整参数梯度、材料/常量列边界和现有训练器集成。数学自检与合成测试不作为Wheat599增益证据。
