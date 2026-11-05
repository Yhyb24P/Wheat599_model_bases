# License Notes

**核验日期：** 2026-10-29。只记录已实际查询到的许可；未确认项标为待核实。

## 数据

| 项 | 许可 / 状态 | 说明 |
|----|-------------|------|
| BGLR 包（含 `wheat.RData`） | GPL-3 | `DESCRIPTION` 字段 `License: GPL-3`。数据通过 `scripts/fetch_data.py` 从官方仓库获取，不在本仓库重新分发。 |
| wheat 数据本身 | 未发现独立许可声明 | 数据来自 CIMMYT，经 BGLR 包发布；使用与再分发遵循 BGLR 的 GPL-3 与来源方要求。发表时引用 BGLR 与 CIMMYT 来源。 |
| Wheat2000公开`wheat_GS.RData`（2026-11-05） | DeepGS包DESCRIPTION为GPL-2或GPL-3；数据无独立许可声明 | 固定`cma2015/DeepGS@33ae2dc1ed7bda4e38b76460b18756649d4478a5`原件SHA `e9ebecc0c73b39f4747bf0eda036da92640f694938046de557f0c9e68f926696`；原件/NPZ只读留`data/raw/wheat2000`，不在Git重新分发。R/Python与源头记录见`results/wheat2000_protocol_20261105/data_provenance.json`。 |

## 代码与权重

材料—位点推断及普通上下文对照的四个 checkpoint 由本项目独立模拟任务训练，未使用第三方预训练权重，未指定独立权重分发许可；仅保留本地，不入 Git。训练来源 `e3c7369`，目录 `results/matrix_inference_learning_20261104/`：

| 本地文件 | SHA-256 |
|---|---|
| `matrix_42.pt` | `09be932e3c86f954c3214696385bc32f34da99dd5cf7f714ac26e0356b7175ed` |
| `matrix_2026.pt` | `8c4ac599026d93a3e9db8a4f1b9c803238b68ba4ed28e2edec7f6b0ff64d0345` |
| `direct_42.pt` | `30c18195a70605b5aa986751b3ee9fbc608ec327f3ef55894703862233064015` |
| `direct_2026.pt` | `4816e2d527e38e23cd59c6521d7d510867ee353ba1f64d474056e1fa0dbceafc` |

DNAwhisper 从外部固定源码 `Marxin1992/Whisperer_of_DNA@7a657cc00d44263c4b1085d3991ecc0e935c14e5` 加载，2026-11-04 核对本地固定副本的 `LICENSE` 为 Apache-2.0；本仓库只提交协议适配器，不复制上游代码或权重。所导入源码及配置的 Git blob 摘要固定在 `configs/dnawhisper_nested.yaml`，运行时逐项核验。没有使用上游预训练权重，各折从头执行训练子集内预训练；如后续保存本项目权重，权重不入 Git，其 SHA 另行记录。

MoEGP 官方仓库在提交 `d3a3c44f649fadf95af34d63f069fc8dfd85af38` 未发现许可证文件；本项目从本地官方副本加载代码，不复制或重新分发该源码。

BiMSGP 官方仓库提交 `71084c8fa5444fbc747b0291ace627b1f2d6ef33` 使用 MIT 许可证；本项目从本地官方副本加载源码。本次安装的 `mamba-ssm` 2.3.2.post1 为 Apache-2.0，`causal-conv1d` 1.6.2.post1 为 BSD 3-Clause，均以安装包附带 LICENSE 为准。

| 项 | 许可 / 状态 | 说明 |
|----|-------------|------|
| TabPFN 推理代码（`PriorLabs/TabPFN`） | Apache-2.0 | GitHub API 核实。 |
| TabPFN-3.5 权重（`Prior-Labs/tabpfn_3_5`） | `tabpfn-3-5-license-v1.0`（非商用） | 明确允许测试、评估、内部基准与学术研究；禁止商业/生产用途、商业决策、客户交付物、付费产品。许可证全文保存在 `docs/licenses/tabpfn-3-5-license-v1.0.txt`。权重不进入本仓库。 |
| `rtdl-revisiting-models` | Apache-2.0 | `yandex-research/rtdl-revisiting-models` 仓库含 LICENSE；PyPI wheel 的 `dist-info` 含 Apache-2.0 LICENSE 与 `License :: OSI Approved :: Apache Software License`。此前的“未检测到许可证”是查错了 404 路径 `tabular-dl-revisiting-models`。 |
| TabM（`yandex-research/tabm`） | Apache-2.0 | GitHub API 核实。 |
| LassoNet（`lasso-net/lassonet`，扩展比较） | MIT | GitHub API 核实。 |
| numpy / scipy / scikit-learn / PyTorch | 各自发行版许可（常见为 BSD 系） | 发布前在 `env/` 锁文件中核对具体版本与许可。 |

## 使用边界

- TabPFN-3.5 权重只用于本项目的非商业研究、内部评估与学术比较；不用于采购竞标基准、客户交付或任何生产决策。
- 不在仓库、论文附件或公开镜像中分发 TabPFN 权重；只记录 checkpoint 名称、SHA 与获取方式。
- 若研究结果对外发布，确认输出（预测、指标、图表）不构成许可证禁止的“客户交付物”或商业决策依据；必要时在论文中声明许可与使用边界。
- `rtdl-revisiting-models` 为 Apache-2.0；保留 LICENSE 与版本记录，不再列为 G4 阻断项。

## 联合交互推断模拟权重（2026-11-04）

本项目模拟任务训练生成，无外部预训练权重；沿用项目代码许可，不据此赋予旧 at_pheno 数据额外许可。权重未入 Git，保留于 omen `/home/hsy/git_files/Wheat599_model_bases/results/joint_inference_20261104/`。代码冻结 `8ce68a6`。

| 文件 | SHA-256 |
|---|---|
| joint_42.pt | 96771761a2ff31ad8cf9ae8013da9e114f6b474451635996875893a4f591fbfa |
| joint_2026.pt | f6cd45f7cff58f062f0102d84227f76201df04b428408768cac3846a7bdb876c |
| independent_42.pt | a4a0c7fe2f1798a01b15eea05dfafb48719a4d0effc167d50758881da4a64781 |
| independent_2026.pt | 146626e6803774705747f6548d0832699daf7398f0bd3cec0db7461c2fd18105 |
| gate.pt | 79666d256fb8bca7268fea7486818357c1c1133f7e37f339950fb8e59abfa986 |

## TabM 任务适配依赖（2026-11-04，正式训练准备）

Wheat2000神经参照准备（2026-11-05）：DeepGS为公开R/mxnet网络的独立PyTorch结构适配，上游`cma2015/DeepGS`提交`33ae2dc1ed7bda4e38b76460b18756649d4478a5`，包许可GPL-2或GPL-3；未复制其数据或R源码入Git。BiMSGP在原地动态加载MIT许可的官方干净checkout，提交`71084c8fa5444fbc747b0291ace627b1f2d6ef33`，omen路径`/home/hsy/overlays/BiMSGP_71084c8`；model.py SHA-256为`0cec01030ac64621602fbbca029749e91202e795abee47074588474f9e1aace0`，LICENSE SHA-256为`c654fa432aadbbbeda0a5d4ed10557661a67f811f879ccdc80c31c3cf6cb0d8d`。未使用外部预训练权重，尚无这两个参照的正式训练权重。

四独立TabM外层评价（2026-11-05）：本项目从头训练80个权重，无外部预训练权重，依赖沿用Apache-2.0官方TabM；未另行指定权重分发许可。权重总计132809168字节，仅保留omen `results/independent_tabm_final_20261105/weights/`，不入Git；80个逐文件SHA见同名结果目录`weights_sha256.txt`，该清单SHA为`fd50608fb67fd09740b2bb919d3c53e1357b081162a6e91773122116fd319f56`。此为文件SHA，不是内存state摘要。

云端后期迁移交接（2026-11-05）：30个本项目固定配置专家checkpoint已逐文件核验，仅留用户交接包，不入Git；原清单见`results/late_transfer_development_20261105/cloud_artifact_sha256.txt`、核验说明见`handoff_verification.json`。未使用外部预训练权重，不将交接数据重新分发。

环境共享研究（2026-11-05）：沿用上述官方Apache-2.0 TabM依赖，无外部预训练权重。新增2400次fit仅内存best state、无持久化权重；state摘要清单为 `results/environment_tabm_development_20261105/checkpoint_sha256.txt`，清单自身SHA-256 `d2cfd53a77265183c72906b54de8900d0227ba4418069f0a73e1d73af95af5ed`。摘要规则为排序key／dtype／shape／tensor bytes，不是checkpoint文件哈希。

2026-11-05 检索开发生成的1440个本项目训练权重留 omen `/home/hsy/git_files/Wheat599_model_bases/results/retrieval_tabm_development_20261105/`，不入Git。逐文件SHA见同名结果目录的 `weights_sha256.txt`，清单SHA-256为 `9389be175483b37280480c84f37ab5f0a92e142e9e49f1f83e7f4dc0f955fe7e`；未使用外部预训练权重。

RAGP 同协议参照（2026-11-05）：固定上游 `l00907l/RAGP` 提交 `85675b1e8054fe3e91fdb3e974150f1dde81eee3` 无 LICENSE 文件。`src/wheat599/ragp.py` 为所发表机制的独立实现，不复制或分发上游源码、example-data 或权重。fit-only PCA／标准化、严格自身排除及 triplet 索引修正等差异记录于模块说明。未使用外部预训练权重；本轮最优 state 只暂存内存，流式 SHA 随轨迹记录，不持久化权重。

RAGP 480条已完成；state哈希清单 `results/ragp_development_20261105/checkpoint_sha256.txt` 自身SHA-256为 `a77fe1523e33414bf564f3fd08241469a322a47d8cc1d5ee05e8f78b97a66307`。该SHA为排序key／dtype／shape／tensor bytes的流式state摘要，不是不存在的checkpoint文件哈希。预测和轨迹原件留omen同名目录，路径与预测SHA见summary.json。

使用官方 `tabm==0.0.3` 与 `rtdl_num_embeddings==0.0.12`，均为 Apache-2.0，来源 https://github.com/yandex-research/tabm 与 PyPI 固定发布包。当前不复制或改写上游骨干源码，不使用预训练权重。模块SHA-256：tabm.py `fc654af6a16bac53d893a8265c79d7af4ebddcb95ad0d600cc6b6bc6b7317ade`；rtdl_num_embeddings.py `48dfc019d447e0cd225deb8b8145b3efa8b42cde9aa314772d6e234753e1cee1`。正式训练480个权重仅留omen `/home/hsy/git_files/Wheat599_model_bases/results/genomic_tabm_development_20261104/`，不入Git；每个权重SHA见 `results/genomic_tabm_development_20261104/weights_sha256.txt`，该清单自身SHA-256为 `e5a7e89d64e71561cf369ee44cbba38af7b03ad156ad15c1cb43ab3bb065bb09`。未使用外部预训练权重。
