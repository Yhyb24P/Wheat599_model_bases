# 待复现方法实现可用性核查（2026-11-15）

判定标准：仓库存在 + 代码完整（训练/评估入口）+ 能对应论文架构，三者缺一即判"待复现"。本记录只读核查，未运行任何训练。

## CWAGS

- 论文：BMC Genomics 2026;27:625，DOI 10.1186/s12864-026-12980-9，作者 Cao C、Mohamed F.B.、Sunar M.S.B.、Chan V.S.；正文报告 Wheat599 四环境平均 PCC 0.5368（十折均值、固定种子、AdamW、400 epoch）。
- 检索路径：Europe PMC/PMC 全文、Springer/BMC 原文页、GitHub 仓库与用户搜索（"CWAGS"、"chunqing cao"）、Zenodo API（0 命中）、GitLab API（返回 500，未取证）、通用网页检索。
- 仓库：未找到。同名 GitHub 仓库 9 个均无关；正文无代码可用性声明，Data availability 仅列数据集（BGLR Wheat599、CIMMYT Wheat33K、Rice44k），无补充代码文件。
- 结论：**待复现**（未找到任何作者实现）。与 2026-11-06 记录一致，本次复查无新发布。

## PNNGS

- 论文：Frontiers in Plant Science 2024;15:1410596，DOI 10.3389/fpls.2024.1410596（检索报告作者 Xie Zhengchao 等，之江实验室）。数据声明为文内/补充材料、可向通讯作者索取。
- 检索路径：GitHub 仓库检索（仓库描述与论文标题一致）。
- 仓库：https://github.com/996184745/code-for-PNNGS 。2026-11-15 复核可达（HTTP 200）；文件树 `main.py`、`model/PNNGS.py`、`model/DNNGP.py`、`requirements.txt`、`LICENSE`，无 `data/`。
- 完整性判断：有训练/评估入口（`main.py`，模型与表型在 `__main__` 内修改）；`model/PNNGS.py` 为核 1–15 并行 Inception1d＋BN＋Dropout＋线性层，10 折 PCC 评估。缺 README 与数据；检索报告指出实码 Dropout=0.3、仅 MSE、分支无残差，与论文的 0.5、L05/L1/L2/L3、残差分支不符。
- 结论：**待复现**。仓库可获取，但"对应论文架构"不成立（分支、超参、损失不一致）且无数据与入口说明，不能作为忠实适配基座；若后续使用须逐项记录差异，不得称原论文复现。

## MDNN

- 论文：MDNN: memetic deep neural network for genomic prediction，Briefings in Bioinformatics 2025，DOI 10.1093/bib/bbaf352（检索报告 PMID 40698862），作者 Mao Y、Peng X、Weng J 等（华南农大等）。数据声明仅给 DNNGP 数据链接，无代码链接。
- 检索路径：Europe PMC 全文、GitHub 检索（MDNN、memetic genomic prediction）、Zenodo。
- 仓库：https://github.com/xxXPCSC/MDNN （自述 "code repository of MDNN"）。2026-11-15 复核可达（HTTP 200）；文件树 `Start_MDNN.py`、`ma_config.py`（CBAM＋树结构解码）、`train.py`、`judge_perfor.py`、`process_nn.py`、`README.md`，无 `data/`、无 requirements。README 未引用论文；所有者身份未能通过作者列表确认。
- 完整性：有训练/评估入口（`Start_MDNN.py` 默认 `data/tomato332`、pop_size=4、num_generations=2）；与论文架构的对应度较高属推断。
- 结论：**待复现**（作者归属未确认、无数据与依赖说明、README 无论文引用）。
- 后续若运行：架构搜索必须完整放在外折开发集内，记录搜索预算、种子与最终架构选择。

## 检索限制

GitLab 搜索接口本次返回 500，未能取证 GitLab 平台。GitHub 仓库可达性与文件树为本机 2026-11-15 直接复核；论文元数据与仓库内容细节来自联网检索阅读；架构对应性判断属推断。
