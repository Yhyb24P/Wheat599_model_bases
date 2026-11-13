#!/usr/bin/env python3
"""Single-source generator for the Wheat599 offline reference library (ref/).

The inline REFERENCES list is the single source of truth. Running this file emits:
  ref/index.html       one self-contained offline page with EVERY reference brief
  ref/references.bib   citable BibTeX for all entries

Per the project simplification request, all briefs live in one HTML page (no
per-paper pages, no hand-maintained duplicates). Design follows the
scientific-presentation contracts (citable/traceable metadata, CJK typography,
colorblind-safe badges with a text second channel, MathML, print) and
modern-web-guidance (semantic landmarks, color-scheme + light-dark() with a
prefers-color-scheme fallback, <details> no-JS disclosure, responsive tables).
Palette matches the sibling pages: primary #0072B2, background #f6f5ef,
text #172e38, white panels.

Honesty rules: historical verification notes from docs/REFERENCES.md are kept;
links taken from project docs are attributed to their source doc; metadata not
verified online this round is explicitly flagged (canonical/pending), never
fabricated. Full text that could not be downloaded is marked, not faked.
"""
from __future__ import annotations

import html
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent

LEVEL_LABEL = {
    'historical': '历史已核验',
    'online': '本轮在线核验',
    'canonical': '规范著录·待复核',
    'pending': '元数据待核',
}
LEVEL_COLOR = {
    'historical': '#0072B2',
    'online': '#009E73',
    'canonical': '#E69F00',
    'pending': '#CC79A7',
}

REFERENCES = [
    # ================= 现役主方法核心出处 =================
    dict(key='wang2026mulpheg2p', group='main', type='article',
         title='Mul-PheG2P: decoupled learning and prediction-space fusion enables robust and interpretable multi-phenotype genomic prediction',
         authors=['Wang, Jiahui', 'Zhang, Yong', 'Li, Bo', 'Piao, Xinglin', 'Zhao, Xiangyu',
                  'Zhang, Dongfeng', 'Wang, Aiwen', 'Zhang, Bob', 'Wang, Kaiyi'],
         year=2026, venue='New Phytologist, 252(1), 503–519', doi='10.1111/nph.71461',
         pmc='PMC13539974', pmid='42552668',
         url='https://doi.org/10.1111/nph.71461',
         brief='两阶段解耦的多表型基因组预测：先训练性状特异 SNP-CNN 编码器保留独特遗传模式，再在预测空间用 Meta-Phenotype Attention 融合标量贡献；声称避免负迁移，提供注意力（宏观）与 Integrated Gradients（微观）可解释链及零微调迁移。',
         relation='现役同协议完整适配候选。本仓固定提交 e20ca68 完整适配（train-only 标准化/插补、允许目标环境编码器参与、标量融合与隐藏融合分列）；此前“仅其他环境作源、目标环境不作源”的适配保留为历史机制对照，不代表完整候选。统一十折外层宏 PCC 0.438225（选配置）/0.443596（默认），低于固定融合 0.526506；称同协议适配重评，不称原论文精确复现。',
         note='发表版 New Phytologist 2026;252(1):503–519（本轮 PMC 一手核验）；仓库 README 著录年份 2025 为代码/预印年份。论文 γ=0.2（MSE+0.2 Pearson）、预训练/微调/融合 500/300/200、H=128；公开 train.py 默认 300/300/200，以代码为准。论文标量融合与公开默认隐藏融合不可混称。',
         verification='本轮在线核验：PMC13539974 一手确认题名/九位作者/New Phytologist 252(1):503–519/DOI 10.1111/nph.71461/PMID 42552668；并据固定提交 e20ca68 README 核对方法。',
         level='online', pdf=None,
         pdf_note='PMC 提供公开 PDF（NPH-252-503.pdf，Wiley Open Access）；本轮自动下载两次返回 HTML 挑战页，未收全文，不冒充。© New Phytologist Foundation，含 TDM/AI 权利保留声明；按需自取。'),
    dict(key='mulpheg2p_repo', group='main', type='software',
         title='Mul-PheG2P 官方 PyTorch 实现（固定提交）',
         authors=['Wang, Jiahui', 'et al.'], year=2025,
         venue='GitHub: wjhhh2928/Mul-PheG2P @ e20ca6847c7dec6ce80386e94df4378ad2d1fbec',
         url='https://github.com/wjhhh2928/Mul-PheG2P/tree/e20ca6847c7dec6ce80386e94df4378ad2d1fbec',
         code='MIT',
         brief='官方实现：SNP-CNN 编码器、直接基线、多环境预训练、冻结编码器目标适配、预测标量/隐藏表示融合。',
         relation='本仓 mulphe.py / mulphe_unified.py 适配的上游源码（固定提交，MIT）；适配差异见 docs/methods/MULPHE_UNIFIED.md。',
         verification='本轮在线核验：固定提交 README 确认 MIT 许可与方法描述；提交哈希与本仓代码 OFFICIAL_COMMIT 一致。',
         level='online', pdf=None),
    dict(key='koenker1978regression', group='main', type='article',
         title='Regression Quantiles',
         authors=['Koenker, R.', 'Bassett, G.'], year=1978,
         venue='Econometrica, 46(1), 33–50', doi='10.2307/1913643',
         url='https://doi.org/10.2307/1913643',
         brief='提出回归分位数：最小化非对称绝对损失（pinball）估计条件分位数，奠定分位数回归。',
         relation='分位数 TabM 的 pinball 损失与 99 分位点（0.001–0.999）输出的方法学基础；本仓 raw 全标记、分位点 pinball 求和、成员/材料平均。',
         verification='本轮：DOI 10.2307/1913643 确认解析至 JSTOR stable/1913643；JSTOR 全文页需 JavaScript，本轮未能逐字在线核验全文；书目据规范著录。',
         level='canonical', pdf=None,
         pdf_note='本轮未取得全文（DOI 解析至 JSTOR，全文页需 JavaScript）；不冒充已获取。', math='pinball'),
    dict(key='huber1964robust', group='main', type='article',
         title='Robust Estimation of a Location Parameter',
         authors=['Huber, P. J.'], year=1964,
         venue='Annals of Mathematical Statistics, 35(1), 73–101', doi='10.1214/aoms/1177703732',
         url='https://doi.org/10.1214/aoms/1177703732',
         brief='提出 Huber M-估计：损失 ρ(t) 在 |t|<k 时为 ½t²、否则为 k|t|−½k²，介于样本均值与中位数之间，在污染正态下渐近最稳健。',
         relation='Huber ICM 的稳健损失出处。本仓 robust_icm.py 为凸 Huber 惩罚风险最小化（材料核 + ICM 任务核），不是完整 Huber GP 贝叶斯算法。',
         verification='本轮在线核验：Project Euclid（开放获取）确认题名/作者/期刊/卷期页/DOI 与摘要；摘要给出的 ρ(t) 即项目 Huber 损失。',
         level='online', pdf=None,
         pdf_note='开放获取（Project Euclid）；本轮自动下载返回 HTML 挑战页而非 PDF，未收全文，不冒充。', math='huber'),
    dict(key='hubergp2023', group='main', type='article',
         title='Robust Gaussian Process Regression with Huber Likelihood',
         authors=['Algikar, P.', 'Mili, L.'], year=2023,
         venue='arXiv:2301.07858（stat.AP；投稿 Annals of Applied Statistics，发表未确认）', arxiv='2301.07858',
         url='https://arxiv.org/abs/2301.07858',
         brief='以 Huber 似然实现稳健高斯过程回归，降低长尾/异常残差的影响。',
         relation='Huber ICM 参考其稳健观测思想；本实现为凸惩罚风险最小化，不复现其完整贝叶斯算法（docs/history/RESEARCH_PLAN.md）。',
         note='arXiv 预印本；页面注明投稿 Annals of Applied Statistics，正式发表状态未确认。',
         verification='本轮在线核验：arXiv:2301.07858 一手摘要确认题名/作者（Algikar & Mili）；期刊发表未确认。',
         level='online', pdf=None),
    dict(key='bonilla2007multitask', group='main', type='inproceedings',
         title='Multi-task Gaussian Process Prediction',
         authors=['Bonilla, E. V.', 'Chai, K. M. A.', 'Williams, C. K. I.'], year=2007,
         venue='NeurIPS 20 (2007)',
         url='https://papers.nips.cc/paper/3189-multi-task-gaussian-process-prediction.pdf',
         brief='多任务高斯过程预测：用可分离（separable）协方差建模多输出，任务间相关用任务核、输入间相关用输入核；intrinsic coregionalization（ICM）是其可分离形式之一。',
         relation='多任务 GP 可分离协方差（含 ICM 形式）的方法学参考；本仓 Huber ICM 的任务核思路与此相关，但实现为凸惩罚风险最小化，非完整贝叶斯多任务 GP。不据本文主张 ICM 起源（docs/history/RESEARCH_PLAN.md）。',
         verification='规范著录；链接 papers.nips.cc/paper/3189 取自 docs/history/RESEARCH_PLAN.md；本轮未在线逐字核验全文。',
         level='canonical', pdf=None),
    dict(key='alvarez2012kernels', group='main', type='article',
         title='Kernels for Vector-Valued Functions: a Review',
         authors=['Alvarez, M. A.', 'Rosasco, L.', 'Lawrence, N. D.'], year=2012,
         venue='arXiv:1106.6251 (v2)；Foundations and Trends in Machine Learning, DOI 10.1561/2200000036',
         arxiv='1106.6251', doi='10.48550/arXiv.1106.6251',
         url='https://arxiv.org/abs/1106.6251',
         brief='综述多输出/向量值函数的核方法，系统连接概率与泛函视角，含共区域化（含 ICM/LMC）等多输出核。',
         relation='Huber ICM 与 BH-LMC 的向量值核方法学出处（docs/methods/Wheat599_BH-LMC.md 引期刊 DOI 10.1561/2200000036）。',
         verification='本轮在线核验：arXiv:1106.6251（v2, 2012）确认题名/作者/摘要；arXiv DOI 10.48550/arXiv.1106.6251。期刊 DOI 10.1561/2200000036 取自 docs/methods/Wheat599_BH-LMC.md。',
         level='online', pdf='pdfs/alvarez2012kernels.pdf', pdf_pages=37),
    dict(key='tabular_regression_repo', group='main', type='software',
         title='tabular-regression：分位数回归实现（固定提交）',
         authors=['Yandex Research'], year=2025,
         venue='GitHub: yandex-research/tabular-regression @ 79e273f1af14c9a9703a1f385e9c9ea8faf5bae0',
         url='https://github.com/yandex-research/tabular-regression/blob/79e273f1af14c9a9703a1f385e9c9ea8faf5bae0/bin/quantile_regression.py',
         code='许可以仓库为准',
         brief='Yandex 表格回归实现，含分位数回归（pinball）入口 bin/quantile_regression.py。',
         relation='分位数 TabM 的分位数回归实现依据（固定提交，docs/methods/QUANTILE_TABM.md）。作者公开示例用 1024 分位点，本项目 99 为有限预算近似网格，非逐项复现。',
         verification='沿用 docs/methods/QUANTILE_TABM.md 已引用的固定提交链接；本轮未重新核验仓库许可。',
         level='historical', pdf=None),

    # ================= 方法与数据（REFERENCES.md #1-11） =================
    dict(key='gorishniy2021revisiting', group='method_data', type='inproceedings',
         title='Revisiting Deep Learning Models for Tabular Data',
         authors=['Gorishniy, Yury', 'Rubachev, Ivan', 'Khrulkov, Valentin', 'Babenko, Artem'],
         year=2021, venue='NeurIPS 2021', arxiv='2106.11959',
         url='https://arxiv.org/abs/2106.11959',
         code='yandex-research/rtdl-revisiting-models（Apache-2.0）',
         brief='系统比较表格数据深度模型（MLP、ResNet 式、FT-Transformer 等），提出 FT-Transformer：把每个特征 tokenize 后用 Transformer 建模特征交互；给出官方默认配置与调参空间。',
         relation='本项目“A 骨干”FT-Transformer 架构标签的出处。本项目 d=16/32、1–2 层是自主小模型选择，不是官方默认。',
         note='论文默认 FT-Transformer 为 3 层、embedding 192、8 头（Appendix E.2 Table 12）；调参空间层数 [1,4]、embedding [64,512]（Table 13）。rtdl 旧仓库名 tabular-dl-revisiting-models 已 404，现行官方仓库为 rtdl-revisiting-models。',
         verification='gb 在线核验；本地核验 wheel LICENSE；law 核验默认/调参空间。本轮据 arXiv:2106.11959 原文 PDF 首页一手核验作者为 Yury Gorishniy 等——旧 REFERENCES.md 写 “Gorishniy, N.” 有误，已修正为 Y.（Yury）。',
         level='historical', pdf='pdfs/gorishniy2021revisiting.pdf', pdf_pages=25),
    dict(key='yu2026fttransformer_sti', group='method_data', type='article',
         title='FT-Transformer 预测 STI 风险（Natsal-3）［应用论文，描述性题名］',
         authors=['Yu, W.', 'Chen, Y.', 'Liu, B.', 'Huang, E.', 'Guo, F.'], year=2026,
         venue='BMC Public Health, 26(1), 1097', pmid='41761163', pmc='PMC13049863',
         url='https://www.ncbi.nlm.nih.gov/pmc/articles/PMC13049863/',
         brief='将 FT-Transformer 应用于 Natsal-3 性传播感染风险预测的应用研究。',
         relation='只作应用引用，不是 FT-Transformer 架构出处；架构出处见 Gorishniy 2021。',
         note='准确英文题名以 PMC13049863 为准；此处为描述性题名，不冒充原文标题。',
         verification='gb 在线核验；只作应用引用，不是 FT-Transformer 架构出处。',
         level='historical', pdf=None),
    dict(key='gorishniy2025tabm', group='method_data', type='inproceedings',
         title='TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling',
         authors=['Gorishniy, Yury', 'Kotelnikov, Akim', 'Babenko, Artem'], year=2025,
         venue='ICLR 2025', arxiv='2410.24210',
         url='https://arxiv.org/abs/2410.24210',
         code='yandex-research/tabm（Apache-2.0）',
         brief='提出 TabM：共享骨干 + 多成员轻量适配的参数高效集成，一个 TabM 高效模仿 MLP 集成、每对象输出多预测，提升表格深度学习性能与效率。',
         relation='现役分位数 TabM 与独立 MSE TabM 的骨架出处。本项目每环境独立训练、99 分位点 pinball 监督、双种子预测平均为自主适配。',
         note='作者为 Yury Gorishniy, Akim Kotelnikov, Artem Babenko；不要写成 Akhanov 等。',
         verification='gb 在线核验。本轮据 arXiv:2410.24210 原文 PDF 首页一手核验作者为 Yury Gorishniy, Akim Kotelnikov, Artem Babenko——旧 REFERENCES.md 写 “Gorishniy, N.” 有误，已修正为 Y.（Yury）。',
         level='historical', pdf='pdfs/gorishniy2025tabm.pdf', pdf_pages=37),
    dict(key='qu2025tabicl', group='method_data', type='inproceedings',
         title='TabICL: A Tabular Foundation Model for In-Context Learning on Tables',
         authors=['Qu, J. T. G.', 'Holzmüller, D.', 'Varoquaux, G.', 'Le Morvan, Z.'], year=2025,
         venue='ICML 2025, PMLR v267 (qu25d)',
         url='https://proceedings.mlr.press/v267/qu25d.html',
         code='soda-inria/tabicl（含 TabICLv2, ICML 2026）；回归权重 jingang/TabICL（BSD-3-Clause）',
         brief='表格基础模型，通过在合成任务上预训练、以上下文学习方式做表格预测。',
         relation='历史比较候选（TabICL 基因组适配），非现役主候选。回归权重 tabicl-regressor-v2-20260212.ckpt（BSD-3-Clause，docs/LICENSE_NOTES.md）。',
         note='TabICL 是 ICML 2025，不是 ICLR；TabICLv2 才是 ICML 2026。',
         verification='gb 在线核验。',
         level='historical', pdf=None),
    dict(key='wu2024gpformer', group='method_data', type='article',
         title='A transformer-based genomic prediction method fused with knowledge-guided module (GPformer)',
         authors=['Wu, C.', 'Zhang, Y.', 'Ying, Z.', 'et al.'], year=2024,
         venue='Briefings in Bioinformatics, 25(1), bbad438', pmc='PMC10701102',
         url='https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10701102/',
         brief='融合知识引导模块的 Transformer 基因组预测方法（GPformer）。',
         relation='历史方法对照。无公开代码（Data Availability：“reasonable request”）。与 GPTransformer（大麦, Front Plant Sci 2021）、Cropformer 是不同工作，引用写全称。',
         verification='gb 在线核验；无公开代码（Data Availability：“reasonable request”）。',
         level='historical', pdf=None),
    dict(key='li2026gpwaiter', group='method_data', type='article',
         title='Leveraging weighted embedding and Transformer architecture to improve phenotype prediction of complex traits for crops (GP-WAITER)',
         authors=['Li, J.', 'Yu, L.', 'Li, M.', 'et al.'], year=2026,
         venue='Nature Communications', doi='10.1038/s41467-026-71035-5',
         url='https://doi.org/10.1038/s41467-026-71035-5',
         code='snowo-w/GP-WAITER',
         brief='加权嵌入 + Transformer 架构的基因组预测方法（GP-WAITER）。',
         relation='历史比较；不作为工程基座。核心为 .so；train-test.py L64/82 reshape(180,4589)、L212–218 test_loss 早停；正文每训练子集重算加权。',
         verification='gb 在线核验；核心 .so；train-test.py L64/82 reshape(180,4589)、L212–218 test_loss 早停；正文每训练子集重算加权。本轮 Crossref 一手核验 DOI/完整题名。',
         level='historical', pdf=None),
    dict(key='fan2024tabular', group='method_data', type='article',
         title='Tabular deep learning: a comparative study applied to multi-task genome-wide prediction',
         authors=['Fan, Y.', 'Waldmann, P.'], year=2024,
         venue='BMC Bioinformatics, 25, 322', pmc='PMC11452967',
         url='https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11452967/',
         code='angelYHF/Tabular-deep-learning-for-GWP',
         brief='表格深度学习在多任务全基因组预测上的比较研究。',
         relation='历史比较。LassoNet 0.691 / LightGBM 0.657 / FT-Transformer 0.602 是合并多任务 r，不是四环境各自精度。law 逐文件核验公开代码缺陷。',
         note='不得写“评审者指出全样本标准化”（无公开出处）。Regression.zip 尚未逐项复核；冻结文本标“待复核”。',
         verification='gb 在线核验论文与数字；law 逐文件核验五项缺陷 + 两项新发现（vendored lassonet score() 正 MSE + argmax；LightGBM 测试集早停/调参），证据见 docs/history/ars/supervision_factcheck_20261029.md。',
         level='historical', pdf=None),
    dict(key='yang2026gsformer', group='method_data', type='article',
         title='Gsformer: a dual-architecture deep learning framework with CNN-self-attention and sparse-attention for genomic selection',
         authors=['Yang, T.', 'Wu, W.', 'Xue, Y.', 'Zhou, L.', 'Kang, H.', 'Liu, J.'], year=2026,
         venue='Genetics Selection Evolution', doi='10.1186/s12711-026-01055-8', pmc='PMC13455463',
         url='https://doi.org/10.1186/s12711-026-01055-8',
         code='hajudien/GSformer',
         brief='Gsformer 基因组预测方法。',
         relation='历史比较。五折外层/三折内层；SVR 最好于 GY1/GY3/GY4，GBLUP 最好于 GY2；目标为 DNNGP 预处理 EBV；输入 SNP/PC，不含谱系。',
         note='GY1–GY4 与本仓列名 "1","2","4","5" 的映射未核；跨论文引用其排名时不假设对应。',
         verification='gb 在线核验；GY1–GY4 与本仓列名 "1","2","4","5" 的映射未核；跨论文引用其排名时不假设对应。',
         level='historical', pdf=None),
    dict(key='priorlabs2026tabpfn35', group='method_data', type='report',
         title='TabPFN-3.5: Technical Report',
         authors=['Prior Labs (Jäger et al.)'], year=2026,
         venue='arXiv technical report', arxiv='2609.17895',
         url='https://arxiv.org/abs/2609.17895',
         code='tabpfn==9.2.0（Apache-2.0）；权重 Prior-Labs/tabpfn_3_5（tabpfn-3-5-license-v1.0，非商用）',
         brief='TabPFN-3.5 技术报告：表格基础模型，上下文学习做小样本表格预测。',
         relation='历史 Stage 2 条件协议冻结 ICL；当前任务不跑 TabPFN。权重许可非商用，全文在 docs/licenses/。',
         verification='本地核验 checkpoint 元数据（8 成员/balanced/768/无全局投影）与许可；gb 在线核验技术报告 §3.2 与 <1000 行微调警告原文。arXiv 与报告对应关系由 PyPI 官方 BibTeX 闭合（Jäger et al., 2026）。',
         level='historical', pdf='pdfs/priorlabs2026tabpfn35.pdf', pdf_pages=33),
    dict(key='bglr_wheat_data', group='method_data', type='dataset',
         title='BGLR wheat.RData（CIMMYT Global Wheat Program）',
         authors=['CIMMYT Global Wheat Program', 'BGLR-R'], year=2026,
         venue='gdlc/BGLR-R@7e92aa2d, data/wheat.RData',
         url='https://github.com/gdlc/BGLR-R/blob/master/data/wheat.RData',
         code='BGLR（GPL-3）',
         brief='599 份小麦材料、1279 标记、4 环境表型（原表型列 1/2/4/5）。man/wheat.Rd 载明 MAF≥0.05 过滤声明与 Bernoulli 插补。',
         relation='本项目唯一数据出处。SHA-256 8710523389007dd8b7c88a5deb74ee16aea0bae158496bc11f0626fb43b5ea21；data/raw/ 只读；R as.vector 列主序，读取 reshape(order="F")。',
         note='“实测 96 列 MAF<0.05”与文档差异并列记录。',
         verification='本地核验；man/wheat.Rd 载明 MAF≥0.05 过滤声明与 Bernoulli 插补；“实测 96 列 MAF<0.05”与文档差异并列记录。',
         level='historical', pdf=None),
    dict(key='lemhadri2021lassonet', group='method_data', type='article',
         title='LassoNet: A Neural Network with Feature Sparsity',
         authors=['Lemhadri, I.', 'et al.'], year=2021, venue='JMLR, 22(20-848)',
         url='https://jmlr.org/papers/v22/20-848.html',
         code='lasso-net/lassonet（MIT）',
         brief='带特征稀疏性的神经网络，结合 L1 正则与隐藏层。',
         relation='扩展比较使用；正式引用前核对。',
         verification='待核验（扩展比较使用；正式引用前核对）。',
         level='pending', pdf=None),

    # ================= GP 基础与方法学（REFERENCES.md，law 核验） =================
    dict(key='meuwissen2001prediction', group='gp', type='article',
         title='Prediction of total genetic value using genome-wide dense marker maps',
         authors=['Meuwissen, T. H. E.', 'Hayes, B. J.', 'Goddard, M. E.'], year=2001,
         venue='Genetics, 157(4), 1819–1829', doi='10.1093/genetics/157.4.1819',
         url='https://doi.org/10.1093/genetics/157.4.1819',
         brief='基因组预测奠基工作：用全基因组稠密标记估计总遗传值，提出 BayesA/BayesB 等。',
         relation='基因组预测方法学基础；RR-BLUP/GBLUP 等基础参照的理论源头。',
         verification='书目沿用 law 在线核验（REFERENCES.md）；DOI 10.1093/genetics/157.4.1819 本轮经 Crossref 一手 API 核验解析（题名/作者一致）。',
         level='historical', pdf=None),
    dict(key='vanraden2008efficient', group='gp', type='article',
         title='Efficient methods to compute genomic predictions',
         authors=['VanRaden, P. M.'], year=2008,
         venue='Journal of Dairy Science, 91(11), 4414–4423', doi='10.3168/jds.2007-0980',
         url='https://doi.org/10.3168/jds.2007-0980',
         brief='提出高效计算基因组预测的方法与基因组关系矩阵 G（VanRaden 法）。',
         relation='GBLUP/关系矩阵基础；本项目谱系矩阵 A 仅用于 Stage 2 亲缘敏感性。',
         verification='书目沿用 law 在线核验（REFERENCES.md）；DOI 10.3168/jds.2007-0980 本轮经 Crossref 一手 API 核验解析。',
         level='historical', pdf=None),
    dict(key='crossa2010prediction', group='gp', type='article',
         title='Prediction of genetic values of quantitative traits in plant breeding using pedigree and molecular markers',
         authors=['Crossa, J.', 'et al.'], year=2010,
         venue='Genetics, 186(2), 713–724', doi='10.1534/genetics.110.118521', pmc='PMC2954475',
         url='https://doi.org/10.1534/genetics.110.118521',
         brief='植物育种中结合谱系与分子标记预测数量性状遗传值；在 599×1279 小麦数据上报告 marker-only RKHS。',
         relation='小麦/植物 GP 基础；CIMMYT 小麦数据语境。docs/history/RESEARCH_PLAN.md 与 RESULTS.md 以 PMC2954475 引用其 RKHS 先例。',
         verification='书目沿用 law 在线核验（REFERENCES.md）；原构造 DOI 10.1093/genetics/186.2.713 经 Crossref 返回 404，本轮更正为真实 DOI 10.1534/genetics.110.118521（Genetics/OUP）；PMC2954475 取自 docs。',
         level='historical', pdf=None),
    dict(key='deloscampos2013whole', group='gp', type='article',
         title='Whole-genome regression and prediction methods applied to plant and animal breeding',
         authors=['de los Campos, G.', 'Hickey, J. M.', 'Pong-Wong, R.', 'Daetwyler, H. D.', 'Calus, M. P. L.'],
         year=2013, venue='Genetics, 193(2), 327–345', doi='10.1534/genetics.112.143313',
         url='https://doi.org/10.1534/genetics.112.143313',
         brief='全基因组回归与预测方法在动植物育种中的综述。',
         relation='全基因组回归方法学基础（含贝叶斯/岭回归类）。',
         verification='书目沿用 law 在线核验（REFERENCES.md）；原构造 DOI 10.1093/genetics/193.2.327 经 Crossref 返回 404，本轮更正为真实 DOI 10.1534/genetics.112.143313（Genetics/OUP）。',
         level='historical', pdf=None),
    dict(key='crossa2017genomic', group='gp', type='article',
         title='Genomic selection in plant breeding: methods, models, and perspectives',
         authors=['Crossa, J.', 'et al.'], year=2017,
         venue='Trends in Plant Science, 22(11), 961–975', doi='10.1016/j.tplants.2017.08.011',
         url='https://doi.org/10.1016/j.tplants.2017.08.011',
         brief='植物育种基因组选择的方法、模型与展望综述。',
         relation='GP 评估规范与方法学语境。',
         verification='书目沿用 law 在线核验（REFERENCES.md）；DOI 10.1016/j.tplants.2017.08.011 本轮经 Crossref 一手 API 核验解析（题名/作者一致）。',
         level='historical', pdf=None),
    dict(key='cawley2010overfitting', group='gp', type='article',
         title='On over-fitting in model selection and subsequent selection bias in performance evaluation',
         authors=['Cawley, G. C.', 'Talbot, N. L. C.'], year=2010, venue='JMLR, 11, 2079–2107',
         url='https://jmlr.org/papers/v11/cawley10a.html',
         brief='论证模型选择中的过拟合及随后性能评估的选择偏差；嵌套验证可缓解。',
         relation='本项目无泄漏底线与嵌套选模的方法学依据；有限验证数据反复选模可能产生选择偏差（docs/methods/MULPHE_UNIFIED.md 引用）。',
         verification='书目沿用 law 在线核验（REFERENCES.md）；JMLR 链接 cawley10a 已在 docs/methods/MULPHE_UNIFIED.md 引用。',
         level='historical', pdf=None),
    dict(key='vabalas2019machine', group='gp', type='article',
         title='Machine learning algorithm validation with a limited sample size',
         authors=['Vabalas, A.', 'Gowen, E.', 'Poliakoff, E.', 'Casson, A. J.'], year=2019,
         venue='PLOS ONE, 14(11), e0224365', doi='10.1371/journal.pone.0224365',
         url='https://doi.org/10.1371/journal.pone.0224365',
         brief='小样本下机器学习算法验证的偏差与交叉验证行为研究。',
         relation='小样本验证方法学；Wheat599 n=599 属小样本语境。',
         verification='书目沿用 law 在线核验（REFERENCES.md）；DOI 10.1371/journal.pone.0224365 本轮经 Crossref 一手 API 核验解析。',
         level='historical', pdf=None),
    dict(key='bengio2004no', group='gp', type='article',
         title='No unbiased estimator of the variance of K-fold cross-validation',
         authors=['Bengio, Y.', 'Grandvalet, Y.'], year=2004, venue='JMLR, 5, 1089–1105',
         url='https://www.jmlr.org/papers/v5/grandvalet04a.html',
         brief='证明 K 折交叉验证方差不存在无偏估计。',
         relation='本项目“10 折不独立，只描述、不做 p/CI”的方法学依据之一（docs/EVALUATION_PROTOCOL.md 引用）。',
         verification='书目沿用 law 在线核验（REFERENCES.md）；JMLR 链接 grandvalet04a 取自 docs/EVALUATION_PROTOCOL.md（修正先前误构的 bengio04a）。',
         level='historical', pdf=None),
    dict(key='nadeau2003inference', group='gp', type='article',
         title='Inference for the generalization error',
         authors=['Nadeau, C.', 'Bengio, Y.'], year=2003,
         venue='Machine Learning, 52(3), 239–281', doi='10.1023/A:1024068626366',
         url='https://doi.org/10.1023/A:1024068626366',
         brief='泛化误差的推断与修正重采样 t 检验。',
         relation='泛化误差推断方法学；本项目不做 p/CI 的相关背景。',
         verification='书目沿用 law 在线核验（REFERENCES.md）；DOI 10.1023/A:1024068626366 本轮经 Crossref 一手 API 核验解析。',
         level='historical', pdf=None),
    dict(key='dietterich1998approximate', group='gp', type='article',
         title='Approximate statistical tests for comparing supervised classification learning algorithms',
         authors=['Dietterich, T. G.'], year=1998,
         venue='Neural Computation, 10(7), 1895–1923', doi='10.1162/089976698300017197',
         url='https://doi.org/10.1162/089976698300017197',
         brief='比较监督学习算法的近似统计检验（含 5×2 折配对 t 检验）。',
         relation='配对检验方法学背景；本项目折均比较只描述、不做 p/CI。',
         verification='书目沿用 law 在线核验（REFERENCES.md）；原构造 DOI …017195 经 Crossref 返回 404，本轮更正为真实 DOI 10.1162/089976698300017197。',
         level='historical', pdf=None),
    dict(key='demsar2006statistical', group='gp', type='article',
         title='Statistical comparisons of classifiers over multiple data sets',
         authors=['Demšar, J.'], year=2006, venue='JMLR, 7, 1–30',
         url='https://jmlr.org/papers/v7/demsar06a.html',
         brief='多数据集分类器比较的 Friedman + Nemenyi 非参数检验框架。',
         relation='多模型比较方法学背景。',
         verification='书目沿用 law 在线核验（REFERENCES.md）；JMLR 链接 demsar06a 本轮经一手摘要页核验（题名/作者一致）。',
         level='historical', pdf=None),
    dict(key='benavoli2017time', group='gp', type='article',
         title='Time for a change: a tutorial for comparing multiple classifiers through Bayesian analysis',
         authors=['Benavoli, A.', 'Corani, G.', 'Demšar, J.', 'Zaffalon, M.'], year=2017,
         venue='JMLR, 18(77), 1–36',
         url='https://www.jmlr.org/papers/v18/16-305.html',
         brief='用贝叶斯分析比较多分类器的教程（替代 NHST）。',
         relation='贝叶斯配对比较方法学背景。',
         verification='书目沿用 law 在线核验（REFERENCES.md）；原链接 16-568.html 经一手核验指向他文（Zhang & Xiao, SPDC, 18(84)），本轮更正为真实 JMLR 链接 16-305.html（题名/作者一致）。',
         level='historical', pdf=None),
    dict(key='bouckaert2004evaluating', group='gp', type='inproceedings',
         title='Evaluating the replicability of significance tests for comparing learning algorithms',
         authors=['Bouckaert, R. R.', 'Frank, E.'], year=2004,
         venue='LNCS, 3056, 3–12', doi='10.1007/978-3-540-24775-3_3',
         url='https://doi.org/10.1007/978-3-540-24775-3_3',
         brief='评估比较学习算法显著性检验的可重复性。',
         relation='重复 CV 检验方法学背景。',
         verification='书目沿用 law 在线核验（REFERENCES.md）；DOI 10.1007/978-3-540-24775-3_3 本轮经 Crossref 一手 API 核验解析。',
         level='historical', pdf=None),
    dict(key='wainer2021nested', group='gp', type='article',
         title='Nested cross-validation when selecting classifiers is overzealous for most practical applications',
         authors=['Wainer, J.', 'Cawley, G.'], year=2021,
         venue='Expert Systems with Applications, 171, 115222', doi='10.1016/j.eswa.2021.115222',
         url='https://doi.org/10.1016/j.eswa.2021.115222',
         brief='讨论嵌套交叉验证在选择分类器时是否过度；支撑“嵌套 CV 给无偏泛化估计”。',
         relation='反方视角：不支撑“嵌套 CV 总是必要”；本项目超参网格大，嵌套属必要侧。',
         verification='书目沿用 law 在线核验（REFERENCES.md）；原构造 DOI 10.1016/j.eswa.2020.115222 经 Crossref 返回 404，本轮更正为真实 DOI 10.1016/j.eswa.2021.115222（ScienceDirect S0957417421006540/作者 UEA 存档确认）。',
         level='historical', pdf=None),

    # ================= docs 方法方案引用的其他来源 =================
    dict(key='rakitsch2013noise', group='plan', type='inproceedings',
         title='It is all in the noise: Efficient multi-task Gaussian process inference with structured residuals',
         authors=['Rakitsch, B.', 'Lippert, C.', 'Borgwardt, K.', 'Stegle, O.'], year=2013, venue='NeurIPS 2013',
         url='https://papers.nips.cc/paper/5089-it-is-all-in-the-noise-efficient-multi-task-gaussian-process-inference-with-structured-residuals',
         brief='区分任务信号与残差协方差的结构化残差多任务高斯过程推断。',
         relation='BH-LMC/GET-ICM 的结构化残差多任务核家族方法定位（docs/RESULTS.md、Wheat599_BH-LMC.md）。',
         verification='本轮在线核验：NeurIPS 2013 一手摘要页确认题名/作者（Rakitsch, Lippert, Borgwardt, Stegle）；链接取自 docs/methods/Wheat599_BH-LMC.md、RESULTS.md。',
         level='online', pdf=None),
    dict(key='gianola2016crossvalidation', group='plan', type='article',
         title='Cross-Validation Without Doing Cross-Validation in Genome-Enabled Prediction',
         authors=['Gianola, D.', 'Schön, C.-C.'], year=2016, venue='G3', doi='10.1534/g3.116.033381',
         pmc='PMC5068934', pmid='27489209',
         url='https://doi.org/10.1534/g3.116.033381',
         brief='基因组预测中无需实际做交叉验证的解析条件 CV / 核预测代数。',
         relation='条件 CV/核预测代数基础；同一 Wheat599 数据上的解析交叉验证与多核（docs/methods/Wheat599_BH-LMC.md、RESULTS.md）。',
         note='RESULTS.md 记录：原报告误标 Genetics、DOI 10.1534/genetics.115.185850；实际发表于 G3，正确 DOI 10.1534/g3.116.033381。',
         verification='题名/期刊/DOI 取自 docs/methods/Wheat599_BH-LMC.md，并经 RESULTS.md 纠错记录确认（G3 非 Genetics）；本轮 Crossref 一手 API 核验 DOI 10.1534/g3.116.033381 解析（题名/作者一致）。',
         level='online', pdf=None),
    dict(key='cuevas2018genomic', group='plan', type='article',
         title='Genomic-Enabled Prediction Kernel Models with Random Intercepts for Multi-environment Trials',
         authors=['Cuevas', 'Granato', 'Fritsche-Neto', 'Montesinos-Lopez', 'Burgueño', 'Bandeira e Sousa', 'Crossa'], year=2018, venue='G3', doi='10.1534/g3.117.300454',
         url='https://doi.org/10.1534/g3.117.300454',
         brief='带随机截距的多环境试验基因组核预测模型。',
         relation='多环境基因组核模型的相关先例（docs/methods/Wheat599_BH-LMC.md）。',
         verification='本轮在线核验：Crossref 一手 API 确认题名/DOI/作者姓氏（Cuevas, Granato, Fritsche-Neto 等）；链接取自 docs/methods/Wheat599_BH-LMC.md。作者名字待补。',
         level='online', pdf=None),
    dict(key='gonen2008localized', group='plan', type='inproceedings',
         title='Localized Multiple Kernel Learning',
         authors=['Gönen, M.', 'Alpaydın, E.'], year=2008, venue='ICML 2008',
         url='https://icml.cc/Conferences/2008/papers/158.pdf',
         brief='输入相关门控组合局部核的多核学习。',
         relation='门控多核方法定位：已使用输入相关门控组合局部核（docs/RESULTS.md）；不能将门控/多核单独宣称首创。',
         verification='链接 icml.cc/Conferences/2008/papers/158.pdf 本轮解析为真实 PDF；题名/作者据通识（Gönen & Alpaydın 2008），PDF 无可提取元数据，本轮未逐字核验，写作引用前复核。',
         level='canonical', pdf=None),
    dict(key='genetic_heterogeneity_wgr', group='plan', type='article',
         title='Incorporating Genetic Heterogeneity in Whole-Genome Regressions Using Interactions',
         authors=['de los Campos, G.', 'Veturi, Y.', 'Vazquez, A. I.', 'Lehermeier, C.', 'Pérez-Rodríguez, P.'],
         year=2015, venue='Journal of Agricultural, Biological and Environmental Statistics, 20(4), 467–490',
         doi='10.1007/s13253-015-0222-5', pmc='PMC4666286', pmid='26660276',
         url='https://pmc.ncbi.nlm.nih.gov/articles/PMC4666286/',
         brief='用交互项在全基因组回归中纳入遗传异质性（群体相关标记效应）。',
         relation='遗传异质性/群体相关标记效应先例（docs/RESULTS.md 方法定位）。',
         verification='本轮在线核验：PMC4666286 一手确认题名/作者（de los Campos 等）/JABES 20(4):467–490/DOI/PMID；年份 2015。',
         level='online', pdf=None),
    dict(key='jiang2015epistasis', group='plan', type='article',
         title='Modeling Epistasis in Genomic Selection',
         authors=['Jiang, Y.', 'Reif, J. C.'], year=2015, venue='Genetics, 201(2), 759–768',
         doi='10.1534/genetics.115.177907', pmc='PMC4596682', pmid='26219298',
         url='https://pmc.ncbi.nlm.nih.gov/articles/PMC4596682/',
         brief='基因组选择中建模上位性；比较加性、加性×加性互作与高斯核等预测。',
         relation='核方法用于基因组预测的先例；高斯核预测贡献不能直接解释为生物学上位性（docs/history/RESEARCH_PLAN.md）。',
         verification='本轮在线核验：PMC4596682 一手确认题名/作者（Jiang & Reif）/Genetics 201(2):759–768/DOI/PMID。',
         level='online', pdf=None),
    dict(key='joint_quantile_rkhs2016', group='plan', type='inproceedings',
         title='Joint quantile regression in vector-valued RKHSs',
         authors=['Sangnier, M.', 'Fercoq, O.', "d'Alché-Buc, F."], year=2016, venue='NeurIPS 2016',
         url='https://papers.neurips.cc/paper_files/paper/2016/hash/dfce06801e1a85d6d06f1fdd4475dacd-Abstract.html',
         brief='向量值 RKHS 中的联合分位数回归。',
         relation='RELATION_QUANTILE 的数学近邻；本实现不宣称 GP 或严格 RKHS 等价（docs/methods/RELATION_QUANTILE.md）。',
         verification='本轮在线核验：NeurIPS 2016 一手摘要页确认题名/作者（Sangnier, Fercoq, d’Alché-Buc）；链接取自 docs/methods/RELATION_QUANTILE.md。',
         level='online', pdf=None),
    dict(key='rudner2023functionspace', group='plan', type='inproceedings',
         title='Function-Space Regularization in Neural Networks: A Probabilistic Perspective',
         authors=['Rudner, T. G. J.', 'Kapoor, S.', 'Qiu, S.', 'Wilson, A. G.'], year=2023,
         venue='ICML 2023, PMLR 202:29275–29290',
         url='https://proceedings.mlr.press/v202/rudner23a.html',
         brief='函数空间正则化的概率视角（FS-EB），将变分推断置于函数空间。',
         relation='RELATION_QUANTILE 的函数空间正则化近邻（docs/methods/RELATION_QUANTILE.md）。',
         verification='本轮在线核验：PMLR v202 rudner23a 一手摘要页确认题名/作者/卷页；链接取自 docs/methods/RELATION_QUANTILE.md。',
         level='online', pdf=None),
    dict(key='rfm_repo', group='plan', type='software',
         title='recursive_feature_machines（LaplaceRFM / AGOP，固定提交）',
         authors=['Radhakrishnan, A.', 'et al.'], year=2022,
         venue='GitHub: aradha/recursive_feature_machines @ 58bd5404bd252d1f4e7e20e01f103a3fcb5a65b2（MIT）',
         url='https://github.com/aradha/recursive_feature_machines/tree/58bd5404bd252d1f4e7e20e01f103a3fcb5a65b2',
         brief='递归特征机：核求解与 AGOP 更新（LaplaceRFM）。',
         relation='RFM_DEVELOPMENT 直接调用作者 LaplaceRFM，完整 1279×1279 特征矩阵、无梯度中心化、每环境单独拟合（docs/methods/RFM_DEVELOPMENT.md）。',
         note='GitHub 仓库 MIT，创建于 2022-12；README 关联 Science 论文 Mechanism for feature learning in neural networks and backpropagation-free machine learning models（DOI 10.1126/science.adi5639）；owner 账号 aradha，作者全名页面未列出。',
         verification='本轮在线核验：GitHub 仓库页确认 MIT 许可/创建年/关联 Science 论文；固定提交链接取自 docs/methods/RFM_DEVELOPMENT.md。',
         level='online', pdf=None),
    dict(key='cwags2026', group='plan', type='article',
         title='CWAGS: multi-trait genomic selection using channel weighted attention convolutional network',
         authors=['Cao', 'Mohamed', 'Sunar', 'Chan'], year=2026, venue='BMC Genomics', doi='10.1186/s12864-026-12980-9',
         url='https://link.springer.com/article/10.1186/s12864-026-12980-9',
         brief='报告 Wheat599 四环境平均 PCC 0.5368，声明十折均值、固定种子、AdamW、400 epoch。',
         relation='未列源码链接/实际折 ID/完整选模信息；GitHub 搜索 CWAGS genomic 未发现匹配仓库（2026-11-06）；暂不能忠实适配或同协议排名，不自行搭建冒充作者实现（docs/methods/SEED_STABILITY.md）。',
         note='作者为 Crossref 姓氏（名字待补）；正式引用前复核。',
         verification='本轮在线核验：Crossref 一手 API 确认题名/DOI/作者姓氏（Cao, Mohamed, Sunar, Chan）；描述取自 docs/methods/SEED_STABILITY.md。',
         level='online', pdf=None),
    dict(key='hybrid_kernel2025', group='plan', type='article',
         title='Enhancing wheat genomic prediction by a hybrid kernel approach',
         authors=['Cuevas, J.', 'Crossa, J.', 'Montesinos-López, A.', 'et al.'], year=2025,
         venue='Frontiers in Plant Science, 16, 1605202', doi='10.3389/fpls.2025.1605202',
         url='https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2025.1605202/full',
         brief='使用标记和谱系关系矩阵的混合核研究，报告不同交叉验证方案。',
         relation='单环境 S5 属额外信息输入类别；不与现有仅标记模型十折结果直接排名；wheat.A 与该论文矩阵/行序一致性须核验；本轮不训练（docs/methods/SEED_STABILITY.md）。',
         verification='本轮在线核验：Frontiers 一手页面（JSON-LD ScholarlyArticle）确认题名/作者（Cuevas, Crossa, Montesinos-López 等）/卷 16 文章号 1605202/DOI；描述取自 docs/methods/SEED_STABILITY.md。',
         level='online', pdf=None),
    dict(key='distill_semiparametric2021', group='plan', type='article',
         title='Knowledge Distillation as Semiparametric Inference',
         authors=['待核（Microsoft Research）'], year=2021, venue='Microsoft Research',
         url='https://www.microsoft.com/en-us/research/publication/knowledge-distillation-as-semiparametric-inference-paper/',
         brief='把知识蒸馏视为半参数推断，讨论教师过拟合与交叉拟合。',
         relation='DISTILL_QUANTILE 的依据之一（知识蒸馏与交叉拟合，而非新神经骨干）（docs/methods/DISTILL_QUANTILE.md）。',
         verification='题名/年份/链接取自 docs/methods/DISTILL_QUANTILE.md；作者待核。',
         level='canonical', pdf=None),
    dict(key='distill_tabular2026', group='plan', type='article',
         title='Distillation of Tabular Foundation Models into Efficient Predictors',
         authors=['待核'], year=2026, venue='arXiv:2610.01435', arxiv='2610.01435',
         url='https://arxiv.org/abs/2610.01435',
         brief='把表格基础模型蒸馏为高效预测器，给出 TabM 等学生的蒸馏证据。',
         relation='其全训练上下文、合成查询条件与本项目不同，不作为本项目全上下文或 CutMix 的依据（docs/methods/DISTILL_QUANTILE.md）。',
         verification='题名/链接取自 docs/methods/DISTILL_QUANTILE.md；作者待核。',
         level='canonical', pdf=None),
    dict(key='xyz_interaction2018', group='plan', type='article',
         title='The xyz algorithm for fast interaction search in high-dimensional data',
         authors=['Thanei, G.-A.', 'Meinshausen, N.', 'Shah, R. D.'], year=2018, venue='JMLR, 19(37), 1–42',
         url='https://www.jmlr.org/papers/volume19/16-515/16-515.pdf',
         brief='高维数据快速交互搜索的 xyz 算法（二值输入交互搜索与全空间 KKT 条件）。',
         relation='交互搜索的成熟算法基础；本项目实现统计参照，不命名为新神经架构（docs/methods/SPARSE_INTERACTION_DIAGNOSTIC.md）。',
         verification='本轮在线核验：JMLR 19(37) 一手摘要页确认题名/作者（Thanei, Meinshausen, Shah）；链接取自 docs/methods/SPARSE_INTERACTION_DIAGNOSTIC.md。',
         level='online', pdf=None),
    dict(key='intht2019', group='plan', type='inproceedings',
         title='Interaction Hard Thresholding: Consistent Sparse Quadratic Regression in Sub-quadratic Time and Space',
         authors=['Yang, S.', 'Shen, Y.', 'Sanghavi, S.'], year=2019, venue='NeurIPS 2019',
         url='https://proceedings.neurips.cc/paper/2019/file/3c7417b8df0daf23f39f445e740c7a43-Paper.pdf',
         brief='交互硬阈值（IntHT）稀疏二次回归，亚二次时间/空间，含设计条件保证。',
         relation='稀疏二次回归的成熟基础（docs/methods/SPARSE_INTERACTION_DIAGNOSTIC.md）。',
         verification='本轮在线核验：NeurIPS 2019 一手摘要页确认题名/作者（Yang, Shen, Sanghavi）；链接取自 docs/methods/SPARSE_INTERACTION_DIAGNOSTIC.md。',
         level='online', pdf=None),
    dict(key='lista2010', group='plan', type='inproceedings',
         title='Learning Fast Approximations of Sparse Coding (LISTA)',
         authors=['Gregor, K.', 'LeCun, Y.'], year=2010, venue='ICML 2010',
         url='https://icml.cc/2010/papers/449.pdf',
         brief='学习 ISTA 的快速近似（LISTA），可学习稀疏推断。',
         relation='可学习稀疏推断的成熟基础（docs/methods/SPARSE_INTERACTION_DIAGNOSTIC.md）。',
         verification='链接 icml.cc/2010/papers/449.pdf 本轮解析为真实 PDF；题名/作者据通识（Gregor & LeCun 2010 LISTA），PDF 无可提取元数据，本轮未逐字核验，写作引用前复核。',
         level='canonical', pdf=None),
    dict(key='mrmash2023', group='plan', type='article',
         title='A flexible empirical Bayes approach to multivariate multiple regression, and its improved accuracy in predicting multi-tissue gene expression from genotypes',
         authors=['Morgante', 'Carbonetto', 'Wang', 'Zou', 'Sarkar', 'Stephens'], year=2023, venue='PLOS Genetics', doi='10.1371/journal.pgen.1010539',
         url='https://journals.plos.org/plosgenetics/article?id=10.1371/journal.pgen.1010539',
         brief='多元多重回归的灵活经验贝叶斯方法（mr.mash），自适应共享/收缩效应，提升从基因型预测多组织基因表达的精度。',
         relation='多元效应共享的成熟基础（docs/methods/SPARSE_INTERACTION_DIAGNOSTIC.md）。',
         note='作者为 Crossref 姓氏（名字待补）；正式引用前复核。',
         verification='本轮在线核验：Crossref 一手 API 确认题名/DOI/作者姓氏（Morgante 等）；PLOS Genetics 2023。',
         level='online', pdf=None),
    dict(key='menet2025', group='plan', type='article',
         title='MeNet: A mixed-effect deep neural network for multi-environment genomic prediction of agronomic traits',
         authors=['Li', 'Ren', 'Li', 'Lee', 'Wan', 'Gan'], year=2025, venue='Plant Communications', doi='10.1016/j.xplc.2025.101620',
         url='https://doi.org/10.1016/j.xplc.2025.101620',
         code='GANLAB/MENET Version2.0 @ 06456e4542ab26716c4db8dd8f17517aa5155ff4（Academic Use Licence，内部非商业）',
         brief='混合效应深度神经网络用于多环境农艺性状基因组预测；表型相关遗传背景嵌入与跨信息特征融合。',
         relation='MENET_QUANTILE 上游：原样动态加载 TraitSpecificEncoderForRepGeno/RepGeno/CrossInformationFeatureFusion；源码/数据/权重不入 Git；Academic Use Licence 限内部非商业学术研究（docs/methods/MENET_QUANTILE.md、RESEARCH_PLAN.md）。',
         note='作者为 Crossref 姓氏（名字待补）；正式引用前复核。',
         verification='本轮在线核验：Crossref 一手 API 确认题名/DOI/作者姓氏（Li 等）；固定提交/许可取自 docs/methods/MENET_QUANTILE.md。',
         level='online', pdf=None),
    dict(key='batchensemble2020', group='plan', type='inproceedings',
         title='BatchEnsemble: An Alternative Approach to Efficient Ensemble and Lifelong Learning',
         authors=['Wen, Y.', 'Tran, D.', 'Ba, J.'], year=2020, venue='ICLR 2020',
         url='https://research.google/pubs/batchensemble-an-alternative-approach-to-efficient-ensemble-and-lifelong-learning/',
         brief='共享矩阵乘性调制的高效集成与终身学习。',
         relation='ENVIRONMENT_TABM 沿用其共享矩阵乘性调制机制，不作为新集成原理（docs/methods/ENVIRONMENT_TABM.md）。',
         verification='本轮在线核验：Google Research 一手页面确认题名/作者（Wen, Tran, Ba）/ICLR 2020；链接取自 docs/methods/ENVIRONMENT_TABM.md。',
         level='online', pdf=None),
    dict(key='tabr2023', group='plan', type='article',
         title='TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023',
         authors=['Gorishniy, Y.', 'Rubachev, I.', 'Kartashev, N.', 'Shlenskii, D.', 'Kotelnikov, A.', 'Babenko, A.'],
         year=2023, venue='arXiv:2307.14338', arxiv='2307.14338',
         url='https://arxiv.org/abs/2307.14338',
         code='yandex-research/tabular-dl-tabr',
         brief='检索增强的表格深度学习：标签编码与差异校正。',
         relation='RETRIEVAL_TABM 依据 TabR 官方代码的标签编码与差异校正；RESEARCH_PLAN 检索机制边界（docs/methods/RETRIEVAL_TABM.md、RESEARCH_PLAN.md）。',
         verification='本轮在线核验：arXiv:2307.14338 一手摘要确认题名/作者（Gorishniy 等）；代码链接取自 docs/methods/RETRIEVAL_TABM.md、RESEARCH_PLAN.md。',
         level='online', pdf=None),
    dict(key='ragp2026', group='plan', type='article',
         title='RAGP: a retrieval-augmented deep learning model for genomic prediction in crop breeding',
         authors=['Zuo', 'Li', 'Wang', 'Man', 'Yang'], year=2026, venue='Plant Methods', doi='10.1186/s13007-026-01574-x',
         url='https://doi.org/10.1186/s13007-026-01574-x',
         code='公开源码 @ 85675b1e8054fe3e91fdb3e974150f1dde81eee3',
         brief='检索增强基因组预测：PCA95 输入、逐环境单输出、Pearson 损失、triplet 对比、cosine top-5、特征融合 0.4、邻居标签乘 0.01。',
         relation='RETRIEVAL_TABM 直接相关参照；已单独同协议适配；PCA 只在 fit 拟合，cosine 库改材料 ID 严格排除并记录差异；论文原划分分数不用于本协议排名（docs/methods/RETRIEVAL_TABM.md）。',
         note='作者为 Crossref 姓氏（名字待补）；正式引用前复核。',
         verification='本轮在线核验：Crossref 一手 API 确认题名/DOI/作者姓氏（Zuo 等）；固定提交/配置取自 docs/methods/RETRIEVAL_TABM.md。',
         level='online', pdf=None),
    dict(key='gpfn2025', group='plan', type='article',
         title='GPFN：基因组预测摊销推断（题名待核）',
         authors=['待核'], year=None, venue='PubMed 40811178',
         url='https://pubmed.ncbi.nlm.nih.gov/40811178/',
         brief='基因组预测的模拟任务摊销推断，报告植物数据结果（据语境）。',
         relation='MATRIX_INFERENCE/JOINT_INFERENCE 的摊销推断先例；跨任务学习不能单独作为原创贡献（docs/methods/MATRIX_INFERENCE.md）。',
         verification='仅 docs/methods/MATRIX_INFERENCE.md 给出 PubMed 40811178 链接；本轮 PubMed 返回 JS 挑战页，未取得题名/作者/年份，保持待核。',
         level='pending', pdf=None),
    dict(key='linearpfn2026', group='plan', type='article',
         title='LinearPFN：主效应与成对交互摊销推断（题名待核）',
         authors=['待核'], year=2026, venue='arXiv:2610.06580', arxiv='2610.06580',
         url='https://arxiv.org/abs/2610.06580',
         code='schiekiera/LinearPFN @ 97ac4f970df655b12d40a3b4a35fa3b2264ce6dd',
         brief='覆盖主效应与成对交互的摊销推断，发布模型预测变量范围 2–30；逐单元双轴注意力、可选原始成对统计量。',
         relation='MATRIX_INFERENCE/JOINT_INFERENCE 的交互摊销推断先例；双向矩阵计算与交互统计量不能单独作为原创贡献（docs/methods/MATRIX_INFERENCE.md、JOINT_INFERENCE.md）。',
         verification='arXiv/代码链接取自 docs/methods/MATRIX_INFERENCE.md、JOINT_INFERENCE.md；确切题名/作者待核。',
         level='pending', pdf=None),
    dict(key='gwkbr2026', group='plan', type='article',
         title='GWKBR: a novel method integrating machine learning and Bayesian inference framework to improve genomic prediction accuracy',
         authors=['Wang', 'Jiang', 'Zhang', 'Zhang'], year=2026, venue='Briefings in Bioinformatics', doi='10.1093/bib/bbag133',
         url='https://doi.org/10.1093/bib/bbag133',
         code='Wangxuer521/GWKBR @ d9019e012cae4d7207eab8c318942450fded67d3',
         brief='K_Model 后验解 (RᵀR + λ_eff·G)⁻¹Rᵀy，G 约束系数精度；有效函数核 R·G⁻¹·Rᵀ。',
         relation='STRUCTURED_COEFFICIENT 依据；原文 Wheat 四环境最终均选 unweighted，不以该结果证明标记加权有效；与加性/RBF 相加有实质区别；文献成绩不与项目结果直接排名（docs/methods/STRUCTURED_COEFFICIENT.md）。',
         note='作者为 Crossref 姓氏（名字待补）；正式引用前复核。',
         verification='本轮在线核验：Crossref 一手 API 确认题名/DOI/作者姓氏（Wang 等）；固定提交/公式取自 docs/methods/STRUCTURED_COEFFICIENT.md。',
         level='online', pdf=None),
    dict(key='subart2024', group='plan', type='article',
         title='Seemingly unrelated Bayesian additive regression trees for cost-effectiveness analyses in healthcare',
         authors=['Esser, J.', 'Maia, M.', 'Parnell, A. C.', 'Bosmans, J.', 'van Dongen, H.', 'Klausch, T.', 'Murphy, K.'],
         year=2024, venue='arXiv:2404.02228（stat.ME）', arxiv='2404.02228',
         url='https://arxiv.org/abs/2404.02228',
         code='MateusMaiaDS/subart @ deddf4ea1290513cd6bce4e7867349d6c32fc2fa（MIT）',
         brief='似不相关贝叶斯加性回归树（subart）：为每个连续/二元结局各建一棵树集成，同时建模结局间依赖（非参数似不相关回归/多元 probit），面向医疗成本效果分析。不是 SoftBART（Chipman 等）。',
         relation='SPARSE_POLYGENIC_BART 复用 subart 仓库的 BART 回归路径（不引入分类/缺失表型/类别重编码）；源码及原许可在 src/wheat599/bayesian_trees/。原文为医疗成本效果的多元似不相关 BART，非基因组学应用（docs/methods/SPARSE_POLYGENIC_BART.md）。',
         note='subart＝seemingly unrelated BART，非 SoftBART；R 包 subart；arXiv v1 2024-04，v3 2025-02。',
         verification='本轮在线核验：arXiv:2404.02228 一手摘要确认题名/作者（Esser, Maia, Parnell 等）；代码/许可取自 docs/methods/SPARSE_POLYGENIC_BART.md。',
         level='online', pdf=None),
    dict(key='dart2018', group='plan', type='article',
         title='Bayesian Regression Trees for High-Dimensional Prediction and Variable Selection',
         authors=['Linero, A. R.'], year=2018, venue='JASA', doi='10.1080/01621459.2016.1264957',
         url='https://doi.org/10.1080/01621459.2016.1264957',
         brief='高维预测与变量选择的贝叶斯回归树（DART 稀疏先验）。',
         relation='SPARSE_POLYGENIC_BART 的稀疏先验来源（docs/methods/SPARSE_POLYGENIC_BART.md）。',
         verification='本轮在线核验：Crossref 一手 API 确认题名/DOI/作者（Linero）；DART 为其稀疏先验方法。',
         level='online', pdf=None),
    dict(key='susie2021', group='plan', type='software',
         title='SuSiE / susieR：单效应之和回归',
         authors=['Wang, G.', 'Sarkar, A.', 'Carbonetto, P.', 'Stephens, M.'], year=2020,
         venue='susieR（stephenslab.github.io/susieR）；主文献 JRSS-B 82:1273–1300',
         url='https://stephenslab.github.io/susieR/',
         brief='相关变量的联合选择与不确定性（sum of single effects）。',
         relation='JOINT_INFERENCE 的先例；本轮不声称这些机制首次提出（docs/methods/JOINT_INFERENCE.md）。',
         note='主文献：Wang, Sarkar, Carbonetto & Stephens (2020) A simple new approach to variable selection in regression, with application to genetic fine mapping. JRSS-B 82:1273–1300；项目页另列 SuSiE 2.0（2025）等。key 年份 2021 与主文献 2020 不一致，以 2020 为准。',
         verification='本轮在线核验：项目页一手确认主文献题名/作者/年份（2020，JRSS-B）；链接取自 docs/methods/JOINT_INFERENCE.md。',
         level='online', pdf=None),
    dict(key='sblgnn2025', group='plan', type='article',
         title='Channel Estimation for XL-MIMO Systems with Decentralized Baseband Processing: Integrating Local Reconstruction with Global Refinement',
         authors=['Tang, A.', 'Wang, J.-B.', 'Pan, Y.', 'Zeng, C.', 'Chen, Y.', 'Yu, H.', 'Xiao, M.', 'de Lamare, R. C.', 'Wang, J.'],
         year=2025, venue='arXiv:2501.17059（接收 IEEE TCOM）', arxiv='2501.17059',
         url='https://arxiv.org/abs/2501.17059',
         brief='SBL-GNN：GNN 增强的稀疏贝叶斯学习，结合神经先验更新与解析后验；原文面向 XL-MIMO 无线信道估计（去中心化基带处理），非基因组学应用。',
         relation='JOINT_INFERENCE 引用其 SBL-GNN（稀疏贝叶斯学习+GNN）机制；原文为无线信道估计应用，作方法参考而非基因组学先例（docs/methods/JOINT_INFERENCE.md）。',
         note='arXiv 2501.17059 一手核验：SBL-GNN 为该无线信道估计论文内部算法；docs 以 SBL-GNN 机制引用，域差异如实标注。',
         verification='本轮在线核验：arXiv:2501.17059 一手摘要确认题名/作者（Tang 等）；SBL-GNN 为其内部算法，应用域为无线信道估计。',
         level='online', pdf=None),
    dict(key='wilson2016dkl', group='plan', type='inproceedings',
         title='Deep Kernel Learning',
         authors=['Wilson, A. G.', 'Hu, Z.', 'Salakhutdinov, R.', 'Xing, E. P.'], year=2016,
         venue='AISTATS 2016, PMLR 51:370–378',
         url='https://proceedings.mlr.press/v51/wilson16.html',
         brief='深核学习：用神经网络学习核函数的输入表示。',
         relation='RESEARCH_PLAN 的深核学习家族；不能把这些成熟技术本身当作原创（docs/history/RESEARCH_PLAN.md）。',
         verification='本轮在线核验：PMLR v51 wilson16 一手摘要页确认题名/作者/卷页；链接取自 docs/history/RESEARCH_PLAN.md。',
         level='online', pdf=None),
    dict(key='bertinetto2019closedform', group='plan', type='article',
         title='Meta-learning with differentiable closed-form solvers',
         authors=['Bertinetto, L.', 'Henriques, J. F.', 'Torr, P. H. S.', 'Vedaldi, A.'], year=2019,
         venue='ICLR 2019（arXiv:1805.08136）', arxiv='1805.08136',
         url='https://arxiv.org/abs/1805.08136',
         brief='可微闭式求解器/读出：元学习中用闭式解（如岭回归）作为可微读出层。',
         relation='RESEARCH_PLAN 的可微闭式读出家族；拟检验差异是位点维保留的自排除背景表示（docs/history/RESEARCH_PLAN.md）。',
         verification='本轮在线核验：arXiv:1805.08136 一手摘要确认题名/作者（Bertinetto 等），页面注明 ICLR 2019；arXiv v1 2018。',
         level='online', pdf=None),
    dict(key='efa2023', group='plan', type='article',
         title='Factorizing polygenic epistasis improves prediction and uncovers biological pathways in complex traits',
         authors=['Tang, D.', 'Freudenberg, J.', 'Dahl, A.'], year=2023,
         venue='American Journal of Human Genetics, 110(11), 1875–1887', doi='10.1016/j.ajhg.2023.10.002',
         pmc='PMC10645564', pmid='37922884',
         url='https://pmc.ncbi.nlm.nih.gov/articles/PMC10645564/',
         brief='因子化多基因上位性（EFA）：低秩分解上位性效应以改善预测并揭示生物学通路。',
         relation='RESEARCH_PLAN 的低秩基因交互先例；本轮考察位点自排除/共享非线性/稀疏调制的具体组合（docs/history/RESEARCH_PLAN.md）。',
         verification='本轮在线核验：PMC10645564 一手确认题名/作者（Tang, Freudenberg, Dahl）/AJHG 110(11):1875–1887/DOI/PMID。',
         level='online', pdf=None),
    dict(key='contextnet2021', group='plan', type='article',
         title='ContextNet: A Click-Through Rate Prediction Framework Using Contextual information to Refine Feature Embedding',
         authors=['Wang, Z.', 'She, Q.', 'Zhang, P.', 'Zhang, J.'], year=2021, venue='arXiv:2107.12025（cs.IR）', arxiv='2107.12025',
         url='https://arxiv.org/abs/2107.12025',
         brief='输入相关的特征调制：用上下文信息精炼特征嵌入（原文面向点击率预测）。',
         relation='RESEARCH_PLAN 的输入相关特征调制先例（docs/history/RESEARCH_PLAN.md）；原文为 CTR 预测应用，作机制参考。',
         note='原文为点击率（CTR）预测框架，非基因组学应用；题名 information 小写为原文 verbatim。',
         verification='本轮在线核验：arXiv:2107.12025 一手摘要确认题名/作者（Wang 等）；应用域为 CTR 预测。',
         level='online', pdf=None),
    dict(key='dnawhisper2026', group='plan', type='article',
         title='DNAwhisper: An Integrated Deep Learning Pyramidal Framework for Multi-Trait Genomic Prediction and Adaptive Marker Prioritisation',
         authors=['Ma, Y.', 'Li, X.', 'Ji, X.', 'Wang, C.', 'Zhang, D.', 'Zhai, T.', 'Wang, H.', 'Liu, P.'],
         year=2026, venue='Plant Biotechnology Journal, 24(6), 4005–4026', doi='10.1111/pbi.70619',
         pmc='PMC13205711', pmid='41757800',
         url='https://pmc.ncbi.nlm.nih.gov/articles/PMC13205711/',
         code='Marxin1992/Whisperer_of_DNA @ 7a657cc00d44263c4b1085d3991ecc0e935c14e5',
         brief='DNA 预训练模型；表 1 的 0.80 是验证集均值，预训练使用全体材料基因型。',
         relation='官方 CSV 入口在划分前按完整表型归一化、预训练配置用全部输入材料、模型代码无条件计算主损失——直接载入将使测试表型参与训练；同协议适配须把预训练与监督限制在训练材料内（docs/history/RESEARCH_PLAN.md）。',
         verification='本轮在线核验：PMC13205711 一手确认题名/作者（Ma 等）/Plant Biotechnol J 24(6):4005–4026/DOI/PMID；固定提交与泄漏分析取自 docs/history/RESEARCH_PLAN.md，结论针对所核查公开代码版本。',
         level='online', pdf=None),
    dict(key='caruana2004ensemble', group='plan', type='inproceedings',
         title='Ensemble selection from libraries of models',
         authors=['Caruana, R.', 'Niculescu-Mizil, A.', 'Crew, G.', 'Ksikes, A.'], year=2004,
         venue='ICML 2004',
         url='https://www.cs.cornell.edu/~caruana/ctp/ct.papers/caruana.icml04.icdm06long.pdf',
         brief='从模型库前向选择集成成员以优化验证分数。',
         relation='模型库预测平均的成熟依据；本项目固定 top-5，不是其完整前向选择算法，不把预测平均等同于对核矩阵求和（docs/history/RESEARCH_PLAN.md）。',
         verification='链接 cs.cornell.edu/~caruana/ctp/ct.papers/caruana.icml04.icdm06long.pdf 本轮解析为真实 PDF；题名/作者据通识（Caruana 等 2004），PDF 无可提取元数据，本轮未逐字核验，写作引用前复核。',
         level='canonical', pdf=None),
    dict(key='mtg2_2016', group='plan', type='article',
         title='MTG2：多性状基因组分析（题名待核）',
         authors=['待核'], year=2016, venue='Bioinformatics, 32(9), 1420',
         url='https://academic.oup.com/bioinformatics/article/32/9/1420/1744518',
         brief='多性状混合模型的协方差估计（MTG2，AI 算法）。',
         relation='RESEARCH_PLAN 参考其协方差估计思想；本实现用 L-BFGS-B 而非 MTG2 的 AI 算法，不宣称复现其软件或新模型家族（docs/history/RESEARCH_PLAN.md）。',
         verification='链接/卷期页取自 docs/history/RESEARCH_PLAN.md；本轮 OUP 返回 403，未取得题名/作者，保持待核。',
         level='pending', pdf=None),
    dict(key='kernelreml2026', group='plan', type='article',
         title='REML implementations of kernel-based genomic prediction models for genotype x environment x management interactions',
         authors=['Melsen, K. A. C.', 'Gezan, S.', 'Tolhurst, D. J.', 'van Eeuwijk, F. A.', 'Peeters, C. F. W.'],
         year=2025, venue='arXiv:2501.06844（stat.ME；v1 2025-01，v3 2026-03）', arxiv='2501.06844',
         url='https://arxiv.org/abs/2501.06844',
         brief='核基因组预测模型的 REML 方差组分估计实现，面向基因型×环境×管理交互。',
         relation='RESEARCH_PLAN 参考；基于材料谱分解的 REML 目标与预测已对照列主序稠密 GLS（误差约 2.2e-16/3.0e-15）（docs/history/RESEARCH_PLAN.md）。',
         note='arXiv v1 2025-01，v3 修订 2026-03；key 年份 2026 对应修订稿。',
         verification='本轮在线核验：arXiv:2501.06844 一手摘要确认题名/作者（Melsen 等）；年份以 v1 2025 计，修订 2026。',
         level='online', pdf=None),
    dict(key='sklearn_pitfalls', group='plan', type='software',
         title='scikit-learn: Common pitfalls and recommended practices',
         authors=['scikit-learn developers'], year=None, venue='scikit-learn 官方文档',
         url='https://scikit-learn.org/stable/common_pitfalls.html',
         brief='训练侧预处理与数据泄漏的常见陷阱及推荐做法。',
         relation='EVALUATION_PROTOCOL 训练侧预处理（scaler/插补只在拟合集拟合）的依据（docs/EVALUATION_PROTOCOL.md）。',
         verification='链接取自 docs/EVALUATION_PROTOCOL.md；官方文档，本轮未重新核验版本。',
         level='canonical', pdf=None),
    dict(key='spyromitrosxioufis2016multitarget', group='plan', type='article',
         title='Multi-Target Regression via Input Space Expansion: Treating Targets as Inputs',
         authors=['Spyromitros-Xioufis, E.', 'Tsoumakas, G.', 'Groves, W.', 'Vlahavas, I.'],
         year=2016, venue='Machine Learning', doi='10.1007/s10994-016-5546-z', arxiv='1211.6581',
         url='https://doi.org/10.1007/s10994-016-5546-z',
         brief='多目标回归的输入空间扩展法（stacked single-target）：把其他目标的预测作为附加输入分阶段学习，以利用目标间依赖。',
         relation='docs/methods/LATE_TRANSFER.md 引用的多目标/目标间依赖先例（stacked multi-target regression）。',
         note='arXiv:1211.6581（2012 初稿，2016 v5）；期刊版 Machine Learning 2016，DOI 10.1007/s10994-016-5546-z。',
         verification='本轮核验：arXiv:1211.6581 一手摘要确认题名/作者（Spyromitros-Xioufis, Tsoumakas, Groves, Vlahavas）；期刊版 Machine Learning 2016 DOI 10.1007/s10994-016-5546-z 据 arXiv journal-ref。',
         level='online', pdf=None),
]

GROUP_TITLE = {
    'main': '一、现役主方法核心出处',
    'method_data': '二、方法与数据（REFERENCES.md 既有）',
    'gp': '三、GP 基础与方法学（REFERENCES.md 既有，law 核验）',
    'plan': '四、docs 方法方案引用的其他来源',
}
GROUP_ORDER = ['main', 'method_data', 'gp', 'plan']

LICENSE_ROWS = [
    ('PriorLabs/TabPFN', 'Apache-2.0（本地核验）'),
    ('Prior-Labs/tabpfn_3_5 权重', 'tabpfn-3-5-license-v1.0，非商用；允许研究/内部基准，禁止商业决策、客户交付、付费产品；全文在 docs/licenses/'),
    ('yandex-research/rtdl-revisiting-models', 'Apache-2.0（仓库 LICENSE 与 wheel dist-info 均核验）'),
    ('yandex-research/tabm', 'Apache-2.0（GitHub API）'),
    ('lasso-net/lassonet', 'MIT（GitHub API，扩展比较）'),
    ('wjhhh2928/Mul-PheG2P', 'MIT（本轮固定提交 README 核验）'),
    ('GANLAB/MENET', 'Academic Use Licence，内部非商业学术研究（docs/methods/MENET_QUANTILE.md）'),
    ('MateusMaiaDS/subart', 'MIT（docs/methods/SPARSE_POLYGENIC_BART.md）'),
    ('jingang/TabICL 回归权重', 'BSD-3-Clause（docs/LICENSE_NOTES.md）'),
    ('BGLR', 'GPL-3'),
]

PITFALLS = [
    'PMC13049863 是 FT-Transformer 用于性传播感染风险预测的应用论文（Natsal-3），不是 FT-Transformer 原始论文；只作应用引用。',
    'TabICL 是 ICML 2025，不是 ICLR；TabICLv2 才是 ICML 2026。',
    'TabM / FT-Transformer 作者为 Yury Gorishniy（Y.）；旧 REFERENCES.md 写 “N.” 有误，本轮据 arXiv 原文 PDF 一手修正。TabM 合著者为 Akim Kotelnikov、Artem Babenko，不要写成 Akhanov 等。',
    'GPformer 与 GPTransformer（大麦 2021）、Cropformer 是不同工作；引用写全称。',
    'Gsformer 的 GY1–GY4 与本仓列名映射未核；不要假设其四任务顺序与 "1","2","4","5" 相同。',
    'Fan & Waldmann 公开代码的四项问题目前是简报审查结论；Regression.zip 尚未逐项复核。冻结文本标“待复核”，不写成已证实事实。',
    '“评审者曾指出全样本标准化”没有找到公开出处；不写入论文或仓库文本，除非给出可引用来源。',
    'Mul-PheG2P 发表版为 New Phytologist 2026;252(1):503–519（DOI 10.1111/nph.71461）；README 著录 2025 为代码/预印年份。标量融合（论文）与隐藏融合（公开默认）不可混称；论文 500/300/200 与公开 train.py 默认 300/300/200 不同，以代码为准。',
    'Gianola & Schön 文章发表于 G3（DOI 10.1534/g3.116.033381），不是 Genetics（旧报告误标 10.1534/genetics.115.185850）。',
    'Huber ICM 是凸 Huber 惩罚风险最小化，不是完整 Huber GP 贝叶斯算法；分位数 TabM 为 raw 全标记、99 个 0.001–0.999 分位点、pinball 求和、成员/材料平均。',
    '第四组多数条目仅据 docs 方法方案引用著录，元数据未在线核验者已标“待核/待复核”；写作引用前须核对一手来源，不编造题名/作者/卷期页。',
]

MATHML = {
    'huber': (
        '<math display="block" xmlns="http://www.w3.org/1998/Math/MathML">'
        '<mrow><mi>ρ</mi><mo>(</mo><mi>t</mi><mo>)</mo><mo>=</mo>'
        '<mrow><mo>{</mo><mtable columnalign="left">'
        '<mtr><mtd><mfrac><mn>1</mn><mn>2</mn></mfrac><msup><mi>t</mi><mn>2</mn></msup></mtd>'
        '<mtd><mrow><mo>|</mo><mi>t</mi><mo>|</mo><mo>&lt;</mo><mi>k</mi></mrow></mtd></mtr>'
        '<mtr><mtd><mi>k</mi><mo>|</mo><mi>t</mi><mo>|</mo><mo>−</mo><mfrac><mn>1</mn><mn>2</mn></mfrac><msup><mi>k</mi><mn>2</mn></msup></mtd>'
        '<mtd><mrow><mo>|</mo><mi>t</mi><mo>|</mo><mo>≥</mo><mi>k</mi></mrow></mtd></mtr>'
        '</mtable></mrow></mrow></math>'
    ),
    'pinball': (
        '<math display="block" xmlns="http://www.w3.org/1998/Math/MathML">'
        '<mrow><msub><mi>L</mi><mi>τ</mi></msub><mo>(</mo><mi>y</mi><mo>,</mo><mover><mi>y</mi><mo>^</mo></mover><mo>)</mo><mo>=</mo>'
        '<mrow><mo>{</mo><mtable columnalign="left">'
        '<mtr><mtd><mi>τ</mi><mo>(</mo><mi>y</mi><mo>−</mo><mover><mi>y</mi><mo>^</mo></mover><mo>)</mo></mtd>'
        '<mtd><mrow><mi>y</mi><mo>≥</mo><mover><mi>y</mi><mo>^</mo></mover></mrow></mtd></mtr>'
        '<mtr><mtd><mo>(</mo><mn>1</mn><mo>−</mo><mi>τ</mi><mo>)</mo><mo>(</mo><mover><mi>y</mi><mo>^</mo></mover><mo>−</mo><mi>y</mi><mo>)</mo></mtd>'
        '<mtd><mrow><mi>y</mi><mo>&lt;</mo><mover><mi>y</mi><mo>^</mo></mover></mrow></mtd></mtr>'
        '</mtable></mrow></mrow></math>'
    ),
}


def esc(text):
    return html.escape(str(text), quote=True)


def rich(text):
    parts = re.split(r'`([^`]+)`', str(text))
    return ''.join(f'<code>{esc(p)}</code>' if i % 2 else esc(p) for i, p in enumerate(parts))


def badge(level):
    return (f'<span class="badge" style="--badge:{LEVEL_COLOR[level]}">'
            f'<span class="dot" aria-hidden="true"></span>{esc(LEVEL_LABEL[level])}</span>')


def apa(ref):
    alist = [a for a in ref['authors'] if a.strip() not in ('', '待核')]
    authors = ', '.join(alist) if alist else '作者待核'
    year = ref['year'] if ref['year'] else 'n.d.'
    bits = [f"{esc(authors)} ({year}).", f"<cite>{esc(ref['title'])}</cite>."]
    if ref.get('venue'):
        bits.append(esc(ref['venue']) + '.')
    return ' '.join(bits)


def bib_authors(ref):
    parts = []
    for a in ref['authors']:
        a = a.strip()
        if a in ('', '待核'):
            continue
        parts.append('others' if a == 'et al.' else a)
    return ' and '.join(parts)


def bibtex(ref):
    key = ref['key']
    authors = bib_authors(ref)
    typ = ref['type']
    fields = [('title', ref['title'])]
    if authors:
        fields.append(('author', authors))
    if ref['year']:
        fields.append(('year', str(ref['year'])))
    if typ in ('software', 'dataset'):
        bibtype = 'misc'
        if ref.get('venue'):
            fields.append(('howpublished', ref['venue']))
    elif typ == 'report':
        bibtype = 'techreport'
        if ref.get('venue'):
            fields.append(('institution', ref['venue']))
    elif typ == 'inproceedings':
        bibtype = 'inproceedings'
        fields.append(('booktitle', ref.get('venue', '')))
    else:
        bibtype = 'article'
        fields.append(('journal', ref.get('venue', '')))
    if ref.get('doi'):
        fields.append(('doi', ref['doi']))
    if ref.get('arxiv'):
        fields.append(('eprint', ref['arxiv']))
        fields.append(('archivePrefix', 'arXiv'))
    if ref.get('url'):
        fields.append(('url', ref['url']))
    notes = []
    if ref.get('code'):
        notes.append(ref['code'])
    if ref.get('level') in ('pending', 'canonical'):
        notes.append('元数据待核，正式引用前复核')
    if notes:
        fields.append(('note', '；'.join(notes)))
    lines = [f"@{bibtype}{{{key},"]
    for i, (name, val) in enumerate(fields):
        comma = ',' if i < len(fields) - 1 else ''
        lines.append(f"  {name} = {{{val}}}{comma}")
    lines.append('}')
    return '\n'.join(lines)


CSS = """
:root {
  --primary-light:#0072B2; --primary-dark:#56B4E9;
  --bg-light:#f6f5ef; --bg-dark:#14181d;
  --text-light:#172e38; --text-dark:#e6e8ea;
  --panel-light:#ffffff; --panel-dark:#1e242b;
  --muted-light:#5a6b73; --muted-dark:#9fb0b8;
  --border-light:#e2e0d6; --border-dark:#323a42;
  --primary:var(--primary-light); --bg:var(--bg-light); --text:var(--text-light);
  --panel:var(--panel-light); --muted:var(--muted-light); --border:var(--border-light);
  color-scheme: light dark;
}
@media (prefers-color-scheme: dark) {
  :root { --primary:var(--primary-dark); --bg:var(--bg-dark); --text:var(--text-dark);
          --panel:var(--panel-dark); --muted:var(--muted-dark); --border:var(--border-dark); }
  .badge { color:var(--text); }
}
@supports (color: light-dark(white, black)) {
  :root { --primary:light-dark(var(--primary-light),var(--primary-dark));
          --bg:light-dark(var(--bg-light),var(--bg-dark));
          --text:light-dark(var(--text-light),var(--text-dark));
          --panel:light-dark(var(--panel-light),var(--panel-dark));
          --muted:light-dark(var(--muted-light),var(--muted-dark));
          --border:light-dark(var(--border-light),var(--border-dark)); }
}
* { box-sizing:border-box; }
html { accent-color:var(--primary); scroll-behavior:smooth; }
body { margin:0; background:var(--bg); color:var(--text);
  font-family:"Source Han Serif SC","Noto Serif CJK SC","Songti SC",Georgia,"Noto Sans CJK SC","PingFang SC","Microsoft YaHei",serif;
  line-height:1.75; text-autospace:normal; overflow-wrap:anywhere; }
:lang(zh) { font-style:normal; }
.wrap { max-width:74rem; margin:0 auto; padding:1.5rem 1.25rem 4rem; }
header.site { border-bottom:2px solid var(--primary); padding-bottom:.75rem; margin-bottom:1.25rem; }
header.site h1 { font-size:1.6rem; margin:0 0 .25rem; line-height:1.4; }
header.site p { margin:.25rem 0; color:var(--muted); }
nav.crumbs { margin:.5rem 0 1.25rem; font-size:.95rem; }
nav.crumbs a { color:var(--primary); text-decoration:none; margin-right:1rem; }
nav.crumbs a:hover, nav.crumbs a:focus-visible { text-decoration:underline; }
nav.toc { background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:.75rem 1.1rem; margin-bottom:1.5rem; }
nav.toc a { color:var(--primary); text-decoration:none; margin-right:1.25rem; white-space:normal; }
nav.toc a:hover { text-decoration:underline; }
h2 { font-size:1.25rem; border-left:4px solid var(--primary); padding-left:.6rem; margin:1.75rem 0 .75rem; scroll-margin-top:1rem; }
a { color:var(--primary); }
.panel { background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:1rem 1.25rem; margin-bottom:1rem; }
.refcard { background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:.9rem 1.15rem; margin-bottom:.9rem; scroll-margin-top:1rem; }
.refcard h3 { margin:0 0 .35rem; font-size:1.04rem; line-height:1.5; }
.refcard .apa { margin:.25rem 0 .5rem; }
.refcard .meta { font-size:.9rem; color:var(--muted); margin:.35rem 0; display:flex; flex-wrap:wrap; gap:.5rem; align-items:center; min-width:0; }
.refcard .meta > * { min-width:0; overflow-wrap:anywhere; }
.refcard p { margin:.4rem 0; }
.label { font-weight:700; color:var(--primary); }
.badge { display:inline-flex; align-items:center; gap:.4em; font-size:.8rem; border:1px solid var(--badge);
  color:var(--badge); border-radius:999px; padding:.1em .7em; white-space:nowrap; }
.badge .dot { width:.6em; height:.6em; border-radius:50%; background:var(--badge); }
code { font-family:"SFMono-Regular",Consolas,"Liberation Mono",monospace; font-size:.88em;
  background:color-mix(in srgb, var(--primary) 10%, transparent); padding:.08em .35em; border-radius:4px; }
details.bib { margin:.5rem 0 .25rem; }
details.bib summary { cursor:pointer; color:var(--primary); font-size:.9rem; }
pre { background:var(--bg); border:1px solid var(--border); border-radius:8px; padding:.7rem .9rem; overflow:auto; margin:.5rem 0 0; }
pre code { background:none; padding:0; }
.table-wrapper { overflow:auto; max-inline-size:100%; container-type:inline-size; border:1px solid var(--border); border-radius:8px; background:var(--panel); }
table { border-collapse:separate; border-spacing:0; width:100%; font-size:.92rem; }
caption { text-align:left; padding:.6rem .8rem; color:var(--muted); font-size:.9rem; }
th, td { text-align:left; padding:.5rem .8rem; border-bottom:1px solid var(--border); vertical-align:top; }
thead th { position:sticky; inset-block-start:0; background:var(--panel); z-index:1; border-bottom:2px solid var(--primary); }
tbody tr:last-child td { border-bottom:none; }
ul.pitfalls li { margin:.45rem 0; }
.note { font-size:.9rem; color:var(--muted); }
math { font-size:1.05em; }
.mathblock { background:var(--bg); border:1px solid var(--border); border-radius:8px; padding:.5rem .9rem; margin:.5rem 0; overflow:auto; }
footer.site { margin-top:3rem; border-top:1px solid var(--border); padding-top:1rem; color:var(--muted); font-size:.9rem; }
@media print {
  :root { --bg:#fff; --text:#000; --panel:#fff; --muted:#333; --border:#999; --primary:#0056b3; }
  nav.crumbs, nav.toc, .noprint { display:none; }
  .refcard, .panel, pre { break-inside:avoid; }
  pre { white-space:pre-wrap; }
  details.bib { display:block; }
  a { color:#000; text-decoration:underline; }
  body { font-size:10.5pt; }
  a[href^="http"]::after { content:" (" attr(href) ")"; font-size:8pt; overflow-wrap:anywhere; }
  thead th { position:static; }
  h2, h3 { break-after:avoid; }
}
"""


def refcard(ref):
    bib = bibtex(ref)
    links = [f'<a href="{esc(ref["url"])}">原文链接</a>']
    if ref.get('doi'):
        links.append(f'<a href="https://doi.org/{esc(ref["doi"])}">DOI</a>')
    if ref.get('pdf'):
        links.append(f'<a href="{esc(ref["pdf"])}">本地 PDF（{ref.get("pdf_pages", "?")} 页）</a>')
    meta = ' · '.join(links)
    code = f'<p><span class="label">代码/许可：</span>{rich(ref["code"])}</p>' if ref.get('code') else ''
    note = f'<p class="note">{rich(ref["note"])}</p>' if ref.get('note') else ''
    math = ''
    if ref.get('math'):
        math = f'<div class="mathblock">{MATHML[ref["math"]]}</div>'
    if ref.get('pdf'):
        pdfline = (f'<p class="note"><span class="label">全文：</span>'
                   f'<a href="{esc(ref["pdf"])}">{esc(ref["pdf"])}</a>（{ref.get("pdf_pages", "?")} 页，本轮下载并核验为完整 PDF）。</p>')
    elif ref.get('pdf_note'):
        pdfline = f'<p class="note"><span class="label">全文可获取性：</span>{rich(ref["pdf_note"])}</p>'
    else:
        pdfline = '<p class="note"><span class="label">全文可获取性：</span>本轮未下载全文；原文链接见上。</p>'
    return f"""<article class="refcard" id="{esc(ref['key'])}">
  <h3>{esc(ref['title'])}</h3>
  <p class="apa">{apa(ref)}</p>
  <p class="meta">{badge(ref['level'])} <span>{meta}</span></p>
  {code}
  <p><span class="label">简讲：</span>{rich(ref['brief'])}</p>
  <p><span class="label">项目关系：</span>{rich(ref['relation'])}</p>
  {note}{math}
  <p><span class="label">核验状态：</span>{rich(ref['verification'])}</p>
  {pdfline}
  <details class="bib"><summary>BibTeX</summary><pre tabindex="0"><code>{esc(bib)}</code></pre></details>
</article>"""


def index_page():
    n_pdf = sum(1 for r in REFERENCES if r.get('pdf'))
    counts = {g: sum(1 for r in REFERENCES if r['group'] == g) for g in GROUP_ORDER}
    toc = ' '.join(f'<a href="#g-{g}">{esc(GROUP_TITLE[g])}（{counts[g]}）</a>' for g in GROUP_ORDER)
    sections = []
    for g in GROUP_ORDER:
        cards = '\n'.join(refcard(r) for r in REFERENCES if r['group'] == g)
        sections.append(f'<section aria-labelledby="g-{g}"><h2 id="g-{g}">{esc(GROUP_TITLE[g])}</h2>{cards}</section>')
    lic_rows = '\n'.join(f'<tr><th scope="row">{esc(a)}</th><td>{rich(b)}</td></tr>' for a, b in LICENSE_ROWS)
    pitfalls = '\n'.join(f'<li>{rich(p)}</li>' for p in PITFALLS)
    legend = ' · '.join(badge(l) + ' ' + esc({
        'historical': '沿用 gb/law 历史核验',
        'online': '本轮在线核验',
        'canonical': '规范著录，写作前复核',
        'pending': '仅文档链接/描述，元数据待核'}[l]) for l in ['historical', 'online', 'canonical', 'pending'])
    body = f"""<div class="wrap">
<header class="site">
  <h1>Wheat599 参考文献库</h1>
  <p>植物基因组表型预测方法论文 · 离线自包含中文研究页 · 共 {len(REFERENCES)} 条文献，{n_pdf} 篇本地全文 PDF。</p>
  <p class="note">单一数据源 <code>ref/build.py</code> 生成；历史核验记录沿用 <code>docs/REFERENCES.md</code>，未覆盖。</p>
</header>
<nav class="crumbs" aria-label="项目导航">
  <a href="../docs/models.html">模型与架构（docs/models.html）</a>
  <a href="../results/index.html">结果导航（results/index.html）</a>
  <a href="references.bib">BibTeX 书目</a>
</nav>
<nav class="toc" aria-label="文献分组目录">{toc}</nav>
<main>
<section aria-label="阅读说明">
  <h2>阅读说明</h2>
  <div class="panel">
    <p>每条文献含<span class="label">简讲</span>、<span class="label">项目关系</span>、<span class="label">原文链接</span>、<span class="label">核验状态</span>与可展开 <span class="label">BibTeX</span>。全部简讲集中于本页（无逐篇独立页面）。</p>
    <p>核验状态徽章（色盲安全 Okabe-Ito 配色，均带文字标签作第二通道）：</p>
    <p>{legend}</p>
    <p class="note">链接与核验说明：核验状态逐条以徽章标注，具体见各卡片“核验状态”。历史核验＝沿用 gb/law 此前一手核验；本轮在线核验＝本轮经一手摘要、Crossref API 或原文核验；规范著录＝完整书目但本轮未在线核验，写作引用前复核；待核＝仅文档链接/描述，题名作者待补。GP 基础文献 DOI 本轮经 Crossref 一手核验，个别返回 404 者已更正为真实 DOI；全文不可获取者明确标注，不冒充全文。</p>
  </div>
</section>
{''.join(sections)}
<section aria-label="代码与权重许可">
  <h2>代码与权重许可</h2>
  <div class="table-wrapper" tabindex="0" role="region" aria-label="代码与权重许可表">
    <table>
      <caption>沿用 docs/REFERENCES.md 许可表，并补本轮与方法文档核验项。</caption>
      <thead><tr><th scope="col">项</th><th scope="col">许可 / 状态</th></tr></thead>
      <tbody>{lic_rows}</tbody>
    </table>
  </div>
</section>
<section aria-label="不引用与易错项">
  <h2>不引用 / 易错项</h2>
  <div class="panel"><ul class="pitfalls">{pitfalls}</ul></div>
</section>
</main>
<footer class="site">
  <p>Wheat599 参考文献库 · 生成自 <code>ref/build.py</code> · 配色与导航对齐 docs/models.html 与 results/index.html。</p>
  <p>核验底线：不编造元数据；未在线核验项明确标注；全文不可获取不冒充。数据只读，本目录不含实验。</p>
</footer>
</div>
</body>
</html>
"""
    head = ('<!DOCTYPE html>\n<html lang="zh-CN">\n<head>\n'
            '<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            '<meta name="color-scheme" content="light dark">\n'
            '<meta name="citation_title" content="Wheat599 参考文献库">\n'
            '<title>Wheat599 参考文献库</title>\n'
            f'<style>{CSS}</style>\n</head>\n<body>\n')
    return "\n".join(line.rstrip() for line in (head + body).splitlines()) + "\n"


def main():
    keys = [r['key'] for r in REFERENCES]
    if len(keys) != len(set(keys)):
        raise ValueError('duplicate reference keys')
    page = index_page()
    bib = '\n\n'.join(bibtex(r) for r in REFERENCES) + '\n'
    (ROOT / 'index.html').write_text(page, encoding='utf-8')
    (ROOT / 'references.bib').write_text(bib, encoding='utf-8')
    print(f"generated index.html + references.bib: {len(REFERENCES)} entries, "
          f"{sum(1 for r in REFERENCES if r.get('pdf'))} local PDFs")
    for g in GROUP_ORDER:
        print(f"  {GROUP_TITLE[g]}: {sum(1 for r in REFERENCES if r['group'] == g)}")


if __name__ == '__main__':
    main()