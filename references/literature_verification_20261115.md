# WS5 文献元数据核验记录（2026-11-15）

任务：AIA REQUEST `req-gb-literature-20261115`（gpt66），为 `docs/models.html` 的【年份＋来源文章＋期刊/DOI】标注提供书目。执行：gb。

方法：Crossref API 一手 DOI 解析（`api.crossref.org/works/<DOI>`）；NeurIPS 官方摘要页与官方 BibTeX；Europe PMC REST（PMID）。复用 `ref/build.py` 既有已核条目，不重复添加。写入：`ref/build.py` 源条目（新增 baseline 组 13 条，升级 Bonilla/Alvarez/CWAGS/RAGP 4 条），`python3 ref/build.py` 重新生成 `references.bib` 与 `index.html`；`tests/test_report.py` 3 项通过。`docs/models.html` 由 gpt66 整合，本轮未改动（测试重建字节一致）。

**结论：24 个模型全部已核，无待核项。**

## ① 主候选/参照

| 模型 | 年份 | 期刊/会议 | DOI/链接 | 状态 | 来源 |
|---|---|---|---|---|---|
| 分位数 TabM / MSE TabM | 2025 | ICLR 2025 | [arXiv:2410.24210](https://arxiv.org/abs/2410.24210) | 已核（复用） | 既有 `gorishniy2025tabm`，历史一手核验＋本地 PDF |
| Huber ICM—Huber 损失 | 1964 | Annals of Mathematical Statistics 35(1), 73–101 | [10.1214/aoms/1177703732](https://doi.org/10.1214/aoms/1177703732) | 已核（复用） | 既有 `huber1964robust`，Project Euclid |
| Huber ICM—多任务 GP（Bonilla） | 2007 | NeurIPS 20 (NIPS 2007) | 无 DOI；[官方摘要页](https://proceedings.neurips.cc/paper_files/paper/2007/hash/66368270ffd51418ec58bd793f2d9b1b-Abstract.html) | 已核（本轮） | NeurIPS 官方摘要页＋官方 BibTeX |
| Huber ICM—向量值核综述（Alvarez） | 2012 | Foundations and Trends in Machine Learning 4(3), 195–266 | [10.1561/2200000036](https://doi.org/10.1561/2200000036) | 已核（本轮） | Crossref；arXiv:1106.6251 既有核验 |
| 固定融合 | — | 项目自有 | — | 无需外部书目 | 预固定等权融合为本项目方法 |
| Mul-PheG2P | 2026 | New Phytologist 252(1), 503–519 | [10.1111/nph.71461](https://doi.org/10.1111/nph.71461) | 已核（复用） | 既有 `wang2026mulpheg2p`，PMC13539974 一手核验 |

## ② 经典

| 模型 | 年份 | 期刊/会议 | DOI/链接 | 状态 | 来源 |
|---|---|---|---|---|---|
| RR-BLUP（Meuwissen） | 2001 | Genetics 157(4), 1819–1829 | [10.1093/genetics/157.4.1819](https://doi.org/10.1093/genetics/157.4.1819) | 已核（复用） | 既有 `meuwissen2001prediction`，Crossref |
| GBLUP（VanRaden） | 2008 | Journal of Dairy Science 91(11), 4414–4423 | [10.3168/jds.2007-0980](https://doi.org/10.3168/jds.2007-0980) | 已核（复用） | 既有 `vanraden2008efficient`，Crossref |
| Gaussian/加性+RBF 核岭（Crossa） | 2010 | Genetics 186(2), 713–724 | [10.1534/genetics.110.118521](https://doi.org/10.1534/genetics.110.118521) | 已核（复用） | 既有 `crossa2010prediction`，Crossref |
| SVR（Cortes & Vapnik） | 1995 | Machine Learning 20(3), 273–297 | [10.1007/BF00994018](https://doi.org/10.1007/BF00994018) | 已核（本轮） | Crossref |
| Random Forest（Breiman） | 2001 | Machine Learning 45(1), 5–32 | [10.1023/A:1010933404324](https://doi.org/10.1023/A:1010933404324) | 已核（本轮） | Crossref |
| XGBoost（Chen & Guestrin） | 2016 | KDD 2016, 785–794 | [10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785) | 已核（本轮） | Crossref（题名含副标题） |
| LightGBM（Ke 等） | 2017 | NeurIPS 30 (NIPS 2017) | 无 DOI；[官方摘要页](https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html) | 已核（本轮） | NeurIPS 官方摘要页＋官方 BibTeX |

## ③ 深度基线

| 模型 | 年份 | 期刊/会议 | DOI/链接 | 状态 | 来源 |
|---|---|---|---|---|---|
| DeepGS（Ma 等） | 2018 | Planta 248(5), 1307–1318 | [10.1007/s00425-018-2976-9](https://doi.org/10.1007/s00425-018-2976-9) | 已核（本轮） | Crossref＋Europe PMC PMID 30101399 |
| DNNGP（Wang 等） | 2023 | Molecular Plant 16(1), 279–293 | [10.1016/j.molp.2022.11.004](https://doi.org/10.1016/j.molp.2022.11.004) | 已核（本轮） | Crossref |
| SoyDNGP（Gao 等） | 2023 | Briefings in Bioinformatics 24(6), bbad349 | [10.1093/bib/bbad349](https://doi.org/10.1093/bib/bbad349) | 已核（本轮） | Crossref＋Europe PMC PMID 37824739 |

## ④ 近期

| 模型 | 年份 | 期刊/会议 | DOI/链接 | 状态 | 来源 |
|---|---|---|---|---|---|
| WheatGP（Wang 等） | 2025 | Briefings in Bioinformatics 26(2), bbaf191 | [10.1093/bib/bbaf191](https://doi.org/10.1093/bib/bbaf191) | 已核（本轮） | Crossref |
| MoEGP（Pan 等） | 2026 | Plant Methods 22(1), 23 | [10.1186/s13007-026-01500-1](https://doi.org/10.1186/s13007-026-01500-1) | 已核（本轮） | Crossref |
| MtCro（Chao 等） | 2025 | Plant Methods 21(1), 12 | [10.1186/s13007-024-01321-0](https://doi.org/10.1186/s13007-024-01321-0) | 已核（本轮，期刊名确认） | Crossref |
| Gsformer（Yang 等） | 2026 | Genetics Selection Evolution | [10.1186/s12711-026-01055-8](https://doi.org/10.1186/s12711-026-01055-8) | 已核（复用） | 既有 `yang2026gsformer`，PMC13455463 |
| RAGP（Zuo 等） | 2026 | Plant Methods 22(1), 98 | [10.1186/s13007-026-01574-x](https://doi.org/10.1186/s13007-026-01574-x) | 已核（本轮补全作者/文章号） | Crossref |
| BiMSGP（Liu 等） | 2026 | Genes 17(9), 1156 | [10.3390/genes17091156](https://doi.org/10.3390/genes17091156) | 已核（本轮，仅书目） | Crossref |

## ⑤ 待复现

| 模型 | 年份 | 期刊/会议 | DOI/链接 | 状态 | 来源 |
|---|---|---|---|---|---|
| PNNGS（Xie 等） | 2024 | Frontiers in Plant Science 15, 1410596 | [10.3389/fpls.2024.1410596](https://doi.org/10.3389/fpls.2024.1410596) | 已核（本轮） | Crossref |
| MDNN（Mao 等） | 2025 | Briefings in Bioinformatics 26(4), bbaf352 | [10.1093/bib/bbaf352](https://doi.org/10.1093/bib/bbaf352) | 已核（本轮） | Crossref |
| CWAGS（Cao 等） | 2026 | BMC Genomics 27(1), 625 | [10.1186/s12864-026-12980-9](https://doi.org/10.1186/s12864-026-12980-9) | 已核（本轮补全作者/文章号） | Crossref |

## 限制与注意（影响引用方式）

1. **DeepGS 发表版题名不含 "DeepGS"**：发表版为 "A deep convolutional neural network approach for predicting phenotypes from genotypes"（Planta 2018）；方法名来自 bioRxiv 预印本（2017，[10.1101/241414](https://doi.org/10.1101/241414)）与 R 包 `cma2015/DeepGS`。引用以发表版为准。
2. **LightGBM 与 Bonilla 2007 不著录页码**：NeurIPS 官方 BibTeX 页码为空；通引的 LightGBM 3146–3154 本轮未采用。
3. **年份与 DOI 后缀不一致两处**：MtCro 发表 2025、DOI 后缀 `s13007-024`；DNNGP 卷期年 2023、DOI 后缀 `molp.2022.11.004`。均按卷期年著录。
4. **WheatGP 单位未核**：请求所给“山东农大”本轮未核验（Crossref 不提供单位），不写入书目。
5. 核验覆盖题名/作者/年份/期刊/卷期页/DOI，未逐字读全文；论文报告成绩不混入本项目同协议结果。
6. BiMSGP 仅书目核验；按 AGENTS.md 硬边界不运行。
7. PNNGS/MDNN/CWAGS 的实现可用性沿用 WS4 核查（`results/repro_availability_20261115/verification.md`），本轮未复查仓库。
