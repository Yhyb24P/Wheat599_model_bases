# SoyBase 分类数据事实与协议

用户授权（2026-11-17）的最小分类基线。现成分类器（XGBoost/Logistic），官方原始数据自行筛选，本机运行。云端新架构/校准/亲缘分组/SoyDNGP 本轮不做。

## 数据来源

- **基因型**：SoySNP50K USDA 大豆种质库，SoyBase `Wm82.gnm2.div.Song_Hyten_2015`（[data.soybase.org](https://data.soybase.org/Glycine/max/diversity/Wm82.gnm2.div.Song_Hyten_2015/)），HapMap `hmp.gz`，20087 材料（PI ID）×41625 SNP（去除 7 个单态位点）。Song et al. 2015 G3（DOI 10.1534/g3.115.019000）。MD5 校验随源 CHECKSUM。
- **表型**：GRIN Global（[grin.soybase.org](https://grin.soybase.org)）observations API，四 qualitative 性状：Flower color(db 51058)、Pubescence density(51056)、Pod color(51057)、Stem termination type(51050)。156889 观测、21066 材料。
- **冷存储**：原始数据与缓存落 E 盘 `/mnt/e/WSL-data/soybase/`，项目 `data/raw/soybase` symlink；不入 Git。`scripts/fetch_soybase.py` 幂等下载＋校验＋记录 SHA（`source.json`）。

## 筛选与编码

- **匹配**：基因型 PI 列与 GRIN cultivar_name 按去空格大写 PI 匹配（19682/20087 匹配）。
- **基准类筛选**：GRIN 原始类别多于基准；仅保留基准类（与 BiMSGP `soybase_classification.json` 一致）：FC {W,P}、P_DENS {Ssp,N}、POD {Tn,Br,Bl}、ST {N,S,D}。
- **冲突处理**：同一材料同性状有多条基准类记录且不一致者**丢弃**（-1），不盲用 keep-first。丢弃数 FC 5/P_DENS 346/POD 79/ST 542。
- **剂量编码**：HapMap 纯合单字母→0/2（allele2 计数），IUPAC 杂合码（R/Y/K/M/S/W）→1，N→NaN。缓存缺失率 0.0003。
- **缓存**：`soybase_cache.npz`（X float32 20087×41625、Y int64 20087×4，-1 不可用）＋metadata SHA。`genopred.data.soybase.prepare/load_soybase` 双 SHA 校验。

## 各类材料数（缓存，SNP 过滤前）

| 性状 | 有效材料 | 类别计数 | 冲突丢弃 |
|---|---:|---|---:|
| FC | 18529 | W 6079 / P 12450 | 5 |
| P_DENS | 18793 | Ssp 6048 / N 12745 | 346 |
| POD | 19094 | Tn 5001 / Br 12238 / Bl 1855 | 79 |
| ST | 17692 | N 7499 / S 1436 / D 8757 | 542 |

ST 的 S 类约 8.1%（少数类）。SNP fit-only 过滤后材料数会略降（call-rate）。与 BiMSGP 论文 Table 2 计数（FC 15731 等）差异源于其进一步基因型缺失过滤，不可直接混用。

## 评价协议

- **划分**：每性状有效材料 StratifiedKFold 10 外折×3 内折（seed 42，内折 seed 42+fold），`protocols/soybase/<trait>_splits_manifest.json`（含 SHA＋各折各类支持度）。分组完整性优先，少数类各折支持度如实记录。
- **fit-only**：SNP call-rate≥0.9、MAF≥0.05 在外层 dev 集计算；缺失填补（dev 均值）、Logistic 标准化均 dev 拟合。
- **分类器**：XGBClassifier（hist，multi:softprob）＋LogisticRegression（lbfgs multinomial，正则 C 搜索）。普通交叉熵，**无类别权重/focal**（基线先行）。
- **种子**：42/2026 先平均概率再 argmax 评分；内层按 BA 选模，完整 dev 重拟合，外层测试一次。
- **指标**：BA（各类召回均值）、Macro-F1、逐类 precision/recall/f1/support、混淆矩阵（`evaluation.metrics.compute_classification`）。
- **运行**：`python -m genopred run --config configs/soybase/<trait>.yaml --out results/soybase/<trait>/...`（先 selection 后 --selection outer）。结果 `results/soybase/<trait>/`（扇出≤6）。

## 本轮不做

SoyDNGP（需上游重克隆＋GPU＋分类头）、概率校准/温度缩放、亲缘分组测试、Precision@K/Recall@K、区域编码/层次概率分解新架构、类别权重/focal、多任务共享编码器。