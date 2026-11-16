# Wheat599 参考文献

打开 [index.html](index.html) 阅读全部参考资料的简讲、项目关系、原文链接和核验状态；[references.bib](references.bib) 可导入文献管理器。资料包含论文、数据来源和官方代码，不把代码仓库计作独立论文。

唯一维护源为 `build.py`，由其生成 HTML 和 BibTeX。修改后在仓库根目录运行：

```bash
python3 ref/build.py
python3 tests/test_report.py
```

生成仅需 Python 标准库。测试使用已安装的 BibTeX 检查真实解析；缺少该程序时跳过对应测试。页面可离线阅读，外部原文链接需联网。

## 来源与核验

覆盖原 `docs/REFERENCES.md` 的方法/数据与 GP 基础书目，以及项目方法方案实际引用的论文和代码；包括后期迁移方案的堆叠多目标回归。每条资料都附项目关系，不把论文报告成绩混入本项目模型比较。

状态逐条标注：历史核验、本轮一手核验、规范著录待复核、元数据待核。核验仅覆盖卡片列出的信息；摘要、书目核对不等于全文审读。部分条目尚无确认的作者或完整出版信息，BibTeX `note` 保留待核提示，**不能直接作为论文定稿引文**。原历史记录留在 `docs/REFERENCES.md`，例如 Gorishniy 作者首字母以本库所引原文的 Yury（Y.）为准。

最优候选是独立分位数 TabM 与 Huber ICM 的预固定融合。Huber ICM 是凸惩罚风险最小化，不是完整 Huber GP 贝叶斯算法；当前完整 Mul-PheG2P 适配包含目标环境编码器，早期仅其他环境作源的适配另作历史说明。详见 [结果与架构](../docs/models.html)。

## 本地全文

`pdfs/` 收录下列公开来源 PDF，首页题名、作者与版本已核对；其余文献通过卡片原文链接阅读。本地有无 PDF 不代表文献质量或适用性。

| 文件 | 原文版本 |
|---|---|
| `gorishniy2021revisiting.pdf` | arXiv:2106.11959v5（2023-10-26；NeurIPS 2021 论文修订版） |
| `gorishniy2025tabm.pdf` | arXiv:2410.24210v3（2025-02-18） |
| `alvarez2012kernels.pdf` | arXiv:1106.6251v2（2012-04-16） |
| `priorlabs2026tabpfn35.pdf` | arXiv:2609.17895v3（2026-11-03；只收文献，未运行模型） |

本轮未取得 Mul-PheG2P、Huber 1964 和 Koenker–Bassett 1978 的本地全文；具体访问情况与来源见各卡片。这不等于这些论文没有可获取全文。第三方论文、代码和权重的许可分别按原来源与 [LICENSE_NOTES](../docs/LICENSE_NOTES.md) 阅读。
