> **历史锚点：** 本清单生成于 Stage 1 收敛前（含 `configs/core.yaml` 旧路径）。Stage 1 修订版将在 iq 完成 stage1 代码与测试后重新生成；冻结以新 `G0_VERSION` 为准。

# G0 Review Version Anchor

**日期：** 2026-10-29。**状态：** 工作树未提交（用户尚未授权本地提交）。本清单是 gb G0 评审的版本锚点；提交后以 commit SHA 为准。

仓库：`/home/yhshy/git_files/Wheat599_model_bases`（旧路径 `RC_CKRR_semisynth` 为指向本仓库的软链接）。
测试：`PYTHONPATH=src python -m pytest -q` → 16 passed。

## 文件 SHA-256

| 文件 | SHA-256 |
|------|---------|
| `AGENTS.md` | `ddc2800498b2ab46788cb05673fa97809ebebac4fb635c927775254d568fdafc` |
| `README.md` | `8124ca9d272bb6ede74c2a2ff28d304ec0ccb5bc2ff800f9dd0ad91d0fd5b9a6` |
| `configs/core.yaml` | `3ae8a76738a53fb6652833bbc4dfb437b21c5b2fd02481fae80d03275de69005` |
| `docs/RESEARCH_PLAN.md` | `4873ef1395b5d95d2460d4a710aae5cbc0aa64f664b994d15dae2c6d0613845a` |
| `docs/REVIEW_PROTOCOL.md` | `06de9094a34cce5c974672320737d277a2a4ba76bcdf65c871fbecda446b6ddf` |
| `docs/DATA_FACTS.md` | `c0f4a7cbfecb847014f3ef1cf648099b2887f56489742724f566b9512e9e5a24` |
| `docs/LICENSE_NOTES.md` | `0c83afa69f4789af3902a62d4d88632e213be081f739c9841cad478710d27094` |
| `docs/REFERENCES.md` | `62d067cf90d00e6802f898827fdea1d79b55d7ef08a197bad1b69ea1d901ec34` |
| `docs/RESULTS.md` | `d4d9e0670c6db305d92a2ef3eb0784bdc9c02d91aa22215bd7add452fcae2a01` |
| `docs/MIGRATION.md` | `daadd0e0d5deba8bc04dceff20a90844838de457d68953045510df2b199a1964` |
| `src/wheat599/splits.py` | `0cf2353093a4f70aeb92e80baf7d362e4dd84b53df1bf49fa44bcefb9743f31a` |
| `src/wheat599/bootstrap.py` | `1540d50c8e6627be721e4b03ac89b6a3e25b37a474b433a7f407799f2760bed4` |
| `src/wheat599/contracts.py` | `e4f39c6a2ed9fe3b3052e8d303cf2f316568c8aa98e7afabf47f880716cd43fd` |
| `src/wheat599/data.py` | `e3a3bb71b88a9aa2b4fa1dde7ed9293b429ca0cc1d5d2b4ce39e6cca29df5046` |
| `src/wheat599/metrics.py` | `efcb4f8b1fd9cfc1735c2b43906027bc43ec03c59f4f2f296ef303a53e6176a2` |
| `src/wheat599/oof.py` | `95c2b4ca4f4fd27dfa087618f471fc626910e35dffbfe7d29a46cd191561145b` |
| `tests/test_scaffold.py` | `e8d86bd1eee197c8993df4b0949668109e5f01a6fe4b8edab842478cccd66cc1` |
| `tests/test_leakage_contract.py` | `267b689eb95fbdd2781ff20d6de8f7e4d67f091a25c61272a4e3a6a169ca0c3d` |
| `env/requirements-bench.txt` | `e63e3dbca8682f0fe59adadc58d38f18ad00231089960d288d66ec53e842ff6d` |
| `env/environment.yml` | `2c1e7a3fdf0884df97e7da4246697147a2f829c09cf83e4353a9274f95313502` |

## 说明

- 版本锚点是未提交工作树；如果 gb 要求 commit 锚点，需要用户授权一次本地提交后再更新本文件。
- `docs/RESULTS.md` 当前只有“尚无正式结果”一行；不是断链。
- `splits.py` 的 report 函数（每折 n、家族大小分布、% 多线家族、train×test 平均 IBS/A）尚待补；不影响 G0 协议文本评审，作为非阻断项记录。
