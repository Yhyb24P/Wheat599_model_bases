# Migration — RC_CKRR_semisynth → Wheat599_model_bases

**日期：** 2026-10-29。用户授权清空旧仓库；按“备份 → 清空 → 原路径软链接 → 迁移 AIA → 删除旧 scope”的顺序执行。

## 已完成

| 项 | 位置 / 值 |
|----|-----------|
| 旧仓库完整备份 | `/mnt/e/RC_CKRR_data/backups/RC_CKRR_semisynth_20261029.tar.gz` |
| 备份 SHA-256 | `41c11f9cc00befc5d6988a5fef72229749f25ccd5f623596aa1294ba935f63a2` |
| 备份条目 | 664（含 `.git`、`.state/aia`、`site/`、`data_tmp/`、`dist/`、`.qoder.site`） |
| AIA 会话史导出 | `/mnt/e/RC_CKRR_data/backups/aia_wheat599_model_bases_20261029.jsonl`（37 条记录） |
| 旧路径 | `/home/yhshy/git_files/RC_CKRR_semisynth` → 软链接到 `/home/yhshy/git_files/Wheat599_model_bases` |
| 新 AIA scope | `/home/yhshy/git_files/Wheat599_model_bases/.state/aia` |
| 主导者显示名 / agent_id | `cn` / `agent-4c5d46d2db6240d6ae694a8acd24513e` |

旧仓库的 git 历史仍在 GitHub（`Yhyb24P/RC_CKRR_semisynth`，`main == 5e42160`）和上述备份中。旧 `.state/aia` 已随备份封存，不再作为活动 scope。

## gb / iq 迁移

旧 AIA 绑定已随旧 scope 退役。请在下次动作时到新仓库执行：

```bash
/home/yhshy/git_files/AIA/.state/product/bin/aia \
  --scope /home/yhshy/git_files/Wheat599_model_bases/.state/aia \
  join --compact
```

拿到 `ready:true` 与新的 `client` 后，在新 scope 给 `cn` 发一条 RESULT/HELLO。旧的 `wheat599-model-bases` 消息史只在备份 JSONL 中，不在新 scope。

## 当前 G0 状态

- iq 的 16 项测试已通过；random 外层 119–120、inner 3 折 159–160；IBS/A 两臂外层 [120,120,120,120,119]；bootstrap 三层与泄漏注入通过。
- gb 已签核 AGENTS.md、R1–R8 与亲缘规格；数字更正已并入 `DATA_FACTS.md`。
- 待办：gb/iq 重新 join 新 scope；gb 对修订版做 G0 评审；用户授权一次本地提交后打冻结标签。未提交前不开始正式 M1 运行。
