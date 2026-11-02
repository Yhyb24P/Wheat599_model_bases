# Finalization notes — bimsgp_direct_20261030

- 运行时参照（interim）：results/bimsgp_ftt_fixed_20261029/per_fold_val_pcc_rule.csv
  SHA-256 0d4de42f4d3b8e2018bfe0a0b412138b4e72789427ee668d161303c83ecd37d4；由早期 A 正式跑转换而来；仅用于审计，禁止作为常规依赖。
- 正式参照（official）：results/bimsgp_ckpt_rules_20261030/per_fold.csv
  SHA-256 94622acf7723b56f453313435e015ba198b771f201a9a29ba1179a42b75905b5；rule=val_pcc；40 行；与 interim 的 40 折键一致、指标最大差 0.004245。
- 正式重算（不重训、不重评 test）：宏均 ΔPCC(vs A)=-0.014958、ΔRMSE=-0.006712、ΔPCC(vs RR)=-0.033540；G1=False、G2=True、G3=False、GO=False。
- 运行时内部 GO（interim 参照）：G1=False、G2=True、G3=False、GO=False；判定与正式一致。
- 诊断口径更正：A(official) pred_sd 宏均 0.552；induced pred_sd 宏均 0.512；direct 0.577；RR 0.432（来自 residual 轮 rr_only）。此前把 RR 的 0.432 误标为 A、把 residual 分支的 0.604 误标为 induced，作废，以本文件为准。
- tokenizer 参数范数 0.577→0.578，仅为参数范数稳定性观测，不构成"尺度问题已消除"的证明。
