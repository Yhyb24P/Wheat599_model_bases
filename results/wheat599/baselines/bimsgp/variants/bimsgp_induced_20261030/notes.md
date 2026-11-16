# Induced FT-Transformer — first complete-model run

**Run:** `results/bimsgp_induced_20261030/`
**Status:** 40/40 folds completed, 0 failures. Pre-registered GO = **False** (G1/G2/G3 all false). No ablation and no retry.

## Fixed configuration

- Input: 1,279 DArT markers; marker-only (no `wheat.A`).
- Tokenizer: `t_j = b_j + x_j U c_j`; `d=64`, rank `r=16`.
- Two independent bidirectional induced-attention blocks: `H=MAB(I,T)`, `T=MAB(T,H)`; `m=32`, 4 heads, Pre-LN residual, ReGLU FFN hidden 128.
- Readout: 4 learnable queries, Pre-LN multi-head attention pooling (no FFN), flatten 4×64, MLP 256→32→1.
- Training: same seed-42 10 folds, train-only scalers, AdamW lr 5e-4, wd 0.01, batch 8, 100 epochs, val-RMSE scheduler, val-PCC checkpoint, test once. Each environment trained separately.
- Initialization: `b_j ~ N(0,0.01²)`, `c_j ~ N(0,1)`, `U` Kaiming-uniform. `b/c/U` are in the weight-decay group; LayerNorm/bias/latents/queries are not.

## Parameters and attention

| Component | Parameters |
|---|---:|
| Shared tokenizer | 103,344 |
| 2 induced blocks | 171,776 |
| Readout (no FFN) | 17,152 |
| Regression head | 8,257 |
| **Total** | **300,529** |

- Relative to A: `300,529 / 247,425 = 1.2146` (+21.5%). No parameter-matched control; noted for interpretation.
- Attention scores per sample including heads: **675,312**.
- A has 4·1280² (full first block) + 4·1280 (CLS-only last block) = **6,558,720**; ratio ≈ **1/9.7**. Counting both A blocks as full would give ≈1/19; the actual rtdl last block is CLS-only.

## Results (fold mean ± sample SD, ddof=1)

| Environment | PCC | RMSE | MAE |
|---|---:|---:|---:|
| 1 | 0.4487 ± 0.1158 | 0.9261 ± 0.0926 | 0.7360 ± 0.0850 |
| 2 | 0.2898 ± 0.1331 | 0.9987 ± 0.1060 | 0.7759 ± 0.0964 |
| 4 | 0.2328 ± 0.1396 | 1.0024 ± 0.1080 | 0.7870 ± 0.0799 |
| 5 | 0.3813 ± 0.0676 | 0.9782 ± 0.0740 | 0.7634 ± 0.0637 |
| **Macro** | **0.3382** | **0.9764** | **0.7656** |

Pooled OOF PCC (per environment 599 materials once): macro **0.3229**; fold mean is the primary table above.

Comparisons (same folds):

- vs A: ΔPCC **−0.0961**, ΔRMSE **+0.0309**, ΔMAE **+0.0269**; PCC better in 7/40 folds, RMSE lower in 10/40; no environment has positive ΔPCC.
- vs RR-BLUP: ΔPCC **−0.1147**, ΔRMSE **+0.0806**, ΔMAE **+0.0688**; PCC better in 7/40, RMSE lower in 6/40.

## Pre-registered GO

| Criterion | Value | Pass |
|---|---:|---|
| G1 vs A: macro ΔPCC ≥ +0.01 and ≥3/4 env ΔPCC > 0 | −0.0961; 0 env positive | No |
| G2 vs A: macro ΔRMSE ≤ +0.01 | +0.0309 | No |
| G3 vs RR: macro ΔPCC ≥ −0.01 | −0.1147 | No |

**GO = False.** Per the pre-registration, no rule/config retry and no component ablation were performed. Whether to continue this architecture is a user decision. This result is specific to this configuration, single seed, and fixed 10-fold protocol; it is not a general rejection of the reconstruction direction, and without ablation it does not identify whether the low-rank tokenizer or induced attention alone is useful.

## Diagnostics

- Five folds have selected prediction SD < 0.1 × truth SD: (env 4, fold 1), (env 5, fold 1), (env 4, fold 4), (env 2, fold 5), (env 4, fold 7); their best epochs are 11, 0, 2, 0, 1 and PCC 0.090, 0.348, 0.207, 0.089, 0.088. The 0.1σ_y threshold is a descriptive diagnostic, not pre-registered; these folds are not excluded and their PCC is included in the macro means. Four of five are in environments 4/5 with best epoch ≤2, suggesting initialization/early-training instability rather than a hidden prediction failure.
- Token identity norm: initial mean 0.0802; final mean 0.1642 (range 0.0838–0.2639). Norms grew rather than collapsed; no evidence that weight decay removed marker identity in this run.
- Best epoch mean/median: 23.8 / 22.0; the model was not uniformly stuck at epoch 0.

## Smoke/test disclosure

The four fold-0 test folds were observed in two smoke runs (before and after the initialization/parameter-group/count corrections); the configuration was frozen after smoke #2 and not changed afterwards. The corrections were implementation-driven (initialization, parameter grouping, attention count), not selected from test values. Fold 0 is therefore not a clean holdout; all 40-fold results are protocol-internal descriptive evaluation, not an independent confirmation.

## Artifacts

`per_fold.csv`, `oof.csv`, `failures.csv`, `summary.json`, `provenance.json`, `splits_manifest.json`; code `scripts/run_induced.py`, model `src/wheat599/induced_ftt.py`, config `configs/bimsgp_induced.yaml`.
