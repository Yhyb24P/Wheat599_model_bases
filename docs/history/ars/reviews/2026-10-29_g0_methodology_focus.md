# ARS Methodology-Focus Panel — G0 Review (2026-10-29)

Panel: Journal-Fit/EIC seat + Methodology seat. Both reviewed the G0_VERSION.md anchor (uncommitted worktree). Role separation is not a claim of independent error processes.

## Methodology Review Card

### Summary
The protocol is unusually self-aware: it separates random from relatedness holdouts, forbids pooling repeats, and scopes claims away from biology and attention hype. But the confirmatory family tests TabPFN only against linear, not against the nonlinear controls named in the main RQ; the primary interval treats 15 correlated fold-level differences as iid; the leakage contract is declarative; and the relatedness arm’s outer family placement is seed-invariant. The model grid and early-stopping/refit rules are still drafts. The design is close in spirit but not G0-ready.

### Blocking Findings
1. **Confirmatory family does not match the main RQ.** `docs/RESEARCH_PLAN.md:7,76`: the main RQ compares TabPFN with FT-Transformer, TabM, RKHS, SVR, and linear, but the Holm family contains only TabPFN vs linear plus three FT-Transformer contrasts. Required fix: pre-register TabPFN vs TabM, RKHS, and SVR (or the best nonlinear control) in the primary family with a multiplicity plan, or restate the main RQ as TabPFN vs linear.
2. **Primary interval assumes iid folds.** `docs/RESEARCH_PLAN.md:72-75`; `src/wheat599/bootstrap.py:153-191`; `AGENTS.md:21`: 15 fold-level differences are correlated; df=14 understates uncertainty. Required fix: repeat-level means with more repeats, a hierarchical/cluster bootstrap over repeats×folds, or a mixed model.
3. **Leakage contract is declarative; inner scope is mis-checked.** `src/wheat599/contracts.py:84-98`; `tests/test_leakage_contract.py:29-75`: `check_adapter` passes `data.X[outer_train]` even for `inner_train`, and checks `n_rows_seen == n_train` (outer_train size); an inner-scoped adapter can train on all outer_train and pass. Required fix: pass inner_train for inner scope, assert size/disjointness, add a closure-based leaky adapter caught by an access guard or output-equivalence check, and add stage-specific tests.
4. **Relatedness-arm repeats are not independent.** `src/wheat599/splits.py:184-207`: families are assigned heaviest-first to the least-loaded fold with no RNG; the 9 IBS families occupy the same folds across all 3 repeats. Required fix: randomize family order and fold offsets per seed; verify assignments differ across repeats; report family×fold table.
5. **Model grid and selection/refit protocol are still drafts.** `docs/RESEARCH_PLAN.md:91-103`; `configs/core.yaml:17,65-112`: no inner selection metric, selection unit, batch size/optimizer/patience, or refit-epoch rule. Required fix: finalize the grid and training schedule, pre-specify selection metric/aggregation/unit, and define refit epochs.

### Non-blocking Findings
1. Tuning-grid counts are wrong: TabM 16 (not 32), FT-Transformer 32 (not 64).
2. TabPFN “full-column coverage” variant is not operationalized; add an exploratory arm or narrow the subquestion.
3. `src/wheat599/splits.py:7-9` docstring still describes K=10 hierarchical clustering; inner family assignment is size-stratified, not size-balanced.
4. Averaged bootstrap endpoints are not a 95% CI; the gate’s adjusted rule is ambiguous.
5. IBS-arm aggregate is dominated by singletons (12.4%); report family-member-only and singleton-only results; rename “near-duplicate” to “high-similarity family holdout” unless justified.

### Dimension Scores
- Research design: 3/5
- Statistical validity: 2/5
- Reproducibility: 3/5
- Claim-evidence alignment: 3/5

### Recommendation
**Major revision before freeze.** The blocking items change the primary inferential method and leakage test; minor edits cannot fix them. Freeze only after a committed revision, a rebuilt leakage suite, exact grid counts, and a pre-registered power/MDE statement. `docs/G0_VERSION.md:3` anchors an uncommitted worktree; tag G0 only on a commit.

## Journal-Fit Review Card

### Summary
The question is worth asking: GP papers keep claiming transformer or TabPFN superiority from random CV on related panels. The protocol’s real novelty is the evaluation frame — family-held-out splits, fold-level paired intervals, leakage-injection tests, explicit claim ladders — not a new algorithm. Positioning is mostly right and unusually disciplined, but the package is not yet journal-ready: no target venue, no account of the closest prior comparisons, and one baseline decision still open. Recommend minor revision before freeze; the scientific core needs no redesign.

### Blocking Concerns
- **B1 Roster/grid counts contradict across documents.** The plan says six families and TabM 32/FT-Transformer 64, but `configs/core.yaml` defines seven model entries including a separate kernel-ridge/GBLUP model, and grids of 16 and 32. Fix counts and state whether GBLUP is a reported seventh arm or an equivalence check.
- **B2 No strong tabular baseline decision.** LightGBM/XGBoost are absent, yet they are the expected yardstick in tabular-model work and were competitive in the closest GP comparison. Decide now: add one GBM as a cheap pre-registered secondary control, or record a claim-scope justification for exclusion.
- **B3 Single-dataset scope is admitted only in Limitations.** Either frame the work explicitly as a Wheat599 case study or commit to a second dataset before freeze.

### Non-blocking Suggestions
- Pre-register a defined estimand for Δ_arm (e.g., rank correlation between arm rankings).
- Add foundational GP/data citations: Crossa et al. 2010; Meuwissen 2001; VanRaden 2008; de los Campos 2013.
- State that TabPFN conclusions are checkpoint-specific and weights are non-commercial but license-obtainable.
- Add a GPU-hour/wall-clock budget and a budget-exhaustion rule.
- Add fixed “non-claims” and “what would change the conclusion” rows to `docs/RESULTS.md`.
- Documentation set is coherent; `G0_VERSION.md` anchors an uncommitted tree; freeze tag requires a commit.

### Overall Quality
**4/5 — Minor revision before freeze.**
