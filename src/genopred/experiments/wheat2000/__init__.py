"""Wheat2000 experiment orchestration (WP-C, gb; ds provides model->file mapping + verification).

Separable run()/dev orchestration from comparators/wheat2000_* and classical wheat2000 runners.
gb adapts its wheat2000_classical fold-resume/model-subset (--models) logic into this structure
(runner convenience only; evaluation protocol/PCC/selection rules unchanged).
"""