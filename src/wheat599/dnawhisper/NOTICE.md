# Upstream notice — DNAwhisper

Model code is used from an external pinned checkout, not copied into this repository:

- project: DNAwhisper (README title "Whisper of DNA")
- repository: https://github.com/Marxin1992/Whisperer_of_DNA
- commit: `7a657cc00d44263c4b1085d3991ecc0e935c14e5` (default branch `master`, dated 2026-02-01)
- obtained as the codeload archive of that commit; every file this package imports or reads is
  checked against the git blob id recorded from that commit's tree (15 entries in
  `configs/dnawhisper_nested.yaml:upstream_blobs`), and the measured ids go to `provenance.json`.
  There is no git working tree to inspect (no tag, no release, history not public beyond two
  commits), so the verification is content-based rather than a repository-state check
- tags / releases / pretrained checkpoints: none published at the time of integration
- license: Apache-2.0 (`LICENSE` in upstream); no additional use clauses
- publication: Ma Y, Li X, Ji X, Wang C, Zhang D, Zhai T, Wang H, Liu P. *Plant Biotechnology Journal* 24(6):4005–4026, 2026. DOI `10.1111/pbi.70619`, PMID 41757800, PMCID PMC13205711, CC-BY.
- dataset card: https://huggingface.co/datasets/Mahahasuper/Deep_GS_datas (`license: mit`)

Integration surface in this project:

- `adapter.py` puts the checkout root on `sys.path` and imports `training.models.DNAWhisper`;
  upstream files are never modified, and the run aborts if any required file is missing or its
  blob hash differs from the manifest in `configs/dnawhisper_nested.yaml`.
- The official architecture configuration is read from `training/config/model_config_csv_wheat.json`
  and `model_config_csv_wheat_pretrain.json` at run time; their SHA-256 is written to
  `provenance.json` with the commit.
- Data assembly, splits, fitting-subset standardization, stage scheduling, epoch selection and
  metrics live in this package only.
- Paper-versus-code differences that change what is actually trained (optimizer key names, the
  ignored `primary_loss.enabled` flag) are listed in `configs/dnawhisper_nested.yaml` under
  `deviations` and copied into every run's `provenance.json`.
