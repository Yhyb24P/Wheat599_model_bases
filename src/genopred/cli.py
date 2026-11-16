"""Unified experiment entry: ``python -m genopred run --config ...`` and ``index``.

Generic model routing (RUNNERS) plus the background-conditioned ridge/RR nested
experiments. Route keys (cfg['model']) are unchanged from the former
scripts/run_background_nested.py so existing run configs need no edits; values
point at the mechanism-experiment split (models/* mechanism, experiments/* runs).
Specialized commands with independent args/external deps stay as scripts
(build_report.py, fetch_data.py, run_moegp_unified.py).
"""
import argparse
from importlib import import_module
from pathlib import Path

import yaml

# Active routes only; frozen experiments are reproduced by checking out their tag.
RUNNERS = {
    'polygenic_simulation': ('genopred.evaluation.polygenic_simulation', 'run'),
    'kernel_gcv': ('genopred.models.kernels.kernel_baselines', 'run_gcv'),
    'extrapolation_diagnostic': ('genopred.evaluation.extrapolation_diagnostic', 'run'),
    'quantile_tabm_final': ('genopred.experiments.wheat599.quantile_tabm_outer', 'run'),
    'member_interaction_freeze': ('genopred.experiments.wheat599.quantile_tabm_outer', 'freeze_member_manifest'),
    'quantile_tabm': ('genopred.experiments.wheat599.quantile_tabm_dev', 'run'),
    'robust_icm': ('genopred.models.kernels.robust_icm', 'run'),
    'genomic_tabm': ('genopred.experiments.wheat599.genomic_tabm_dev', 'run'),
    'independent_tabm_final': ('genopred.experiments.wheat599.independent_tabm_outer', 'run'),
    'heterogeneous_icm': ('genopred.models.kernels.heterogeneous_icm', 'run'),
    'reml_icm': ('genopred.models.kernels.reml_icm', 'run'),
    'get_icm': ('genopred.models.kernels.get_icm', 'run'),
    'bh_lmc': ('genopred.models.kernels.bh_lmc', 'run'),
    'multienv_kernel': ('genopred.models.kernels.multienv_kernel', 'run'),
    'kernel_baselines': ('genopred.models.kernels.kernel_baselines', 'run'),
    'classical_baselines': ('genopred.models.classical.classical_baselines', 'run'),
    'ragp_outer': ('genopred.experiments.wheat599.ragp_dev', 'run_outer'),
    'deepgs': ('genopred.models.comparators.deep_baselines', 'run'),
    'dnngp': ('genopred.models.comparators.deep_baselines', 'run'),
    'soydngp': ('genopred.models.comparators.deep_baselines', 'run'),
    'wheatgp_unified': ('genopred.models.comparators.wheatgp_unified', 'run'),
    'mtcro_unified': ('genopred.models.comparators.mtcro_unified', 'run'),
    'gsformer_csa': ('genopred.models.comparators.gsformer_unified', 'run'),
    'gsformer_nsa': ('genopred.models.comparators.gsformer_unified', 'run'),
    'wheat2000_gblup': ('genopred.models.kernels.wheat2000_gblup', 'run'),
}


def run(argv=None):
    p = argparse.ArgumentParser(prog='genopred run')
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--selection', type=Path, default=None, help='Frozen development-only selections; omit for selection phase')
    p.add_argument('--resume-summary', action='store_true',
                   help='Independent TabM only: finalize saved fits and predictions without training')
    args = p.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text())
    if args.resume_summary and cfg.get('model') != 'independent_tabm_final':
        p.error('--resume-summary requires model independent_tabm_final')
    model = cfg.get('model')
    if model is not None and not isinstance(model, str):
        p.error('model must be a name, not a nested architecture configuration')
    runner_path = Path(__file__)
    if args.selection is not None and model == 'quantile_tabm' and cfg.get('optimizer_variant', '').startswith('shape_'):
        return import_module('genopred.experiments.wheat599.shape_quantile_outer').run(cfg, args, runner_path)
    if model in RUNNERS:
        module, function = RUNNERS[model]
        return getattr(import_module(module), function)(cfg, args, runner_path)
    if model not in (None, 'background_rbf'):
        p.error(f'Unknown model: {model!r}; check --config')
    from genopred.experiments.wheat599 import background_ridge
    if model == 'background_rbf':
        return background_ridge.run_hybrid(cfg, args, runner_path)
    return background_ridge.run_nested(cfg, args, runner_path)


def main(argv=None):
    parser = argparse.ArgumentParser(prog='genopred')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('run', help='Run an experiment from a config')
    sub.add_parser('index', help='Build results/run_index.json from run artifacts')
    args, rest = parser.parse_known_args(argv)
    if args.command == 'run':
        return run(rest)
    from genopred.evaluation import run_index
    return run_index.main(rest)


if __name__ == '__main__':
    raise SystemExit(main())