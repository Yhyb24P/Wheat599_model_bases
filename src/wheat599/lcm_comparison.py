"""Paired material-fold metrics for LCM-GP experiments."""

import csv
import math

import numpy as np


METRICS = ('pcc', 'rmse', 'mae')


def read_fold_metrics(path, variant=None):
    """Read one model's metrics and reject missing, duplicate or non-finite cells."""
    values = {}
    with path.open(newline='') as handle:
        for row in csv.DictReader(handle):
            if variant is not None and row.get('variant') != variant:
                continue
            key = (int(row['fold']), str(row['environment']))
            if key in values:
                raise ValueError(f'Duplicate environment-fold result: {key}')
            entry = {metric: float(row[metric]) for metric in METRICS}
            if not all(math.isfinite(value) for value in entry.values()):
                raise ValueError(f'Non-finite environment-fold result: {key}')
            values[key] = entry
    if not values:
        raise ValueError(f'No fold metrics found in {path}')
    return values


def paired(candidate, reference, expected_folds=10, expected_environments=('1', '2', '4', '5')):
    """Summarize candidate minus reference without treating folds as independent."""
    if candidate.keys() != reference.keys():
        raise ValueError('Candidate and reference environment-fold keys differ')
    environments = sorted({environment for _, environment in candidate})
    folds = {fold for fold, _ in candidate}
    if (len(folds) != expected_folds or set(environments) != set(expected_environments)
            or len(candidate) != len(folds) * len(environments)):
        raise ValueError('Incomplete environment-fold matrix')
    differences = {
        key: {metric: candidate[key][metric] - reference[key][metric] for metric in METRICS}
        for key in sorted(candidate)
    }
    per_environment = {
        environment: {metric: float(np.mean([values[metric] for (fold, env), values in differences.items()
                                            if env == environment])) for metric in METRICS}
        for environment in environments
    }
    return {
        'n_environment_folds': len(differences),
        'macro': {metric: float(np.mean([v[metric] for v in per_environment.values()])) for metric in METRICS},
        'per_environment': per_environment,
        'pcc_improved': sum(v['pcc'] > 0 for v in differences.values()),
        'pcc_equal': sum(v['pcc'] == 0 for v in differences.values()),
        'fold_differences': [dict(fold=fold, environment=environment, **entry)
                             for (fold, environment), entry in differences.items()],
    }
