import csv
import json

import pytest

from wheat599 import independent_freeze as freeze


ENVIRONMENTS = ('1', '2', '4', '5')
CONFIGURATIONS = [{'n_blocks': 2 + i % 3, 'd_block': 128, 'dropout': 0.1, 'k': 16 + i,
                   'learning_rate': 0.001, 'weight_decay': 0.001} for i in range(8)]


def build_splits():
    folds, historical = [], []
    for k in range(10):
        test = list(range(k * 60, (k + 1) * 60)) if k < 9 else list(range(540, 599))
        dev = [i for i in range(599) if i not in set(test)]
        val = dev[-180:]
        folds.append({'fold': k, 'dev_idx': dev, 'test_idx': test,
                      'inner': [{'train_idx': dev[:-180], 'val_idx': val}] * 3})
        val_historical = dev[485:] if len(dev) == 539 else dev[486:]
        historical.append({'fold': k, 'train_idx': dev[:len(dev) - len(val_historical)],
                           'val_idx': val_historical, 'test_idx': test})
    return folds, historical


def build_best_epochs(chosen):
    out = {}
    for k in range(10):
        for inner in range(3):
            for seed in (42, 2026):
                for config in range(8):
                    for env in range(4):
                        out[(k, inner, seed, config, env)] = (
                            1000 * k + 100 * inner + 10 * env + (0 if seed == 42 else 1) + config)
    return out


def build_selection(chosen):
    records = []
    for k in range(10):
        for training in ('shared', 'independent'):
            for policy in ('common', 'per_environment'):
                indices = chosen[k] if (training, policy) == ('independent', 'per_environment') \
                    else [0, 0, 0, 0]
                records.append({'fold': k, 'training': training, 'selection': policy,
                                'config_indices': indices})
    return records


def build_original(chosen):
    return {str(k): {'independent': {str(e): {'config_index': chosen[k][e]} for e in range(4)}}
            for k in range(10)}


def build_chosen():
    return {k: [k % 8, (k + 1) % 8, (k + 2) % 8, (k + 3) % 8] for k in range(10)}


def generate(chosen=None, best_epochs=None):
    folds, historical = build_splits()
    chosen = build_chosen() if chosen is None else chosen
    best = build_best_epochs(chosen) if best_epochs is None else best_epochs
    return freeze.generate(
        ENVIRONMENTS, 599, folds, historical, best, chosen, [42, 2026], CONFIGURATIONS,
        'data-sha', 'splits-sha', 'predictions-sha', 'trajectories-sha', 'selection-sha', 'commit')


def test_median_epoch_and_trajectory_validation():
    assert freeze.median_epoch([72, 14, 33]) == 33
    assert freeze.median_epoch([0, 5, 3]) == 3
    assert freeze.median_epoch([0, 0, 4]) == 0
    with pytest.raises(ValueError):
        freeze.median_epoch([1, 2])
    with pytest.raises(ValueError):
        freeze.median_epoch([1, -1, 2])
    record = {'trajectory': {'variant': 'independent', 'fold': 0, 'inner': 0, 'seed': 42,
                             'config_index': 0, 'environment': 0}, 'best_epoch': 7}
    assert freeze.build_best_epochs([record]) == {(0, 0, 42, 0, 0): 7}
    with pytest.raises(ValueError):
        freeze.build_best_epochs([record, record])
    with pytest.raises(ValueError):
        freeze.build_best_epochs([{**record, 'best_epoch': -1}])


def test_generate_uses_the_selected_configuration_and_seed_isolation():
    rows, verification = generate()
    assert len(rows) == 80
    assert verification['epoch_rule'].startswith('median')
    chosen = build_chosen()
    row = next(r for r in rows if r['fold'] == 3 and r['environment'] == '4' and r['seed'] == 2026)
    config = chosen[3][2]
    expected = [1000 * 3 + 0 + 20 + 1 + config, 1000 * 3 + 100 + 20 + 1 + config,
                1000 * 3 + 200 + 20 + 1 + config]
    assert row['environment_index'] == 2 and row['config_index'] == config
    assert row['inner_best_epochs'] == expected
    assert row['median_epochs'] == expected[1]
    assert row['configuration'] == CONFIGURATIONS[config]
    other = next(r for r in rows if r['fold'] == 3 and r['environment'] == '4' and r['seed'] == 42)
    assert other['median_epochs'] == row['median_epochs'] - 1
    folds, _ = build_splits()
    assert row['fit_ids'] == folds[3]['dev_idx']
    assert row['test_ids'] == folds[3]['test_idx']
    assert row['fit_ids_sha256'] == freeze.ids_sha256(folds[3]['dev_idx'])
    assert row['test_count'] == 60
    fold9 = next(r for r in rows if r['fold'] == 9)
    assert fold9['fit_count'] == 540 and fold9['test_count'] == 59
    with pytest.raises(ValueError):
        generate(best_epochs={k: v for k, v in build_best_epochs(build_chosen()).items()
                              if k != (4, 2, 2026, chosen[4][1], 1)})


def test_selection_and_split_mismatches_are_rejected():
    folds, historical = build_splits()
    chosen = build_chosen()
    assert freeze.chosen_configs(build_selection(chosen), folds) == chosen
    freeze.cross_check_selection(chosen, build_original(chosen), folds)
    altered = {**chosen, 2: [0, 0, 0, 0]}
    with pytest.raises(ValueError):
        freeze.cross_check_selection(altered, build_original(chosen), folds)
    freeze.verify_splits(folds, historical, 599)
    swapped = json.loads(json.dumps(historical))
    swapped[4]['test_idx'] = list(reversed(swapped[4]['test_idx']))
    with pytest.raises(ValueError):
        freeze.verify_splits(folds, swapped, 599)
    missing = json.loads(json.dumps(historical))
    missing[1]['train_idx'] = missing[1]['train_idx'][1:]
    with pytest.raises(ValueError):
        freeze.verify_splits(folds, missing, 599)
    overlap = json.loads(json.dumps(folds))
    overlap[0]['dev_idx'][0] = overlap[0]['test_idx'][0]
    with pytest.raises(ValueError):
        freeze.verify_splits(overlap, historical, 599)


def test_selection_schema_rejects_duplicates_and_bad_configurations():
    folds, historical = build_splits()
    chosen = build_chosen()
    records = build_selection(chosen)
    duplicated = records + [r for r in records
                            if r['training'] == 'independent'
                            and r['selection'] == 'per_environment' and r['fold'] == 0]
    with pytest.raises(ValueError):
        freeze.chosen_configs(duplicated, folds)
    missing = [r for r in records
               if not (r['training'] == 'independent' and r['selection'] == 'per_environment'
                       and r['fold'] == 3)]
    with pytest.raises(ValueError):
        freeze.chosen_configs(missing, folds)
    best = build_best_epochs(chosen)
    with pytest.raises(ValueError):
        freeze.generate(ENVIRONMENTS, 599, folds, historical, best, {**chosen, 0: [99, 1, 2, 3]},
                        [42, 2026], CONFIGURATIONS, 'd', 's', 'p', 't', 'sel', 'c')
    with pytest.raises(ValueError):
        freeze.generate(ENVIRONMENTS, 599, folds, historical, best, chosen, [42, 42],
                        CONFIGURATIONS, 'd', 's', 'p', 't', 'sel', 'c')


def test_source_configuration_and_seeds_are_enforced():
    source = {'config': {'configurations': [{**c, 'context_size': 32} for c in CONFIGURATIONS],
                         'seeds': [42, 2026]}}
    freeze.verify_source_config(source, CONFIGURATIONS, [42, 2026])
    with pytest.raises(ValueError):
        freeze.verify_source_config(source, CONFIGURATIONS[:7], [42, 2026])
    with pytest.raises(ValueError):
        freeze.verify_source_config(source, CONFIGURATIONS, [42, 7])


def test_write_manifest_roundtrip(tmp_path):
    rows, verification = generate()
    out = tmp_path / 'manifest'
    digest = freeze.write_manifest(out, rows, verification, {'source_run': 'x'})
    assert json.loads((out / 'failures.json').read_text()) == []
    payload = json.loads((out / 'manifest.json').read_text())
    assert payload['row_count'] == 80 and payload['rows_sha256'] == digest
    assert payload['verification'] == json.loads(json.dumps(verification))
    with (out / 'manifest.csv').open() as handle:
        reader = list(csv.reader(handle))
    assert reader[0][:6] == ['fold', 'environment_index', 'environment', 'seed',
                             'config_index', 'n_blocks']
    assert reader[0][-1] == 'source_git_commit'
    assert len(reader) == 81
    first = rows[0]
    assert reader[1][0] == str(first['fold']) and reader[1][2] == first['environment']
    assert reader[1][17] == ' '.join(map(str, first['fit_ids']))
    assert reader[1][18] == ' '.join(map(str, first['test_ids']))
    assert reader[1][14] == str(first['median_epochs'])
    provenance = json.loads((out / 'provenance.json').read_text())
    assert provenance['scope'].startswith('Frozen refit manifest only')
