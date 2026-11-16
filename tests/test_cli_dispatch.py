"""CLI dispatch must reject unrelated configs before loading data or training."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from genopred import cli

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def runner():
    return cli


def test_all_routes_have_existing_implementations(runner):
    for name, (module, function) in runner.RUNNERS.items():
        path = ROOT/'src'/Path(*module.split('.')).with_suffix('.py')
        definitions = ast.parse(path.read_text()).body
        assert any(isinstance(node, ast.FunctionDef) and node.name == function for node in definitions), name


@pytest.mark.parametrize('name,function', [
    ('robust_icm', 'run'),
    ('quantile_tabm_final', 'run'),
    ('member_interaction_freeze', 'freeze_member_manifest'),
])
def test_dispatch_is_lazy_and_keeps_cli_arguments(runner, monkeypatch, tmp_path, name, function):
    config = tmp_path/'config.yaml'
    config.write_text(json.dumps({'model': name}))
    seen = []
    def call(cfg, args, script):
        seen.append((cfg, args.out, script))
        return 'dispatched'
    def load_module(actual):
        assert actual == runner.RUNNERS[name][0]
        return SimpleNamespace(**{function: call})
    monkeypatch.setattr(runner, 'import_module', load_module)
    assert runner.run(['--config', str(config), '--out', str(tmp_path/'out')]) == 'dispatched'
    assert seen == [({'model': name}, tmp_path/'out', Path(runner.__file__))]
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('model', ['robust_icm_typo', {'d_block': 64}, ''])
def test_wrong_model_never_falls_through_to_training(runner, monkeypatch, tmp_path, model):
    config = tmp_path/'wrong.yaml'
    config.write_text(json.dumps({'model': model}))
    def forbidden(*args, **kwargs):
        pytest.fail('Wrong configuration reached data loading or model import')
    monkeypatch.setattr(runner, 'import_module', forbidden)
    with pytest.raises(SystemExit) as error:
        runner.run(['--config', str(config), '--out', str(tmp_path/'out')])
    assert error.value.code == 2
    assert not (tmp_path/'out').exists()