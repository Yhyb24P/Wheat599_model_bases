"""CLI dispatch must reject unrelated configs before loading data or training."""
import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    spec = importlib.util.spec_from_file_location('nested_cli', ROOT/'scripts/run_background_nested.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_all_routes_have_existing_implementations(runner):
    for name, (module, function) in runner.RUNNERS.items():
        path = ROOT/'src/wheat599'/Path(*module.split('.')).with_suffix('.py')
        definitions = ast.parse(path.read_text()).body
        assert any(isinstance(node, ast.FunctionDef) and node.name == function for node in definitions), name


@pytest.mark.parametrize('name,module,function', [
    ('robust_icm','robust_icm','run'),
    ('quantile_tabm_final','quantile_tabm_final','run'),
    ('mulphe_unified','mulphe_unified','run'),
    ('member_interaction_freeze','quantile_tabm_final','freeze_member_manifest'),
    ('target_icm','joint_kernel_selection','run_target'),
    ('wheat2000_deepgs','wheat2000_neural_run','run'),
])
def test_dispatch_is_lazy_and_keeps_cli_arguments(runner, monkeypatch, tmp_path, name, module, function):
    config = tmp_path/'config.yaml'
    config.write_text(json.dumps({'model': name}))
    seen = []
    def call(cfg, args, script):
        seen.append((cfg, args.out, script))
        return 'dispatched'
    def load_module(actual):
        assert actual == 'wheat599.' + module
        return SimpleNamespace(**{function: call})
    monkeypatch.setattr(runner, 'import_module', load_module)
    monkeypatch.setattr('sys.argv', ['runner', '--config', str(config), '--out', str(tmp_path/'out')])
    assert runner.main() == 'dispatched'
    assert seen == [({'model':name}, tmp_path/'out', Path(runner.__file__))]
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('model', ['robust_icm_typo', {'d_block':64}, ''])
def test_wrong_model_never_falls_through_to_training(runner, monkeypatch, tmp_path, model):
    config=tmp_path/'wrong.yaml'
    config.write_text(json.dumps({'model':model}))
    monkeypatch.setattr('sys.argv', ['runner','--config',str(config),'--out',str(tmp_path/'out')])
    def forbidden(*args, **kwargs):
        pytest.fail('Wrong configuration reached data loading or model import')
    monkeypatch.setattr(runner,'load',forbidden)
    monkeypatch.setattr(runner,'import_module',forbidden)
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2
    assert not (tmp_path/'out').exists()
