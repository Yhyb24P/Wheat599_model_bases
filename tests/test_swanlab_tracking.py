"""SwanLab tracking helper is opt-in and a safe no-op when disabled."""
from genopred.evaluation import swanlab_tracking


def test_disabled_is_noop(monkeypatch):
    monkeypatch.delenv('SWANLAB_ENABLE', raising=False)
    assert swanlab_tracking.init_run({'model': 'deepgs'}, '/tmp/x') is None
    swanlab_tracking.finish_run(None, '/tmp/x')  # must not raise


def test_enable_flag_parsing(monkeypatch):
    monkeypatch.setenv('SWANLAB_ENABLE', '0')
    assert swanlab_tracking.init_run({'model': 'deepgs'}, '/tmp/x') is None


def test_flatten_numeric_leaves():
    flat = swanlab_tracking._flatten({'primary': {'pcc': 0.61, 'rmse': 0.77}, 'n': 80})
    assert flat == {'primary/pcc': 0.61, 'primary/rmse': 0.77, 'n': 80}


def test_flatten_skips_nonfinite_and_strings():
    flat = swanlab_tracking._flatten({'a': float('nan'), 'b': 'text', 'c': 1.5, 'd': True})
    assert flat == {'c': 1.5}


def test_dataset_tag():
    assert swanlab_tracking._dataset({'data': 'data/raw/wheat2000/wheat_GS.RData'}) == 'wheat2000'
    assert swanlab_tracking._dataset({'source': 'data/raw/soybase/source.json'}) == 'soybase'
    assert swanlab_tracking._dataset({'model': 'deepgs'}) == 'wheat599'