import pytest
from wheat599.seed_stability import quantile_rule


def test_epoch_rule_uses_all_inner_epochs_and_fixed_config():
    rows=[{'seed':42,'config_index':2,'inner_best_epochs':[3,5,9]}, {'seed':2026,'config_index':2,'inner_best_epochs':[4,6,10]}]
    assert quantile_rule(rows)==(2,6)
    rows[1]['config_index']=3
    with pytest.raises(ValueError): quantile_rule(rows)
