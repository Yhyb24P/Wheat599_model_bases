import numpy as np
import torch
from sklearn.model_selection import KFold
from wheat599.wheatgp_rebuilt import splits


def test_author_style_split_is_reproducible_and_disjoint():
    folds=splits(599,555,42)
    assert folds==splits(599,555,42)
    assert sorted(i for f in folds for i in f['test_idx'])==list(range(599))
    for f,(dev,test) in zip(folds,KFold(10,shuffle=True,random_state=555).split(np.arange(599))):
        assert f['dev_idx']==dev.tolist() and f['test_idx']==test.tolist()
        permutation=torch.randperm(len(dev),generator=torch.Generator().manual_seed(42)).numpy()
        n=int(.9*len(dev))
        assert f['train_idx']==dev[permutation[:n]].tolist()
        assert f['val_idx']==dev[permutation[n:]].tolist()
        assert not set(f['train_idx']) & set(f['test_idx'])
