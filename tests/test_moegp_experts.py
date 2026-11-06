import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path('scripts').resolve()))
from run_moegp_nested import build_model


def test_expert_sharing_switch(monkeypatch):
    from types import ModuleType
    utils=ModuleType('utils');utils.__path__=[]
    moe=ModuleType('utils.moe')
    class Gating(torch.nn.Module):
        def __init__(self,p,n,model,top_k,loss_coef):
            super().__init__();self.experts=torch.nn.ModuleList([model for _ in range(n)])
    moe.Gating=Gating
    monkeypatch.setitem(sys.modules,'utils',utils);monkeypatch.setitem(sys.modules,'utils.moe',moe)
    def factory(p,out,hidden,dropout,batch_norm): return torch.nn.Linear(p,out)
    cand={'hidden':[4],'dropout':.3,'experts':3};cfg={'top_k':2,'loss_coef':.1}
    shared=build_model(5,cand,{**cfg,'independent_experts':False},factory)
    independent=build_model(5,cand,cfg,factory)
    assert shared.experts[0] is shared.experts[1] is shared.experts[2]
    assert len({id(m) for m in independent.experts})==3
