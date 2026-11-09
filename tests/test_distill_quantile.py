import copy
from pathlib import Path
import numpy as np
import torch
import yaml

from wheat599 import distill_quantile as dq
from wheat599 import quantile_tabm as qt


def test_distillation_is_per_member_mean_without_spread_penalty():
    q=torch.linspace(.001,.999,99)
    outputs=torch.randn(5,4,99,requires_grad=True)
    y=torch.randn(5,1);teacher=torch.randn(5,1)
    full,pin,kd=dq.distillation_loss(outputs,y,teacher,q,.1)
    torch.testing.assert_close(pin,qt.quantile_loss(outputs,y,q))
    torch.testing.assert_close(full,pin+.1*kd)
    grad=torch.autograd.grad(kd,outputs)[0]
    torch.testing.assert_close(grad[:,:,0],grad[:,:,98])
    offsets=torch.randn_like(outputs);offsets-=offsets.mean(-1,keepdim=True)
    _,_,same=dq.distillation_loss(outputs+offsets,y,teacher,q,.1)
    torch.testing.assert_close(same,kd)
    zero,_,_=dq.distillation_loss(outputs,y,teacher,q,0.)
    torch.testing.assert_close(zero,pin)


def test_crossfit_excludes_all_heldout_labels(monkeypatch,tmp_path):
    x=np.arange(18)[:,None].astype(float)
    y=np.column_stack([100+np.arange(18)+e*1000 for e in range(4)])
    calls=[]
    def teacher(tx,ty,query,configs,cfg,kcfg,seed,device,record,out):
        train=tx[:,0].astype(int); held=query[:,0].astype(int)
        assert not set(train)&set(held)
        np.testing.assert_array_equal(ty,y[train])
        assert ty.shape==(len(train),4)
        calls.append((train,held))
        pred=np.repeat(held[:,None],4,axis=1).astype(float)
        return pred,pred+2.,{}
    monkeypatch.setattr(dq,'fit_teacher',teacher)
    pred=dq.crossfit_teacher(x,y,np.arange(18)+100,[],{}, {},42,'cpu',lambda *a:None,tmp_path)
    np.testing.assert_array_equal(pred,np.repeat((np.arange(18)+1)[:,None],4,axis=1))
    assert len(calls)==3
    assert sorted(np.concatenate([c[1] for c in calls]))==list(range(18))


def test_huber_query_phenotypes_are_absent(monkeypatch):
    def features(x,y,tr,query,base):
        assert np.isnan(y[query]).all()
        assert np.isfinite(y[tr]).all()
        assert set(tr).isdisjoint(query)
        return 'ok'
    monkeypatch.setattr(dq,'features',features)
    assert dq.prepare_query(np.ones((8,3)),np.ones((8,4)),np.zeros((2,3)),{})=='ok'


def test_finetune_preserves_warm_start_and_reproducibility():
    torch.set_num_threads(1)
    rng=np.random.default_rng(4); x=rng.normal(size=(24,8)); y=rng.normal(size=24)
    cfg=yaml.safe_load(Path('configs/distill_quantile.yaml').read_text())
    cfg['training']['batch_size']=8
    cfg['finetuning'].update(max_epochs=2,patience=2)
    config=dict(n_blocks=2,d_block=16,dropout=.1,k=4,learning_rate=.001,weight_decay=.001)
    model=dq.fixed_fit(x[:16],y[:16],config,42,0,2,cfg,'cpu')
    original=dq.predict(model,x[16:],'cpu');before={k:v.clone() for k,v in model.state_dict().items()}
    args=(x[:16],y[:16],y[:16]*.5,x[16:],y[16:],.1,config,42,0,cfg,'cpu')
    p1,warm,info=dq.finetune(copy.deepcopy(model),*args)
    p2,_,_=dq.finetune(copy.deepcopy(model),*args)
    np.testing.assert_array_equal(warm,original)
    np.testing.assert_array_equal(p1,p2)
    assert info['best_pcc']>=dq.pcc(y[16:],original)-1e-12
    assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items())
