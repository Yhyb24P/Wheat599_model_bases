import inspect

import numpy as np
import pytest
import torch

from wheat599.query_quantile import QueryQuantile
from wheat599.quantile_tabm import quantile_loss, train


def data():
    rng = np.random.default_rng(14)
    x = rng.integers(0, 2, (20, 17)).astype('float32')
    x[:, 0] = 0
    y = (x[:, 1] - x[:, 2] + rng.normal(size=20)).astype('float32')
    return x, y


def model():
    torch.manual_seed(42)
    x, y = data()
    return QueryQuantile(x, y, width=16, heads=4, dropout=0), torch.tensor(x)


def test_queries_are_independent_of_order_duplicates_grid_and_batch():
    m, x = model()
    m.eval()
    q = torch.tensor([.1, .5, .9])
    reference = m(x[:3], q)
    torch.testing.assert_close(m(x[:3], [.9,.1,.5]), reference[:,:,[2,0,1]])
    torch.testing.assert_close(m(x[:3], [.5,.5]), reference[:,:,1:2].expand(-1,-1,2))
    torch.testing.assert_close(m(x[:1], [.5]), reference[:1,:,1:2], atol=1e-6, rtol=1e-5)
    expanded = m(x[:3], [.01,.1,.25,.5,.75,.9,.99])
    torch.testing.assert_close(expanded[:,:,[1,3,5]], reference, atol=1e-6, rtol=1e-5)
    assert not torch.allclose(reference[:,:,0], reference[:,:,2])


def test_full_network_gradients_and_no_query_label_argument():
    m,x = model()
    y = torch.tensor(data()[1])[:,None]
    result=m(x)
    assert result.shape==(20,1,99)
    quantile_loss(result,m.standardize_y(y),m.quantiles).backward()
    for name,p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(),name
    assert m.identity.grad.norm()>0 and m.state.grad.norm()>0
    assert set(inspect.signature(m.forward).parameters)=={'x','tau'}
    for block in m.reads:
        assert block.attention.in_proj_weight.grad.norm()>0


def test_fit_statistics_are_fixed_and_constant_marker_is_safe():
    m,x=model();m.eval()
    before={name:value.clone() for name,value in m.named_buffers()}
    changed=x.clone();changed[:,0]=100
    torch.testing.assert_close(m(x),m(changed))
    m(x*10)
    for name,value in m.named_buffers():torch.testing.assert_close(value,before[name])
    # A missing marker state is still a token: every one of the 17 markers is read.
    shapes=[]
    hook=m.reads[0].attention.register_forward_pre_hook(lambda _,args: shapes.append(args[1].shape))
    m(torch.zeros_like(x[:1]));hook.remove()
    assert shapes[0][1]==17


@pytest.mark.parametrize('tau',[[],[0],[1],[float('nan')],[[.5]]])
def test_bad_quantile_queries_fail(tau):
    m,x=model()
    with pytest.raises(ValueError):m(x,tau)


def test_training_uses_existing_loop_and_resets_initialization_seed():
    x,y=data()
    cfg=dict(width=16,heads=4,dropout=0,learning_rate=.001,weight_decay=.01)
    settings=dict(optimizer_variant='query',n_quantiles=99,quantile_min=.001,quantile_max=.999,
                  training=dict(max_epochs=2,patience=1,batch_size=8,gradient_clip=5,tie_tolerance=1e-12))
    a,info=train(x[:14],y[:14],x[14:],y[14:],cfg,42,0,settings,torch.device('cpu'))
    torch.randn(100)
    b,_=train(x[:14],y[:14],x[14:],y[14:],cfg,42,0,settings,torch.device('cpu'))
    np.testing.assert_array_equal(a,b)
    assert np.isfinite(a).all() and a.shape==(6,)
    assert 0<=info['best_epoch']<=2


def test_checkpoint_and_marker_identity_permutation(tmp_path):
    m,x=model();m.eval()
    path=tmp_path/'model.pt'
    torch.save(m.state_dict(),path)
    restored,_=model()
    restored.load_state_dict(torch.load(path,weights_only=True));restored.eval()
    torch.testing.assert_close(restored(x),m(x))
    torch.testing.assert_close(m(x.to(torch.int64),[.5]),m(x,[.5]))
    permutation=torch.randperm(x.shape[1])
    with torch.no_grad():
        for name in ['identity','state','x_mean','x_std','x_constant']:
            value=getattr(restored,name)
            value.copy_(value[permutation].clone())
    torch.testing.assert_close(restored(x[:,permutation]),m(x),atol=1e-6,rtol=1e-5)


def test_development_runner_saves_seed_predictions_and_original_references(tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace
    from wheat599 import quantile_tabm as runner
    from wheat599.data import sha256_of
    n=20
    x=np.arange(n,dtype=float)[:,None]
    y=np.repeat(x,4,axis=1)
    folds=[]
    for f in range(10):
        test=[2*f,2*f+1];dev=[i for i in range(n) if i not in test]
        folds.append(dict(fold=f,test_idx=test,dev_idx=dev,inner=[
            dict(train_idx=[i for i in dev if i not in dev[j::3]],val_idx=dev[j::3]) for j in range(3)]))
    splits=tmp_path/'splits.json';splits.write_text(json.dumps({'folds':folds}))
    architectures=tmp_path/'architectures.yaml'
    architectures.write_text(json.dumps({'configurations':[{'index':i} for i in range(8)]}))
    source=tmp_path/'reference';source.mkdir()
    (source/'provenance.json').write_text(json.dumps({'sha256':{str(splits):sha256_of(splits)}}))
    baseline=tmp_path/'baseline';baseline.mkdir();hashes={}
    for fold in folds:
        for j,split in enumerate(fold['inner']):
            truth=y[split['val_idx']]
            np.savez(source/f"references_{fold['fold']}_{j}.npz",huber=truth,
                     independent_tabm=truth,fixed_fusion=truth)
            path=baseline/f"predictions_{fold['fold']}_{j}.npz"
            np.savez(path,**{f'e{e}_c{c}_s{seed}':truth[:,e] for e in range(4) for c in range(8) for seed in [42,2026]})
            hashes[str(path)]=sha256_of(path)
    (baseline/'summary.json').write_text(json.dumps({'folds':[dict(fold=f,selected=[0]*4) for f in range(10)]}))
    audit=tmp_path/'audit.json';audit.write_text(json.dumps({'sha256':hashes}))
    config=dict(optimizer_variant='query',device='cpu',cpu_threads=1,data='synthetic',data_sha256='synthetic',
                splits=str(splits),architecture_config=str(architectures),reference_source=str(source),
                baseline_source=str(baseline),baseline_audit=str(audit),seeds=[42,2026])
    config_file=tmp_path/'config.yaml';config_file.write_text(json.dumps(config))
    monkeypatch.setattr(runner,'load',lambda *_:SimpleNamespace(X=x,Y=y))
    calls=[]
    def trainer(x_fit,y_fit,x_val,y_val,cfg,seed,index,settings,device):
        assert set(x_fit[:,0]).isdisjoint(x_val[:,0])
        calls.append(seed)
        # Opposite seed errors cancel only when predictions, not scores, are averaged.
        sign=1 if seed==42 else -1
        return y_val+sign*(index+1)*x_val[:,0]**2,dict(best_epoch=2)
    out=tmp_path/'out'
    runner.run(config,SimpleNamespace(selection=None,out=out,config=config_file),__file__,trainer=trainer)
    assert len(calls)==1920
    assert json.loads((out/'execution_status.json').read_text())==dict(status='complete',completed=1920,expected=1920)
    summary=json.loads((out/'summary.json').read_text())
    assert summary['macro']['quantile']['pcc']==pytest.approx(1)
    assert summary['macro']['quantile']['rmse']==0
    assert 'original_fusion' in summary['macro']
    assert all(r['selected']==[0]*4 for r in summary['folds'])
    assert len(list(out.glob('predictions_*.npz')))==30
    provenance=json.loads((out/'provenance.json').read_text())
    assert any(path.endswith('/query_quantile.py') for path in provenance['sha256'])
    assert any(path.endswith('/induced_ftt.py') for path in provenance['sha256'])


def test_fitted_buffers_are_detached_from_input_graphs():
    x,y=data()
    m=QueryQuantile(torch.tensor(x,requires_grad=True),torch.tensor(y,requires_grad=True),width=16)
    assert all(not value.requires_grad for value in m.buffers())
