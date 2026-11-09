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


def test_real_teacher_records_keep_student_and_teacher_inner_separate(tmp_path):
    torch.set_num_threads(1)
    rng=np.random.default_rng(37)
    x=rng.normal(size=(18,8));y=rng.normal(size=(18,4));query=rng.normal(size=(4,8))
    cfg=yaml.safe_load(Path('configs/distill_quantile.yaml').read_text())
    cfg['seeds']=[42];cfg['training'].update(max_epochs=1,patience=1,batch_size=8)
    configs=[dict(n_blocks=1,d_block=8,dropout=0.,k=2,learning_rate=.001,weight_decay=.001)]
    kernel=yaml.safe_load(Path('configs/robust_icm.yaml').read_text())
    kernel.update(bandwidths=[1.],additive_fractions=[.5],ridge=[1.],etas=[.3],huber_deltas=[1.345],ridge_multipliers=[1.])
    records=[];context={'fold':7,'inner':2,'teacher_fold':1}
    def record(kind, detail):
        records.append(dict(kind=kind,**context,**detail))
    qp,hp,_=dq.fit_teacher(x,y,query,configs,cfg,kernel,17,'cpu',record,tmp_path/'teacher')
    assert qp.shape==hp.shape==(4,4)
    assert len(records)==20
    assert all(r['fold']==7 and r['inner']==2 and r['teacher_fold']==1 for r in records)
    assert {r['teacher_inner'] for r in records if 'teacher_inner' in r}=={0,1,2}


def test_unit_executor_records_and_warm_start(tmp_path, monkeypatch):
    from types import SimpleNamespace
    torch.set_num_threads(1)
    rng=np.random.default_rng(82)
    data=SimpleNamespace(X=rng.normal(size=(18,8)),Y=rng.normal(size=(18,4)))
    cfg=yaml.safe_load(Path('configs/distill_quantile.yaml').read_text())
    cfg.update(device='cpu',seeds=[42],strengths=[0.,.1],baseline_source=str(tmp_path/'baseline'))
    cfg['finetuning'].update(max_epochs=1,patience=1);cfg['execution']={'workers':1,'cpu_threads_per_worker':1}
    (tmp_path/'baseline').mkdir();(tmp_path/'out').mkdir()
    config=dict(n_blocks=1,d_block=8,dropout=.1,k=2,learning_rate=.001,weight_decay=.001)
    f={'fold':0,'inner':[{'train_idx':list(range(12)),'val_idx':list(range(12,18))}]}
    old={};saved={}
    for e in range(4):
        m=dq.fixed_fit(data.X[:12],data.Y[:12,e],config,42,0,0,cfg,'cpu')
        saved[f'e{e}_c0_s42']=dq.predict(m,data.X[12:],'cpu')
        old[0,0,e,0,42]={'best_epoch':0}
    np.savez(tmp_path/'baseline/predictions_0_0.npz',**saved)
    monkeypatch.setattr(dq,'crossfit_teacher',lambda x,y,*a: y*.9)
    records=dq.execute_unit((data,f,0,cfg,[config],{},[0]*4,old,str(tmp_path/'out')))
    assert len(records)==12
    assert dq.read(tmp_path/'out/unit_0_0/execution_status.json')['status']=='complete'
    assert dq.read(tmp_path/'out/unit_0_0/trajectories.json')==records
    assert len(np.load(tmp_path/'out/predictions_0_0.npz').files)==12


def test_resume_rejects_changed_recipe_and_skips_partial_units(tmp_path):
    import pytest
    src=Path(dq.__file__)
    dq.write(tmp_path/'provenance.json',{'config':{},'sha256':{str(src):dq.sha256_of(src)}})
    dq.write(tmp_path/'trajectories.json',[{'fold':0,'inner':0,'kind':'teacher_selection'}])
    folds=[{'fold':0,'inner':[{'val_idx':[1,2]}]*3}]
    assert dq.reusable_units(tmp_path,{'execution':{'workers':2}},folds)=={}
    with pytest.raises(ValueError,match='scientific configuration'):
        dq.reusable_units(tmp_path,{'strengths':[1.]},folds)


def test_correlation_loss_invariance_and_gradients():
    torch.manual_seed(1)
    output=torch.randn(12,3,9,dtype=torch.float64,requires_grad=True)
    y=torch.randn(12,1,dtype=torch.float64); t=torch.randn(12,1,dtype=torch.float64)
    q=torch.linspace(.01,.99,9,dtype=torch.float64)
    total,pin,kd=dq.correlation_distillation_loss(output,y,t,q,.1)
    pred=output.mean((1,2));r=torch.corrcoef(torch.stack((pred,t[:,0])))[0,1]
    torch.testing.assert_close(kd,9*(1-r))
    torch.testing.assert_close(pin,qt.quantile_loss(output,y,q))
    torch.testing.assert_close(total,pin+.1*kd)
    _,_,shift=dq.correlation_distillation_loss(3*output+2,y,5*t-8,q,.1)
    torch.testing.assert_close(kd,shift)
    offsets=torch.randn_like(output); offsets-=offsets.mean((1,2),keepdim=True)
    _,_,diverse=dq.correlation_distillation_loss(output+offsets,y,t,q,.1)
    torch.testing.assert_close(kd,diverse)
    grad=torch.autograd.grad(kd,output)[0]
    torch.testing.assert_close(grad.sum(),torch.zeros((),dtype=grad.dtype),atol=1e-12,rtol=0)
    torch.testing.assert_close((grad*output).sum(),torch.zeros((),dtype=grad.dtype),atol=1e-12,rtol=0)
    assert torch.autograd.gradcheck(lambda x:dq.correlation_distillation_loss(x,y,t,q,.1)[2],(output,))
    constant=torch.ones_like(output,requires_grad=True)
    loss=dq.correlation_distillation_loss(constant,y,t,q,.1)[0];loss.backward()
    assert torch.isfinite(constant.grad).all()


def test_full_block_finetuning_uses_complete_groups(monkeypatch):
    torch.set_num_threads(1)
    rng=np.random.default_rng(62);x=rng.normal(size=(24,8));y=rng.normal(size=24)
    cfg=yaml.safe_load(Path('configs/distill_correlation.yaml').read_text())
    cfg['finetuning'].update(max_epochs=2,patience=2)
    config=dict(n_blocks=1,d_block=8,dropout=.1,k=2,learning_rate=.001,weight_decay=.001)
    model=dq.fixed_fit(x[:18],y[:18],config,42,0,1,cfg,'cpu')
    blocks=[np.arange(0,5),np.arange(5,11),np.arange(11,18)];calls=[]
    original=dq.correlation_distillation_loss
    def checked(output,target,teacher,*args):
        calls.append(len(output));return original(output,target,teacher,*args)
    monkeypatch.setattr(dq,'correlation_distillation_loss',checked)
    pred,warm,info=dq.finetune(model,x[:18],y[:18],y[:18]*.5,x[18:],y[18:],0.,config,42,0,cfg,'cpu',blocks=blocks)
    assert sorted(calls)==[5,5,6,6,7,7]
    assert info['best_pcc']>=dq.pcc(y[18:],warm)-1e-12


def test_saved_teacher_boundary_and_hash(tmp_path):
    import pytest
    unit=tmp_path/'unit_0_0';unit.mkdir();ids=np.arange(9)
    np.savez(unit/'teacher_targets.npz',fit_ids=ids,fusion=np.ones((9,4)))
    parts=[dict(train_ids=ids[~np.isin(ids,b)].tolist(),heldout_ids=b.tolist()) for b in np.array_split(ids,3)]
    dq.write(unit/'teacher_partitions.json',parts)
    def audit():return {str(Path('results')/tmp_path.name/unit.name/n):dq.sha256_of(unit/n) for n in ('teacher_targets.npz','teacher_partitions.json')}
    teacher,blocks=dq.load_teacher_blocks(tmp_path,0,0,ids,audit());assert teacher.shape==(9,4) and len(blocks)==3
    with pytest.raises(ValueError,match='hash mismatch'):dq.load_teacher_blocks(tmp_path,0,0,ids,{k:'bad' for k in audit()})
    parts[0]['train_ids'].append(0);dq.write(unit/'teacher_partitions.json',parts)
    with pytest.raises(ValueError,match='boundary'):dq.load_teacher_blocks(tmp_path,0,0,ids,audit())
