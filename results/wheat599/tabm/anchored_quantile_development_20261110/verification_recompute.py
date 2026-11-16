import json
from pathlib import Path
import numpy as np,torch,yaml
from wheat599.data import load,sha256_of
from wheat599.metrics import pcc,rmse,mae
from wheat599.robust_icm import features
from wheat599.quantile_tabm import build_model
p=Path('results/anchored_quantile_development_20261110');read=lambda q:json.loads(Path(q).read_text());prov=read(p/'provenance.json');cfg=prov['config'];status=read(p/'execution_status.json');summary=read(p/'summary.json');data=load(Path(cfg['data']),cfg['data_sha256']);folds=read(cfg['splits'])['folds'];configs=yaml.safe_load(Path(cfg['architecture_config']).read_text())['configurations'];torch.set_num_threads(1)
assert status==dict(status='complete',completed=180,expected=180) and not read(p/'failures.json')
for name,digest in prov['sha256'].items():assert sha256_of(Path(name))==digest
records=[];all_metrics={name:[] for name in ('joint','initial_fusion','original_quantile')};max_replay=0.;all_scores=[];hashes={}
for fold in folds:
 f=fold['fold'];unit_arrays=[np.load(p/f'unit_{f}_{j}/predictions.npz') for j in range(3)];result=next(v for v in summary['folds'] if v['fold']==f)
 scores=[]
 for c in range(3):
  for t in range(cfg['updates']+1):
   vals=[]
   for j,inner in enumerate(fold['inner']):
    pred=np.mean([unit_arrays[j][f'c{c}_s{seed}_t{t}'] for seed in cfg['seeds']],axis=0)
    assert pred.shape==(len(inner['val_idx']),4) and np.isfinite(pred).all()
    vals.extend([pcc(data.Y[inner['val_idx'],e],pred[:,e]) for e in range(4)])
   scores.append(dict(config_index=c,step=t,pcc=float(np.mean(vals))))
 chosen=next(v for v in scores if max(r['pcc'] for r in scores)-v['pcc']<=cfg['training']['tie_tolerance'])
 assert chosen['config_index']==result['selected']['config_index'] and chosen['step']==result['selected']['step']
 np.testing.assert_allclose([v['pcc'] for v in scores],[v['pcc'] for v in result['scores']],atol=1e-12,rtol=0)
 for method in all_metrics:
  vals=[]
  for j,inner in enumerate(fold['inner']):
   if method=='original_quantile':pred=np.mean([unit_arrays[j][f'original_s{seed}'] for seed in cfg['seeds']],axis=0)
   else:
    c,t=(chosen['config_index'],chosen['step']) if method=='joint' else (0,0)
    pred=np.mean([unit_arrays[j][f'c{c}_s{seed}_t{t}'] for seed in cfg['seeds']],axis=0)
   vals.append([[fn(data.Y[inner['val_idx'],e],pred[:,e]) for fn in (pcc,rmse,mae)] for e in range(4)])
  mean=np.mean(vals,axis=0);np.testing.assert_allclose(mean,result['metrics'][method],atol=1e-12,rtol=0);all_metrics[method].append(mean)
 for j,inner in enumerate(fold['inner']):
  unit=p/f'unit_{f}_{j}';saved=read(unit/'states.json');assert len(saved)==6*(cfg['updates']+1)
  assert {(r['config_index'],r['seed'],r['step']) for r in saved}=={(c,seed,t) for c in range(3) for seed in cfg['seeds'] for t in range(cfg['updates']+1)}
  assert read(unit/'initialization.json')['huber_max_difference']<=1e-8
  for r in saved:
   path=unit/f"c{r['config_index']}_s{r['seed']}_t{r['step']}.pt";assert sha256_of(path)==r['weight_sha256']
   if r['step']==0:assert r['initialization_max_difference']<=1e-6 and r['kernel_certificate'] is None
   else:assert -1e-10<=r['kernel_certificate']['relative_gap']<=cfg['solver']['tolerance']
  records.extend(saved)
  hashes[str(unit/'predictions.npz')]=sha256_of(unit/'predictions.npz')
  for seed in cfg['seeds']:
   for replay_c,replay_t in [(chosen['config_index'],chosen['step'])]+[(c,cfg['updates']) for c in range(3)]:
    key=f"c{replay_c}_s{seed}_t{replay_t}";state=torch.load(unit/f'{key}.pt',map_location='cpu',weights_only=False)
    tr=np.array(inner['train_idx']);va=np.array(inner['val_idx']);assert state['fit_ids']==inner['train_idx']
    prepared=features(data.X,data.Y,tr,va,state['base']);D=state['kernel_train_features'];cross=prepared['target_features'];hs=state['kernel_initial_state'];models=[]
    for e,c in enumerate(state['config_ids']):
     m=build_model(data.X[tr],data.Y[tr,e],configs[c],seed,cfg,'cpu');m.load_state_dict(state['neural'][e]);m.eval();models.append(m)
    means=np.array([float(m.y_mean) for m in models]);scales=np.array([float(m.y_std) for m in models]);anchor=.5*((cross@hs['weights']@hs['factor'].T+hs['intercept'])*state['kernel_scale']+state['kernel_mean']-means)/scales
    correction=torch.tensor(cross@D.T)@state['kernel_delta_coefficients']@state['kernel_environment']
    with torch.no_grad():
     replay=np.column_stack([m.inverse_y((m(torch.tensor(data.X[va],dtype=torch.float32)).mean((1,2)).double()+torch.tensor(anchor[:,e])+correction[:,e])[:,None])[:,0].numpy() for e,m in enumerate(models)])
    error=float(np.max(abs(replay-unit_arrays[j][key])));max_replay=max(max_replay,error);assert error<=1e-6
for method,values in all_metrics.items():np.testing.assert_allclose(np.mean(values,axis=(0,1)),[summary['macro'][method][k] for k in ('pcc','rmse','mae')],atol=1e-12,rtol=0)
certs=[r['kernel_certificate'] for r in records if r['step']>0];paired={}
for baseline in ('initial_fusion','original_quantile'):
 delta=np.mean(np.array(all_metrics['joint'])-np.array(all_metrics[baseline]),axis=1)
 paired[baseline]=dict(macro_delta=delta.mean(0).tolist(),pcc_higher_folds=int((delta[:,0]>1e-12).sum()),rmse_lower_folds=int((delta[:,1]<-1e-12).sum()),mae_lower_folds=int((delta[:,2]<-1e-12).sum()))
audit=dict(trajectories=180,synchronized_states=len(records),selected_states_replayed=60,updated_states_replayed=180,weight_replay_max_difference=max_replay,initialization_max_difference=max(r['initialization_max_difference'] for r in records if r['step']==0),max_kernel_relative_gap=max(r['relative_gap'] for r in certs),max_kernel_iterations=max(r['iterations'] for r in certs),paired=paired,source_sha256=hashes)
(p/'verification.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(summary['macro']));print(json.dumps(audit))
