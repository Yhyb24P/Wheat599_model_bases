import json,csv,torch,yaml
from pathlib import Path
from collections import Counter
import numpy as np
from wheat599.data import load,sha256_of
from wheat599.metrics import pcc,rmse,mae
from wheat599 import quantile_tabm as qt
from wheat599.distill_quantile import predict
p=Path('results/distill_correlation_final_20261109');read=lambda x:json.loads(Path(x).read_text());state=read(p/'execution_status.json');assert state['status']=='complete' and state['completed']==state['expected'] and not read(p/'failures.json')
prov=read(p/'provenance.json');cfg=prov['config'];manifest=read(cfg['manifest']);data=load(Path(cfg['data']),cfg['data_sha256']);folds=read(cfg['splits'])['folds'];rows=read(p/'trajectories.json');summary=read(p/'summary.json');preds=np.load(p/'seed_predictions.npz');stored=list(csv.DictReader((p/'oof.csv').open()));assert len(stored)==2396
assert Counter(r['kind'] for r in rows)==Counter(state['expected'])
keys={(int(r['fold']),r['environment'],int(r['material_index'])) for r in stored};assert len(keys)==2396
torch.set_num_threads(1)
settings=yaml.safe_load(Path(cfg['development_config']).read_text()); configs=yaml.safe_load(Path(settings['architecture_config']).read_text())['configurations']; replay_max=0.;teacher_weights=0
methods=['quantile','original_quantile','huber','original_fusion','mse_tabm'];metrics=[];sources={};teacher_sizes=set()
for f in folds:
 unit=p/f"fold_{f['fold']}";t=np.load(unit/'teacher_targets.npz');assert np.array_equal(t['fit_ids'],f['dev_idx']);parts=read(unit/'teacher_partitions.json');held=[]
 for part in parts:
  tr,va=set(part['train_ids']),set(part['heldout_ids']);assert tr.isdisjoint(va) and tr|va==set(f['dev_idx']);assert not (tr|va)&set(f['test_idx']);held+=part['heldout_ids'];teacher_sizes.add((len(tr),len(va)))
  for inner in part['inner']:assert set(inner['train']).isdisjoint(inner['valid']) and set(inner['train'])|set(inner['valid'])==set(range(len(tr)))
 assert len(held)==len(set(held))==len(f['dev_idx']);assert np.allclose(t['fusion'],.5*(t['quantile']+t['huber']),rtol=0,atol=1e-12)
 for part in parts:
  for record in part['refits']:
   file=unit/f"teacher_{part['teacher_fold']}"/f"e{record['environment']}_s{record['seed']}.pt"
   assert sha256_of(file)==record['weight_sha256'];teacher_weights+=1
 for r in [r for r in rows if r['kind']=='student_finetune' and r['fold']==f['fold']]:
  frozen=next(v for v in manifest['rows'] if (v['fold'],v['environment'],v['seed'])==(r['fold'],r['environment'],r['seed']))
  assert r['epochs']==frozen['finetuning_epochs'] and r['strength']==frozen['strength']
  model=qt.build_model(data.X[f['dev_idx']],data.Y[f['dev_idx'],r['environment']],configs[frozen['config_index']],r['seed'],settings,'cpu')
  model.load_state_dict(torch.load(unit/f"e{r['environment']}_s{r['seed']}.pt",map_location='cpu',weights_only=True))
  replay=predict(model,data.X[f['test_idx']],'cpu');delta=float(np.max(abs(replay-preds[f"f{f['fold']}_e{r['environment']}_s{r['seed']}"])))
  replay_max=max(replay_max,delta);assert delta<=1e-6
  history=read(unit/f"e{r['environment']}_s{r['seed']}_training.json");assert len(history)==r['epochs'];assert all('val_pcc' not in v for v in history)
  weight=unit/f"e{r['environment']}_s{r['seed']}.pt";assert sha256_of(weight)==r['weight_sha256']
 for e,name in enumerate(data.environments):
  selected=[r for r in stored if int(r['fold'])==f['fold'] and r['environment']==str(name)];assert [int(r['material_index']) for r in selected]==f['test_idx']
  y=data.Y[f['test_idx'],e];assert np.allclose([float(r['y_true']) for r in selected],y,atol=1e-12,rtol=0)
  bank=np.mean([preds[f"f{f['fold']}_e{e}_s{seed}"] for seed in [42,2026]],axis=0)
  assert np.allclose(bank,[float(r['quantile']) for r in selected],rtol=0,atol=1e-12)
  vals={m:dict(zip(('pcc','rmse','mae'),[fn(y,np.array([float(r[m]) for r in selected])) for fn in (pcc,rmse,mae)])) for m in methods}
  metrics.append(dict(fold=f['fold'],environment=str(name),metrics=vals))
for method in methods:
 for metric in ('pcc','rmse','mae'):assert abs(np.mean([r['metrics'][method][metric] for r in metrics])-summary['macro'][method][metric])<1e-12
for method in methods:
 for name in data.environments:
  subset=[r for r in metrics if r['environment']==str(name)]
  raw=[r for r in stored if r['environment']==str(name)]
  truth=np.array([float(r['y_true']) for r in raw]);prediction=np.array([float(r[method]) for r in raw])
  for k,fn in zip(('pcc','rmse','mae'),(pcc,rmse,mae)):
   assert abs(np.mean([r['metrics'][method][k] for r in subset])-summary['per_environment'][method][str(name)][k])<1e-12
   assert abs(fn(truth,prediction)-summary['pooled_by_environment'][method][str(name)][k])<1e-12
for r in metrics:
 original=next(v for v in summary['per_fold'] if v['fold']==r['fold'] and v['environment']==r['environment'])
 for method in methods:
  for k in ('pcc','rmse','mae'):assert abs(r['metrics'][method][k]-original['metrics'][method][k])<1e-12
paired={}
for baseline in ('original_quantile','huber','original_fusion'):
 deltas=np.array([[np.mean([r['metrics']['quantile'][k]-r['metrics'][baseline][k] for r in metrics if r['fold']==f['fold']]) for k in ('pcc','rmse','mae')] for f in folds])
 paired[baseline]=dict(pcc_higher_folds=int((deltas[:,0]>0).sum()),rmse_lower_folds=int((deltas[:,1]<0).sum()),mae_lower_folds=int((deltas[:,2]<0).sum()),macro_delta=deltas.mean(0).tolist(),fold_deltas=deltas.tolist())
audit=dict(checked_fold_and_pooled_metrics=True,checked_teacher_weights=teacher_weights,student_weight_replay_max_difference=replay_max,checked_outer_material_environment_keys=2396,checked_students=80,teacher_train_holdout_sizes=sorted(teacher_sizes),warm_max_difference=max(r['max_difference'] for r in rows if r['kind']=='student_initialization'),zero_update_students=sum(r['finetuning_epochs']==0 for r in manifest['rows']),paired=paired,source_sha256={str(p/n):sha256_of(p/n) for n in ('oof.csv','seed_predictions.npz','trajectories.json','summary.json')})
(p/'verification.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(summary['macro']));print(json.dumps(audit))
