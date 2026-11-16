import json,hashlib
from pathlib import Path
import numpy as np
from wheat599.data import load,sha256_of
from wheat599.metrics import pcc,rmse,mae
p=Path('results/distill_quantile_development_20261109_parallel');read=lambda q:json.loads(q.read_text());cfg=read(p/'provenance.json')['config'];folds=read(Path(cfg['splits']))['folds'];data=load(Path(cfg['data']),cfg['data_sha256']);summary=read(p/'summary.json')
rows=[];teachers=[];hashes={};sizes=set(); max_identity=0

def stats(y,pred):
 global max_identity
 y=np.asarray(y,dtype=np.float64);pred=np.asarray(pred,dtype=np.float64)
 r=pcc(y,pred);sy=float(y.std());sp=float(pred.std());bias=float((pred.mean()-y.mean())**2);scale=(sp-r*sy)**2;residual=sy**2*(1-r*r);mse=float(np.mean((pred-y)**2))
 max_identity=max(max_identity,abs(mse-bias-scale-residual));assert abs(mse-bias-scale-residual)<1e-12
 return dict(pcc=r,rmse=rmse(y,pred),mae=mae(y,pred),mse=mse,bias_squared=bias,scale_mismatch=scale,correlation_residual=residual,pred_mean=float(pred.mean()),truth_mean=float(y.mean()),pred_sd=sp,truth_sd=sy)
for f in folds:
 chosen=next(r['selected_strengths'] for r in summary['folds'] if r['fold']==f['fold'])
 for j,sp in enumerate(f['inner']):
  unit=p/f"unit_{f['fold']}_{j}";predfile=p/f"predictions_{f['fold']}_{j}.npz";a=np.load(predfile);t=np.load(unit/'teacher_targets.npz');parts=read(unit/'teacher_partitions.json');oldquality=read(unit/'teacher_quality.json');ids=t['fit_ids'];assert np.array_equal(ids,sp['train_idx']);fit=set(ids.tolist());held=[]
  for part in parts:
   tr=set(part['train_ids']);va=set(part['heldout_ids']);assert not tr&va and tr|va==fit;held+=part['heldout_ids'];sizes.add((len(fit),len(tr)))
   for inner in part['inner']:
    assert not set(inner['train'])&set(inner['valid']);assert set(inner['train'])|set(inner['valid'])==set(range(len(tr)))
  assert len(held)==len(set(held))==len(fit) and set(held)==fit
  assert np.allclose(t['fusion'],.5*(t['quantile']+t['huber']),rtol=0,atol=1e-12)
  for name in ('quantile','huber','fusion'):
   for e in range(4):
    v=stats(data.Y[ids,e],t[name][:,e]);assert np.allclose([v[k] for k in ('pcc','rmse','mae')],oldquality[name][e],rtol=0,atol=1e-12)
    teachers.append(dict(fold=f['fold'],inner=j,environment=e,method=name,aggregation='pooled_crossfit_within_student_fit',**v))
    for part in parts:
     idx=np.flatnonzero(np.isin(ids,part['heldout_ids']));teachers.append(dict(fold=f['fold'],inner=j,environment=e,teacher_fold=part['teacher_fold'],method=name,aggregation='within_teacher_holdout_block',**stats(data.Y[ids[idx],e],t[name][idx,e])))
  for e in range(4):
   for method,suffix in [('original','original'),('selected',f'l{chosen[e]}')]+[(f'lambda_{l}',f'l{l}') for l in cfg['strengths']]:
    pred=np.mean([a[f'e{e}_s{seed}_{suffix}'] for seed in cfg['seeds']],axis=0)
    rows.append(dict(fold=f['fold'],inner=j,environment=e,method=method,**stats(data.Y[sp['val_idx'],e],pred)))
  for file in [predfile,unit/'teacher_targets.npz',unit/'teacher_partitions.json',unit/'teacher_quality.json']:hashes[str(file)]=sha256_of(file)
keys=list(stats(np.array([1.,2.,3.]),np.array([.5,2.,2.6])))
def macro(records):return {k:float(np.mean([r[k] for r in records])) for k in keys}
methods=list(dict.fromkeys(r['method'] for r in rows));macros={m:macro([r for r in rows if r['method']==m]) for m in methods}
paired={}
for m in methods:
 ds=[]
 for f in folds:
  sel=[r for r in rows if r['method']==m and r['fold']==f['fold']];base=[r for r in rows if r['method']=='original' and r['fold']==f['fold']]
  ds.append({k:float(np.mean([r[k] for r in sel])-np.mean([r[k] for r in base])) for k in keys})
 paired[m]={'fold_deltas':ds,'pcc_higher_folds':sum(d['pcc']>1e-12 for d in ds),'rmse_higher_folds':sum(d['rmse']>1e-12 for d in ds),'mae_higher_folds':sum(d['mae']>1e-12 for d in ds)}
tmac={agg:{m:macro([r for r in teachers if r['aggregation']==agg and r['method']==m]) for m in ('quantile','huber','fusion')} for agg in ('pooled_crossfit_within_student_fit','within_teacher_holdout_block')}
report=dict(student_macro=macros,teacher_macro=tmac,paired_vs_original=paired,student_fit_teacher_train_sizes=sorted(sizes),max_mse_identity_residual=max_identity,validation_rows=rows,teacher_rows=teachers,source_sha256=hashes,analysis_script_sha256=sha256_of(Path(__file__)),notes=['All SDs use ddof=0; decomposition terms averaged per validation/environment unit, not squared macro RMSE.','Fixed lambda still uses validation checkpoint selection; no calibration or training performed.','Teacher metrics are crossfit on student fit materials, not paired with student validation or historical outer metrics.'])
(p/'diagnostics.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ('student_macro','teacher_macro','student_fit_teacher_train_sizes','max_mse_identity_residual')},indent=2))
print('fold counts', {m:{k:v for k,v in x.items() if k!='fold_deltas'} for m,x in paired.items()})
