import json,numpy as np
from pathlib import Path
from wheat599.data import load,sha256_of
from wheat599.metrics import pcc,rmse,mae
from wheat599.distill_quantile import load_teacher_blocks
p=Path('results/distill_quantile_correlation_development_20261109');read=lambda x:json.loads(x.read_text());s=read(p/'summary.json');c=read(p/'provenance.json')['config'];folds=read(Path(c['splits']))['folds'];data=load(Path(c['data']),c['data_sha256']);state=read(p/'execution_status.json');rows=read(p/'trajectories.json')
assert state['status']=='complete' and state['completed']=={'student_initialization':240,'student_finetune':960} and not read(p/'failures.json')
assert {k:sum(r['kind']==k for r in rows) for k in state['expected']}==state['expected']
methods=['student','extra_training_lambda0','original_quantile']+[f'fixed_lambda_{l}' for l in c['strengths']];out={m:[] for m in methods};details=[];sources={};maxdiff=0.;weights=0
for f in folds:
 fold=f['fold'];arrays=[np.load(p/f'predictions_{fold}_{j}.npz') for j in range(3)];record=next(r for r in s['folds'] if r['fold']==fold);chosen=[]
 for e in range(4):
  scores=[np.mean([pcc(data.Y[sp['val_idx'],e],np.mean([arrays[j][f'e{e}_s{seed}_l{l}'] for seed in c['seeds']],axis=0)) for j,sp in enumerate(f['inner'])]) for l in c['strengths']]
  chosen.append(c['strengths'][int(np.argmax(scores))])
 assert chosen==record['selected_strengths']
 for j,sp in enumerate(f['inner']):
  unit=p/f'unit_{fold}_{j}';load_teacher_blocks(c['teacher_source'],fold,j,np.array(sp['train_idx']),read(Path(c['teacher_audit']))['source_sha256'])
  records=read(unit/'student_training.json');assert len(records)==32
  for r in records:
   file=unit/f"student_e{r['environment']}_s{r['seed']}_l{r['strength']}.pt"
   assert sha256_of(file)==r['weight_sha256'];weights+=1
   assert r['best_epoch']<=r['epochs_run']<=100
  file=p/f'predictions_{fold}_{j}.npz';sources[str(file)]=sha256_of(file)
 for m in methods:
  values=[]
  for j,sp in enumerate(f['inner']):
   envs=[]
   for e in range(4):
    suffix='original' if m=='original_quantile' else 'l0.0' if m=='extra_training_lambda0' else 'l'+m.removeprefix('fixed_lambda_') if m.startswith('fixed_lambda_') else f'l{chosen[e]}'
    pred=np.mean([arrays[j][f'e{e}_s{seed}_{suffix}'] for seed in c['seeds']],axis=0);y=data.Y[sp['val_idx'],e];assert np.isfinite(pred).all()
    metrics=[fn(y,pred) for fn in (pcc,rmse,mae)];envs.append(metrics);details.append(dict(fold=fold,inner=j,environment=e,method=m,pcc=metrics[0],rmse=metrics[1],mae=metrics[2],pred_sd=float(pred.std())))
   values.append(envs)
  result=np.mean(values,axis=0);maxdiff=max(maxdiff,float(np.max(abs(result-record['metrics'][m]))));out[m].append(result)
for m,v in out.items():assert np.allclose(np.mean(v,axis=(0,1)),list(s['macro'][m].values()),rtol=0,atol=1e-12)
student=s['macro']['student'];baseline=s['macro']['original_quantile'];gate=student['pcc']>max(s['macro'][k]['pcc'] for k in ('original_quantile','original_fusion','extra_training_lambda0','previous_fixed_mse_lambda1')) and student['rmse']<=baseline['rmse'] and student['mae']<=baseline['mae'];assert gate==s['outer_gate_passed']
audit=dict(units=30,checked_weights=weights,selection_matches=40,max_metric_difference=maxdiff,warm_prediction_max_difference=max(r['max_difference'] for r in rows if r['kind']=='student_initialization'),mean_prediction_sd={m:float(np.mean([r['pred_sd'] for r in details if r['method']==m])) for m in methods},outer_gate_passed=gate,source_sha256=sources,validation_units=details)
(p/'verification.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(s['macro'],indent=2));print('gate',gate,'error',maxdiff,'sd',audit['mean_prediction_sd'])
