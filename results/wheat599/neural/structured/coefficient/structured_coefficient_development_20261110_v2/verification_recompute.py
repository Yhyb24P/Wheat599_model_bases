import json
from pathlib import Path
import numpy as np,torch
from wheat599.data import load,sha256_of
from wheat599.metrics import pcc,rmse,mae
from wheat599.structured_coefficient import expansion
p=Path('results/structured_coefficient_development_20261110_v2');read=lambda q:json.loads(Path(q).read_text());prov=read(p/'provenance.json');cfg=prov['config'];status=read(p/'execution_status.json');summary=read(p/'summary.json');data=load(Path(cfg['data']),cfg['data_sha256']);folds=read(cfg['splits'])['folds'];torch.set_num_threads(1)
assert status['status']=='complete' and status['completed_units']==30 and status['candidate_fits']==15120 and status['reference_four_target_fits']==1890 and not read(p/'failures.json')
for path,digest in prov['sha256'].items():assert sha256_of(Path(path))==digest
cases=summary['configurations'];bases=summary['reference_configurations'];macros={k:[] for k in summary['macro']};all_diags=[];max_replay=0.;max_objective=0.;sources={}
for fold in folds:
 f=fold['fold'];record=next(r for r in summary['folds'] if r['fold']==f);arrays=[np.load(p/f'unit_{f}_{j}/predictions.npz') for j in range(3)]
 scores=[np.mean([pcc(data.Y[sp['val_idx'],e],arrays[j][f'c{c}'][:,e]) for j,sp in enumerate(fold['inner']) for e in range(4)]) for c in range(len(cases))]
 chosen=next(c for c,v in enumerate(scores) if max(scores)-v<=cfg['tie_tolerance']);assert chosen==record['selected_index'];np.testing.assert_allclose(scores,record['candidate_pcc_by_config'],atol=1e-12,rtol=0)
 reference=[]
 for e in range(4):
  scores_b=[np.mean([pcc(data.Y[sp['val_idx'],e],arrays[j][f'b{b}'][:,e]) for j,sp in enumerate(fold['inner'])]) for b in range(len(bases))];reference.append(int(np.argmax(scores_b)))
 assert reference==record['reference_selected_indices']
 metrics={k:[] for k in macros};qchoices=next(r['selected'] for r in read(Path(cfg['quantile_source'])/'summary.json')['folds'] if r['fold']==f)
 for j,sp in enumerate(fold['inner']):
  unit=p/f'unit_{f}_{j}';verification=read(unit/'verification.json');diags=read(unit/'optimization.json');assert len(diags)==504 and {r['config_index'] for r in diags}==set(range(504));all_diags.extend(diags)
  for diag in diags:
   c=diag['config_index'];assert diag['number_observations']==len(sp['train_idx'])*4
   assert np.isclose(diag['normalized_lambda'],cases[c]['lambda'],rtol=1e-14,atol=0)
   assert np.isclose(diag['ridge'],cases[c]['lambda']*len(sp['train_idx'])*4,rtol=1e-14,atol=0)
  for name,digest in verification['sha256'].items():assert sha256_of(unit/name)==digest
  weights=np.load(unit/'coefficients.npz');y=data.Y[sp['train_idx']];np.testing.assert_array_equal(weights['mean'],y.mean(0));np.testing.assert_array_equal(weights['scale'],y.std(0))
  sources[str(unit/'predictions.npz')]=sha256_of(unit/'predictions.npz')
  tr=sp['train_idx'];va=sp['val_idx'];target=(y-weights['mean'])/weights['scale'];cov=np.corrcoef(y,rowvar=False)
  for h,eps in {(v['bandwidth'],v['epsilon']) for v in cases}:
   _,_,meta=expansion(data.X[tr],data.X[va],h,eps)
   assert meta['min_precision_eigenvalue']>=eps-1e-9
   for c in [i for i,v in enumerate(cases) if v['bandwidth']==h and v['epsilon']==eps]:
    a=weights[f'c{c}_coefficient'];b=weights[f'c{c}_intercept'];pred=(meta['query_rbf']@a+b)*weights['scale']+weights['mean'];err=float(np.max(abs(pred-arrays[j][f'c{c}'])));max_replay=max(max_replay,err);assert err<=1e-6
    if c==chosen:
     residual=meta['rbf']@a+b-target;diag=diags[c];delta=np.array(diag['thresholds']);B=(1-cases[c]['eta'])*np.eye(4)+cases[c]['eta']*cov
     loss=np.where(abs(residual)<=delta,.5*residual**2,delta*(abs(residual)-.5*delta)).mean()+.5*cases[c]['lambda']*np.trace(np.linalg.solve(B,a.T@meta['precision']@a))
     discrepancy=abs(loss-diag['primal']);max_objective=max(max_objective,discrepancy);assert discrepancy<1e-8
   for b in [i for i,v in enumerate(bases) if v['bandwidth']==h and v['epsilon']==eps]:
    pred=(meta['query_rbf']@weights[f'b{b}_coefficient'])*weights['scale']+weights['mean'];err=float(np.max(abs(pred-arrays[j][f'b{b}'])));max_replay=max(max_replay,err);assert err<=1e-6
  assert len(arrays[j].files)==567
  old=np.load(Path(cfg['quantile_source'])/f'predictions_{f}_{j}.npz');q=np.column_stack([np.mean([old[f'e{e}_c{c}_s{seed}'] for seed in cfg['seeds']],axis=0) for e,c in enumerate(qchoices)]);hub=np.load(Path(cfg['huber_reference'])/f'references_{f}_{j}.npz')['huber']
  predictions=dict(candidate=arrays[j][f'c{chosen}'],unweighted_posterior=np.column_stack([arrays[j][f'b{reference[e]}'][:,e] for e in range(4)]),original_quantile=q,huber=hub,fixed_fusion=.5*(q+hub))
  for name,pred in predictions.items():metrics[name].append([[fn(data.Y[va,e],pred[:,e]) for fn in (pcc,rmse,mae)] for e in range(4)])
 for name,values in metrics.items():
  value=np.mean(values,axis=0);np.testing.assert_allclose(value,record['metrics'][name],atol=1e-12,rtol=0);macros[name].append(value)
for name,v in macros.items():np.testing.assert_allclose(np.mean(v,axis=(0,1)),[summary['macro'][name][k] for k in ('pcc','rmse','mae')],atol=1e-12,rtol=0)
assert all(-1e-10<=d['relative_gap']<=cfg['relative_gap_tolerance'] and d['dual_zero_sum_residual']<=1e-12 and d['dual_box_violation']<=1e-12 and d['gradient_inf']<=cfg['optimizer']['accepted_gradient'] for d in all_diags)
paired={}
for name in ('unweighted_posterior','original_quantile','huber','fixed_fusion'):
 delta=np.mean(np.array(macros['candidate'])-np.array(macros[name]),axis=1)
 paired[name]=dict(macro_delta=delta.mean(0).tolist(),fold_deltas=delta.tolist(),pcc_higher_folds=int((delta[:,0]>0).sum()),rmse_lower_folds=int((delta[:,1]<0).sum()),mae_lower_folds=int((delta[:,2]<0).sum()))
audit=dict(refined_solutions=sum(d.get('refinement_solver')=='trust-ncg' for d in all_diags),verified_units=30,candidate_fits=15120,reference_four_target_fits=1890,max_coefficient_prediction_replay_difference=max_replay,max_selected_primal_objective_difference=max_objective,max_relative_gap=max(d['relative_gap'] for d in all_diags),max_iterations=max(d['iterations'] for d in all_diags),max_gradient_inf=max(d['gradient_inf'] for d in all_diags),max_dual_zero_sum_residual=max(d['dual_zero_sum_residual'] for d in all_diags),max_dual_box_violation=max(d['dual_box_violation'] for d in all_diags),paired=paired,source_sha256=sources)
(p/'verification.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(summary['macro']));print(json.dumps(audit))
