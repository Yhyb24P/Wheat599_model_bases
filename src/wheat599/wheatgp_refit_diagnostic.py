"""Replay fixed WheatGP fits; inspect training without using outer test labels."""
import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
import yaml

from wheat599.data import load, sha256_of
from wheat599.wheatgp_nested import segments, validation_pcc


def replay(model_type, xf, yf, xq, epochs, cfg, device):
    torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed'])
    model=model_type(8*(xf.shape[1]-20)).to(device)
    x=torch.as_tensor(xf,dtype=torch.float32,device=device);q=torch.as_tensor(xq,dtype=torch.float32,device=device)
    center,scale=float(yf.mean()),float(yf.std())
    if scale<=0:raise ValueError('Constant training target')
    y=torch.as_tensor((yf-center)[:,None]/scale,dtype=torch.float32,device=device)
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=90,gamma=.1)
    generator=torch.Generator(device='cpu').manual_seed(cfg['seed']); history=[]
    for epoch in range(1,epochs+1):
        model.train(); losses=[];norms=[]
        for idx in torch.randperm(len(x),generator=generator).split(cfg['batch_size']):
            idx=idx.to(device);optimizer.zero_grad(set_to_none=True)
            loss=torch.nn.functional.mse_loss(model(*segments(x[idx])),y[idx])
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
            losses.append(float(loss.detach()));norms.append(float(norm))
        scheduler.step();model.eval()
        with torch.no_grad(): train=model(*segments(x)).flatten().cpu().numpy()*scale+center
        history.append({'epoch':epoch,'train_prediction_std':float(np.std(train,dtype=np.float64)),
                        'train_unique':int(len(np.unique(train))),'batch_loss_mean':float(np.mean(losses)),
                        'gradient_norm_max':float(np.max(norms))})
    model.eval()
    with torch.no_grad(): pred=model(*segments(q)).flatten().cpu().numpy()*scale+center
    if not np.isfinite(pred).all():raise ValueError('Nonfinite prediction')
    return pred,{'target_mean':center,'target_std':scale,'history':history}


def run(cfg,args,runner):
    if args.selection is not None:raise ValueError('Diagnostic only')
    source=json.loads(Path(cfg['selection']).read_text());folds=source['splits'];settings=yaml.safe_load(Path(cfg['wheatgp_config']).read_text())
    if sha256_of(Path(settings['source']))!=settings['source_sha256']:raise ValueError('Official source mismatch')
    spec=importlib.util.spec_from_file_location('wheatgp_official',settings['source']);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    data=load(Path(settings['data']),settings['data_sha256']);device=torch.device(settings['device']);torch.set_num_threads(settings['cpu_threads'])
    if device.type=='cuda':torch.cuda.set_per_process_memory_fraction(settings['gpu_memory_fraction'])
    args.out.mkdir(parents=True,exist_ok=False);out=[]
    paths=[Path(cfg['selection']),Path(cfg['wheatgp_config']),Path(__file__),Path(runner),Path(args.config)]
    for f,e in cfg['cases']:
        fold=folds[f];choice=source['folds'][f]['wheatgp'][e]
        if fold['fold']!=f or choice['kind']!='raw':raise ValueError('Unexpected fixed case')
        chosen={**settings,'lr':choice['learning_rate']};epochs=choice['best_epoch']
        tr=np.array(fold['train_idx']);val=np.array(fold['val_idx']);dev=np.array(fold['dev_idx']);test=np.array(fold['test_idx'])
        a,training=replay(module.wheatGP_base,data.X[tr],data.Y[tr,e],data.X[val],epochs,chosen,device)
        score=validation_pcc(data.Y[val,e],a)
        if abs(score-choice['pcc'])>1e-5:raise ValueError('Selection replay mismatch')
        b,refit=replay(module.wheatGP_base,data.X[dev],data.Y[dev,e],data.X[test],epochs,chosen,device)
        path=Path(cfg['outer_predictions'])/f'predictions_{f}.npz';paths.append(path)
        saved=np.load(path)['wheatgp'][:,e];difference=float(np.max(np.abs(b-saved)))
        if difference>1e-6:raise ValueError('Refit replay mismatch')
        out.append({'fold':f,'environment':data.environments[e],'learning_rate':choice['learning_rate'],
                    'epochs':epochs,'selection_validation_pcc':float(score),'refit_saved_prediction_max_abs':difference,
                    'selected_training':training,'full_development_refit':refit})
    (args.out/'diagnostic.json').write_text(json.dumps({'cases':out,'test_labels_used':False,
        'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'sha256':{str(p):sha256_of(p) for p in paths}},indent=2,allow_nan=False)+'\n')
