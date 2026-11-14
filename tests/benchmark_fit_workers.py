"""Optional synthetic throughput check; never reads Wheat599 or reports model accuracy."""
import json,time,multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import torch
import hashlib
from wheat599.quantile_tabm import _init_fit_worker,_fit_worker

def main():
 rng=np.random.default_rng(91)
 x=rng.integers(0,2,(540,1279)).astype('float32');y=rng.normal(size=(540,4)).astype('float32')
 folds=[dict(fold=0,inner=[dict(train_idx=list(range(360)),val_idx=list(range(360,540)))])]
 configurations=[dict(spins=12,width=128,learning_rate=.001,weight_decay=.01)]
 settings=dict(device='cuda',gpu_memory_fraction=.3,cpu_threads=2,optimizer_variant='ising',n_quantiles=1,quantile_min=.001,quantile_max=.999,
               training=dict(objective='mse',max_epochs=10,patience=10,batch_size=128,gradient_clip=5,tie_tolerance=1e-12))
 tasks=[(0,0,i%4,0,42+i) for i in range(128)]
 _init_fit_worker(x,y,folds,configurations,settings,1)
 _fit_worker(tasks[0])
 start=time.perf_counter()
 expected=[_fit_worker(task) for task in tasks]
 serial=time.perf_counter()-start
 start=time.perf_counter()
 with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn'),initializer=_init_fit_worker,initargs=(x,y,folds,configurations,settings,4)) as pool:
  list(pool.map(_fit_worker,tasks[:16]))
  startup=time.perf_counter()-start
  start=time.perf_counter();actual=list(pool.map(_fit_worker,tasks));parallel=time.perf_counter()-start
 for a,b in zip(expected,actual):
  np.testing.assert_array_equal(a[3],b[3])
  assert a[4]['best_epoch']==b[4]['best_epoch']
 result=dict(synthetic_only=True,predictions_identical=True,seed=91,torch_version=str(torch.__version__),gpu=torch.cuda.get_device_name(0),
             implementation_sha256=hashlib.sha256(Path('src/wheat599/quantile_tabm.py').read_bytes()).hexdigest(),materials=540,markers=1279,spins=12,epochs=10,tasks=128,workers=4,
             serial_seconds=serial,parallel_seconds=parallel,pool_startup_and_warmup_seconds=startup,steady_speedup=serial/parallel)
 Path('.state').mkdir(exist_ok=True)
 Path('.state/ising_parallel_timing.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
