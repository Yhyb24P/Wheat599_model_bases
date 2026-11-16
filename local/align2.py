import subprocess
from pathlib import Path
def gmv(a,b):
    a=Path(a); b=Path(b)
    if not a.exists(): return False
    b.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run(['git','mv',str(a),str(b)],check=True,capture_output=True)
    return True
n=0
for sub in ['tabular','deep']:
    for f in sorted(Path(f'src/genopred/models/{sub}').glob('*.py')):
        if f.name=='__init__.py': continue
        n+=gmv(str(f), f'src/genopred/models/neural/{f.name}')
libmap={'base.py':'src/genopred/models/base.py','linear.py':'src/genopred/models/classical/linear.py','kernel_ridge_gblup.py':'src/genopred/models/classical/kernel_ridge_gblup.py','svr_rbf.py':'src/genopred/models/classical/svr_rbf.py','rkhs_gaussian.py':'src/genopred/models/kernels/rkhs_gaussian.py','ft_transformer.py':'src/genopred/models/neural/ft_transformer.py','tabm.py':'src/genopred/models/neural/tabm.py','tabpfn_v3_5.py':'src/genopred/models/neural/tabpfn_v3_5.py'}
for a,b in libmap.items(): n+=gmv(f'src/genopred/models/library/{a}',b)
admap={'wheat2000_classical.py':'classical','wheat2000_gblup.py':'kernels','wheat2000_huber.py':'kernels','wheat2000_candidate.py':'neural','wheat2000_gsformer.py':'neural','wheat2000_moegp.py':'neural','wheat2000_mtcro.py':'neural','wheat2000_wheatgp.py':'neural'}
for a,fam in admap.items(): n+=gmv(f'src/genopred/adapters/{a}',f'src/genopred/models/{fam}/{a}')
n+=gmv('src/genopred/adapters/dnawhisper','third_party/dnawhisper')
n+=gmv('src/genopred/adapters/bayesian_trees','third_party/bayesian_trees')
n+=gmv('vendor/WheatGP','third_party/WheatGP')
for sub in ['tabm','structured']:
    for d in sorted(Path(f'results/wheat599/{sub}').iterdir()):
        if d.is_dir(): n+=gmv(str(d), f'results/wheat599/neural/{d.name}')
for d in sorted(Path('results/diagnostics').iterdir()):
    if d.is_dir(): n+=gmv(str(d), f'results/wheat599/diagnostics/{d.name}')
for p in ['get_icm_protocol_20261103','bimsgp_ftt_fixed_20261029','mulphe_splits_20261105']:
    n+=gmv(f'results/wheat599/diagnostics/{p}', f'protocols/{p}')
n+=gmv('results/wheat2000/wheat2000_protocol_20261105','protocols/wheat2000_protocol_20261105')
n+=gmv('ref','references')
print('git mv done:',n)
