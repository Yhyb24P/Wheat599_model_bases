import subprocess, shutil, json
from pathlib import Path
def mv(a,b):
    a=Path(a); b=Path(b)
    if not a.exists(): return False
    b.parent.mkdir(parents=True,exist_ok=True)
    if a.is_dir():
        # move contents into target dir (merge)
        for child in a.iterdir():
            if child.name=='__pycache__': shutil.rmtree(child); continue
            dest=b/child.name
            if dest.exists(): 
                if child.is_dir():
                    for c in child.iterdir(): shutil.move(str(c),str(dest/c.name))
                    shutil.rmtree(child)
                else: shutil.move(str(child),str(dest))
            else: shutil.move(str(child),str(dest))
        shutil.rmtree(a)
    else:
        shutil.move(str(a),str(b))
    return True
moves=[]
# models: tabular+deep -> neural
for sub in ['tabular','deep']:
    for f in Path(f'src/genopred/models/{sub}').glob('*.py'):
        if f.name=='__init__.py': continue
        moves.append((str(f), f'src/genopred/models/neural/{f.name}'))
# library distribute
libmap={'base.py':'src/genopred/models/base.py','linear.py':'src/genopred/models/classical/linear.py','kernel_ridge_gblup.py':'src/genopred/models/classical/kernel_ridge_gblup.py','svr_rbf.py':'src/genopred/models/classical/svr_rbf.py','rkhs_gaussian.py':'src/genopred/models/kernels/rkhs_gaussian.py','ft_transformer.py':'src/genopred/models/neural/ft_transformer.py','tabm.py':'src/genopred/models/neural/tabm.py','tabpfn_v3_5.py':'src/genopred/models/neural/tabpfn_v3_5.py'}
for n,t in libmap.items(): moves.append((f'src/genopred/models/library/{n}',t))
# wheat2000 adapters distribute
admap={'wheat2000_classical.py':'classical','wheat2000_gblup.py':'kernels','wheat2000_huber.py':'kernels','wheat2000_candidate.py':'neural','wheat2000_gsformer.py':'neural','wheat2000_moegp.py':'neural','wheat2000_mtcro.py':'neural','wheat2000_wheatgp.py':'neural'}
for n,fam in admap.items(): moves.append((f'src/genopred/adapters/{n}',f'src/genopred/models/{fam}/{n}'))
# vendored -> third_party
moves.append(('src/genopred/adapters/dnawhisper','third_party/dnawhisper'))
moves.append(('src/genopred/adapters/bayesian_trees','third_party/bayesian_trees'))
moves.append(('vendor/WheatGP','third_party/WheatGP'))
# results: tabm+structured -> neural; diagnostics -> wheat599/diagnostics
for sub in ['tabm','structured']:
    for d in Path(f'results/wheat599/{sub}').iterdir():
        if d.is_dir(): moves.append((str(d), f'results/wheat599/neural/{d.name}'))
for d in Path('results/diagnostics').iterdir():
    if d.is_dir(): moves.append((str(d), f'results/wheat599/diagnostics/{d.name}'))
# splits -> protocols
for n in ['get_icm_protocol_20261103','bimsgp_ftt_fixed_20261029','mulphe_splits_20261105']:
    moves.append((f'results/wheat599/diagnostics/{n}', f'protocols/{n}'))
moves.append(('results/wheat2000/wheat2000_protocol_20261105','protocols/wheat2000_protocol_20261105'))
# ref -> references
moves.append(('ref','references'))
done=0
for a,b in moves:
    if mv(a,b): done+=1
# cleanup empty dirs + __init__
for empty in ['src/genopred/models/tabular','src/genopred/models/deep','src/genopred/models/library','src/genopred/adapters','results/wheat599/tabm','results/wheat599/structured','results/diagnostics','vendor']:
    p=Path(empty)
    if p.exists(): shutil.rmtree(p)
# ensure __init__ for neural
Path('src/genopred/models/neural/__init__.py').touch()
print('moves done:',done)
print('models now:', sorted(x.name for x in Path('src/genopred/models').iterdir() if x.is_dir()))
print('third_party:', sorted(x.name for x in Path('third_party').iterdir()))
print('protocols:', sorted(x.name for x in Path('protocols').iterdir()))
print('results/wheat599:', sorted(x.name for x in Path('results/wheat599').iterdir()))
