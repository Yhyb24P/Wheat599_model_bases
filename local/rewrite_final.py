import re
from pathlib import Path
# dotted import replacements (current WP-A/B -> final aligned), longest first
dotted={
 'genopred.models.library.kernel_ridge_gblup':'genopred.models.classical.kernel_ridge_gblup',
 'genopred.models.library.tabpfn_v3_5':'genopred.models.neural.tabpfn_v3_5',
 'genopred.models.library.ft_transformer':'genopred.models.neural.ft_transformer',
 'genopred.models.library.rkhs_gaussian':'genopred.models.kernels.rkhs_gaussian',
 'genopred.models.library.svr_rbf':'genopred.models.classical.svr_rbf',
 'genopred.models.library.linear':'genopred.models.classical.linear',
 'genopred.models.library.base':'genopred.models.base',
 'genopred.models.library.tabm':'genopred.models.neural.tabm',
 'genopred.adapters.wheat2000_classical':'genopred.models.classical.wheat2000_classical',
 'genopred.adapters.wheat2000_gblup':'genopred.models.kernels.wheat2000_gblup',
 'genopred.adapters.wheat2000_huber':'genopred.models.kernels.wheat2000_huber',
 'genopred.adapters.wheat2000_candidate':'genopred.models.neural.wheat2000_candidate',
 'genopred.adapters.wheat2000_gsformer':'genopred.models.neural.wheat2000_gsformer',
 'genopred.adapters.wheat2000_moegp':'genopred.models.neural.wheat2000_moegp',
 'genopred.adapters.wheat2000_mtcro':'genopred.models.neural.wheat2000_mtcro',
 'genopred.adapters.wheat2000_wheatgp':'genopred.models.neural.wheat2000_wheatgp',
 'genopred.adapters.dnawhisper':'third_party.dnawhisper',
 'genopred.adapters.bayesian_trees':'third_party.bayesian_trees',
 'genopred.models.tabular':'genopred.models.neural',
 'genopred.models.deep':'genopred.models.neural',
}
# path replacements
paths={
 'src/genopred/models/library/kernel_ridge_gblup.py':'src/genopred/models/classical/kernel_ridge_gblup.py',
 'src/genopred/models/library/tabpfn_v3_5.py':'src/genopred/models/neural/tabpfn_v3_5.py',
 'src/genopred/models/library/ft_transformer.py':'src/genopred/models/neural/ft_transformer.py',
 'src/genopred/models/library/rkhs_gaussian.py':'src/genopred/models/kernels/rkhs_gaussian.py',
 'src/genopred/models/library/svr_rbf.py':'src/genopred/models/classical/svr_rbf.py',
 'src/genopred/models/library/linear.py':'src/genopred/models/classical/linear.py',
 'src/genopred/models/library/base.py':'src/genopred/models/base.py',
 'src/genopred/models/library/tabm.py':'src/genopred/models/neural/tabm.py',
 'src/genopred/adapters/wheat2000_classical.py':'src/genopred/models/classical/wheat2000_classical.py',
 'src/genopred/adapters/wheat2000_gblup.py':'src/genopred/models/kernels/wheat2000_gblup.py',
 'src/genopred/adapters/wheat2000_huber.py':'src/genopred/models/kernels/wheat2000_huber.py',
 'src/genopred/adapters/wheat2000_candidate.py':'src/genopred/models/neural/wheat2000_candidate.py',
 'src/genopred/adapters/wheat2000_gsformer.py':'src/genopred/models/neural/wheat2000_gsformer.py',
 'src/genopred/adapters/wheat2000_moegp.py':'src/genopred/models/neural/wheat2000_moegp.py',
 'src/genopred/adapters/wheat2000_mtcro.py':'src/genopred/models/neural/wheat2000_mtcro.py',
 'src/genopred/adapters/wheat2000_wheatgp.py':'src/genopred/models/neural/wheat2000_wheatgp.py',
 'src/genopred/models/tabular/':'src/genopred/models/neural/',
 'src/genopred/models/deep/':'src/genopred/models/neural/',
 'results/wheat599/tabm/':'results/wheat599/neural/',
 'results/wheat599/structured/':'results/wheat599/neural/',
 'results/wheat599/diagnostics/get_icm_protocol_20261103':'protocols/get_icm_protocol_20261103',
 'results/wheat599/diagnostics/bimsgp_ftt_fixed_20261029':'protocols/bimsgp_ftt_fixed_20261029',
 'results/wheat599/diagnostics/mulphe_splits_20261105':'protocols/mulphe_splits_20261105',
 'results/wheat2000/wheat2000_protocol_20261105':'protocols/wheat2000_protocol_20261105',
 'results/diagnostics/':'results/wheat599/diagnostics/',
 'ref/index.html':'references/index.html',
 'ref/build.py':'references/build.py',
 'ref/references.bib':'references/references.bib',
}
def rewrite(t):
    for old in sorted(dotted,key=len,reverse=True):
        t=t.replace(old,dotted[old])
    for old in sorted(paths,key=len,reverse=True):
        t=t.replace(old,paths[old])
    return t
files=list(Path('src/genopred').rglob('*.py'))+list(Path('scripts').glob('*.py'))+list(Path('tests').glob('*.py'))+list(Path('configs').rglob('*.yaml'))+list(Path('docs').glob('*.md'))
n=0
for p in files:
    t=p.read_text(); nt=rewrite(t)
    if nt!=t: p.write_text(nt); n+=1
# library relative import fix: distributed files use from .base -> from ..base
for p in Path('src/genopred/models').rglob('*.py'):
    if p.name=='base.py': continue
    t=p.read_text()
    nt=t.replace('from .base import','from ..base import').replace('from . import base','from .. import base')
    if nt!=t: p.write_text(nt)
print('rewritten files:',n)
