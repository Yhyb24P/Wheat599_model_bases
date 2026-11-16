import json, re
from pathlib import Path
M=json.loads(Path('docs/PATH_MIGRATION.json').read_text())
IMP=M['imports']
def map_dotted(d):
    # longest prefix match in imports map
    for old in sorted(IMP,key=len,reverse=True):
        if d==old or d.startswith(old+'.'):
            return IMP[old]+d[len(old):]
    return None
def rewrite(text):
    # bare: from wheat599 import a, b  -> regroup by family
    def bare(m):
        names=[n.strip() for n in m.group(1).split(',')]
        groups={}
        for n in names:
            nn=n.split(' as ')[0].strip()
            mapped=map_dotted('wheat599.'+nn)
            if mapped is None: return m.group(0)
            parent=mapped.rsplit('.',1)[0]
            groups.setdefault(parent,[]).append(n)
        return '; '.join(f'from {p} import {", ".join(ns)}' for p,ns in groups.items())
    text=re.sub(r'from wheat599 import ([^\n]+)', bare, text)
    # dotted: from wheat599.x.y import -> from mapped import
    text=re.sub(r'from (wheat599\.[a-z0-9_.]+) import', lambda m: f'from {map_dotted(m.group(1)) or m.group(1)} import', text)
    # string refs 'wheat599.x...' / "wheat599.x..."
    def strref(m):
        q=m.group(1); d=m.group(2); mapped=map_dotted(d)
        return f'{q}{mapped or d}{q}'
    text=re.sub(r'([\'"])(wheat599\.[a-z0-9_.]+)\1', strref, text)
    return text
import sys
changed=0
for p in list(Path('src/genopred').rglob('*.py'))+list(Path('scripts').glob('run_*.py')):
    t=p.read_text(); nt=rewrite(t)
    if nt!=t: p.write_text(nt); changed+=1
print('files rewritten:',changed)
