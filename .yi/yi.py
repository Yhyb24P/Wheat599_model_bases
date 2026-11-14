#!/usr/bin/env python3
"""YI's staged growth budget. No checkout, model call, or implicit Git staging."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys


def git(root, *args, optional=False):
    result = subprocess.run(['git', '-C', str(root), *args], capture_output=True)
    if result.returncode and not optional:
        raise ValueError(result.stderr.decode(errors='replace').strip())
    return None if result.returncode else result.stdout


def staged(root):
    entries = {}
    for row in git(root, 'ls-files', '--stage', '-z').split(b'\0'):
        if not row:
            continue
        info, path = row.split(b'\t', 1)
        mode, oid, stage = info.decode().split()
        if stage != '0':
            raise ValueError('Resolve index conflicts before measuring')
        entries[path.decode('utf-8')] = (mode, oid)
    return entries


def contents(root, entries, name):
    if name not in entries:
        return None
    mode, oid = entries[name]
    if mode not in ('100644', '100755'):
        raise ValueError(f'{name}: configuration must be a regular file')
    return git(root, 'cat-file', 'blob', oid)


def default_policy(measure, cap):
    def group(patterns, extensions=None):
        return {'include': patterns, 'exclude': [], 'extensions': extensions or []}
    return {'version': 1, 'max_file_lines': cap, 'slack': 0,
            'groups': {
                'tests': group(list(measure.TEST_GLOBS), sorted(measure.CODE_EXTS)),
                'docs': group(['**'], ['.md', '.rst', '.txt']),
                'maintenance': group(['scripts/**', '.yi/**', 'configs/**', '.github/**', '.githooks/**', 'templates/**',
                                      '*.toml', '*.json', '*.yaml', '*.yml', '*.lock',
                                      '*.sh', '.gitignore', '.gitattributes', 'Makefile'],
                                     sorted(measure.CODE_EXTS | {'.toml', '.json', '.yaml', '.yml', '.lock', ''})),
                'source': group(['**'], sorted(measure.CODE_EXTS))},
            'ceilings': {}, 'file_ceilings': {}}


def measure_index(root, entries, cfg, measure):
    metadata = {'.ratchet.json', '.ratchet-grants.jsonl'}
    paths = [p for p in entries if p not in metadata]
    classified = measure.classify(paths, cfg['groups'])
    sizes = {g: {} for g in cfg['groups']}
    outside = []
    for path in paths:
        mode, oid = entries[path]
        if mode not in ('100644', '100755'):
            if path in classified:
                raise ValueError(f'{path}: budgeted symlinks/submodules need a separate policy')
            outside.append(path)
        elif path not in classified:
            outside.append(path)
        else:
            data = git(root, 'cat-file', 'blob', oid)
            if data.startswith(b'version https://git-lfs.github.com/spec/v1'):
                raise ValueError(f'{path}: budgeted LFS objects cannot be measured from pointer text')
            sizes[classified[path]][path] = measure.count_text(data)
    return sizes, outside


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-C', '--project', default='.')
    parser.add_argument('action', choices=['init', 'budget', 'check', 'tighten'])
    parser.add_argument('--max-file-lines', type=int, default=2000,
                        help='initial file cap; calibrate for the project')
    parser.add_argument('--base', help='baseline commit, defaults to HEAD')
    parser.add_argument('--accept-policy-change', action='store_true',
                        help='explicitly accept reviewed counting-scope changes (not authentication)')
    args = parser.parse_args(argv)
    config_error = ValueError
    try:
        if importlib.metadata.version('ratchet-size') != '0.2.3':
            raise ValueError('Install ratchet-size==0.2.3')
        from ratchet import config, measure, rules
        config_error = config.ConfigError
        root = Path(git(args.project, 'rev-parse', '--show-toplevel').decode().strip())
        entries = staged(root)
        raw = contents(root, entries, config.CONFIG)
        policy = root / config.CONFIG
        base_ref = args.base or ('HEAD' if git(root, 'rev-parse', '--verify', 'HEAD', optional=True) else None)
        if base_ref:
            git(root, 'rev-parse', '--verify', base_ref + '^{commit}')
        if args.action == 'init':
            if base_ref and git(root, 'show', f'{base_ref}:{config.CONFIG}', optional=True):
                raise ValueError('Baseline already committed; restore the policy instead of init')
            if raw is not None or policy.exists() or policy.is_symlink():
                raise ValueError('Budget already exists; use tighten, not re-baselining')
            if args.max_file_lines <= 0:
                raise ValueError('File cap must be positive')
            cfg = default_policy(measure, args.max_file_lines)
        else:
            if raw is None:
                raise ValueError('Stage .ratchet.json before measuring')
            cfg = config.parse(raw.decode())
        sizes, outside = measure_index(root, entries, cfg, measure)
        totals = rules.totals(sizes)
        if args.action == 'init':
            if not any(sizes.values()):
                raise ValueError('Stage a runnable initial implementation before init')
            cfg['ceilings'] = totals
            cfg['file_ceilings'] = {p: n for files in sizes.values()
                                    for p, n in files.items() if n > args.max_file_lines}
        log = config.parse_grants((contents(root, entries, config.GRANTS) or b'').decode())
        base = None
        base_log = []
        if base_ref:
            old = git(root, 'show', f'{base_ref}:{config.CONFIG}', optional=True)
            if old:
                base = config.parse(old.decode())
            old_log = git(root, 'show', f'{base_ref}:{config.GRANTS}', optional=True)
            base_log = config.parse_grants((old_log or b'').decode())
        if base and list(base['groups'].items()) != list(cfg['groups'].items()):
            scope = hashlib.sha256(json.dumps(list(cfg['groups'].items()),
                                   sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            approved = any(g.get('scope_sha256') == scope and g.get('reason') and g.get('by')
                           for g in log[len(base_log):])
            if not approved and not args.accept_policy_change:
                raise ValueError(f'Counting scope changed; review scope SHA-256 {scope} and record its approval')
        violations = rules.check(cfg, sizes, base, base_log, log)
        print(json.dumps({'groups': {g: {'lines': n, 'ceiling': cfg['ceilings'][g],
                                             'remaining': cfg['ceilings'][g] - n} for g, n in totals.items()},
                          'outside_budget': sorted(outside),
                          'violations': [v.to_dict() for v in violations]}, ensure_ascii=False, indent=2))
        if violations:
            return 1
        if args.action == 'init':
            with policy.open('x') as output:
                json.dump(cfg, output, ensure_ascii=False, indent=2)
                output.write('\n')
            print('Created .ratchet.json from staged blobs; review scope and stage the file', file=sys.stderr)
        if args.action == 'tighten':
            if policy.is_symlink() or policy.read_bytes() != raw:
                raise ValueError('Stage the working policy first; refusing to overwrite it')
            updated, _ = rules.tighten(cfg, sizes)
            config.save(str(root), updated)
            print('Tightened staged totals; stage .ratchet.json with the cleanup', file=sys.stderr)
        return 0
    except (ValueError, OSError, KeyError, TypeError, config_error, importlib.metadata.PackageNotFoundError) as error:
        print(f'yi: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
