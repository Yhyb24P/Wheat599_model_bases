"""WS2 established deep baselines: DeepGS, DNNGP and SoyDNGP.

All three follow the nested Wheat599 protocol: inner-fold selection on
seed-averaged validation PCC, per-seed median best epoch and per-epoch median
learning-rate sequence frozen from the three inner runs, full-development
refit, and a single outer scoring pass.  Upstream identities:

* DeepGS — official R/mxnet network (cma2015/DeepGS commit 33ae2dc1, package
  1.2, GPL-2|GPL-3) as the independent PyTorch structure adaptation frozen in
  tag ``wheat2000-deepgs-protocol-20261105`` (src/wheat599/wheat2000_baselines.py).
  Only the network structure and official training defaults are reused; the
  Wheat2000 run records are not migrated.
* DNNGP — official compiled trainer (AIBreeding/DNNGP commit 3bbac096) loaded
  in a pinned Python 3.9 / TF 2.6 environment through ``dnngp_worker.py``.
  The official KFold split is bypassed; protocol splits are passed directly
  to ``dnngp.dnngp_model``.  Input is fit-set-only PCA at 95% explained
  variance (the official pc95 recipe refitted inside each training set).
* SoyDNGP — official SoyDNGPNext source (IndigoFloyd/SoyDNGPNext commit
  ebda01d0), model.yaml network built by the official ``remodel``; official
  one-hot/resize encoding, SmoothL1 loss, Adam and xavier init.  DArT 0/1
  markers map to VCF codes 0/0 and 1/1.  Target min/max comes from the fit
  set only (upstream uses the whole dataset).
"""
import importlib.util
import json
import multiprocessing
import os
import subprocess
import sys
import time
import types
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from genopred.data.wheat599 import load, sha256_of
from genopred.data.wheat2000 import load_wheat2000
from genopred.evaluation.metrics import pcc, rmse, mae

TIE_TOLERANCE = 1e-12
_AMP = False


# -- DeepGS (official structure, PyTorch adaptation) ------------------------
class DeepGS(nn.Module):
    """Official cma2015/DeepGS network: Conv(1x18, 8) ReLU MaxPool(1x4),
    Dropout, FC(32) Sigmoid, Dropout, FC(1), Dropout, linear output."""

    def __init__(self, n_markers, conv_channels=8, kernel_size=18, pool_size=4,
                 hidden=32, dropouts=(0.2, 0.1, 0.05)):
        super().__init__()
        self.n_markers = int(n_markers)
        self.pooled = (self.n_markers - kernel_size + 1 - pool_size) // pool_size + 1
        if self.pooled < 1:
            raise ValueError('Marker length is shorter than one pooled window')
        self.conv = nn.Conv1d(1, conv_channels, kernel_size, stride=1)
        self.pool = nn.MaxPool1d(pool_size, stride=pool_size)
        self.drop_input = nn.Dropout(float(dropouts[0]))
        self.fc1 = nn.Linear(conv_channels * self.pooled, hidden)
        self.drop_hidden = nn.Dropout(float(dropouts[1]))
        self.fc2 = nn.Linear(hidden, 1)
        self.drop_output = nn.Dropout(float(dropouts[2]))

    def forward(self, x):
        z = self.pool(torch.relu(self.conv(x.unsqueeze(1))))
        z = self.drop_input(z).flatten(1)
        z = torch.sigmoid(self.fc1(z))
        z = self.fc2(self.drop_hidden(z))
        return self.drop_output(z).squeeze(1)


def _sgd_epoch(model, optimizer, x, target, generator, batch_size):
    model.train()
    order = torch.randperm(len(x), generator=generator).to(x.device)
    for batch in order.split(batch_size):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=x.device.type, dtype=torch.float16,
                            enabled=x.device.type == 'cuda' and _AMP):
            loss = nn.functional.mse_loss(model(x[batch]).float(), target[batch])
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite DeepGS loss')
        loss.backward()
        optimizer.step()


@torch.no_grad()
def _torch_predict(model, x, batch_size=256):
    model.eval()
    out = torch.cat([model(x[i:i + batch_size]) for i in range(0, len(x), batch_size)])
    out = out.float().cpu().numpy().astype(np.float64)
    if not np.isfinite(out).all():
        raise FloatingPointError('Nonfinite prediction')
    return out


def _deepgs_tensors(X, y, fit, query, device):
    y_fit = np.asarray(y[fit], dtype=np.float64)
    y_mean, y_std = float(y_fit.mean()), float(y_fit.std())
    if y_std <= 0:
        raise ValueError('Fit phenotype standard deviation must be positive')
    x_fit = torch.as_tensor(X[fit], dtype=torch.float32, device=device)
    x_query = torch.as_tensor(X[query], dtype=torch.float32, device=device)
    target = torch.as_tensor((y_fit - y_mean) / y_std, dtype=torch.float32, device=device)
    return x_fit, x_query, target, y_mean, y_std


def _deepgs_model(n_markers, spec, seed, training, device):
    torch.manual_seed(int(seed))
    model = DeepGS(n_markers, **spec['architecture'])
    scale = float(training['initialization_scale'])
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                nn.init.uniform_(module.weight, -scale, scale)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    return model.to(device)


def _deepgs_optimizer(model, spec):
    opt = spec['optimizer']
    return torch.optim.SGD(model.parameters(), lr=float(opt['lr']),
                           momentum=float(opt['momentum']),
                           weight_decay=float(opt['weight_decay']))


def deepgs_inner(X, y, fit, val, spec, seed, cfg, device):
    training = cfg['training']
    x_fit, x_val, target, y_mean, y_std = _deepgs_tensors(X, y, fit, val, device)
    model = _deepgs_model(X.shape[1], spec, seed, training, device)
    optimizer = _deepgs_optimizer(model, spec)
    generator = torch.Generator().manual_seed(int(seed))
    best_pcc, best_epoch, best_pred, lrs = -np.inf, 0, None, []
    start = time.perf_counter()
    for epoch in range(1, int(training['max_epochs']) + 1):
        lrs.append(float(optimizer.param_groups[0]['lr']))
        _sgd_epoch(model, optimizer, x_fit, target, generator, int(training['batch_size']))
        pred = _torch_predict(model, x_val) * y_std + y_mean
        score = pcc(y[val], pred)
        if not np.isfinite(score):
            raise ValueError('Nonfinite DeepGS validation PCC')
        if score > best_pcc + TIE_TOLERANCE:
            best_pcc, best_epoch, best_pred = score, epoch, pred
        if epoch - best_epoch >= int(training['early_stopping_patience']):
            break
    return {'prediction': best_pred, 'best_epoch': best_epoch, 'lr_sequence': lrs,
            'best_pcc': best_pcc, 'seconds': time.perf_counter() - start}


def deepgs_refit(X, y, dev, test, spec, seed, epochs, lr_sequence, cfg, device):
    training = cfg['training']
    x_dev, x_test, target, y_mean, y_std = _deepgs_tensors(X, y, dev, test, device)
    model = _deepgs_model(X.shape[1], spec, seed, training, device)
    optimizer = _deepgs_optimizer(model, spec)
    generator = torch.Generator().manual_seed(int(seed))
    for epoch in range(int(epochs)):
        for group in optimizer.param_groups:
            group['lr'] = float(lr_sequence[epoch])
        _sgd_epoch(model, optimizer, x_dev, target, generator, int(training['batch_size']))
    return _torch_predict(model, x_test) * y_std + y_mean


# -- SoyDNGP (official SoyDNGPNext source) ----------------------------------
def _load_upstream_module(checkout, name):
    """Import a SoyDNGPNext module without the package __init__ (cupy/matplotlib)."""
    full = f'soydngpnext.{name}'
    if full in sys.modules:
        return sys.modules[full]
    if 'soydngpnext' not in sys.modules:
        pkg = types.ModuleType('soydngpnext')
        pkg.__path__ = [str(Path(checkout) / 'soydngpnext')]
        sys.modules['soydngpnext'] = pkg
    spec = importlib.util.spec_from_file_location(full, Path(checkout) / 'soydngpnext' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[full] = module
    spec.loader.exec_module(module)
    return module


def soydngp_encode(X, checkout):
    """Official one-hot + np.resize encoding; DArT 0/1 -> VCF codes 3 (0/0) / 1 (1/1)."""
    reader = _load_upstream_module(checkout, 'reader_cpu')
    codes = np.where(np.asarray(X) > 0, 1, 3).astype(np.int32)
    return reader.one_hot_CPU(codes)


def _soydngp_model(checkout, seed, device):
    remodel = _load_upstream_module(checkout, 'remodel').remodel
    torch.manual_seed(int(seed))
    net, _ = remodel(str(Path(checkout) / 'soydngpnext' / 'data' / 'model.yaml'), 1,
                     show_structure=False)
    for module in net.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(module.weight)
    return net.to(device)


def _soy_epoch(model, optimizer, loss_fn, x, target, generator, batch_size, device):
    model.train()
    order = torch.randperm(len(x), generator=generator)
    for batch in order.split(batch_size):
        optimizer.zero_grad(set_to_none=True)
        xb = torch.as_tensor(x[batch.numpy()], dtype=torch.float32, device=device)
        # SoyDNGP stays FP32 even when amp is on: the official SmoothL1/Adam
        # recipe diverges under fp16 autocast on this network (verified
        # 2026-11-15); AMP remains enabled for DeepGS only.
        loss = loss_fn(model(xb).reshape(-1), target[batch.to(device)])
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite SoyDNGP loss')
        loss.backward()
        optimizer.step()


def _soy_predict(model, x, device, batch_size=32):
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.as_tensor(x[start:start + batch_size], dtype=torch.float32, device=device)
            chunks.append(model(xb).reshape(-1).float().cpu().numpy())
    out = np.concatenate(chunks).astype(np.float64)
    if not np.isfinite(out).all():
        raise FloatingPointError('Nonfinite SoyDNGP prediction')
    return out


def _soy_tensors(encoded, y, fit, query, device):
    y_fit = np.asarray(y[fit], dtype=np.float64)
    lo, hi = float(y_fit.min()), float(y_fit.max())
    if hi <= lo:
        raise ValueError('Fit phenotype range must be positive')
    target = torch.as_tensor((y_fit - lo) / (hi - lo), dtype=torch.float32, device=device)
    return encoded[fit], encoded[query], target, lo, hi


def soydngp_inner(encoded, y, fit, val, spec, seed, cfg, device):
    training = cfg['training']
    x_fit, x_val, target, lo, hi = _soy_tensors(encoded, y, fit, val, device)
    model = _soydngp_model(cfg['upstream']['path'], seed, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(spec['lr']),
                                 weight_decay=float(spec['weight_decay']))
    loss_fn = nn.SmoothL1Loss()
    generator = torch.Generator().manual_seed(int(seed))
    best_pcc, best_epoch, best_pred, lrs = -np.inf, 0, None, []
    start = time.perf_counter()
    for epoch in range(1, int(training['max_epochs']) + 1):
        lrs.append(float(optimizer.param_groups[0]['lr']))
        _soy_epoch(model, optimizer, loss_fn, x_fit, target, generator,
                   int(training['batch_size']), device)
        pred = _soy_predict(model, x_val, device) * (hi - lo) + lo
        score = pcc(y[val], pred)
        if not np.isfinite(score):
            raise ValueError('Nonfinite SoyDNGP validation PCC')
        if score > best_pcc + TIE_TOLERANCE:
            best_pcc, best_epoch, best_pred = score, epoch, pred
        if epoch - best_epoch >= int(training['early_stopping_patience']):
            break
    return {'prediction': best_pred, 'best_epoch': best_epoch, 'lr_sequence': lrs,
            'best_pcc': best_pcc, 'seconds': time.perf_counter() - start}


def soydngp_refit(encoded, y, dev, test, spec, seed, epochs, lr_sequence, cfg, device):
    training = cfg['training']
    x_dev, x_test, target, lo, hi = _soy_tensors(encoded, y, dev, test, device)
    model = _soydngp_model(cfg['upstream']['path'], seed, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(spec['lr']),
                                 weight_decay=float(spec['weight_decay']))
    loss_fn = nn.SmoothL1Loss()
    generator = torch.Generator().manual_seed(int(seed))
    for epoch in range(int(epochs)):
        for group in optimizer.param_groups:
            group['lr'] = float(lr_sequence[epoch])
        _soy_epoch(model, optimizer, loss_fn, x_dev, target, generator,
                   int(training['batch_size']), device)
    return _soy_predict(model, x_test, device) * (hi - lo) + lo


# -- DNNGP (official compiled trainer in a pinned environment) --------------
class DnngpClient:
    """Persistent worker process running the official dnngp module."""

    def __init__(self, cfg, scratch):
        self.scratch = Path(scratch)
        self.scratch.mkdir(parents=True, exist_ok=True)
        worker = Path(__file__).with_name('dnngp_worker.py')
        self.proc = subprocess.Popen(
            [cfg['upstream']['python'], str(worker), cfg['upstream']['path'], str(self.scratch)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=(self.scratch / 'worker.log').open('w'), text=True)
        self.counter = 0
        hello = self._read()
        if hello.get('status') != 'ready':
            raise RuntimeError(f'DNNGP worker failed to start: {hello}')
        self.tf_version = hello['tensorflow']

    def _read(self):
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError('DNNGP worker exited unexpectedly')
        return json.loads(line)

    def submit(self, task):
        self.counter += 1
        tag = f'task{self.counter}'
        arrays = task.pop('arrays')
        np.savez(self.scratch / f'{tag}.npz', **arrays)
        task.update(tag=tag, input=str(self.scratch / f'{tag}.npz'))
        self.proc.stdin.write(json.dumps(task) + '\n')
        self.proc.stdin.flush()
        reply = self._read()
        if reply.get('status') != 'ok':
            raise RuntimeError(f"DNNGP fit failed: {reply.get('error')}")
        prediction = np.load(self.scratch / f'{tag}_pred.npy')
        for name in (f'{tag}.npz', f'{tag}_pred.npy'):
            (self.scratch / name).unlink()
        reply['prediction'] = prediction
        return reply

    def close(self):
        if self.proc.poll() is None:
            self.proc.stdin.write('{"action": "shutdown"}\n')
            self.proc.stdin.flush()
            self.proc.wait(timeout=60)


def _dnngp_pcs(X, fit, query, variance):
    from sklearn.decomposition import PCA
    centered = X - X[fit].mean(axis=0)
    full = PCA(svd_solver='full').fit(centered[fit])
    cumulative = np.cumsum(full.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumulative, variance) + 1)
    scores = full.transform(centered[np.r_[fit, query]])[:, :n_components]
    return scores[:len(fit)], scores[len(fit):], n_components


def dnngp_inner(client, X, y, fit, val, spec, seed, cfg, device):
    pc_fit, pc_val, n_pc = _dnngp_pcs(X, fit, val, cfg['pca_variance'])
    start = time.perf_counter()
    reply = client.submit({
        'action': 'fit', 'seed': int(seed), 'n_features': n_pc,
        'epoch': int(cfg['training']['max_epochs']),
        'patience': int(spec['patience']), 'batch_size': int(spec['batch_size']),
        'lr': float(spec['lr']), 'dropout1': float(spec['dropout1']),
        'dropout2': float(spec['dropout2']),
        'earlystopping': int(cfg['training']['early_stopping']),
        'arrays': {'x_fit': pc_fit, 'x_val': pc_val,
                   'y_fit': np.asarray(y[fit], dtype=np.float64),
                   'y_val': np.asarray(y[val], dtype=np.float64)}})
    reply['seconds'] = time.perf_counter() - start
    reply['n_pc'] = n_pc
    reply['best_pcc'] = pcc(y[val], reply['prediction'])
    if not np.isfinite(reply['best_pcc']):
        raise ValueError('Nonfinite DNNGP validation PCC')
    return reply


def dnngp_refit(client, X, y, dev, test, spec, seed, epochs, lr_sequence, cfg, device):
    pc_dev, pc_test, n_pc = _dnngp_pcs(X, dev, test, cfg['pca_variance'])
    reply = client.submit({
        'action': 'refit', 'seed': int(seed), 'n_features': n_pc,
        'epochs': int(epochs), 'lr_sequence': [float(v) for v in lr_sequence],
        'patience': int(spec['patience']), 'batch_size': int(spec['batch_size']),
        'lr': float(spec['lr']), 'dropout1': float(spec['dropout1']),
        'dropout2': float(spec['dropout2']),
        'arrays': {'x_fit': pc_dev, 'x_test': pc_test,
                   'y_fit': np.asarray(y[dev], dtype=np.float64)}})
    return reply['prediction']


# -- shared nested driver ----------------------------------------------------
def frozen_lr(sequences, epochs):
    """Per-epoch median of three inner learning-rate sequences; pad short runs."""
    padded = []
    for sequence in sequences:
        values = [float(v) for v in sequence]
        if not values or not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError('Learning-rate sequence must be finite and positive')
        if len(values) < epochs:
            values = values + [values[-1]] * (epochs - len(values))
        padded.append(values[:epochs])
    return [float(np.median([values[i] for values in padded])) for i in range(epochs)]


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def write_csv(path, fields, rows):
    import csv
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def _validate(cfg):
    if cfg['model'] not in ('deepgs', 'dnngp', 'soydngp'):
        raise ValueError('model must be deepgs, dnngp or soydngp')
    seeds = [int(s) for s in cfg['seeds']]
    if len(seeds) != 2 or len(set(seeds)) != 2:
        raise ValueError('Protocol requires two distinct seeds')
    if not cfg['configurations']:
        raise ValueError('configurations must be non-empty')
    if cfg['model'] in ('dnngp', 'soydngp'):
        checkout = Path(cfg['upstream']['path'])
        head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=checkout, text=True).strip()
        if head != cfg['upstream']['commit']:
            raise ValueError(f'Upstream checkout {checkout} is at {head}, expected {cfg["upstream"]["commit"]}')


def _upstream_files(cfg):
    if cfg['model'] == 'dnngp':
        base = Path(cfg['upstream']['path']) / 'Scripts'
        names = ['dnngp.cpython-39-x86_64-linux-gnu.so', 'config_dnngp.cpython-39-x86_64-linux-gnu.so',
                 'dnngp_runner.py']
    elif cfg['model'] == 'soydngp':
        base = Path(cfg['upstream']['path']) / 'soydngpnext'
        names = ['SoyDNGP.py', 'remodel.py', 'reader_cpu.py', 'train.py', 'data/model.yaml']
    else:
        return {}
    return {str(base / name): sha256_of(base / name) for name in names}


def run(cfg, args, runner):
    global _AMP
    _validate(cfg)
    _AMP = bool(cfg.get('amp', False))
    torch.set_num_threads(cfg['cpu_threads'])
    device = torch.device(cfg['device'])
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(cfg['gpu_memory_fraction'])
    if 'cache_sha256' in cfg:
        d = load_wheat2000(Path(cfg['data']), cfg['data_sha256'],
                           Path(cfg['cache']), cfg['cache_sha256'])
        data = SimpleNamespace(X=d.X, Y=d.Y, n=int(d.X.shape[0]),
                               n_env=int(d.Y.shape[1]), environments=tuple(d.trait_names))
    else:
        data = load(Path(cfg['data']), cfg['data_sha256'])
    folds = json.loads(Path(cfg['splits']).read_text())['folds']
    split_hash = sha256_of(Path(cfg['splits']))
    if len(folds) != 10 or any(len(f['inner']) != 3 for f in folds):
        raise ValueError('Expected ten outer folds with three inner splits')
    configs = cfg['configurations']
    seeds = [int(s) for s in cfg['seeds']]
    frozen = json.loads(args.selection.read_text()) if args.selection else None
    if frozen is not None:
        # Execution knobs may change between development and refit (e.g. a GPU
        # resume); the scientific configuration must stay identical.
        execution_keys = ('workers', 'cpu_threads', 'device', 'gpu_memory_fraction', 'amp', 'resume')
        strip = lambda c: {k: v for k, v in c.items() if k not in execution_keys}
        if strip(frozen['config']) != strip(cfg) or frozen['splits_sha256'] != split_hash:
            raise ValueError('Frozen selection does not match configuration or splits')
    args.out.mkdir(parents=True, exist_ok=cfg.get('resume', False))
    encoded = soydngp_encode(data.X, cfg['upstream']['path']) if cfg['model'] == 'soydngp' else None
    workers = int(cfg.get('workers', 1))
    # In parallel mode each fit worker owns its DNNGP client; the main process
    # only needs one for the serial path and the final refit phase.
    client = DnngpClient(cfg, args.out / 'scratch') \
        if cfg['model'] == 'dnngp' else None
    sources = [Path(__file__), Path(args.config), Path(runner), Path(cfg['splits'])]
    if cfg['model'] == 'dnngp':
        sources.append(Path(__file__).with_name('dnngp_worker.py'))
    provenance = {
        'config': cfg, 'splits_sha256': split_hash,
        'selection_sha256': sha256_of(args.selection) if args.selection else None,
        'sha256': {str(p): sha256_of(p) for p in sources},
        'upstream_files': _upstream_files(cfg),
        'torch_version': str(torch.__version__),
        'dnngp_tensorflow': client.tf_version if client else None,
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'git_status_porcelain_tracked': subprocess.check_output(
            ['git', 'status', '--porcelain', '--untracked-files=no'], text=True).strip(),
        'workers': workers, 'amp': _AMP,
        'rule': 'mean inner PCC of seed-averaged predictions; first configuration on tie; '
                'per-seed median best epoch and per-epoch median learning rate; full dev refit; '
                'outer test scored once',
        'adaptation': cfg.get('adaptation_notes', []),
    }
    write_json(args.out / 'provenance.json', provenance)
    records, failures = [], []
    start = time.perf_counter()
    try:
        if frozen is None:
            _development(cfg, args, data, folds, configs, seeds, device, encoded, client, records)
        else:
            _final(cfg, args, data, folds, configs, seeds, device, encoded, client, frozen, records)
    except Exception as exc:
        failures.append({'stage': 'selection' if frozen is None else 'final', 'error': repr(exc)})
        raise
    finally:
        if client is not None:
            client.close()
        write_json(args.out / 'failures.json', failures)
        write_json(args.out / 'timing.json', {'seconds': time.perf_counter() - start})


def _inner_fit(cfg, data, encoded, client, fit, val, env, spec, seed, device):
    y = data.Y[:, env]
    if cfg['model'] == 'deepgs':
        return deepgs_inner(data.X, y, fit, val, spec, seed, cfg, device)
    if cfg['model'] == 'soydngp':
        return soydngp_inner(encoded, y, fit, val, spec, seed, cfg, device)
    return dnngp_inner(client, data.X, y, fit, val, spec, seed, cfg, device)


_WORKER = None


def _init_worker(cfg, X, Y, encoded):
    global _WORKER, _AMP
    import atexit
    torch.set_num_threads(cfg['cpu_threads'])
    _AMP = bool(cfg.get('amp', False))
    device = torch.device(cfg['device'])
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(
            cfg['gpu_memory_fraction'] / cfg.get('workers', 1))
    data = SimpleNamespace(X=X, Y=Y, n=X.shape[0], n_env=Y.shape[1])
    client = DnngpClient(cfg, Path(cfg['_scratch']) / f'worker{os.getpid()}') \
        if cfg['model'] == 'dnngp' else None
    if client is not None:
        atexit.register(client.close)
    _WORKER = (cfg, data, encoded, client, device)


def _worker_fit(task):
    inner, env, c, seed, fit, val = task
    cfg, data, encoded, client, device = _WORKER
    result = _inner_fit(cfg, data, encoded, client, fit, val, env,
                        cfg['configurations'][c], seed, device)
    return inner, env, c, seed, result




def _refit(cfg, data, encoded, client, dev, test, env, spec, seed, epochs, lr_sequence, device):
    y = data.Y[:, env]
    if cfg['model'] == 'deepgs':
        return deepgs_refit(data.X, y, dev, test, spec, seed, epochs, lr_sequence, cfg, device)
    if cfg['model'] == 'soydngp':
        return soydngp_refit(encoded, y, dev, test, spec, seed, epochs, lr_sequence, cfg, device)
    return dnngp_refit(client, data.X, y, dev, test, spec, seed, epochs, lr_sequence, cfg, device)


def _development(cfg, args, data, folds, configs, seeds, device, encoded, client, records):
    selected = []
    workers = int(cfg.get('workers', 1))
    resume = bool(cfg.get('resume', False))
    done = {}
    if resume:
        traj_path = args.out / 'trajectories.json'
        if traj_path.is_file():
            records.extend(json.loads(traj_path.read_text()))
        for fold in folds:
            for inner in range(3):
                path = args.out / f'predictions_{fold["fold"]}_{inner}.npz'
                if path.is_file():
                    with np.load(path) as archive:
                        for key in archive.files:
                            done[(fold['fold'], inner, key)] = archive[key]
        print(f'resume: {len(done)} cached prediction arrays, {len(records)} trajectories', flush=True)
    pool = None
    if workers > 1:
        pool = ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
            initializer=_init_worker,
            initargs=({**cfg, '_scratch': str(args.out / 'scratch')}, data.X, data.Y, encoded))
    for fold in folds:
        f = fold['fold']
        dev, test = np.asarray(fold['dev_idx']), np.asarray(fold['test_idx'])
        if set(dev) & set(test) or len(set(dev) | set(test)) != data.n:
            raise ValueError('Invalid outer split')
        predictions = {}
        for inner, split in enumerate(fold['inner']):
            fit, val = np.asarray(split['train_idx']), np.asarray(split['val_idx'])
            if set(fit) & set(val) or set(fit) | set(val) != set(dev):
                raise ValueError('Invalid inner split')
            tasks = [(inner, env, c, seed, fit, val)
                     for env in range(data.n_env) for c in range(len(configs)) for seed in seeds]
            if resume:
                predictions = {key: done[(f, inner, key)] for key in
                               (f'e{e}_c{c}_s{s}' for e in range(data.n_env)
                                for c in range(len(configs)) for s in seeds)
                               if (f, inner, key) in done}
                tasks = [t for t in tasks if f'e{t[1]}_c{t[2]}_s{t[3]}' not in predictions]
                if not tasks:
                    continue
            if pool is None:
                outputs = ((_i, _e, _c, _s,
                            _inner_fit(cfg, data, encoded, client, _fit, _val, _e,
                                       configs[_c], _s, device))
                           for _i, _e, _c, _s, _fit, _val in tasks)
            else:
                outputs = pool.map(_worker_fit, tasks, chunksize=1)
            for inner_i, env, c, seed, result in outputs:
                key = f'e{env}_c{c}_s{seed}'
                predictions[key] = result['prediction']
                records.append({'fold': f, 'inner': inner_i, 'environment': env,
                                'config_index': c, 'seed': seed,
                                'best_epoch': int(result['best_epoch']),
                                'best_pcc': float(result['best_pcc']),
                                'lr_sequence': result['lr_sequence'],
                                'seconds': result['seconds'],
                                **({'n_pc': result['n_pc']} if 'n_pc' in result else {})})
                print(f'fold {f} inner {inner_i} env {env} config {c} seed {seed}: '
                      f'pcc={result["best_pcc"]:.4f} epoch={result["best_epoch"]} '
                      f'({result["seconds"]:.1f}s)', flush=True)
            np.savez_compressed(args.out / f'predictions_{f}_{inner}.npz', **predictions)
            write_json(args.out / 'trajectories.json', records)
            predictions = {}
        arrays = [np.load(args.out / f'predictions_{f}_{j}.npz') for j in range(3)]
        for env in range(data.n_env):
            scores = [np.mean([pcc(data.Y[split['val_idx'], env],
                                   np.mean([arrays[j][f'e{env}_c{c}_s{seed}'] for seed in seeds], axis=0))
                               for j, split in enumerate(fold['inner'])])
                      for c in range(len(configs))]
            if not np.isfinite(scores).all():
                raise ValueError('Nonfinite configuration selection score')
            winner = int(np.argmax(scores))
            seed_blocks = {}
            for seed in seeds:
                rows = [r for r in records if r['fold'] == f and r['environment'] == env
                        and r['config_index'] == winner and r['seed'] == seed]
                if len(rows) != 3:
                    raise ValueError('Expected three inner trajectories per seed')
                best_epochs = sorted(int(r['best_epoch']) for r in rows)
                median_epoch = best_epochs[1]
                seed_blocks[str(seed)] = {
                    'per_inner_best_epochs': [int(r['best_epoch']) for r in rows],
                    'median_epoch': median_epoch,
                    'lr_sequence': frozen_lr([r['lr_sequence'] for r in rows], median_epoch)}
            selected.append({'fold': f, 'environment': env, 'config_index': winner,
                             'inner_pcc': float(scores[winner]),
                             'all_config_inner_pcc': [float(s) for s in scores],
                             'seeds': seed_blocks})
            write_json(args.out / 'selection.json',
                       {'config': cfg, 'splits_sha256': sha256_of(Path(cfg['splits'])),
                        'selected': selected})
        print(f'fold {f}: selection complete', flush=True)
    if pool is not None:
        pool.shutdown(wait=True)
    if len(selected) != len(folds) * data.n_env:
        raise RuntimeError('Incomplete development selection')
    print('Development selection complete; no test predictions made.', flush=True)


def _final(cfg, args, data, folds, configs, seeds, device, encoded, client, frozen, records):
    expected = {(f['fold'], e) for f in folds for e in range(data.n_env)}
    keys = [(v['fold'], v['environment']) for v in frozen['selected']]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError('Frozen selection is incomplete or duplicated')
    rows, seed_rows, oof, seed_oof = [], [], [], []
    for fold in folds:
        f = fold['fold']
        dev, test = np.asarray(fold['dev_idx']), np.asarray(fold['test_idx'])
        for env in range(data.n_env):
            entry = next(v for v in frozen['selected'] if v['fold'] == f and v['environment'] == env)
            spec = configs[entry['config_index']]
            preds = {}
            for seed in seeds:
                block = entry['seeds'][str(seed)]
                pred = _refit(cfg, data, encoded, client, dev, test, env, spec, seed,
                              block['median_epoch'], block['lr_sequence'], device)
                if pred.shape != (len(test),) or not np.isfinite(pred).all():
                    raise ValueError('Invalid refit prediction')
                preds[seed] = pred
                records.append({'fold': f, 'environment': env, 'seed': seed,
                                'config_index': entry['config_index'],
                                'epochs': block['median_epoch']})
            np.savez_compressed(args.out / f'test_predictions_{f}_{env}.npz',
                                **{f's{seed}': preds[seed] for seed in seeds})
            truth = data.Y[test, env]
            mean_pred = np.mean([preds[seed] for seed in seeds], axis=0)
            metrics = {'pcc': pcc(truth, mean_pred), 'rmse': rmse(truth, mean_pred),
                       'mae': mae(truth, mean_pred)}
            rows.append({'fold': f, 'environment': data.environments[env], **metrics,
                         'config_index': entry['config_index'], 'inner_pcc': entry['inner_pcc']})
            for seed in seeds:
                seed_rows.append({'fold': f, 'environment': data.environments[env], 'seed': seed,
                                  'pcc': pcc(truth, preds[seed]), 'rmse': rmse(truth, preds[seed]),
                                  'mae': mae(truth, preds[seed])})
                seed_oof.extend({'fold': f, 'environment': data.environments[env], 'seed': seed,
                                 'material_index': int(i), 'y_true': float(t), 'y_pred': float(p)}
                                for i, t, p in zip(test, truth, preds[seed]))
            oof.extend({'fold': f, 'environment': data.environments[env],
                        'material_index': int(i), 'y_true': float(t), 'y_pred': float(p)}
                       for i, t, p in zip(test, truth, mean_pred))
            print(f'fold {f} env {data.environments[env]}: pcc={metrics["pcc"]:.4f}', flush=True)
        write_csv(args.out / 'per_fold.csv',
                  ['fold', 'environment', 'pcc', 'rmse', 'mae', 'config_index', 'inner_pcc'], rows)
        write_csv(args.out / 'per_fold_seed.csv',
                  ['fold', 'environment', 'seed', 'pcc', 'rmse', 'mae'], seed_rows)
        write_csv(args.out / 'oof.csv',
                  ['fold', 'environment', 'material_index', 'y_true', 'y_pred'], oof)
        write_csv(args.out / 'oof_seed.csv',
                  ['fold', 'environment', 'seed', 'material_index', 'y_true', 'y_pred'], seed_oof)
        write_json(args.out / 'refit_records.json', records)
    if len(rows) != len(folds) * data.n_env or len(oof) != data.n_env * data.n:
        raise RuntimeError('Incomplete outer evaluation')
    summary = {'n_runs': len(rows), 'n_failures': 0, 'per_environment': {}, 'pooled': {}}
    for env in data.environments:
        env_rows = [r for r in rows if r['environment'] == env]
        summary['per_environment'][env] = {
            m: {'mean': float(np.mean([r[m] for r in env_rows])),
                'sd': float(np.std([r[m] for r in env_rows], ddof=1))}
            for m in ('pcc', 'rmse', 'mae')}
        pooled = [r for r in oof if r['environment'] == env]
        summary['pooled'][env] = {
            'pcc': pcc([r['y_true'] for r in pooled], [r['y_pred'] for r in pooled]),
            'rmse': rmse([r['y_true'] for r in pooled], [r['y_pred'] for r in pooled]),
            'mae': mae([r['y_true'] for r in pooled], [r['y_pred'] for r in pooled])}
    summary['macro'] = {m: float(np.mean([summary['per_environment'][e][m]['mean']
                                          for e in data.environments]))
                        for m in ('pcc', 'rmse', 'mae')}
    write_json(args.out / 'summary.json', summary)
    print(json.dumps(summary['macro']), flush=True)
