"""MeNet relation features and bidirectional fusion with full-marker quantile TabM.

Copyright, GANLAB: upstream modules are loaded unchanged from a pinned local
academic-use checkout; they are not redistributed with this project.
"""
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from wheat599 import quantile_tabm
from wheat599.data import sha256_of
from wheat599.metrics import pcc


def official_modules(settings):
    root = Path(settings['upstream_path'])
    modules = []
    for relative, expected in settings['upstream_sha256'].items():
        if sha256_of(root / relative) != expected:
            raise ValueError(f'MeNet source hash mismatch: {relative}')
    for name in ('contrastive_learning', 'menet'):
        spec = importlib.util.spec_from_file_location(f'menet_official_{name}', root / 'network' / f'{name}.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules.append(module)
    return modules


def triplet_loss(anchor, a_y, s, s_y, t, t_y, margin):
    """Author's softplus triplet objective, with exactly one margin per triplet."""
    a_y, s_y, t_y = (y.reshape(-1) for y in (a_y, s_y, t_y))
    near_s = (a_y - s_y).abs() <= (a_y - t_y).abs()
    positive = torch.where(near_s[:, None], s, t)
    negative = torch.where(near_s[:, None], t, s)
    near = torch.where(near_s, s_y, t_y)
    far = torch.where(near_s, t_y, s_y)
    adaptive = (margin * (a_y - far).abs() / ((a_y - near).abs() + .01)).clamp(.1, .9)
    a, pos, neg = (F.normalize(v, dim=1) for v in (anchor, positive, negative))
    d_pos, d_neg = F.pairwise_distance(a, pos), F.pairwise_distance(a, neg)
    if adaptive.shape != d_pos.shape or d_neg.shape != d_pos.shape:
        raise ValueError('Triplet distances and margins must be one-dimensional')
    return F.softplus(d_pos - d_neg + adaptive).mean()


def sample_triplets(n, generator):
    if n < 2:
        raise ValueError('Triplets require at least two materials')
    anchor = torch.arange(n)
    # The official ungrouped sampler excludes anchor, but permits s == t.
    s = (anchor + torch.randint(1, n, (n,), generator=generator)) % n
    t = (anchor + torch.randint(1, n, (n,), generator=generator)) % n
    return anchor, s, t


def state_sha(model):
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        value = value.detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class RelationMap(nn.Module):
    def __init__(self, encoder, x_fit):
        super().__init__()
        self.encoder = encoder.eval().requires_grad_(False)
        with torch.no_grad():
            reference = F.normalize(encoder.forward_once(x_fit[:, None, :]), dim=1)
            scale = torch.cdist(reference, reference).max()
        if not torch.isfinite(reference).all() or not torch.isfinite(scale) or scale <= 0:
            raise ValueError('Degenerate fitting-reference embedding')
        self.register_buffer('reference', reference)
        self.register_buffer('distance_scale', scale)

    @torch.no_grad()
    def forward(self, x):
        self.encoder.eval()
        embedding = F.normalize(self.encoder.forward_once(x[:, None, :]), dim=1)
        # Queries never supply reference columns or alter distance scaling.
        return 1 - torch.cdist(embedding, self.reference) / self.distance_scale


def train_relation(x_fit, y_fit, x_val, y_val, seed, settings, device, module):
    start = time.perf_counter()
    spec = settings['relation_training']
    torch.manual_seed(int(seed))
    encoder = module.TraitSpecificEncoderForRepGeno(x_fit.shape[1], spec['stride'], spec['out_dim']).to(device)
    x = torch.as_tensor(x_fit, dtype=torch.float32, device=device)
    xv = torch.as_tensor(x_val, dtype=torch.float32, device=device)
    y = torch.as_tensor(y_fit, dtype=torch.float32, device=device)
    yv = torch.as_tensor(y_val, dtype=torch.float32, device=device)
    mean, scale = y.mean(), y.std(unbiased=False).clamp_min(1e-12)
    y, yv = (y - mean) / scale, (yv - mean) / scale
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=spec['learning_rate'], weight_decay=spec['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    generator = torch.Generator().manual_seed(int(seed) + 700000)
    validation_generator = torch.Generator().manual_seed(int(seed) + 800000)
    validation = tuple(v.to(device) for v in sample_triplets(len(xv), validation_generator))
    best_loss, best_state, best_epoch, history = np.inf, None, 0, []
    for epoch in range(1, spec['epochs'] + 1):
        encoder.train()
        triplets = tuple(v.to(device) for v in sample_triplets(len(x), generator))
        order = torch.randperm(len(x), generator=generator).to(device)
        training_loss, seen = 0., 0
        for batch in order.split(spec['batch_size']):
            if len(batch) < 2:  # Official BatchNorm requires at least two examples.
                continue
            a, s, t = (v[batch] for v in triplets)
            optimizer.zero_grad(set_to_none=True)
            ae, se, te = encoder(x[a, None, :], x[s, None, :], x[t, None, :])
            loss = triplet_loss(ae, y[a], se, y[s], te, y[t], spec['margin'])
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite relation loss')
            loss.backward()
            optimizer.step()
            training_loss += loss.item() * len(batch)
            seen += len(batch)
        encoder.eval()
        with torch.no_grad():
            a, s, t = validation
            ae, se, te = encoder(xv[a, None, :], xv[s, None, :], xv[t, None, :])
            val_loss = triplet_loss(ae, yv[a], se, yv[s], te, yv[t], spec['margin']).item()
        if not np.isfinite(val_loss):
            raise FloatingPointError('Nonfinite relation validation loss')
        scheduler.step(val_loss)
        history.append({'epoch': epoch, 'train_loss': training_loss / seen, 'val_loss': val_loss})
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()}
    encoder.load_state_dict(best_state)
    relation = RelationMap(encoder, x)
    fit_r, val_r = relation(x), relation(xv)
    details = dict(best_epoch=best_epoch, best_val_loss=best_loss, history=history,
                   seconds=time.perf_counter() - start, state_sha256=state_sha(relation),
                   reference_count=len(x), distance_scale=relation.distance_scale.item())
    return relation, fit_r, val_r, details


class DualQuantile(nn.Module):
    def __init__(self, marker, references, width, module):
        super().__init__()
        self.marker = marker
        self.markers = marker.markers
        self.head = marker.backbone.output
        marker.backbone.output = None
        self.relation = module.RepGeno(references, {'embedding_dim': [4 * width, width], 'dropout': .6})
        self.marker_update = module.CrossInformationFeatureFusion(width)
        self.relation_update = module.CrossInformationFeatureFusion(width)
        self.readout = nn.Sequential(nn.Linear(2 * width, width), nn.BatchNorm1d(width),
                                     nn.LeakyReLU(.05), nn.Dropout(.6))

    def forward(self, inputs):
        hv = self.marker(inputs[:, :self.markers])
        hr = self.relation(inputs[:, self.markers:])[:, None, :].expand_as(hv)
        shape = hv.shape
        v, r = hv.reshape(-1, shape[-1]), hr.reshape(-1, shape[-1])
        joint = torch.cat((v, r), dim=-1)
        # Both residual updates see the same pre-update pair, as in MeNet.
        v_new = v + self.marker_update(joint)
        r_new = r + self.relation_update(joint)
        hidden = self.readout(torch.cat((v_new, r_new), dim=-1)).reshape(shape)
        return self.head(hidden)

    def standardize_y(self, y):
        return self.marker.standardize_y(y)

    def inverse_y(self, y):
        return self.marker.inverse_y(y)


class Trainer:
    def __init__(self, cfg, out):
        self.encoder_module, self.fusion_module = official_modules(cfg)
        self.out = Path(out)
        self.scope, self.cache, self.pretraining = None, {}, []

    def __call__(self, x_fit, y_fit, x_val, y_val, config, seed, index, settings, device):
        start = time.perf_counter()
        scope = hashlib.sha256(b''.join(np.ascontiguousarray(a).tobytes() for a in (x_fit, y_fit, x_val, y_val))).hexdigest()
        if scope != self.scope:
            self.scope, self.cache = scope, {}
        if seed not in self.cache:
            relation, fit_r, val_r, details = train_relation(x_fit, y_fit, x_val, y_val, seed, settings, device, self.encoder_module)
            stem = f'{scope}_s{seed}'
            weights = self.out / 'relations' / f'{stem}.pt'
            weights.parent.mkdir(exist_ok=True)
            torch.save(relation.state_dict(), weights)
            details.update(scope=scope, seed=seed, weights_sha256=sha256_of(weights))
            self.pretraining.append(details)
            (self.out / 'pretraining.json').write_text(json.dumps(self.pretraining, indent=2, allow_nan=False) + '\n')
            self.cache[seed] = (fit_r, val_r, stem, details)
        fit_r, val_r, stem, relation_details = self.cache[seed]
        marker = quantile_tabm.build_model(x_fit, y_fit, config, seed, settings, device)
        model = DualQuantile(marker, len(x_fit), config['d_block'], self.fusion_module).to(device)
        x = torch.cat((torch.as_tensor(x_fit, dtype=torch.float32, device=device), fit_r), dim=1)
        xv = torch.cat((torch.as_tensor(x_val, dtype=torch.float32, device=device), val_r), dim=1)
        y = model.standardize_y(torch.as_tensor(y_fit[:, None], dtype=torch.float32, device=device))
        q = torch.linspace(settings['quantile_min'], settings['quantile_max'], settings['n_quantiles'], device=device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        generator = torch.Generator().manual_seed(int(seed) * 1000 + index)
        spec = settings['training']
        best_score, best_epoch = -np.inf, -1
        best_curves, best_state = None, None
        for epoch in range(spec['max_epochs'] + 1):
            if epoch:
                quantile_tabm.update_epoch(model, x, y, q, optimizer, generator, spec, device)
            model.eval()
            with torch.no_grad():
                curves = model.inverse_y(model(xv).mean(dim=1)).cpu().numpy()
                prediction = curves.mean(axis=1)
            score = pcc(y_val, prediction)
            if not np.isfinite(score):
                raise ValueError('Nonfinite dual-representation validation PCC')
            if score > best_score + spec['tie_tolerance']:
                best_score, best_epoch = score, epoch
                best_curves = curves.copy()
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            elif epoch - best_epoch >= spec['patience']:
                break
        model.load_state_dict(best_state)
        curves_dir = self.out / 'curves'
        curves_dir.mkdir(exist_ok=True)
        np.savez_compressed(curves_dir / f'{stem}_c{index}.npz', quantiles=best_curves)
        return best_curves.mean(axis=1), dict(best_pcc=best_score, best_epoch=best_epoch, epochs_run=epoch,
            parameters=sum(p.numel() for p in model.parameters()), seconds=time.perf_counter() - start,
            state_sha256=state_sha(model), relation_scope=scope, relation_best_epoch=relation_details['best_epoch'])


def run(cfg, args, runner):
    trainer = Trainer(cfg, args.out)
    sources = [str(Path(__file__))] + [str(Path(cfg['upstream_path']) / p) for p in cfg['upstream_sha256']]
    quantile_tabm.run(cfg, args, runner, trainer=trainer, extra_sources=sources)
