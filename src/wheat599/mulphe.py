"""Wheat599 re-evaluation of Mul-PheG2P with strict train/val/test separation.

Official source: wjhhh2928/Mul-PheG2P, commit
e20ca6847c7dec6ce80386e94df4378ad2d1fbec (MIT). The SNP encoder, direct
baseline, pretraining recipe, frozen-encoder adaptation, and embedding fusion
follow the public implementation. Adaptations required here:

* train-only phenotype scaling and missing-value imputation (the official
  direct/fusion paths fit MinMaxScaler on all rows);
* no test phenotype is accepted by any fitting function;
* source environments never include the target environment;
* prediction-scalar fusion (``scalar_prediction``) uses source heads and is
  the paper's prediction-space fusion; the official fine_tune_predict
  mean-embedding MetaTraitFusion path is not an arm in this round because it is
  not prediction-space fusion and would add a mixed control;
* the embedding fusion keeps full frozen source embeddings and is a separate,
  higher-capacity attribution control.

The module returns per-material test predictions; it never evaluates them.
The caller owns test labels and final scoring.
"""
from __future__ import annotations

import hashlib
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from wheat599.metrics import mae, pcc, rmse

OFFICIAL_REPO = 'wjhhh2928/Mul-PheG2P'
OFFICIAL_COMMIT = 'e20ca6847c7dec6ce80386e94df4378ad2d1fbec'
ARMS = ('direct', 'target_pretrain', 'frozen_sources',
        'scalar_prediction', 'linear_prediction', 'embedding')
def _state_sha256(*states):
    digest = hashlib.sha256()
    for state in states:
        for key in sorted(state):
            tensor = state[key].detach().cpu().contiguous()
            digest.update(key.encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _parameter_count(*modules):
    return int(sum(parameter.numel() for module in modules
                   for parameter in module.parameters()))


def _array_sha256(value):
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    return hashlib.sha256(array.tobytes()).hexdigest()


DEFAULT_CONFIG = {
    'hidden_dim': 128, 'conv_channels': 32, 'kernel_size': 7, 'stride': 2,
    'batch_size': 32, 'lr': 1e-3,
    'direct_epochs': 300, 'pretrain_epochs': 300, 'finetune_epochs': 300,
    'fusion_epochs': 200, 'direct_head_hidden': 64, 'fusion_hidden': 64,
    'dropout': 0.5, 'weight_decay': 5e-4, 'patience': 50,
    'attn_entropy_lambda': 1e-3, 'grad_clip': 5.0, 'seed': 42,
}


# -- official modules (adapted from the MIT repository) --------------------
class SNPEncoderCNN(nn.Module):
    def __init__(self, input_len, conv_channels=32, kernel_size=7, stride=2,
                 hidden_dim=128):
        super().__init__()
        self.conv1 = nn.Conv1d(1, conv_channels, kernel_size=kernel_size,
                               stride=stride, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(conv_channels, conv_channels * 2, kernel_size=5,
                               stride=stride, padding=2)
        self.pool = nn.AdaptiveAvgPool1d(16)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(conv_channels * 2 * 16, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() == 3:
            if x.size(1) != 1:
                raise ValueError('SNPEncoderCNN expects one input channel')
        else:
            raise ValueError('SNPEncoderCNN expects [B,L] or [B,1,L]')
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        return self.relu(self.fc(x))


class MetaTraitFusion(nn.Module):
    """Official prediction-scalar attention fusion (models.py MetaTraitFusion)."""

    def __init__(self, num_traits, hidden_dim=64):
        super().__init__()
        self.num_traits = int(num_traits)
        self.attn = nn.Sequential(
            nn.Linear(self.num_traits, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, self.num_traits), nn.Softmax(dim=1))
        self.out = nn.Sequential(nn.Linear(1, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, 1))

    def forward(self, trait_preds):
        if trait_preds.dim() != 2 or trait_preds.size(1) != self.num_traits:
            raise ValueError('MetaTraitFusion input shape mismatch')
        weights = self.attn(trait_preds)
        fused = torch.sum(weights * trait_preds, dim=1, keepdim=True)
        return self.out(fused).squeeze(1), weights


class EmbeddingAttentionFusion(nn.Module):
    """Official embedding-space attention fusion (fusion.py)."""

    def __init__(self, hidden_dim, num_traits, dropout=0.5):
        super().__init__()
        self.num_traits = int(num_traits)
        self.ln = nn.LayerNorm(hidden_dim)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1))
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Dropout(p=dropout), nn.Linear(hidden_dim, 1))

    def forward(self, emb):
        emb = self.ln(emb)
        scores = self.scorer(emb)
        weights = torch.softmax(scores, dim=1)
        fused = (emb * weights).sum(dim=1)
        return self.reg_head(fused).squeeze(-1), weights.squeeze(-1)


def pearson_loss(prediction, truth):
    vx = prediction - prediction.mean()
    vy = truth - truth.mean()
    return 1 - (vx * vy).sum() / (
        torch.sqrt((vx ** 2).sum()) * torch.sqrt((vy ** 2).sum()) + 1e-8)


def selection_loss(prediction, truth, combined):
    """Validation objective: official pretraining/fine-tuning uses MSE+0.2*Pearson."""
    prediction = torch.as_tensor(prediction, dtype=torch.float64).reshape(-1)
    truth = torch.as_tensor(truth, dtype=torch.float64).reshape(-1)
    mse = F.mse_loss(prediction, truth)
    if not combined:
        return float(mse)
    return float(mse + 0.2 * pearson_loss(prediction, truth))


def batch_validation_loss(model, head, x, y, device, batch_size, combined):
    """Official validation: mean over batches of MSE (+0.2*Pearson)."""
    model.eval()
    head.eval()
    values, mse_values = [], []
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
                        batch_size=int(batch_size), shuffle=False)
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            prediction = head(model(batch_x)).squeeze(1)
            mse = F.mse_loss(prediction, batch_y)
            mse_values.append(float(mse))
            values.append(float(mse + 0.2 * pearson_loss(prediction, batch_y))
                          if combined else float(mse))
    return float(np.mean(values)), float(np.mean(mse_values))


# -- fit-only preprocessing ------------------------------------------------
def _fit_impute(x):
    x = np.asarray(x, dtype=np.float64)
    finite = np.where(np.isfinite(x), x, np.nan)
    medians = np.nanmedian(finite, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    return medians


def _apply_impute(x, medians):
    x = np.asarray(x, dtype=np.float64).copy()
    bad = ~np.isfinite(x)
    if bad.any():
        x[bad] = np.take(medians, np.where(bad)[1])
    return x.astype(np.float32)


def _target_fill(y, fallback):
    y = np.asarray(y, dtype=np.float64).reshape(-1).copy()
    bad = ~np.isfinite(y)
    if bad.any():
        y[bad] = float(fallback)
    return y.astype(np.float32)


class _TargetScaler:
    def __init__(self, mode, y_train):
        self.mode = mode
        self.low = float(np.min(y_train))
        self.high = float(np.max(y_train))
        if self.mode == 'minmax' and self.high <= self.low:
            raise ValueError('Constant training phenotype')
        self.scope = 'train'

    def transform(self, y):
        y = np.asarray(y, dtype=np.float32)
        if self.mode == 'none':
            return y
        return ((y - self.low) / (self.high - self.low)).astype(np.float32)

    def inverse(self, y):
        y = np.asarray(y, dtype=np.float64)
        if self.mode == 'none':
            return y
        return y * (self.high - self.low) + self.low

    def info(self):
        return {'mode': self.mode, 'scope': 'train',
                'low': self.low, 'high': self.high}


def _metrics(truth, prediction):
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    return {'pcc': float(pcc(truth, prediction)),
            'rmse': float(rmse(truth, prediction)),
            'mae': float(mae(truth, prediction))}


# -- training utilities ----------------------------------------------------
def _loader(x, y, batch_size, shuffle, seed):
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
                      batch_size=int(batch_size), shuffle=shuffle,
                      generator=generator if shuffle else None)


def _seed(seed):
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _evaluate(model, head, x, device):
    model.eval()
    head.eval()
    chunks = []
    with torch.no_grad():
        for batch in DataLoader(TensorDataset(torch.from_numpy(x)),
                                batch_size=256, shuffle=False):
            chunks.append(head(model(batch[0].to(device))).squeeze(1).cpu())
    return torch.cat(chunks).numpy().astype(np.float64)


def _encoder_for(config, markers):
    return SNPEncoderCNN(
        input_len=int(markers),
        conv_channels=int(config['conv_channels']),
        kernel_size=int(config['kernel_size']),
        stride=int(config['stride']),
        hidden_dim=int(config['hidden_dim']))


def _train_predictor(x_train, y_train, x_val, y_val, config, seed, device,
                     head_kind, epochs_key, criterion_kind, selection,
                     fixed_epochs=None):
    _seed(seed)
    model = _encoder_for(config, x_train.shape[1]).to(device)
    if head_kind == 'direct':
        head = nn.Sequential(
            nn.Linear(int(config['hidden_dim']), int(config['direct_head_hidden'])),
            nn.ReLU(),
            nn.Linear(int(config['direct_head_hidden']), 1)).to(device)
    else:
        head = nn.Linear(int(config['hidden_dim']), 1).to(device)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(head.parameters()),
                                 lr=float(config['lr']))
    loader = _loader(x_train, y_train, config['batch_size'], True, seed)
    best = {'score': -np.inf if selection == 'pearson' else np.inf,
            'epoch': 0, 'model': None, 'head': None}
    history = []
    epochs = int(fixed_epochs) if fixed_epochs is not None else int(config[epochs_key])
    for epoch in range(1, epochs + 1):
        model.train()
        head.train()
        total = 0.0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = head(model(batch_x)).squeeze(1)
            if criterion_kind == 'mse_pearson':
                loss = F.mse_loss(prediction, batch_y) \
                    + 0.2 * pearson_loss(prediction, batch_y)
            else:
                loss = F.mse_loss(prediction, batch_y)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(batch_x)
        if fixed_epochs is not None:
            history.append({'epoch': epoch, 'train_loss': total / len(x_train)})
            continue
        if selection == 'pearson':
            prediction = _evaluate(model, head, x_val, device)
            score = float(pcc(y_val, prediction))
            better = score > best['score']
            val_mse = float(np.mean((prediction - y_val) ** 2))
        else:
            score, val_mse = batch_validation_loss(
                model, head, x_val, y_val, device, config['batch_size'], combined=True)
            better = score < best['score']
        history.append({'epoch': epoch, 'train_loss': total / len(x_train),
                        'val_score': score, 'val_mse': val_mse})
        if better:
            best = {'score': score, 'epoch': epoch,
                    'model': {key: value.detach().cpu().clone()
                              for key, value in model.state_dict().items()},
                    'head': {key: value.detach().cpu().clone()
                             for key, value in head.state_dict().items()}}
    if fixed_epochs is not None:
        best = {'score': None, 'epoch': epochs,
                'model': {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()},
                'head': {key: value.detach().cpu().clone()
                         for key, value in head.state_dict().items()}}
    model.load_state_dict(best['model'])
    head.load_state_dict(best['head'])
    return model, head, best, history


def _frozen_embeddings(encoders, x, device, batch_size=256):
    outputs = []
    with torch.no_grad():
        for batch in DataLoader(TensorDataset(torch.from_numpy(x)),
                                batch_size=batch_size, shuffle=False):
            input_tensor = batch[0].to(device)
            outputs.append(torch.stack([encoder(input_tensor) for encoder in encoders],
                                       dim=1).cpu())
    return torch.cat(outputs).numpy().astype(np.float32)


def train_source_encoders(x_train, y_train, x_val, y_val, source_names, config,
                          seed, device, fixed_epochs=None):
    """Official pretraining recipe for every source environment."""
    models = []
    for position, name in enumerate(source_names):
        y_train_s = _target_fill(y_train[:, position], np.median(
            y_train[np.isfinite(y_train[:, position]), position]))
        if fixed_epochs is None:
            fixed = None
        elif isinstance(fixed_epochs, dict):
            if str(name) not in fixed_epochs:
                raise ValueError(f'Missing fixed source epochs for {name}')
            fixed = int(fixed_epochs[str(name)])
        else:
            fixed = int(fixed_epochs)
        if y_val is None:
            y_val_s = None
        else:
            y_val_s = _target_fill(y_val[:, position], np.median(
                y_train[np.isfinite(y_train[:, position]), position]))
        model, head, best, history = _train_predictor(
            x_train, y_train_s, x_val, y_val_s, config, int(seed) + position,
            device, 'linear', 'pretrain_epochs', 'mse_pearson', 'combined',
            fixed_epochs=fixed)
        models.append({'name': str(name), 'model': model, 'head': head,
                       'best_epoch': best['epoch'], 'history': history})
    return models


# -- arms ------------------------------------------------------------------
def _source_predictions(source_models, x, device):
    outputs = []
    for entry in source_models:
        encoder, head = entry['model'], entry['head']
        encoder.eval()
        head.eval()
        with torch.no_grad():
            outputs.append(head(encoder(torch.from_numpy(x).to(device))).squeeze(1).cpu().numpy())
    return np.stack(outputs, axis=1).astype(np.float32)


def run_arm(arm, x_train, y_train, x_val, y_val, x_test, config, seed, device,
            source_models=None, source_names=None, fixed_epoch=None):
    if arm not in ARMS:
        raise ValueError(f'Unknown arm: {arm}')
    started = time.perf_counter()
    config = {**DEFAULT_CONFIG, **config}
    y_train = np.asarray(y_train, dtype=np.float32).reshape(-1)
    y_val = None if y_val is None else np.asarray(y_val, dtype=np.float32).reshape(-1)
    if (x_val is None) != (y_val is None):
        raise ValueError('x_val and y_val must both be present or both absent')
    if arm not in ('linear_prediction',) and fixed_epoch is None and y_val is None:
        raise ValueError('No-validation fits require fixed_epoch')
    if arm in ('direct', 'target_pretrain'):
        mode = 'minmax' if arm == 'direct' else 'none'
        scaler = _TargetScaler(mode, y_train)
        y_train_s = scaler.transform(y_train)
        y_val_s = None if y_val is None else scaler.transform(y_val)
        if arm == 'direct':
            model, head, best, history = _train_predictor(
                x_train, y_train_s, x_val, y_val_s, config, seed, device,
                'direct', 'direct_epochs', 'mse', 'pearson',
                fixed_epochs=fixed_epoch)
        else:
            model, head, best, history = _train_predictor(
                x_train, y_train_s, x_val, y_val_s, config, seed, device,
                'linear', 'pretrain_epochs', 'mse_pearson', 'combined',
                fixed_epochs=fixed_epoch)
        val_prediction = None if y_val is None else scaler.inverse(
            _evaluate(model, head, x_val, device))
        prediction = scaler.inverse(_evaluate(model, head, x_test, device))
        return {'arm': arm, 'prediction': prediction,
                'val_prediction': val_prediction,
                'val_metrics': None if y_val is None else _metrics(y_val, val_prediction),
                'best_epoch': best['epoch'], 'epochs_run': len(history),
                'history': history, 'scaling': scaler.info(),
                'source_names': [], 'source_pretraining': None,
                'fusion_input': None, 'state_sha256': _state_sha256(
                    best['model'], best['head']),
                'parameters': _parameter_count(model, head),
                'total_parameters': _parameter_count(model, head),
                'train_seconds': time.perf_counter() - started,
                'amp': False, 'device': str(device), 'seed': int(seed),
                'dtype': 'float32', 'mode': 'fixed_full_dev' if y_val is None
                else 'validation'}
    if source_models is None or source_names is None:
        raise ValueError('Source arms require pretrained source models')
    if any(str(name) == str(config.get('target_name')) for name in source_names):
        raise ValueError('Target environment cannot be a source')
    encoders = [entry['model'] for entry in source_models]
    for encoder in encoders:
        encoder.eval()
        for parameter in encoder.parameters():
            parameter.requires_grad = False
    scaler = _TargetScaler('minmax' if arm == 'embedding' else 'none', y_train)
    y_train_s = scaler.transform(y_train)
    y_val_s = None if y_val is None else scaler.transform(y_val)
    if arm == 'frozen_sources':
        train_embedding = _frozen_embeddings(encoders, x_train, device)
        val_embedding = None if x_val is None else _frozen_embeddings(
            encoders, x_val, device)
        test_embedding = _frozen_embeddings(encoders, x_test, device)
        mean_train = train_embedding.mean(axis=1)
        mean_val = None if val_embedding is None else val_embedding.mean(axis=1)
        mean_test = test_embedding.mean(axis=1)
        _seed(seed)
        head = nn.Linear(int(config['hidden_dim']), 1).to(device)
        optimizer = torch.optim.Adam(head.parameters(), lr=float(config['lr']))
        x_val_t = None if mean_val is None else torch.from_numpy(mean_val).to(device)
        y_val_t = None if y_val_s is None else torch.from_numpy(y_val_s).to(device)
        best = {'loss': np.inf, 'epoch': 0, 'state': None}
        history = []
        loader = _loader(mean_train, y_train_s, config['batch_size'], True, seed)
        epochs = int(fixed_epoch) if fixed_epoch is not None else int(config['finetune_epochs'])
        for epoch in range(1, epochs + 1):
            head.train()
            total = 0.0
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad(set_to_none=True)
                prediction = head(batch_x).squeeze(1)
                loss = F.mse_loss(prediction, batch_y) \
                    + 0.2 * pearson_loss(prediction, batch_y)
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(batch_x)
            if y_val_t is None:
                history.append({'epoch': epoch, 'train_loss': total / len(mean_train)})
                continue
            with torch.no_grad():
                val_values = []
                for batch_x, batch_y in _loader(mean_val, y_val_s, config['batch_size'],
                                                False, seed):
                    prediction = head(batch_x.to(device)).squeeze(1)
                    batch_y = batch_y.to(device)
                    val_values.append(float(F.mse_loss(prediction, batch_y)
                                            + 0.2 * pearson_loss(prediction, batch_y)))
                loss = float(np.mean(val_values))
            history.append({'epoch': epoch, 'val_loss': loss})
            if loss < best['loss']:
                best = {'loss': loss, 'epoch': epoch,
                        'state': {key: value.detach().cpu().clone()
                                  for key, value in head.state_dict().items()}}
        if y_val_t is None:
            best = {'loss': None, 'epoch': epochs,
                    'state': {key: value.detach().cpu().clone()
                              for key, value in head.state_dict().items()}}
        head.load_state_dict(best['state'])
        with torch.no_grad():
            val_prediction = None if mean_val is None else scaler.inverse(
                head(torch.from_numpy(mean_val).to(device)).squeeze(1).cpu().numpy())
            prediction = scaler.inverse(
                head(torch.from_numpy(mean_test).to(device)).squeeze(1).cpu().numpy())
        return {'arm': arm, 'prediction': prediction,
                'val_prediction': val_prediction,
                'val_metrics': None if y_val is None else _metrics(y_val, val_prediction),
                'best_epoch': best['epoch'],
                'epochs_run': len(history), 'history': history,
                'scaling': scaler.info(),
                'source_names': [str(name) for name in source_names],
                'source_pretraining': [{'name': entry['name'],
                                        'best_epoch': entry['best_epoch']}
                                       for entry in source_models],
                'fusion_input': 'mean_embedding',
                'state_sha256': _state_sha256(
                    *[entry['model'].state_dict() for entry in source_models],
                    best['state']),
                'parameters': _parameter_count(head),
                'total_parameters': _parameter_count(
                    *[entry['model'] for entry in source_models], head),
                'train_seconds': time.perf_counter() - started,
                'amp': False, 'device': str(device), 'seed': int(seed),
                'dtype': 'float32', 'mode': 'fixed_full_dev' if y_val is None
                else 'validation'}
    train_embedding = _frozen_embeddings(encoders, x_train, device)
    val_embedding = None if x_val is None else _frozen_embeddings(
        encoders, x_val, device)
    test_embedding = _frozen_embeddings(encoders, x_test, device)
    if arm == 'embedding':
        _seed(seed)
        model = EmbeddingAttentionFusion(
            int(config['hidden_dim']), len(source_models),
            float(config['dropout'])).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(config['lr']),
                                     weight_decay=float(config['weight_decay']))
        x_train_t = torch.from_numpy(train_embedding).to(device)
        x_val_t = None if val_embedding is None else torch.from_numpy(val_embedding).to(device)
        x_test_t = torch.from_numpy(test_embedding).to(device)
        y_train_t = torch.from_numpy(y_train_s).to(device)
        y_val_t = None if y_val_s is None else torch.from_numpy(y_val_s).to(device)
        best = {'pearson': -np.inf, 'epoch': 0, 'state': None}
        wait = 0
        history = []
        epochs = int(fixed_epoch) if fixed_epoch is not None else int(config['fusion_epochs'])
        for epoch in range(1, epochs + 1):
            model.train()
            prediction, weights = model(x_train_t)
            loss = F.smooth_l1_loss(prediction, y_train_t, beta=0.5)
            if float(config['attn_entropy_lambda']) > 0:
                entropy = (weights * torch.log(weights + 1e-8)).sum(1).mean()
                loss = loss + float(config['attn_entropy_lambda']) * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if y_val_t is None:
                history.append({'epoch': epoch, 'train_loss': float(loss.detach())})
                continue
            model.eval()
            with torch.no_grad():
                val_prediction_s, _ = model(x_val_t)
                val_prediction = scaler.inverse(val_prediction_s.cpu().numpy())
            score = float(pcc(y_val, val_prediction))
            history.append({'epoch': epoch, 'train_loss': float(loss.detach()),
                            'val_pearson': score})
            if score > best['pearson']:
                best = {'pearson': score, 'epoch': epoch,
                        'state': {key: value.detach().cpu().clone()
                                  for key, value in model.state_dict().items()}}
                wait = 0
            else:
                wait += 1
                if wait >= int(config['patience']):
                    break
        if y_val_t is None:
            best = {'pearson': None, 'epoch': epochs,
                    'state': {key: value.detach().cpu().clone()
                              for key, value in model.state_dict().items()}}
        model.load_state_dict(best['state'])
        model.eval()
        with torch.no_grad():
            val_prediction = None if x_val_t is None else scaler.inverse(
                model(x_val_t)[0].cpu().numpy())
            prediction = scaler.inverse(model(x_test_t)[0].cpu().numpy())
        return {'arm': arm, 'prediction': prediction,
                'val_prediction': val_prediction,
                'val_metrics': None if y_val is None else _metrics(y_val, val_prediction),
                'best_epoch': best['epoch'],
                'epochs_run': len(history), 'history': history,
                'scaling': scaler.info(),
                'source_names': [str(name) for name in source_names],
                'source_pretraining': [{'name': entry['name'],
                                        'best_epoch': entry['best_epoch']}
                                       for entry in source_models],
                'fusion_input': 'embedding',
                'state_sha256': _state_sha256(
                    *[entry['model'].state_dict() for entry in source_models],
                    best['state']),
                'parameters': _parameter_count(model),
                'total_parameters': _parameter_count(
                    *[entry['model'] for entry in source_models], model),
                'train_seconds': time.perf_counter() - started,
                'amp': False, 'device': str(device), 'seed': int(seed),
                'dtype': 'float32', 'mode': 'fixed_full_dev' if y_val is None
                else 'validation'}
    # True prediction-space fusion: scalar source-head predictions.
    scalar_train = _source_predictions(source_models, x_train, device)
    scalar_val = None if x_val is None else _source_predictions(
        source_models, x_val, device)
    scalar_test = _source_predictions(source_models, x_test, device)
    source_mean = scalar_train.mean(0)
    source_std = scalar_train.std(0)
    source_std = np.where(source_std > 0, source_std, 1.0)
    scalar_train = (scalar_train - source_mean) / source_std
    scalar_val = None if scalar_val is None else (
        scalar_val - source_mean) / source_std
    scalar_test = (scalar_test - source_mean) / source_std
    if arm == 'linear_prediction':
        design = np.column_stack([np.ones(len(scalar_train)), scalar_train])
        coefficients = np.linalg.lstsq(design, np.asarray(y_train, dtype=np.float64),
                                       rcond=None)[0]
        val_prediction = None if scalar_val is None else scaler.inverse(
            np.column_stack([np.ones(len(scalar_val)), scalar_val]) @ coefficients)
        prediction = scaler.inverse(
            np.column_stack([np.ones(len(scalar_test)), scalar_test]) @ coefficients)
        return {'arm': arm, 'prediction': prediction,
                'val_prediction': val_prediction,
                'val_metrics': None if y_val is None else _metrics(y_val, val_prediction),
                'best_epoch': 0, 'epochs_run': 0,
                'history': [{'method': 'ols', 'epochs': 0}],
                'scaling': scaler.info(),
                'source_names': [str(name) for name in source_names],
                'source_pretraining': [{'name': entry['name'],
                                        'best_epoch': entry['best_epoch']}
                                       for entry in source_models],
                'fusion_input': 'source_prediction_linear',
                'source_scalars': {'train': scalar_train, 'val': scalar_val,
                                   'test': scalar_test},
                'mode': 'fixed_full_dev' if y_val is None else 'validation',
                'coefficients': [float(value) for value in coefficients],
                'state_sha256': hashlib.sha256(
                    (_state_sha256(*[entry['model'].state_dict()
                                     for entry in source_models])
                     + _array_sha256(coefficients)).encode()).hexdigest(),
                'parameters': int(coefficients.size),
                'total_parameters': _parameter_count(
                    *[entry['model'] for entry in source_models]) + int(coefficients.size),
                'train_seconds': time.perf_counter() - started,
                'amp': False, 'device': str(device), 'seed': int(seed),
                'dtype': 'float32'}
    fusion_input = 'source_prediction'
    _seed(seed)
    fusion = MetaTraitFusion(len(source_models), int(config['fusion_hidden'])).to(device)
    head = nn.Sequential(nn.Linear(1, int(config['fusion_hidden'])), nn.ReLU(),
                         nn.Linear(int(config['fusion_hidden']), 1)).to(device)
    optimizer = torch.optim.Adam(list(fusion.parameters()) + list(head.parameters()),
                                 lr=float(config['lr']))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-5)
    x_val_t = None if scalar_val is None else torch.from_numpy(
        scalar_val.astype(np.float32)).to(device)
    y_val_t = None if y_val_s is None else torch.from_numpy(y_val_s).to(device)
    x_test_t = torch.from_numpy(scalar_test.astype(np.float32)).to(device)
    best = {'loss': np.inf, 'epoch': 0, 'fusion': None, 'head': None}
    history = []
    epochs = int(fixed_epoch) if fixed_epoch is not None else int(config['fusion_epochs'])
    for epoch in range(1, epochs + 1):
        fusion.train()
        head.train()
        total = 0.0
        for batch_x, batch_y in _loader(scalar_train, y_train_s, config['batch_size'],
                                        True, seed):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            fused, _ = fusion(batch_x)
            prediction = head(fused.unsqueeze(-1)).squeeze(1)
            loss = F.mse_loss(prediction, batch_y) \
                + 0.2 * pearson_loss(prediction, batch_y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if config.get('grad_clip'):
                torch.nn.utils.clip_grad_norm_(
                    list(fusion.parameters()) + list(head.parameters()),
                    float(config['grad_clip']))
            optimizer.step()
            total += float(loss.detach()) * len(batch_x)
        fusion.eval()
        head.eval()
        if y_val_t is None:
            history.append({'epoch': epoch, 'train_loss': total / len(scalar_train)})
            continue
        with torch.no_grad():
            val_values = []
            for batch_x, batch_y in _loader(scalar_val, y_val_s, config['batch_size'],
                                            False, seed):
                fused_val, _ = fusion(batch_x.to(device))
                prediction_val = head(fused_val.unsqueeze(-1)).squeeze(1)
                val_values.append(float(F.mse_loss(prediction_val,
                                                   batch_y.to(device))))
            val_loss = float(np.mean(val_values))
        scheduler.step(val_loss)
        history.append({'epoch': epoch, 'train_loss': total / len(scalar_train),
                        'val_loss': val_loss})
        if val_loss < best['loss']:
            best = {'loss': val_loss, 'epoch': epoch,
                    'fusion': {key: value.detach().cpu().clone()
                               for key, value in fusion.state_dict().items()},
                    'head': {key: value.detach().cpu().clone()
                             for key, value in head.state_dict().items()}}
    if y_val_t is None:
        best = {'loss': None, 'epoch': epochs,
                'fusion': {key: value.detach().cpu().clone()
                           for key, value in fusion.state_dict().items()},
                'head': {key: value.detach().cpu().clone()
                         for key, value in head.state_dict().items()}}
    fusion.load_state_dict(best['fusion'])
    head.load_state_dict(best['head'])
    fusion.eval()
    head.eval()
    with torch.no_grad():
        val_prediction = None if x_val_t is None else scaler.inverse(
            head(fusion(x_val_t)[0].unsqueeze(-1)).squeeze(1).cpu().numpy())
        prediction = scaler.inverse(
            head(fusion(x_test_t)[0].unsqueeze(-1)).squeeze(1).cpu().numpy())
    return {'arm': arm, 'prediction': prediction,
            'val_prediction': val_prediction,
            'val_metrics': None if y_val is None else _metrics(y_val, val_prediction),
            'best_epoch': best['epoch'],
            'epochs_run': len(history), 'history': history,
            'scaling': scaler.info(),
            'source_names': [str(name) for name in source_names],
            'source_pretraining': [{'name': entry['name'],
                                    'best_epoch': entry['best_epoch']}
                                   for entry in source_models],
            'fusion_input': fusion_input,
            'mode': 'fixed_full_dev' if y_val is None else 'validation',
            'source_scalars': {'train': scalar_train, 'val': scalar_val,
                               'test': scalar_test},
            'state_sha256': _state_sha256(
                *[entry['model'].state_dict() for entry in source_models],
                best['fusion'], best['head']),
            'parameters': _parameter_count(fusion, head),
            'total_parameters': _parameter_count(
                *[entry['model'] for entry in source_models], fusion, head),
            'train_seconds': time.perf_counter() - started,
            'amp': False, 'device': str(device), 'seed': int(seed),
            'dtype': 'float32'}


def run_arms(x_train, y_train, x_val, y_val, x_test, *, target_name, source_names,
             source_train, source_val, config=None, seed=42, device='cpu',
             arms=ARMS, source_models=None, fixed_epochs=None,
             source_fixed_epochs=None):
    """Run the requested arms without ever receiving test phenotypes."""
    config = {**DEFAULT_CONFIG, **(config or {})}
    config['target_name'] = str(target_name)
    x_train = np.asarray(x_train, dtype=np.float64)
    x_test = np.asarray(x_test, dtype=np.float64)
    val_mode = x_val is not None
    if val_mode != (y_val is not None):
        raise ValueError('x_val and y_val must both be present or both absent')
    if val_mode:
        x_val = np.asarray(x_val, dtype=np.float64)
        if x_val.ndim != 2 or x_train.ndim != 2 or x_test.ndim != 2 \
                or x_train.shape[1] != x_val.shape[1] or x_train.shape[1] != x_test.shape[1]:
            raise ValueError('Marker matrices must share the marker axis')
        if fixed_epochs is not None or source_fixed_epochs is not None:
            raise ValueError('fixed epochs are only valid in no-validation refit mode')
    else:
        if x_train.ndim != 2 or x_test.ndim != 2 or x_train.shape[1] != x_test.shape[1]:
            raise ValueError('Marker matrices must share the marker axis')
        if fixed_epochs is None:
            raise ValueError('Fixed full-dev mode requires fixed_epochs')
        for arm in arms:
            if arm != 'linear_prediction' and arm not in fixed_epochs:
                raise ValueError(f'Missing fixed epochs for {arm}')
        if source_fixed_epochs is None and source_models is None \
                and any(arm in ARMS[2:] for arm in arms):
            raise ValueError('Fixed source pretraining requires source_fixed_epochs')
    if source_names is None or source_train is None:
        raise ValueError('Source phenotypes and names are required')
    source_names = [str(name) for name in source_names]
    if str(target_name) in source_names:
        raise ValueError('Target environment cannot be a source')
    source_train = np.asarray(source_train, dtype=np.float64)
    if source_train.ndim != 2 or source_train.shape[1] != len(source_names):
        raise ValueError('Source phenotype shapes disagree')
    if val_mode:
        source_val = np.asarray(source_val, dtype=np.float64)
        if source_val.shape != (len(x_val), len(source_names)):
            raise ValueError('Source validation phenotype shapes disagree')
    else:
        if source_val is not None:
            raise ValueError('Full-dev refit cannot receive source validation phenotypes')
        source_val = None
    medians = _fit_impute(x_train)
    x_train = _apply_impute(x_train, medians)
    if val_mode:
        x_val = _apply_impute(x_val, medians)
    x_test = _apply_impute(x_test, medians)
    y_train = _target_fill(y_train, np.median(
        np.asarray(y_train, dtype=np.float64)[np.isfinite(y_train)]))
    y_val = None if not val_mode else _target_fill(y_val, np.median(
        np.asarray(y_train, dtype=np.float64)[np.isfinite(y_train)]))
    source_seconds = 0.0
    if any(arm in ARMS[2:] for arm in arms):
        if source_models is None:
            source_started = time.perf_counter()
            source_models = train_source_encoders(
                x_train, source_train, x_val if val_mode else None,
                source_val if val_mode else None, source_names, config,
                int(seed) + 1000, device,
                fixed_epochs=None if val_mode else source_fixed_epochs)
            source_seconds = time.perf_counter() - source_started
        else:
            mapping = {str(entry['name']): entry for entry in source_models}
            missing = [name for name in source_names if name not in mapping]
            if missing:
                raise ValueError(f'Missing pretrained source models: {missing}')
            source_models = [mapping[name] for name in source_names]
    results = {}
    for arm in arms:
        results[arm] = run_arm(
            arm, x_train, y_train, x_val, y_val, x_test, config, seed, device,
            source_models=source_models, source_names=source_names,
            fixed_epoch=None if (val_mode or arm == 'linear_prediction')
            else int(fixed_epochs[arm]))
    return {'arms': results, 'source_names': source_names,
            'mode': 'validation' if val_mode else 'fixed_full_dev',
            'fixed_epochs': None if val_mode else {
                str(arm): int(value) for arm, value in fixed_epochs.items()},
            'source_fixed_epochs': None if val_mode else source_fixed_epochs,
            'source_pretraining': None if source_models is None else [
                {'name': entry['name'], 'best_epoch': entry['best_epoch']}
                for entry in source_models],
            'impute_scope': 'train', 'seed': int(seed),
            'source_pretraining_seconds': source_seconds,
            'official': {'repo': OFFICIAL_REPO, 'commit': OFFICIAL_COMMIT,
                         'license': 'MIT'},
            'notes': ['train-only scaling/imputation',
                      'no test phenotype accepted',
                      'scalar_prediction uses source head predictions (prediction-space fusion)']}
