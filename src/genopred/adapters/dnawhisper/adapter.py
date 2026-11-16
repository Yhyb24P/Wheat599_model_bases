"""Adapter around the pinned official DNAwhisper checkout.

Upstream model code is used unchanged from an external checkout at a fixed
commit; this module only owns data assembly, fold-restricted fitting, stage
scheduling and metric recording.

Two upstream behaviours cannot be reproduced as stated in the paper and are
handled explicitly here, and both are written into the run provenance:

* the shipped YAML names ``optimizer.type`` / ``learning_rate`` /
  ``scheduler.type``, while ``DNAWhisper.configure_optimizers`` reads
  ``optimizer.name`` and ``optimizer.params.lr``.  With the released files the
  effective optimizer is AdamW at lr 1e-4 with weight decay 0.01 and a cosine
  schedule over all steps; the lion/1e-5 pair named in the text is never
  constructed.  We pass values through the keys upstream actually reads.
* ``loss_config.primary_loss.enabled`` is never read: the supervised MSE term is
  computed and added unconditionally.  Pretraining therefore consumes
  phenotypes, which is why it is restricted to the current fitting subset and
  is reported as in-training supervised pretraining, not label-free
  pretraining.
"""

import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl

CONFIG_FILES = ['model_config_csv_wheat.json', 'model_config_csv_wheat_pretrain.json',
                'training_config_csv_wheat.yml', 'training_config_csv_wheat_pretrain.yml']


class MarkerDataset(Dataset):
    """Official batch contract: features [markers, 1] plus phenotype [environments]."""

    def __init__(self, markers, phenotypes):
        self.features = torch.as_tensor(np.asarray(markers, dtype=np.float32)[:, :, None])
        self.phenotypes = torch.as_tensor(np.asarray(phenotypes, dtype=np.float32))
        if self.features.ndim != 3 or self.features.shape[2] != 1:
            raise ValueError('Marker matrix must be materials by markers')
        if self.phenotypes.ndim != 2 or self.phenotypes.shape[0] != self.features.shape[0]:
            raise ValueError('Phenotype matrix must match the marker rows')
        if len(self) < 2:
            raise ValueError('Stage dataset needs at least two materials')
        if not torch.isfinite(self.features).all() or not torch.isfinite(self.phenotypes).all():
            raise ValueError('Non-finite genotype or phenotype input')

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, index):
        return {'features': self.features[index], 'phenotype': self.phenotypes[index]}


def loader(dataset, batch_size, shuffle):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def official_files(checkout):
    out = {}
    for name in CONFIG_FILES:
        path = Path(checkout) / 'training' / 'config' / name
        raw = path.read_bytes()
        out[name] = (json.loads(raw) if name.endswith('.json') else yaml.safe_load(raw),
                     hashlib.sha256(raw).hexdigest())
    return out


def import_official(checkout):
    root = str(Path(checkout).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from training.models.DNAWhisper import DNAWhisper

    return DNAWhisper, root


def standardize(phenotypes, fit):
    """Per-environment centering and scaling fitted on the current fitting subset."""
    values = np.asarray(phenotypes[fit], dtype=np.float64)
    mean, scale = values.mean(0), values.std(0)
    if np.any(scale <= 0):
        raise ValueError('Constant environment in the fitting subset')
    return mean, scale


class EpochValidationPCC(pl.Callback):
    """Per-environment validation PCC computed from the raw batches of each epoch.

    Upstream logs its macro validation metric from the module hook, which Lightning
    calls after callback hooks, so reading ``trainer.callback_metrics`` here would lag
    by an epoch.  Upstream also *silently drops* any environment whose predictions are
    constant when averaging, which conflicts with the project rule of not filtering
    failures; computing the metric from collected batches keeps every environment and
    fails loudly instead.
    """

    def __init__(self, environments):
        self.environments = int(environments)
        self.series = []
        self.per_environment = []
        self._predictions = []
        self._targets = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if trainer.sanity_checking:
            return
        if not isinstance(outputs, dict) or 'preds' not in outputs or 'labels' not in outputs:
            raise RuntimeError('Upstream validation_step did not return predictions and labels')
        self._predictions.append(outputs['preds'].detach().float().cpu().numpy().astype(np.float64))
        self._targets.append(outputs['labels'].detach().float().cpu().numpy().astype(np.float64))

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking or not self._predictions:
            self._predictions, self._targets = [], []
            return
        prediction = np.concatenate(self._predictions, 0)
        target = np.concatenate(self._targets, 0)
        self._predictions, self._targets = [], []
        if prediction.shape != target.shape or prediction.shape[1] != self.environments:
            raise RuntimeError(f'Validation predictions have shape {prediction.shape}, '
                               f'expected [{target.shape[0]}, {self.environments}]')
        values = []
        for column in range(self.environments):
            first, second = prediction[:, column], target[:, column]
            if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
                raise RuntimeError(f'Non-finite validation prediction in environment {column}')
            if first.std() <= 0 or second.std() <= 0:
                raise RuntimeError(f'Constant validation prediction or target in environment {column}')
            values.append(float(np.corrcoef(first, second)[0, 1]))
        if not all(np.isfinite(values)):
            raise RuntimeError('Non-finite validation PCC')
        self.per_environment.append(values)
        self.series.append(float(np.mean(values)))


class StageCost(pl.Callback):
    def __init__(self):
        self.seconds = 0.0
        self.steps = 0
        self.peak_megabytes = 0.0
        self._start = None

    def on_train_start(self, trainer, pl_module):
        self._start = time.perf_counter()
        if trainer.strategy.root_device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()

    def on_train_epoch_end(self, trainer, pl_module):
        self.steps += len(trainer.train_dataloader)

    def on_train_end(self, trainer, pl_module):
        self.seconds += time.perf_counter() - self._start
        if trainer.strategy.root_device.type == 'cuda':
            self.peak_megabytes = float(torch.cuda.max_memory_allocated() / 2 ** 20)

    def record(self):
        return {'seconds': round(self.seconds, 3), 'optimizer_steps': self.steps,
                'peak_megabytes': round(self.peak_megabytes, 1)}


def build_module(DNAWhisper, config, seed, cfg):
    # Initialisation must not depend on whatever RNG state the caller consumed.
    pl.seed_everything(int(seed), workers=True)
    return DNAWhisper(config={**copy.deepcopy(config), 'random_seed': int(seed)},
                      optimizer_config=copy.deepcopy(cfg['optimizer_config']),
                      scheduler_config=copy.deepcopy(cfg['scheduler_config']))


def make_trainer(cfg, epochs, callbacks):
    if cfg['device'] not in ('cuda', 'cpu'):
        raise ValueError('Configured device must be cuda or cpu')
    return pl.Trainer(
        accelerator='gpu' if cfg['device'] == 'cuda' else 'cpu',
        devices=1,
        max_epochs=int(epochs),
        precision=cfg['precision'],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        callbacks=callbacks,
    )


def pretrain(checkout, cfg, markers, phenotypes, seed):
    """In-training supervised pretraining on the current fitting rows only."""
    DNAWhisper, _ = import_official(checkout)
    config = official_files(checkout)['model_config_csv_wheat_pretrain.json'][0]
    cost = StageCost()
    module = build_module(DNAWhisper, config, seed, cfg)
    parameters = sum(p.numel() for p in module.parameters())
    make_trainer(cfg, cfg['pretrain']['epochs'], [cost]).fit(
        module, train_dataloaders=loader(MarkerDataset(markers, phenotypes), cfg['batch_size'], True))
    record = {'stage': 'pretrain', 'seed': int(seed), 'rows': int(len(markers)),
              'parameters': int(parameters), **cost.record()}
    weights = {key: (value.detach().cpu().clone() if torch.is_tensor(value) else value)
               for key, value in module.state_dict().items()}
    del module
    return weights, int(parameters), record


def finetune_curve(checkout, cfg, pretrained, train_rows, val_rows, seed, epochs=None):
    """Fine-tune from fold-restricted pretraining and return the epoch curve."""
    DNAWhisper, _ = import_official(checkout)
    config = official_files(checkout)['model_config_csv_wheat.json'][0]
    module = build_module(DNAWhisper, config, seed, cfg)
    result = module.load_state_dict(pretrained, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError(f'Pretrained weights do not match the fine-tune graph: '
                         f'{len(result.missing_keys)} missing, {len(result.unexpected_keys)} unexpected')
    limit = int(cfg['finetune']['epochs'] if epochs is None else epochs)
    recorder, cost = EpochValidationPCC(cfg['environments']), StageCost()
    make_trainer(cfg, limit, [recorder, cost]).fit(
        module,
        train_dataloaders=loader(MarkerDataset(*train_rows), cfg['batch_size'], True),
        val_dataloaders=loader(MarkerDataset(*val_rows), cfg['batch_size'], False))
    if len(recorder.series) != limit:
        raise RuntimeError(f'Fine-tune recorded {len(recorder.series)} epochs, expected {limit}')
    record = {'stage': 'finetune', 'seed': int(seed), 'train_rows': len(train_rows[0]),
              'val_rows': len(val_rows[0]), 'epochs': limit,
              'validation_pcc_per_environment': recorder.per_environment, **cost.record()}
    return module, recorder.series, record


class StopAtEpoch(pl.Callback):
    """Stop inside a full-length schedule so the refit trajectory matches the selection run.

    The selection phase trains every inner unit for the whole fine-tune budget and reads
    the learning-rate schedule from that budget.  Refitting for the selected epoch count
    with a shortened ``max_epochs`` would rebuild a different cosine schedule, so the
    budget is kept and training is cut at the selected epoch instead.
    """

    def __init__(self, stop_after):
        self.stop_after = int(stop_after)

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch + 1 >= self.stop_after:
            trainer.should_stop = True


def refit(checkout, cfg, pretrained, train_rows, seed, epochs, schedule_epochs=None):
    """Refit the development set for the selected epochs under the selection schedule."""
    DNAWhisper, _ = import_official(checkout)
    config = official_files(checkout)['model_config_csv_wheat.json'][0]
    module = build_module(DNAWhisper, config, seed, cfg)
    result = module.load_state_dict(pretrained, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError('Pretrained weights do not match the fine-tune graph')
    budget = int(cfg['finetune']['epochs'] if schedule_epochs is None else schedule_epochs)
    if not 1 <= int(epochs) <= budget:
        raise ValueError('Selected epoch must lie inside the fine-tune budget')
    cost = StageCost()
    make_trainer(cfg, budget, [StopAtEpoch(epochs), cost]).fit(
        module, train_dataloaders=loader(MarkerDataset(*train_rows), cfg['batch_size'], True))
    return module, {'stage': 'refit', 'seed': int(seed), 'train_rows': len(train_rows[0]),
                    'epochs': int(epochs), 'schedule_epochs': budget, **cost.record()}


def prediction_batches(count, batch_size):
    """Index batches that all hold at least two materials (InstanceNorm needs a batch)."""
    size = max(2, min(int(batch_size), count))
    batches = [np.arange(start, min(start + size, count)) for start in range(0, count, size)]
    if batches and len(batches[-1]) == 1:
        if len(batches) == 1:
            raise ValueError('Need at least two materials per prediction batch')
        batches[-2] = np.arange(batches[-2][0], batches[-1][-1] + 1)
        batches.pop()
    return batches


def predict(module, markers, environments, batch_size=256, autocast=False):
    """Official forward pass under the training autocast, scattered back to row order.

    Upstream evaluates through the Lightning precision plugin, so weights trained with
    bf16-mixed are always forwarded inside an autocast region; a bare float32 forward
    would take a different attention branch (flash-attn refuses float32) and could
    change the numbers.  ``autocast`` must therefore mirror the training precision.
    """
    module.eval()
    device = next(module.parameters()).device
    rows = np.asarray(markers, dtype=np.float32)
    prediction = np.zeros((rows.shape[0], environments), dtype=np.float64)
    filled = np.zeros(rows.shape[0], dtype=bool)
    context = (torch.autocast(device_type='cuda', dtype=torch.bfloat16)
               if autocast and device.type == 'cuda' else torch.autocast('cpu', enabled=False))
    with torch.no_grad(), context:
        for index in prediction_batches(rows.shape[0], batch_size):
            batch = torch.as_tensor(rows[index][:, :, None], dtype=torch.float32, device=device)
            values = module(batch)['final_pred'].detach().float().cpu().numpy().astype(np.float64)
            if values.shape != (len(index), environments):
                raise ValueError('Prediction shape does not match the requested environments')
            if not np.isfinite(values).all():
                raise RuntimeError('Non-finite prediction under the requested autocast regime')
            prediction[index] = values
            filled[index] = True
    if not filled.all():
        raise RuntimeError('Some materials were never predicted')
    return prediction
