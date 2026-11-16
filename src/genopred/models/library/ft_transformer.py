"""FT-Transformer adapter — rtdl-revisiting-models (real training loop).

Protocol encoding (RESEARCH_PLAN §5, core.yaml ``marker_encoding``): each
binary marker j is embedded by rtdl's ``LinearEmbeddings`` as
``token_j = b_j + x_j * w_j``, i.e. ``t_j(0)=b_j`` and ``t_j(1)=b_j+w_j`` —
verified elementwise against the installed module (0.0.2). So the STANDARD
``FTTransformer(n_cont_features=p, cat_cardinalities=[])`` construction IS
the per-marker two-vector encoding; no custom tokenizer.

Runner<->adapter channels (NOT tuned grid params; underscore keys, stripped
from selection records):
  * ``_val`` = (X_val, y_val): inner-fold validation slice for early stopping.
    Early stopping monitors VAL MSE per epoch (never the outer-valid fold).
  * ``_refit_epochs``: int — final refit on outer_train runs exactly this many
    epochs (median of inner best epochs, capped at max_epochs), no early stop.
Without ``_val`` (and without ``_refit_epochs``) the fit trains
``max_epochs`` epochs on the given slice only (used by contract no-op-free
paths and tests).

Selection/aggregation happens in scripts/run_stage1.py (inner 3-fold mean
RMSE); this adapter fits only the slice it is handed. Determinism: torch seed
is taken from ``_seed`` (default 0) and applied to weight init + batch order.
"""

from __future__ import annotations

import numpy as np

from .base import Adapter, ParamGrid, register


@register
class FTTransformerAdapter(Adapter):
    name = "ft_transformer"

    @staticmethod
    def check_deps() -> None:
        try:
            import torch  # noqa: F401
            import rtdl_revisiting_models  # noqa: F401
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise ImportError(
                "ft_transformer requires torch + rtdl-revisiting-models in the "
                "wheat599-bench env (see env/lock.txt)"
            ) from exc

    def param_grid(self, smoke: bool = False) -> ParamGrid:
        grid = {
            "d_block": [16, 32],
            "n_blocks": [1, 2],
            "dropout": [0.0, 0.1],
            "learning_rate": [0.001, 0.003],
            "weight_decay": [0.00001, 0.0001],
            "max_epochs": [200],
            "patience": [20],
            "batch_size": [64],
        }
        pg = ParamGrid(grid)
        return pg.smoke() if smoke else pg

    def fit(self, X: np.ndarray, y: np.ndarray, params: dict) -> "FTTransformerAdapter":
        import torch

        self.check_deps()
        d_block = int(params.get("d_block", 16))
        n_blocks = int(params.get("n_blocks", 1))
        dropout = float(params.get("dropout", 0.0))
        lr = float(params.get("learning_rate", 0.001))
        wd = float(params.get("weight_decay", 1e-5))
        max_epochs = int(params.get("max_epochs", 200))
        patience = int(params.get("patience", 20))
        batch_size = int(params.get("batch_size", 64))
        seed = int(params.get("_seed", 0))
        val = params.get("_val")
        refit_epochs = params.get("_refit_epochs")

        n, p = X.shape
        torch.manual_seed(seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device

        model = _build(torch, d_block, n_blocks, dropout, p).to(device)

        Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
        yt = torch.as_tensor(y, dtype=torch.float32, device=device).view(-1)
        if val is not None:
            Xv = torch.as_tensor(val[0], dtype=torch.float32, device=device)
            yv = torch.as_tensor(val[1], dtype=torch.float32, device=device).view(-1)

        groups = model.make_parameter_groups()
        for g in groups:
            g["lr"] = lr
        if len(groups) > 1:
            groups[1]["weight_decay"] = 0.0
        opt = torch.optim.AdamW(groups, weight_decay=wd)
        loss_fn = torch.nn.MSELoss()

        epochs = int(refit_epochs) if refit_epochs is not None else max_epochs
        gen = torch.Generator()
        gen.manual_seed(seed)
        best_state, best_val, best_epoch = None, float("inf"), 0
        bad = 0
        for epoch in range(epochs):
            model.train()
            perm = torch.randperm(n, generator=gen).to(device)
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                opt.zero_grad()
                pred = model(x_cont=Xt[idx], x_cat=None).squeeze(-1)
                loss = loss_fn(pred, yt[idx])
                loss.backward()
                opt.step()
            if val is not None:
                model.eval()
                with torch.no_grad():
                    vloss = float(loss_fn(
                        model(x_cont=Xv, x_cat=None).squeeze(-1), yv))
                if vloss < best_val - 1e-7:
                    best_val, best_epoch = vloss, epoch
                    best_state = {k: v.detach().clone()
                                  for k, v in model.state_dict().items()}
                    bad = 0
                else:
                    bad += 1
                    if bad >= patience:
                        break
        if best_state is not None:
            model.load_state_dict(best_state)
        self._model = model
        self._best_epoch = best_epoch if val is not None else epochs - 1
        self._best_val = best_val if val is not None else float("nan")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch

        self._model.eval()
        Xt = torch.as_tensor(X, dtype=torch.float32, device=self._device)
        with torch.no_grad():
            pred = self._model(x_cont=Xt, x_cat=None).squeeze(-1)
        return pred.cpu().numpy().astype(np.float64)

    def best_epoch(self) -> int:
        return getattr(self, "_best_epoch", 0)

    def inner_metric(self) -> float:
        return getattr(self, "_best_val", float("nan"))


def _build(torch, d_block: int, n_blocks: int, dropout: float, p: int):
    from rtdl_revisiting_models import FTTransformer

    return FTTransformer(
        n_cont_features=p,
        cat_cardinalities=[],
        d_out=1,
        n_blocks=n_blocks,
        d_block=d_block,
        attention_n_heads=4,
        attention_dropout=dropout,
        ffn_d_hidden_multiplier=4 / 3,
        ffn_dropout=dropout,
        residual_dropout=0.0,
    )
