"""MoEGP fitting primitives and joblib workers (upstream gating model)."""
import copy
import math
import random

import numpy as np
import torch
from joblib import Parallel, delayed  # noqa: F401  (re-exported for callers)

from wheat599.metrics import pcc

WORKER = {}


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(p, cand, mcfg, creat_mlp_layer):
    from utils.moe import Gating

    def factory():
        return creat_mlp_layer(p, 1, list(cand["hidden"]),
                               dropout=float(cand["dropout"]), batch_norm=True)

    model = Gating(p, int(cand["experts"]), factory(),
                   top_k=int(mcfg["top_k"]), loss_coef=float(mcfg["loss_coef"]))
    model.experts = torch.nn.ModuleList(
        [factory() for _ in range(int(cand["experts"]))])
    return model


def fit_moegp(X, y, train_idx, val_idx, cand, mcfg, tcfg, epochs, device,
              seed, creat_mlp_layer, lr_schedule=None):
    """One MoEGP fit; checkpoint by inner-val PCC when val_idx is given.

    Y tensors are built ONLY for the train/val subsets handed in — the outer
    test phenotypes are never materialized or processed. When ``lr_schedule``
    is given (refit phase), the per-epoch learning rate is replayed from the
    median inner-fold trajectory instead of running the scheduler.
    """
    seed_all(seed)
    mu, sd = float(y[train_idx].mean()), float(y[train_idx].std())
    if sd == 0:
        raise ValueError("constant training target")
    Ytr = torch.as_tensor((y[train_idx] - mu) / sd, dtype=torch.float32,
                          device=device).view(-1, 1)
    Yval = (torch.as_tensor((y[val_idx] - mu) / sd, dtype=torch.float32,
                            device=device).view(-1, 1)
            if val_idx is not None else None)
    model = build_model(X.shape[1], cand, mcfg, creat_mlp_layer).to(device)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X[train_idx], Ytr),
        batch_size=int(tcfg["batch_size"]), shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=float(cand["lr"]),
                           weight_decay=float(mcfg["weight_decay"]))
    sched = (torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min")
             if val_idx is not None else None)
    best_pcc, best_epoch, best_state = -math.inf, None, None
    epoch_lrs = []
    for epoch in range(int(epochs)):
        if lr_schedule is not None:
            opt.param_groups[0]["lr"] = float(
                lr_schedule[min(epoch, len(lr_schedule) - 1)])
        epoch_lrs.append(float(opt.param_groups[0]["lr"]))
        model.train()
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            pred, aux = model(xb)
            loss = torch.nn.functional.mse_loss(pred, yb) + aux
            if not torch.isfinite(loss):
                raise ValueError("non-finite training loss")
            loss.backward()
            opt.step()
        if val_idx is not None:
            model.eval()
            with torch.no_grad():
                pv, aux = model(X[val_idx])
                val_mse = torch.nn.functional.mse_loss(pv, Yval).item()
            if not np.isfinite(val_mse):
                raise ValueError("non-finite validation loss")
            sched.step(val_mse + float(aux))
            val_pcc = float(pcc(y[val_idx],
                                pv.float().cpu().numpy().ravel() * sd + mu))
            if np.isfinite(val_pcc) and val_pcc > best_pcc:
                best_pcc, best_epoch = val_pcc, epoch
                best_state = copy.deepcopy(model.state_dict())
    if val_idx is not None:
        if best_state is None:
            raise ValueError("no finite validation PCC during training")
        model.load_state_dict(best_state)
    model.eval()
    val_pred = None
    if val_idx is not None:
        with torch.no_grad():
            pv, _ = model(X[val_idx])
        val_pred = pv.float().cpu().numpy().ravel() * sd + mu
    return {"model": model, "mu": mu, "sd": sd, "best_epoch": best_epoch,
            "best_pcc": best_pcc, "lrs": epoch_lrs, "val_pred": val_pred,
            "n_params": int(sum(p.numel() for p in model.parameters()))}


def init_worker(x, y, candidates, mcfg, tcfg, source, device_str, fraction):
    import sys
    sys.path.insert(0, source)
    sys.dont_write_bytecode = True
    from utils.moe import creat_mlp_layer
    device = torch.device(device_str)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(fraction)
    WORKER.update(x=torch.as_tensor(x, dtype=torch.float32, device=device),
                  y=y, candidates=candidates, mcfg=mcfg, tcfg=tcfg,
                  device=device, create=creat_mlp_layer)


def inner_job(environment_index, candidate_index, train_idx, val_idx, seed, epochs):
    try:
        y = WORKER["y"][:, environment_index]
        res = fit_moegp(WORKER["x"], y, np.asarray(train_idx), np.asarray(val_idx),
                        WORKER["candidates"][candidate_index], WORKER["mcfg"],
                        WORKER["tcfg"], epochs, WORKER["device"], seed, WORKER["create"])
        return {"best_epoch": res["best_epoch"], "best_pcc": res["best_pcc"],
                "lrs": res["lrs"], "val_pred": res["val_pred"]}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def refit_job(environment_index, candidate_index, dev_idx, test_idx, seed, epochs, lr_schedule):
    try:
        import time as _time
        start = _time.perf_counter()
        y = WORKER["y"][:, environment_index]
        res = fit_moegp(WORKER["x"], y, np.asarray(dev_idx), None,
                        WORKER["candidates"][candidate_index], WORKER["mcfg"],
                        WORKER["tcfg"], epochs, WORKER["device"], seed, WORKER["create"],
                        lr_schedule=lr_schedule)
        rows = torch.as_tensor(np.asarray(test_idx), device=WORKER["device"])
        with torch.no_grad():
            p = res["model"](WORKER["x"][rows])[0].detach().float().cpu().numpy().ravel()
        pred = p * res["sd"] + res["mu"]
        n_params = res["n_params"]
        del res
        return {"pred": pred, "n_params": n_params, "seconds": round(_time.perf_counter() - start, 1)}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}
