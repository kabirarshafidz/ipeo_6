"""Training helpers. Typical use from the notebook:

    cfg = make_cfg(data="data/processed", modality="s1s2", fusion="mid", name="s1s2_mid")
    log = train_model(cfg)
"""
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bands import S1_BANDS, S2_BANDS, expand
from dataset import AGB_SCALE, BiomassDataset, load_stats, split_files
from metrics import RegressionMetrics
from models import build_model

DEFAULTS = dict(
    fusion="early",   # "early" (concat channels) or "mid" (one encoder per sensor)
    drop=(),          # bands or groups to exclude, see bands.GROUPS
    use_clp=False,    # feed the cloud-probability layer as input
    loss="mse",       # "mse" | "huber" | "l1"
    epochs=50, bs=8, lr=1e-3, wd=1e-4,
    base=32, depth=4, seed=0,
    workers=0 if sys.platform == "darwin" else 4,  # DataLoader workers are unreliable on macOS
    max_train=None,   # limit #training tiles (debugging)
    runs_dir="runs",
)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_cfg(data, modality, name, **kw):
    """modality: "s1" | "s2" | "s1s2". Unknown keyword arguments raise an error."""
    unknown = set(kw) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown config keys: {unknown}")
    cfg = {**DEFAULTS, **kw, "data": str(data), "modality": modality, "name": name}
    cfg["runs_dir"] = str(cfg["runs_dir"])
    cfg["modalities"] = {"s1": ["s1"], "s2": ["s2"], "s1s2": ["s1", "s2"]}[modality]
    pool = (S1_BANDS if "s1" in cfg["modalities"] else []) + (S2_BANDS if "s2" in cfg["modalities"] else [])
    drop = set(expand(cfg["drop"]))
    if not cfg["use_clp"]:
        drop.add("CLP")
    cfg["bands"] = [b for b in pool if b not in drop]
    cfg["drop"] = list(cfg["drop"])
    return cfg


def build_from_cfg(cfg, stats, files, augment=False):
    ds = BiomassDataset(files, stats, cfg["modalities"], cfg["bands"], augment=augment)
    in_ch = [len(ds.idx[m]) for m in cfg["modalities"]]
    if 0 in in_ch:
        raise ValueError("A modality has no bands left; change `modality` instead of dropping a whole sensor.")
    model = build_model(cfg["fusion"], in_ch, cfg["base"], cfg["depth"])
    return ds, model


def masked_loss(pred, y, valid, kind):
    p, t = pred[valid], y[valid]
    if kind == "mse":
        return F.mse_loss(p, t)
    if kind == "huber":
        return F.smooth_l1_loss(p, t, beta=0.5)
    return F.l1_loss(p, t)


@torch.no_grad()
def evaluate(model, loader, device):
    """Pixel-level metrics in Mg/ha over a DataLoader."""
    model.eval()
    m = RegressionMetrics()
    for xs, y, valid in loader:
        xs = [x.to(device) for x in xs]
        pred = model(xs).clamp_min(0).cpu() * AGB_SCALE
        m.update(pred, y * AGB_SCALE, valid)
    return m.compute()


def train_model(cfg):
    """Train, keep the checkpoint with the best val RMSE. Returns the per-epoch log."""
    torch.manual_seed(cfg["seed"])
    device = get_device()
    out = Path(cfg["runs_dir"]) / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))

    stats = load_stats(cfg["data"])
    tr_files, va_files, _ = split_files(cfg["data"], seed=cfg["seed"])
    if cfg["max_train"]:
        tr_files = tr_files[: cfg["max_train"]]
    tr_ds, model = build_from_cfg(cfg, stats, tr_files, augment=True)
    va_ds = BiomassDataset(va_files, stats, cfg["modalities"], cfg["bands"])
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[{cfg['name']}] device={device} train={len(tr_ds)} val={len(va_ds)} params={n_params:.2f}M")
    print(f"  bands={cfg['bands']}")

    tr_dl = DataLoader(tr_ds, cfg["bs"], shuffle=True, num_workers=cfg["workers"], drop_last=True)
    va_dl = DataLoader(va_ds, cfg["bs"], num_workers=cfg["workers"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg["lr"], total_steps=cfg["epochs"] * len(tr_dl))

    best, log = float("inf"), []
    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        t0, tot = time.time(), 0.0
        for xs, y, valid in tr_dl:
            xs, y, valid = [x.to(device) for x in xs], y.to(device), valid.to(device)
            loss = masked_loss(model(xs), y, valid, cfg["loss"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        val = evaluate(model, va_dl, device)
        row = {"epoch": ep, "train_loss": tot / len(tr_dl), **{f"val_{k}": v for k, v in val.items()}}
        log.append(row)
        print(f"  ep {ep:3d} loss {row['train_loss']:.4f} val RMSE {val['rmse']:.2f} R2 {val['r2']:.3f} ({time.time()-t0:.0f}s)")
        if val["rmse"] < best:
            best = val["rmse"]
            torch.save(model.state_dict(), out / "best.pt")
        (out / "log.json").write_text(json.dumps(log, indent=2))
    print(f"  best val RMSE {best:.2f} Mg/ha -> {out/'best.pt'}")
    return log
