"""Evaluation helpers: load a trained run, compute test metrics, plot predictions."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from dataset import AGB_SCALE, load_stats, split_files
from train import DEFAULTS, build_from_cfg, evaluate, get_device


def load_run(run, split="test"):
    """Returns (cfg, dataset, model in eval mode, device) for a run directory."""
    run = Path(run)
    cfg = json.loads((run / "config.json").read_text())
    stats = load_stats(cfg["data"])
    tr, va, te = split_files(cfg["data"], seed=cfg["seed"])
    files = {"train": tr, "val": va, "test": te}[split]
    ds, model = build_from_cfg(cfg, stats, files)
    device = get_device()
    model.load_state_dict(torch.load(run / "best.pt", map_location=device))
    return cfg, ds, model.to(device).eval(), device


def evaluate_run(run, split="test", bs=8, workers=DEFAULTS["workers"]):
    """Compute metrics for a run and save them to <run>/metrics_<split>.json."""
    _, ds, model, device = load_run(run, split)
    res = evaluate(model, DataLoader(ds, batch_size=bs, num_workers=workers), device)
    (Path(run) / f"metrics_{split}.json").write_text(json.dumps(res, indent=2))
    return res


@torch.no_grad()
def predict_tile(model, ds, i, device):
    """Returns (reference, prediction) for tile i, both (H,W) in Mg/ha."""
    xs, y, _ = ds[i]
    p = model([x[None].to(device) for x in xs]).clamp_min(0)[0, 0].cpu() * AGB_SCALE
    return y[0] * AGB_SCALE, p


def plot_predictions(run, idx=(0, 1, 2), split="test"):
    _, ds, model, device = load_run(run, split)
    fig, axes = plt.subplots(len(idx), 3, figsize=(12, 4 * len(idx)), squeeze=False)
    for row, i in zip(axes, idx):
        y, p = predict_tile(model, ds, i, device)
        vmax = float(max(y.max(), p.max()))
        for a, img, title in zip(row, [y, p, p - y], ["reference AGB", "predicted AGB", "pred - ref"]):
            if title == "pred - ref":
                kw = dict(cmap="RdBu_r", vmin=-vmax / 2, vmax=vmax / 2)
            else:
                kw = dict(cmap="viridis", vmin=0, vmax=vmax)
            im = a.imshow(img, **kw)
            a.set_title(f"{ds.files[i].stem}: {title}")
            a.axis("off")
            plt.colorbar(im, ax=a, fraction=0.046, label="Mg/ha")
    fig.tight_layout()
    return fig
