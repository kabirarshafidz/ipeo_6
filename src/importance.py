"""Inference-time contribution analysis per band and per band group.

method="permute": shuffle the selected channels across tiles in the batch (keeps each
                  channel's marginal distribution, breaks its link to the target).
method="mean":    replace the selected channels with their training mean (0 after z-scoring).

Importance = RMSE(perturbed) - RMSE(baseline).
Single-band scores understate bands with correlated neighbours. Compare with group
scores and with retraining ablations (make_cfg(..., drop=[...])) before drawing conclusions.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from bands import GROUPS
from dataset import AGB_SCALE
from evaluate import load_run
from train import DEFAULTS
from metrics import RegressionMetrics


@torch.no_grad()
def _rmse(model, dl, device, targets=None, method="permute", gen=None):
    """targets: list of (modality_index, channel_index) to perturb."""
    m = RegressionMetrics()
    for xs, y, valid in dl:
        xs = [x.to(device).clone() for x in xs]
        if targets:
            perm = torch.randperm(xs[0].shape[0], generator=gen).to(device)
            for mi, ci in targets:
                xs[mi][:, ci] = xs[mi][perm, ci] if method == "permute" else 0.0
        pred = model(xs).clamp_min(0).cpu() * AGB_SCALE
        m.update(pred, y * AGB_SCALE, valid)
    return m.compute()["rmse"]


def compute_importance(run, method="permute", split="test", repeats=3, bs=16, workers=DEFAULTS["workers"]):
    """Returns {"baseline_rmse", "importance": {name: {"delta_rmse", "runs"}}}.
    Group names are written in brackets, e.g. "[SWIR]". Also saved to <run>/importance_<method>.json."""
    if method == "permute" and bs < 2:
        raise ValueError("permutation needs batch size >= 2")
    cfg, ds, model, device = load_run(run, split)
    # shuffle=True so permutations pair different tiles in each batch
    dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=workers)

    loc = {b: (mi, ci) for mi, m in enumerate(cfg["modalities"]) for ci, b in enumerate(ds.names[m])}
    units = {b: [b] for b in loc}
    units.update({f"[{g}]": bs_ for g, bs_ in GROUPS.items() if all(b in loc for b in bs_)})

    base = _rmse(model, dl, device)
    reps = repeats if method == "permute" else 1
    gen = torch.Generator().manual_seed(0)
    results = {}
    for name, bands in units.items():
        vals = [_rmse(model, dl, device, [loc[b] for b in bands], method, gen) - base for _ in range(reps)]
        results[name] = {"delta_rmse": sum(vals) / reps, "runs": vals}

    out = {"baseline_rmse": base, "method": method, "split": split, "importance": results}
    (Path(run) / f"importance_{method}.json").write_text(json.dumps(out, indent=2))
    return out


def plot_importance(res, title=""):
    imp = res["importance"]
    names = sorted(imp, key=lambda k: imp[k]["delta_rmse"])
    fig, ax = plt.subplots(figsize=(7, 0.3 * len(names) + 1))
    ax.barh(names, [imp[n]["delta_rmse"] for n in names],
            color=["#999999" if n.startswith("[") else "#2a9d8f" for n in names])
    ax.set_xlabel("increase in RMSE (Mg/ha)")
    ax.set_title(f"{title} {res['method']} importance (groups in grey)".strip())
    fig.tight_layout()
    return fig
