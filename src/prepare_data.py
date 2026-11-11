"""Convert raw tiles into one compact .npz per sample and compute normalization stats.

Raw layout (course extract of BioMassters, one acquisition per tile):

    <raw>/AGB_training_set/metadata.csv            chip_id, s1_file, s2_file, agb_file, mean_agb
    <raw>/AGB_training_set/<chip>/S1/<chip>_S1_10.tif   (256, 256, 4)  float32 dB, -9999 = missing orbit
    <raw>/AGB_training_set/<chip>/S2/<chip>_S2_10.tif   (256, 256, 11) uint16, band 11 = CLP (0-100, 255 = missing)
    <raw>/AGB_training_set/<chip>/AGBM/<chip>_agbm.tif  (256, 256)     float32, Mg/ha
    <raw>/AGB_test_set/...  (same structure)

Missing values are stored as NaN:
    S1: -9999 (about 1 in 9 tiles lacks the ASC or DSC orbit)
    S2: pixels where all reflectance bands are 0; CLP == 255
    optional: S2 reflectance where CLP > clp_thresh (mask_clouds=True)

Output:
    <out>/<split>/<chip>.npz with s1 (4,H,W), s2 (11,H,W), agb (H,W), all float32
    <out>/<split>/tiles.csv  per-tile data-quality summary
    <out>/stats.json         per-channel mean/std from the training split
"""
import csv
import json
from pathlib import Path

import numpy as np
import tifffile
from tqdm.auto import tqdm

SPLIT_DIRS = {"train": "AGB_training_set", "test": "AGB_test_set"}


def read_chw(path):
    a = np.asarray(tifffile.imread(path), dtype=np.float32)
    if a.ndim == 3 and a.shape[-1] < a.shape[0]:  # HWC -> CHW
        a = a.transpose(2, 0, 1)
    return a


def find_samples(raw, split):
    root = Path(raw) / SPLIT_DIRS[split]
    with open(root / "metadata.csv") as f:
        return [{"chip": r["chip_id"], "s1": root / r["s1_file"], "s2": root / r["s2_file"],
                 "agb": root / r["agb_file"]} for r in csv.DictReader(f)]


def load_s1(path):
    s1 = read_chw(path)
    s1[s1 <= -9999] = np.nan
    return s1


def load_s2(path, mask_clouds=False, clp_thresh=50):
    s2 = read_chw(path)
    refl, clp = s2[:10], s2[10]
    refl[:, (refl == 0).all(0)] = np.nan
    clp[clp == 255] = np.nan
    if mask_clouds:
        refl[:, clp > clp_thresh] = np.nan
    return s2


def prepare(raw, out, splits=("train", "test"), mask_clouds=False, clp_thresh=50, overwrite=False):
    """Write one .npz per tile, a tiles.csv per split, and stats.json."""
    out = Path(out)
    for split in splits:
        out_dir = out / split
        out_dir.mkdir(parents=True, exist_ok=True)
        samples = find_samples(raw, split)
        qc = []
        for s in tqdm(samples, desc=split):
            dst = out_dir / f"{s['chip']}.npz"
            if dst.exists() and not overwrite:
                d = np.load(dst)
                s1, s2, agb = d["s1"], d["s2"], d["agb"]
            else:
                s1, s2 = load_s1(s["s1"]), load_s2(s["s2"], mask_clouds, clp_thresh)
                agb = read_chw(s["agb"]).squeeze()
                np.savez_compressed(dst, s1=s1, s2=s2, agb=agb)
            qc.append({
                "chip": s["chip"],
                "asc_missing": float(np.isnan(s1[0]).mean()),
                "dsc_missing": float(np.isnan(s1[2]).mean()),
                "s2_missing": float(np.isnan(s2[0]).mean()),
                "clp_mean": float(np.nanmean(s2[10])) if np.isfinite(s2[10]).any() else np.nan,
                "agb_mean": float(agb.mean()),
                "agb_max": float(agb.max()),
            })
        with open(out_dir / "tiles.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(qc[0]))
            w.writeheader()
            w.writerows(qc)
        print(f"{split}: {len(samples)} tiles -> {out_dir}")
    return compute_stats(out)


def compute_stats(out):
    """Per-channel mean/std over all training tiles (pixel-subsampled, NaN ignored)."""
    out = Path(out)
    acc = {"s1": [], "s2": []}
    for f in sorted((out / "train").glob("*.npz")):
        d = np.load(f)
        for k in acc:
            acc[k].append(d[k].reshape(d[k].shape[0], -1)[:, ::16])
    stats = {}
    for k, v in acc.items():
        v = np.concatenate(v, axis=1)
        stats[k] = {"mean": np.nanmean(v, 1).tolist(), "std": (np.nanstd(v, 1) + 1e-6).tolist()}
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    return stats
