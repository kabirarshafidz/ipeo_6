"""Convert raw tiles into one compact .npz per sample and compute normalization stats.

Expected raw layout (original BioMassters convention). Adapt `find_samples` if the
course extract is organised differently:

    <raw>/train_features/<chip>_S1_<MM>.tif   (256, 256, 4)  float32, dB, -9999 = nodata
    <raw>/train_features/<chip>_S2_<MM>.tif   (256, 256, 11) uint16, band 11 = CLP (0-100, 255 = missing)
    <raw>/train_agbm/<chip>_agbm.tif          (256, 256)     float32, Mg/ha
    <raw>/test_features/..., <raw>/test_agbm/...

A single file per modality (no month suffix) also works; the temporal reduction is then a no-op.

Output:
    <out>/<split>/<chip>.npz with s1 (4,H,W), s2 (11,H,W), agb (H,W), all float32 (NaN = missing)
    <out>/stats.json with per-channel mean/std from the training split
"""
import json
import random
import warnings
from pathlib import Path

import numpy as np
import tifffile
from tqdm.auto import tqdm


def read_chw(path):
    a = np.asarray(tifffile.imread(path), dtype=np.float32)
    if a.ndim == 3 and a.shape[-1] < a.shape[0]:  # HWC -> CHW
        a = a.transpose(2, 0, 1)
    return a


def find_samples(raw, split):
    raw = Path(raw)
    feat, agb = raw / f"{split}_features", raw / f"{split}_agbm"
    if not agb.exists():
        raise FileNotFoundError(f"{agb} not found. Edit find_samples() to match your data layout.")
    samples = []
    for f in sorted(agb.glob("*_agbm.tif")):
        chip = f.name.replace("_agbm.tif", "")
        samples.append({
            "chip": chip,
            "s1": sorted(feat.glob(f"{chip}_S1*.tif")),
            "s2": sorted(feat.glob(f"{chip}_S2*.tif")),
            "agb": f,
        })
    return samples


def composite_s1(paths):
    stack = np.stack([read_chw(p) for p in paths])  # (T,4,H,W)
    stack[stack <= -9999] = np.nan
    return np.nanmedian(stack, axis=0)


def composite_s2(paths, clp_thresh=30):
    stack = np.stack([read_chw(p) for p in paths])  # (T,11,H,W)
    clp = stack[:, 10]
    bad = (clp > clp_thresh) | (clp == 255)
    refl = stack[:, :10].copy()
    refl[np.broadcast_to(bad[:, None], refl.shape)] = np.nan
    out_refl = np.nanmedian(refl, axis=0)
    out_clp = np.nanmean(np.where(clp == 255, np.nan, clp), axis=0)  # mean cloudiness over time
    return np.concatenate([out_refl, out_clp[None]], axis=0)


def prepare(raw, out, splits=("train", "test"), clp_thresh=30, stats_samples=300):
    """Write one .npz per tile and stats.json. Existing .npz files are skipped."""
    out = Path(out)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "All-NaN slice|Mean of empty slice", RuntimeWarning)
        for split in splits:
            out_dir = out / split
            out_dir.mkdir(parents=True, exist_ok=True)
            samples = find_samples(raw, split)
            print(f"{split}: {len(samples)} samples")
            for s in tqdm(samples, desc=split):
                dst = out_dir / f"{s['chip']}.npz"
                if dst.exists():
                    continue
                if not s["s1"] or not s["s2"]:
                    print(f"skip {s['chip']}: missing S1 or S2")
                    continue
                np.savez_compressed(
                    dst,
                    s1=composite_s1(s["s1"]).astype(np.float32),
                    s2=composite_s2(s["s2"], clp_thresh).astype(np.float32),
                    agb=read_chw(s["agb"]).squeeze().astype(np.float32),
                )
    return compute_stats(out, stats_samples)


def compute_stats(out, n_samples=300):
    """Per-channel mean/std from (a subset of) the training split."""
    out = Path(out)
    files = sorted((out / "train").glob("*.npz"))
    random.Random(0).shuffle(files)
    acc = {"s1": [], "s2": []}
    for f in files[:n_samples]:
        d = np.load(f)
        for k in acc:
            acc[k].append(d[k].reshape(d[k].shape[0], -1)[:, ::16])  # subsample pixels
    stats = {}
    for k, v in acc.items():
        v = np.concatenate(v, axis=1)
        stats[k] = {"mean": np.nanmean(v, 1).tolist(), "std": (np.nanstd(v, 1) + 1e-6).tolist()}
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    return stats
