import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from bands import S1_BANDS, S2_BANDS

AGB_SCALE = 100.0  # model predicts AGB / AGB_SCALE
AGB_MAX = 500.0    # reference pixels above this (Mg/ha) are ignored in loss and metrics (~0.02% of pixels)


class BiomassDataset(Dataset):
    """Returns (inputs, target, valid_mask).

    inputs: list of tensors, one per modality in `modalities` (e.g. ["s1", "s2"]),
            each (C,H,W), z-scored, restricted to `bands`.
    target: (1,H,W) AGB / AGB_SCALE.
    """

    def __init__(self, files, stats, modalities, bands, augment=False):
        self.files = list(files)
        self.modalities = modalities
        self.augment = augment
        all_names = {"s1": S1_BANDS, "s2": S2_BANDS}
        self.idx, self.names, self.mean, self.std = {}, {}, {}, {}
        for m in modalities:
            keep = [i for i, b in enumerate(all_names[m]) if b in bands]
            self.idx[m] = keep
            self.names[m] = [all_names[m][i] for i in keep]
            self.mean[m] = np.array(stats[m]["mean"], np.float32)[keep, None, None]
            self.std[m] = np.array(stats[m]["std"], np.float32)[keep, None, None]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        d = np.load(self.files[i])
        xs = []
        for m in self.modalities:
            x = (d[m][self.idx[m]] - self.mean[m]) / self.std[m]
            xs.append(np.nan_to_num(x, nan=0.0))
        agb = d["agb"][None]
        # a few reference pixels reach thousands of Mg/ha (LiDAR artifacts): exclude them
        valid = np.isfinite(agb) & (agb <= AGB_MAX)
        y = agb / AGB_SCALE
        y = np.nan_to_num(y, nan=0.0)

        if self.augment:
            k = random.randint(0, 3)
            flip = random.random() < 0.5
            def aug(a):
                a = np.rot90(a, k, axes=(-2, -1))
                return a[..., ::-1] if flip else a
            xs, y, valid = [aug(x) for x in xs], aug(y), aug(valid)

        to_t = lambda a: torch.from_numpy(np.ascontiguousarray(a))
        return [to_t(x).float() for x in xs], to_t(y).float(), to_t(valid)


def split_files(data_dir, val_frac=0.1, seed=0):
    data_dir = Path(data_dir)
    train = sorted((data_dir / "train").glob("*.npz"))
    random.Random(seed).shuffle(train)
    n_val = max(1, int(len(train) * val_frac))
    test = sorted((data_dir / "test").glob("*.npz"))
    return train[n_val:], train[:n_val], test


def load_stats(data_dir):
    return json.loads((Path(data_dir) / "stats.json").read_text())
