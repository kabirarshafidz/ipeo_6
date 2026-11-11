import numpy as np
import torch

BINS = [0, 50, 100, 150, 200, 300, np.inf]  # Mg/ha


class RegressionMetrics:
    """Streaming pixel-level metrics in Mg/ha."""

    def __init__(self):
        self.n = 0
        self.se = self.ae = self.err = self.sy = self.sy2 = 0.0
        self.bin_se = np.zeros(len(BINS) - 1)
        self.bin_n = np.zeros(len(BINS) - 1)

    @torch.no_grad()
    def update(self, pred, target, valid):
        p = pred[valid].double().cpu().numpy()
        t = target[valid].double().cpu().numpy()
        e = p - t
        self.n += t.size
        self.se += (e**2).sum()
        self.ae += np.abs(e).sum()
        self.err += e.sum()
        self.sy += t.sum()
        self.sy2 += (t**2).sum()
        idx = np.digitize(t, BINS) - 1
        np.add.at(self.bin_se, idx, e**2)
        np.add.at(self.bin_n, idx, 1)

    def compute(self):
        n = max(self.n, 1)
        var_t = self.sy2 / n - (self.sy / n) ** 2
        out = {
            "rmse": float(np.sqrt(self.se / n)),
            "mae": float(self.ae / n),
            "bias": float(self.err / n),
            "r2": float(1 - (self.se / n) / max(var_t, 1e-12)),
        }
        for i in range(len(BINS) - 1):
            if self.bin_n[i] > 0:
                out[f"rmse_{BINS[i]:g}-{BINS[i+1]:g}"] = float(np.sqrt(self.bin_se[i] / self.bin_n[i]))
        return out
