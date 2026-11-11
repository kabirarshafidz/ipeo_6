# Above-ground forest biomass estimation from multisensor satellite imagery

ENV-540 Image Processing for Earth Observation, Fall 2026, Project 2.

## Task

Estimate above-ground biomass (AGB, Mg/ha) per pixel from Sentinel-1 SAR and Sentinel-2 multispectral imagery, then work out **which sensors and bands drive the predictions** and whether that agrees with the physics of each sensor.

Data come from the BioMassters benchmark (Nascetti et al., NeurIPS 2023) and cover forests in Finland. Each sample is a 2.56 × 2.56 km tile on a 256 × 256 grid at 10 m resolution.

| Source | Channels | Measures |
|---|---|---|
| Sentinel-1 (C-band SAR) | `ASC_VV`, `ASC_VH`, `DSC_VV`, `DSC_VH` | Backscatter. VH (cross-pol) responds to volume scattering in the canopy; VV carries more surface/soil signal. C-band saturates at moderate biomass. |
| Sentinel-2 (MSI) | `B2 B3 B4` (visible), `B5 B6 B7` (red edge), `B8 B8A` (NIR), `B11 B12` (SWIR) | Surface reflectance: chlorophyll, leaf structure, canopy water and shadowing. Sees the top of the canopy only. |
| Sentinel-2 | `CLP` | Cloud probability (data-quality layer, not a vegetation signal). |
| Target | `AGB` | Biomass from airborne LiDAR plus field plots. |

## Expected outcomes

1. Regression models for **S1 only**, **S2 only**, and **S1 + S2**, comparing fusion strategies (early fusion vs. mid fusion with one encoder per sensor).
2. Test-set evaluation with RMSE, MAE, R², bias, plus error broken down by biomass range (to show saturation).
3. Contribution analysis per band and per group of bands, with the method chosen and justified.
4. Discussion that ties the contributions to sensor physics, and covers complementarity, limitations and failure modes.

## Repository layout

```
README.md
requirements.txt
biomass.ipynb       main notebook: runs every experiment, figure and table
src/                helper functions imported by the notebook
  bands.py          channel names and physically meaningful band groups
  prepare_data.py   prepare(): raw tiles -> one .npz per sample, quality table, normalization stats
  dataset.py        PyTorch Dataset (normalization, band selection, augmentation)
  models.py         U-Net (single input / early fusion) and two-encoder mid-fusion U-Net
  metrics.py        streaming RMSE / MAE / R2 / bias, per-AGB-bin RMSE
  train.py          make_cfg(), train_model()
  evaluate.py       load_run(), evaluate_run(), plot_predictions()
  importance.py     compute_importance(), plot_importance()
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab biomass.ipynb
```

1. The data go in `data/raw/AGB_training_set` and `data/raw/AGB_test_set`.
2. Run the notebook with `DEBUG = True` first. That trains tiny models for 2 epochs to check the pipeline end to end.
3. Set `DEBUG = False` for the real runs (a GPU is recommended).

Configurations are plain keyword arguments to `make_cfg`, for example:

```python
make_cfg(DATA, name="s1s2_mid_noSWIR", modality="s1s2", fusion="mid", drop=["SWIR"])
```

`drop` takes band names or group names from `src/bands.py` (for example `VH`, `RedEdge`, `CLP`, `ASC`).

## Methodology notes

**Data.** The extract is already split: `data/raw/AGB_training_set` (240 tiles) and `data/raw/AGB_test_set` (60 tiles), each with a `metadata.csv`. Every tile has one acquisition (month 10) and its own `S1/`, `S2/` and `AGBM/` folder. 24 training tiles are held out for validation.

**Data quality** (written to `data/processed/<split>/tiles.csv`):
- S1: about 1 in 9 tiles has the ASC or DSC orbit missing (-9999).
- S2: a few tiles are mostly empty (all bands 0) or have CLP = 255.
- AGB: a handful of reference pixels reach thousands of Mg/ha (the 99.99th percentile is about 550). Pixels above 500 Mg/ha are excluded from the loss and metrics (`AGB_MAX` in `dataset.py`).

Missing inputs are stored as NaN and fall back to the channel mean after normalization. Cloudy S2 pixels are kept by default, since with one date there is nothing to replace them with. `prepare(..., mask_clouds=True)` blanks them instead.

**Normalization.** Per-channel z-score using training-set statistics. S1 is already in dB. Missing values (S1 nodata, fully clouded S2 pixels) are set to 0 after normalization, which is the channel mean.

**Loss.** MSE by default (it optimizes RMSE directly). Huber and L1 are available. The target is divided by a fixed scale (100 Mg/ha) for numerical stability, and metrics are reported in Mg/ha.

**Metrics.** RMSE (the BioMassters headline metric), MAE, R², mean bias, and RMSE per AGB bin. The binned RMSE is key for showing S1 saturation and the underestimation of high-biomass stands.

**Contribution analysis.** Two complementary approaches:
- *Inference-time perturbation* (`compute_importance`): permute a band or group across test tiles, or replace it with its mean, and measure the increase in RMSE. It is cheap and needs no retraining, but correlated bands can stand in for each other, so single-band scores understate the importance of redundant bands. Group-level scores reduce this effect.
- *Retraining ablation* (`make_cfg(..., drop=[...])`): measures how much performance the model *cannot recover* without a group. This is more expensive but more trustworthy when bands are correlated.

Comparing the two tells you whether a band carries unique information or is only one of several redundant carriers. A possible extension is gradient-based XAI (Integrated Gradients through Captum) to get spatial attribution maps.

**Physical hypotheses to test.**
- S2 NIR, red edge and SWIR should matter more than visible bands (canopy structure, leaf water, shadowing).
- S1 VH should matter more than VV (volume scattering).
- ASC vs. DSC should be largely redundant (same physics, different geometry).
- CLP should carry little information. If it scores high, look for leakage or a dataset artifact.
- S1+S2 should help mostly in the high-AGB range where both sensors saturate differently.

## Things to watch for

- Saturation: both C-band SAR and optical reflectance lose sensitivity at high biomass, so expect regression to the mean and underestimation of dense stands.
- Clouds and shadows in the single-date S2 image, and missing S1 orbits.
- Spatial autocorrelation: keep train, val and test tiles separate (the provided split is tile-based).
- Small dataset (240 training tiles): expect variance between runs, so repeat key experiments with different seeds.

## References

Nascetti, A., Yadav, R., Brodt, K., Qu, Q., Fan, H., Shendryk, Y., Shah, I., & Chung, C. (2023). *BioMassters: A benchmark dataset for forest biomass estimation using multi-modal satellite time-series.* NeurIPS Datasets and Benchmarks.
