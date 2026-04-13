# Neural Network-Assisted Initial Parameter Estimation for Gravitational Lens Mass Modelling

A machine learning pipeline for predicting initial lens model parameters from quadruply-lensed image configurations, designed to accelerate high-fidelity mass model optimisation for X-ray astrometry.

---

## Overview

Modelling the mass distribution of gravitational lenses requires optimising a parameter space that is both high-dimensional and computationally expensive to evaluate. A poor choice of initial parameters leads to slow convergence or divergence to local minima. This repository provides a complete pipeline that:

1. **Generates** a large synthetic dataset of quadruply-lensed systems across a grid of lens parameters using forward modelling with `lensmodel`.
2. **Extracts** rotationally and scale-invariant geometric features from the four image positions and flux ratios.
3. **Trains** deep neural networks to predict the lens mass parameter and ellipticity directly from these features.
4. **Applies** the trained networks to real observational image data to produce accurate initial parameter estimates.
5. **Optimises** the lens model iteratively using these predictions as a starting point, converging to a high-accuracy fit to the observed image positions.

This approach significantly reduces the number of iterations required by the downstream optimiser by starting in the vicinity of the true solution rather than at an arbitrary point in parameter space.

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────┐
│  generate_training_data.py                   │
│  • Grid over mass (0.1–5.5) × ellipticity   │
│    (0.05–0.60) → 660 lens configurations    │
│  • Place sources on nested astroids inside   │
│    each inner caustic (discrete gridding)    │
│  • Run lensmodel findimg3 → quad image sets  │
│  Output: <name>.parquet                      │
│          rejected_sources.parquet            │
└────────────────────┬────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│  compute_features.py                         │
│  • Order images clockwise from flux maximum  │
│  • Compute geometric invariant features:     │
│    normalised distance ratios, triangle       │
│    areas, flux ratios, intersection-centred  │
│    coordinates (14 features → NN_INPUT_COLS) │
│  Output: <name>_features.parquet             │
└────────────────────┬────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│  train_networks.py                           │
│  • 70 / 20 / 10 % train / test / val split  │
│    (data shuffled before splitting)          │
│  • Separate or combined dense networks with  │
│    dropout regularisation and early stopping │
│  • Predicts: mass parameter, ellipticity     │
│  Output: model_<name>.keras                  │
│          loss_curve_*.png                    │
│          loss_history_*.csv                  │
│          training_metrics_*.csv              │
└────────────────────┬────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│  evaluate_predictions.py                     │
│  • Applies trained networks to full dataset  │
│  • Generates violin plots with 2.5th/97.5th  │
│    percentile bands per parameter value      │
│  Output: <name>_features_predictions.parquet  │
│          mass_violin_plot_*.png              │
│          ellipticity_violin_plot_*.png       │
└─────────────────────────────────────────────┘

  Requires trained model_<name>.keras files ──┐
                                               ▼
┌─────────────────────────────────────────────┐
│  fit_lens_model.py  (standalone)             │
│  • Ingests real observational image data     │
│  • Applies feature extraction pipeline       │
│  • Loads trained models → initial estimates  │
│  • Iteratively optimises via lensmodel using │
│    stochastic binary parameter masking       │
│  • Convergence on χ² < threshold            │
│  Output: Results_*/optimisation_results.csv  │
│          Results_*/<ID>/best.start           │
└─────────────────────────────────────────────┘
```

---

## Repository Structure

```
.
├── pipeline/
│   ├── config.py                         # All user-facing settings (edit here)
│   ├── columns.py                        # Shared column-name schema (import in all scripts)
│   ├── generate_training_data.py
│   ├── compute_features.py
│   ├── train_networks.py
│   ├── evaluate_predictions.py
│   ├── fit_lens_model.py
│   ├── validate_sources.py
│   ├── SourceGridding.dat
│   ├── SourceGridding.input
│   └── SourceGridding.start
├── data/
│   └── observations/
│       └── quad_lens_observations.txt   # Observational image data (arcsec, relative to Image A)
├── lensmodel                  # lensmodel compiled binary
├── run_pipeline.sh            # Runs stages 1–4 (generate → features → train → evaluate)
├── run_fit_lens_model.sh      # Runs fit_lens_model.py with pre-flight checks
├── requirements.txt           # Pinned Python dependencies
├── .gitignore
└── README.md
```

> **`config.py`** is the single place to edit for routine pipeline runs — it holds all overwrite toggles, file names, and mode settings. Every pipeline script imports its user-facing settings from `config.py` rather than defining them locally.
>
> **`columns.py`** is the single source of truth for all shared column-name lists
> (`DIST_RAT_COLS`, `NEW_COORD_COLS`, `NN_INPUT_COLS`, etc.). Every pipeline script
> imports from it rather than redefining lists locally.
>
> Run all scripts from the `pipeline/` directory so both imports resolve correctly.

---

## Pre-Trained Paper Models

The trained neural network weights used to produce the results in the accompanying paper are included in this repository:

```
data/models/model_mass_parameter_paper.keras
data/models/model_ellipticity_paper.keras
```

These are the exact models described and evaluated in the paper. They were trained on the synthetic dataset generated using the discrete astroid gridding method (`USE_PROBABILITY_SAMPLING = False`) across the full 660-configuration parameter grid (mass 0.1–5.5 × ellipticity 0.05–0.60), and accept the 14-feature `NN_INPUT_COLS` vector as input.

### Using the paper models

The pipeline scripts load models by the default filenames (`model_mass_parameter.keras`, `model_ellipticity.keras`). To use the paper models instead of running your own training, update the model name constants at the top of the relevant script:

**`fit_lens_model.py`** (lines 38–39):
```python
MODEL_NAME_MASS  = 'model_mass_parameter_paper.keras'
MODEL_NAME_ELLIP = 'model_ellipticity_paper.keras'
```

**`evaluate_predictions.py`** (constants `MODEL_MASS` / `MODEL_ELLIP`):
```python
MODEL_MASS  = 'model_mass_parameter_paper.keras'
MODEL_ELLIP = 'model_ellipticity_paper.keras'
```

The paper model files use the `_paper` suffix so they are never overwritten by a pipeline training run, which always writes to the default names.

---

## Dependencies

### External Software

| Software | Purpose | Notes |
|----------|---------|-------|
| [`lensmodel`](https://www.physics.rutgers.edu/~keeton/gravlens/manual.pdf) | Gravitational lens forward modelling | Must be compiled and accessible on `PATH` |

### Python Packages

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | 2.4.4 | Numerical array operations |
| `pandas` | 3.0.2 | Tabular data I/O and manipulation |
| `scipy` | 1.17.1 | Optimisation (astroid fitting, ellipse fitting) |
| `tensorflow` | 2.21.0 | Neural network training and inference |
| `matplotlib` | 3.10.8 | Loss curves and violin plots |
| `seaborn` | 0.13.2 | Violin plot styling |
| `tqdm` | 4.67.3 | Progress bars |
| `pyarrow` | 23.0.1 | Parquet file I/O (required by pandas) |

Install into the project virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **GPU support:** TensorFlow GPU acceleration is optional. Set `USE_GPU = True` in `train_networks.py` to enable it.
> - **NVIDIA:** install CUDA + cuDNN, then use the standard `tensorflow` package above.
> - **AMD:** install ROCm, then replace `tensorflow` with `tensorflow-rocm` in `requirements.txt`.
> - **CPU-only / VM:** leave `USE_GPU = False` (default).

---

## Usage

### 1. Generate Synthetic Training Data

```bash
python "pipeline/generate_training_data.py"
```

> **Approximate runtime:** ~5–10 minutes (CPU, no GPU required)

Iterates over all combinations of mass parameter (0.1–5.5, step 0.1) and ellipticity (0.05–0.60, step 0.05), giving 660 lens configurations. For each, `lensmodel` computes the inner caustic, an astroid is fitted to it, and source positions are placed on a sequence of nested astroids shrinking inward from 99.8% of the caustic boundary. The number of sources per astroid scales with its perimeter, and the step size grows by a factor of 1.2 at each shell, stopping when the semi-major axis falls below 1% of the original. Only systems producing exactly four images are retained. Output is written to `<SAVE_FILE_NAME>.parquet`; rejected sources are written to `rejected_sources.parquet`.

> **Source sampling mode (`USE_PROBABILITY_SAMPLING` in `config.py`):**
> The default and paper-validated setting is `False` — the discrete astroid gridding described above. Setting it to `True` switches to an alternative truncated-normal distribution that draws a fixed 10,000 source positions per configuration, concentrated probabilistically near the outer caustic edge rather than uniformly across the interior. The discrete method (`False`) produces the results reported in the accompanying paper and is recommended.

### 2. Extract Geometric Features

```bash
python "pipeline/compute_features.py"
```

> **Approximate runtime:** ~10–15 seconds (CPU, no GPU required)

Reads the synthetic dataset and computes rotationally and scale-invariant features per system, including normalised pairwise distance ratios, triangle areas formed by image triplets, normalised flux ratios, and coordinates recentred on the diagonal intersection point (the 14 features in `NN_INPUT_COLS`). Output is written to `<FILE_NAME>_features.parquet`.

### 3. Train Neural Networks

```bash
python "pipeline/train_networks.py"
```

> **Approximate runtime:** ~1–2 hours on CPU; significantly faster with a GPU

Trains neural networks using the feature dataset, with data shuffled before the 70/20/10% train/test/val split. The models trained depend on `TRAINING_MODE` (see [User Configuration](#user-configuration)). Training employs early stopping (patience = 100 epochs) and Adam optimisation. Per model, the following are saved: `model_<name>.keras`, `model_summary_<name>.txt`, `model_config_<name>.json`, `loss_curve_<name>.png`, `loss_history_<name>.csv`, and `training_metrics_<name>.csv`.

### 4. Validate Predictions

```bash
python "pipeline/evaluate_predictions.py"
```

> **Approximate runtime:** ~1–2 minutes (CPU, no GPU required)

Applies the trained networks to the dataset and generates violin plots showing the prediction distribution as a function of each lens parameter, with 2.5th and 97.5th percentile bands overlaid. The models used depend on `PREDICTION_MODE` (see [User Configuration](#user-configuration)). Output is written to `<name>_features_predictions.parquet` (or `<name>_features_predictions_combined.parquet` in combined mode), `mass_violin_plot_*.png`, and `ellipticity_violin_plot_*.png`.

### 5. Apply to Observational Data and Optimise

```bash
bash run_fit_lens_model.sh
# or directly:
python "pipeline/fit_lens_model.py"
```

Reads observational image data from `data/observations/quad_lens_observations.txt`, applies the feature extraction pipeline, uses the trained networks to predict initial mass and ellipticity, then runs the iterative `lensmodel` χ²-minimisation loop until convergence. Output is written to `data/observations/Results_quad_lens_observations/optimisation_results.csv`.

#### Observational data file format

The input file `quad_lens_observations.txt` contains one block per lensed system:

```
#<system_name>
RA  Dec  ErrorRA  ErrorDec  Flux  FluxError
<image A values>
<image B values>
<image C values>
<image D values>
```

- **Coordinates** are in **arcseconds**, relative to Image A (which is set to RA = 0, Dec = 0). The image with the highest flux should be placed first (Image A); the remaining three follow in clockwise order.
- **Errors** (ErrorRA, ErrorDec) are positional uncertainties in arcseconds.
- **Flux** is in counts or any self-consistent unit; only flux ratios are used internally.
- **FluxError** is the uncertainty on flux; set to a large value (e.g. `100000`) to effectively ignore flux errors during optimisation.
- Lines beginning with `#` are treated as system name labels; lines beginning with `#` followed by a header row (RA, Dec, …) are skipped automatically.

`INPUT_COORDINATE_UNITS` in `fit_lens_model.py` must be set to `'arcseconds'` to match this file. If absolute sky coordinates in degrees are provided instead, set it to `'degrees'` and the script will convert internally.

---

## User Configuration

All user-facing settings are collected in **`pipeline/config.py`** — this is the only file that needs to be edited for routine runs. No other pipeline script defines these values; they are imported from `config.py` at runtime.

### Output file names

`SAVE_FILE_NAME` and `FILE_NAME` in `config.py` control what files are read and written. Change them when you want to produce a new dataset or model run alongside an existing one rather than overwriting it. `FILE_NAME` must match `SAVE_FILE_NAME` so that downstream scripts find the features file automatically.

| Variable in `config.py` | Default | Controls |
|--------------------------|---------|---------|
| `SAVE_FILE_NAME` | `"quad_lens_sources"` | `generate_training_data.py` writes `<name>.parquet` and `rejected_sources.parquet` in `data/sources/` |
| `FILE_NAME` | `"quad_lens_sources"` | `compute_features.py` reads `<name>.parquet`, writes `<name>_features.parquet`; `train_networks.py` and `evaluate_predictions.py` read `<name>_features.parquet` from `data/sources/` |

> The features file name is derived automatically by appending `_features` to `FILE_NAME`. If you change `SAVE_FILE_NAME`, update `FILE_NAME` in `config.py` to match.

### Overwrite protection

By default, **no script will overwrite an existing output file**. If the expected output already exists, the script prints a message and exits without doing any work. This protects against accidentally discarding a long-running generation or training run.

To regenerate an output, you have two options:

1. **Set the overwrite toggle to `True` in `config.py`** — the existing file will be overwritten when the script runs.
2. **Rename or move the existing file** (recommended when you want to keep the old result).

| Toggle in `config.py` | Default | Output protected |
|-----------------------|---------|-----------------|
| `OVERWRITE_SOURCES` | `False` | `<SAVE_FILE_NAME>.parquet` |
| `OVERWRITE_FEATURES` | `False` | `<FILE_NAME>_features.parquet` |
| `OVERWRITE_TRAINING` | `False` | `model_*.keras` (checked per model) |
| `OVERWRITE_PREDICTIONS` | `False` | `<FILE_NAME>_features_predictions.parquet` |

### Source sampling method

`USE_PROBABILITY_SAMPLING` in `config.py` controls how source positions are distributed inside each inner caustic:

| `USE_PROBABILITY_SAMPLING` | Method | Paper results |
|---|---|---|
| `False` (default) | **Discrete astroid gridding** — sources placed on nested astroids shrinking inward from 99.8% of the caustic. Source density scales with astroid perimeter; step size grows by ×1.2 per shell, stopping at 1% of the original semi-major axis. Produces variable source counts per configuration (typically 7,000–10,000). | **Yes — matches published results** |
| `True` | **Probability sampling** — draws a fixed 10,000 positions per configuration from a truncated normal distribution, concentrating sources near the outer caustic edge rather than across the full interior. | No |

The discrete method is the default and reproduces the training data described in the paper. The probability sampling option is provided as an alternative if uniform source counts per configuration are preferred.

### Training and prediction mode

`TRAINING_MODE` and `PREDICTION_MODE` in `config.py` must be set consistently — training in one mode and evaluating in the other will fail at model load time:

| Mode | Value in `config.py` | Models trained / used |
|------|----------------------|-----------------------|
| Separate | `'separate'` (default) | `model_mass_parameter.keras` + `model_ellipticity.keras` |
| Combined | `'combined'` | `model_mass_ellipticity.keras` |

In `'combined'` mode, `evaluate_predictions.py` writes its output to `<FILE_NAME>_features_predictions_combined.parquet` so the two modes never overwrite each other.

---

## Configuration Files

The `SourceGridding.*` files define the lens model template used throughout the data generation and optimisation stages.

| File | Description |
|------|-------------|
| `SourceGridding.input` | `lensmodel` command file — sets grid mode, chi-squared mode, numerical tolerance, and calls `findimg3` |
| `SourceGridding.dat` | Observational data specification passed to `lensmodel` |
| `SourceGridding.start` | Initial lens model parameters and optimisation flags (1 = free, 0 = fixed) |

The lens model uses a power-law mass profile (`alpha`) parameterised by: mass parameter, lens centre (x, y), ellipticity, ellipticity position angle, external shear, and shear angle.

---

## Lens Model Parameterisation

The mass model follows a singular power-law ellipsoid with external shear, consistent with the standard `lensmodel` `alpha` model:

| Parameter | Symbol | Range (training grid) |
|-----------|--------|----------------------|
| Mass parameter | *b* | 0.1 – 5.5 |
| Ellipticity | *e* | 0.05 – 0.60 |
| Ellipticity position angle | *θ_e* | 0 (fixed) |
| External shear | *γ* | 0 (fixed) |
| Shear angle | *θ_γ* | 0 (fixed) |

---

## Feature Engineering

`compute_features.py` derives the following columns from the four image positions and flux ratios. All features are designed to be invariant to rotation and overall scale, so the networks generalise across different observing orientations.

**Geometric features stored in `_features.parquet`:**

- **Normalised distance ratios** (6): pairwise separations AB, AC, AD, BC, BD, CD, each divided by their sum — `DIST_RAT_COLS`
- **Adjacent distance sums** (2): AB + DA, BC + CD
- **Triangle areas** (4): areas of triangles ABD, ABC, BCD, ACD via Heron's formula
- **Normalised flux ratios** (4): flux of each image divided by total flux
- **Intersection-centred coordinates** (8): image positions translated to the intersection of diagonals AC and BD, stored as paired (RA, Dec) per image — `NEW_COORD_COLS`

**Neural network input (`NN_INPUT_COLS`, 14 features total):**

The networks receive `DIST_RAT_COLS + NEW_COORD_COLS` — the 6 normalised distance ratios and the 8 intersection-centred coordinate components. All other derived columns are available in the parquet file but are not fed to the networks.

> **Column ordering note:** `NEW_COORD_COLS` uses paired (RA, Dec) ordering — `[New A RA, New A Dec, New B RA, New B Dec, ...]` — matching the `_features.parquet` column layout. Training and prediction both import this list from `columns.py`; they must never be defined separately.

---

## Method Summary

A synthetic dataset of approximately 6 million quadruply-lensed systems is generated by forward modelling across a grid of 660 lens configurations (mass 0.1–5.5 arcsec × ellipticity 0.05–0.60). Source positions are distributed inside each inner caustic using a discrete astroid gridding scheme, placing sources on nested astroids with density proportional to perimeter and step size growing geometrically inward. Geometric features invariant to rotation and scale are extracted from each image configuration. Feed-forward neural networks — either separate models for mass and ellipticity, or a single combined model — are trained to predict the lens parameters from these features. When applied to real observations, the network predictions provide initial parameter estimates sufficiently close to the true solution to enable rapid convergence of the `lensmodel` χ²-minimisation optimiser, bypassing the need for exhaustive grid searches or manual initialisation.

---

## Citation

If you use this code in your research, please cite the accompanying paper:

```bibtex
@article{TBD,
  author  = {},
  title   = {},
  journal = {},
  year    = {},
  volume  = {},
  pages   = {},
  doi     = {}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Contact

For questions regarding the code or methodology, please open an issue on this repository.
