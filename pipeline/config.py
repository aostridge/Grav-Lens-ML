"""
Pipeline configuration — edit this file to control pipeline behaviour.

All user-facing settings are collected here. No other pipeline script
needs to be edited for routine runs. Settings are imported by each script
at startup; values here are the single source of truth.
"""

from pathlib import Path

# =============================================================================
# Filesystem paths
# =============================================================================
# PROJECT_ROOT is derived from this file's location (pipeline/config.py) so
# the pipeline works immediately after a git clone, with no path editing.
# Every other script imports its paths from here rather than hardcoding them.
PROJECT_ROOT     = Path(__file__).resolve().parent.parent
PIPELINE_DIR     = PROJECT_ROOT / 'pipeline'      # lens_model.* templates
DATA_DIR         = PROJECT_ROOT / 'data'
SOURCES_DIR      = DATA_DIR / 'sources'           # generate_training_data.py, compute_features.py
MODELS_DIR       = DATA_DIR / 'models'            # train_networks.py, evaluate_predictions.py, fit_lens_model.py
OBSERVATIONS_DIR = DATA_DIR / 'observations'      # fit_lens_model.py
LENSMODEL_PATH   = PROJECT_ROOT / 'lensmodel'     # gravlens lensmodel binary

# =============================================================================
# Output file names
# =============================================================================
# Base name written by generate_training_data.py.
# compute_features.py reads <SAVE_FILE_NAME>.parquet and writes
# <SAVE_FILE_NAME>_features.parquet. FILE_NAME below must match SAVE_FILE_NAME
# so that downstream scripts find the correct files automatically.
SAVE_FILE_NAME = "quad_lens_sources"
FILE_NAME      = "quad_lens_sources"

# =============================================================================
# Overwrite protection
# =============================================================================
# By default no script overwrites an existing output. Set a toggle to True
# to force regeneration, or rename/move the existing file instead.
OVERWRITE_SOURCES     = False   # generate_training_data.py → <SAVE_FILE_NAME>.parquet
OVERWRITE_FEATURES    = False    # compute_features.py       → <FILE_NAME>_features.parquet
OVERWRITE_TRAINING    = False    # train_networks.py         → model_*.keras
OVERWRITE_PREDICTIONS = False    # evaluate_predictions.py   → *_predictions.parquet

# =============================================================================
# Source sampling method  (generate_training_data.py)
# =============================================================================
# False (default, paper-validated): discrete astroid gridding — sources placed
#   on nested astroids shrinking inward; density ∝ perimeter; step ×1.2/shell.
# True: truncated-normal probability sampling — fixed 10 000 positions per
#   configuration, concentrated near the outer caustic edge.
USE_PROBABILITY_SAMPLING = False

# =============================================================================
# Training mode  (train_networks.py)
# =============================================================================
# 'separate' (default, paper-validated): train individual mass and ellipticity
#   models — the architecture reported in the paper (Table 1).
# 'combined': train a single two-output model — retained for comparison; the
#   paper found this configuration less accurate (Table 1) and did not use it
#   for the reported results.
TRAINING_MODE = 'separate'

# =============================================================================
# Prediction mode  (evaluate_predictions.py)
# =============================================================================
# Must match TRAINING_MODE for the models you intend to evaluate — evaluating
# in 'combined' mode requires a model trained with TRAINING_MODE = 'combined',
# and vice versa.
# 'separate' (default): load model_mass_parameter*.keras + model_ellipticity*.keras.
# 'combined':            load model_mass_ellipticity*.keras.
PREDICTION_MODE = 'separate'
