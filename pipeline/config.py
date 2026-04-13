"""
Pipeline configuration — edit this file to control pipeline behaviour.

All user-facing settings are collected here. No other pipeline script
needs to be edited for routine runs. Settings are imported by each script
at startup; values here are the single source of truth.
"""

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
# Training / prediction mode  (train_networks.py + evaluate_predictions.py)
# =============================================================================
# 'separate' (default): train/use individual mass and ellipticity models.
# 'combined':           train/use a single two-output model.
# Must be set consistently in both train_networks.py and evaluate_predictions.py.
TRAINING_MODE   = 'separate'
PREDICTION_MODE = 'separate'
