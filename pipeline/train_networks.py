import io
import json
import os
import logging

# =============================================================================
# GPU configuration — must be set before TensorFlow is imported.
#
# USE_GPU = True  — use GPU if available (NVIDIA via CUDA, or AMD via ROCm)
# USE_GPU = False — force CPU-only training regardless of available hardware
#
# For NVIDIA: install CUDA + cuDNN, then use the standard `tensorflow` package.
# For AMD:    install ROCm, then replace `tensorflow` with `tensorflow-rocm`.
# In a VM:    GPU is not accessible without PCIe passthrough — set False.
# =============================================================================
USE_GPU = False  # change to True if a supported GPU and drivers are available

if not USE_GPU:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

logging.getLogger("tensorflow").setLevel(logging.ERROR)

import tensorflow as tf
import numpy as np
import pandas as pd
import random
import time
from tqdm import tqdm
from pathlib import Path
from matplotlib import pyplot as plt
from columns import DIST_RAT_COLS, NEW_COORD_COLS, NN_INPUT_COLS, NN_MASS_OUTPUT_COLS, NN_ELLIP_OUTPUT_COLS
from config import FILE_NAME, OVERWRITE_TRAINING, TRAINING_MODE

# =============================================================================
# Constants
# =============================================================================
MASS_LAYER_SIZES  = [100, 40]
ELLIP_LAYER_SIZES = [100, 50]
BOTH_LAYER_SIZES  = [100, 50]
DROPOUT_RATE = 0.2

TRAIN_FRACTION = 0.7
TEST_FRACTION  = 0.9

EARLY_STOPPING_PATIENCE  = 100
EARLY_STOPPING_MIN_DELTA = 1e-5
BATCH_SIZE = 10_000
MAX_EPOCHS = 1_000_000
EVAL_SAMPLES = 10_000

# =============================================================================
# Paths
# =============================================================================
FILE_PATH   = Path('/home/alex/Documents/Grav Lens/data')
SOURCE_PATH = 'sources'
MODEL_PATH  = 'models'

# =============================================================================
# Column definitions  (imported from columns.py — single source of truth)
# DIST_RAT_COLS, NEW_COORD_COLS, and NN_INPUT_COLS are defined centrally to
# guarantee the same column ordering is used in both training and prediction.
# NOTE: NEW_COORD_COLS uses paired RA/Dec order (A RA, A Dec, B RA, B Dec, ...)
# matching compute_features.py output — do NOT use RA-grouped ordering.
# =============================================================================

# =============================================================================
# Training configuration
# =============================================================================
MODEL_MASS_NAME  = 'mass_parameter'
MODEL_ELLIP_NAME = 'ellipticity'
MODEL_BOTH_NAME  = 'mass_ellipticity'

input_columns = NN_INPUT_COLS

save_model            = True
save_training_results = True
shuffle_data          = True

# User configuration — edit config.py
# FILE_NAME, OVERWRITE_TRAINING, and TRAINING_MODE are imported from config.py.

# =============================================================================
# Load data once — all output columns read up front so the training loop can
# reuse the same split without reloading from disk for each model type.
# =============================================================================
columns_to_read = input_columns + NN_MASS_OUTPUT_COLS + NN_ELLIP_OUTPUT_COLS
data = pd.read_parquet(
    FILE_PATH / SOURCE_PATH / f'{FILE_NAME}_features.parquet',
    columns=columns_to_read,
)

if shuffle_data:
    shuffled_data = data.sample(frac=1, random_state=1234).reset_index(drop=True)
else:
    shuffled_data = data

n_samples       = len(shuffled_data)
train_split_idx = int(TRAIN_FRACTION * n_samples)
test_split_idx  = int(TEST_FRACTION  * n_samples)

train_df = shuffled_data[:train_split_idx]
test_df  = shuffled_data[train_split_idx:test_split_idx]
val_df   = shuffled_data[test_split_idx:]

# =============================================================================
# Directory setup  (exist_ok — safe to call once before the loop)
# =============================================================================
model_save_dir = FILE_PATH / MODEL_PATH
model_save_dir.mkdir(parents=True, exist_ok=True)
(model_save_dir / 'plots').mkdir(exist_ok=True)

# =============================================================================
# Model definition
# =============================================================================

def get_uncompiled_model(layer_sizes, dropout, outputs):
    """Build a sequential dense network with dropout regularisation.

    Args:
        layer_sizes: List of neuron counts per hidden layer.
        dropout: Dropout fraction applied after each hidden layer.
        outputs: List of output column names (determines output layer size).

    Returns:
        Uncompiled tf.keras.Sequential model.
    """
    model = tf.keras.models.Sequential()
    for size in layer_sizes:
        model.add(tf.keras.layers.Dense(size, activation='relu'))
        model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(len(outputs)))
    return model


def get_compiled_model(layer_sizes, dropout, outputs):
    """Build and compile the model with Adam optimiser and MSE loss.

    Args:
        layer_sizes: List of neuron counts per hidden layer.
        dropout: Dropout fraction.
        outputs: List of output column names.

    Returns:
        Compiled tf.keras.Sequential model.
    """
    model = get_uncompiled_model(layer_sizes, dropout, outputs)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
    return model


# =============================================================================
# Training loop — trains models according to TRAINING_MODE
# =============================================================================
if TRAINING_MODE == 'separate':
    _training_targets = ['Mass', 'Ellip']
elif TRAINING_MODE == 'combined':
    _training_targets = ['Both']
else:
    raise ValueError(f"TRAINING_MODE must be 'separate' or 'combined', got {TRAINING_MODE!r}")

for train_for in _training_targets:

    if train_for == 'Mass':
        target_cols = NN_MASS_OUTPUT_COLS
        model_name  = MODEL_MASS_NAME
        layer_sizes = MASS_LAYER_SIZES
    elif train_for == 'Ellip':
        target_cols = NN_ELLIP_OUTPUT_COLS
        model_name  = MODEL_ELLIP_NAME
        layer_sizes = ELLIP_LAYER_SIZES
    else:
        target_cols = NN_MASS_OUTPUT_COLS + NN_ELLIP_OUTPUT_COLS
        model_name  = MODEL_BOTH_NAME
        layer_sizes = BOTH_LAYER_SIZES

    if not OVERWRITE_TRAINING and (model_save_dir / f'model_{model_name}.keras').exists():
        print(
            f"  Skipping: model_{model_name}.keras already exists. "
            "Set OVERWRITE_TRAINING = True in config.py, or rename the file, to retrain."
        )
        continue

    print(f"\n{'='*60}")
    print(f"  Training: {model_name}")
    print(f"{'='*60}")

    # ── Prepare arrays ────────────────────────────────────────────────────────
    # target_col_idx recomputed each iteration as target_cols changes.
    input_col_idx  = [shuffled_data.columns.get_loc(col) for col in input_columns]
    target_col_idx = [shuffled_data.columns.get_loc(col) for col in target_cols]

    X_train = train_df.iloc[:, input_col_idx].to_numpy()
    Y_train = train_df.iloc[:, target_col_idx].to_numpy()

    X_test = test_df.iloc[:, input_col_idx].to_numpy()
    Y_test = test_df.iloc[:, target_col_idx].to_numpy()

    X_val = val_df.iloc[:, input_col_idx].to_numpy()
    Y_val = val_df.iloc[:, target_col_idx].to_numpy()

    # Fresh callbacks each iteration — ensures EarlyStopping patience resets.
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=EARLY_STOPPING_MIN_DELTA,
            patience=EARLY_STOPPING_PATIENCE,
            verbose=0,
        )
    ]

    # ── Train ─────────────────────────────────────────────────────────────────
    nn_model = get_compiled_model(layer_sizes, DROPOUT_RATE, target_col_idx)

    history = nn_model.fit(
        X_train, Y_train,
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, Y_val),
        callbacks=callbacks,
    )

    if save_model:
        nn_model.save(model_save_dir / f'model_{model_name}.keras')

        summary_buf = io.StringIO()
        nn_model.summary(print_fn=lambda x: summary_buf.write(x + '\n'))
        with open(model_save_dir / f'model_summary_{model_name}.txt', 'w') as f:
            f.write(summary_buf.getvalue())

    test_loss = nn_model.evaluate(X_test, Y_test, verbose=0)
    print(f'  Test loss: {test_loss}')

    # ── Loss curve plot ───────────────────────────────────────────────────────
    train_loss_history = history.history['loss']
    val_loss_history   = history.history['val_loss']
    epoch_range        = range(1, len(train_loss_history) + 1)

    plt.plot(epoch_range, train_loss_history, 'bo', label='loss')
    plt.plot(epoch_range, val_loss_history,         label='val_loss')
    plt.legend()
    plt.savefig(model_save_dir / 'plots' / f'loss_curve_{model_name}.png')
    plt.close()

    loss_history_df = pd.DataFrame(
        {'Loss': train_loss_history, 'Validation Loss': val_loss_history},
        index=epoch_range,
    )
    loss_history_df.to_csv(model_save_dir / f'loss_history_{model_name}.csv')

    # ── Evaluation on test set  (batched prediction) ──────────────────────────
    time_a = time.time()

    eval_idx = [random.randrange(len(Y_test)) for _ in range(EVAL_SAMPLES)]

    X_eval = X_test[eval_idx]
    Y_eval = Y_test[eval_idx]

    predictions = nn_model.predict(X_eval, verbose=0)

    abs_errors  = np.abs(predictions - Y_eval)
    sq_errors   = abs_errors ** 2
    pct_errors  = 100 * np.abs((predictions - Y_eval) / Y_eval)

    mean_abs_error  = np.mean(abs_errors, axis=0)
    rms_errors      = np.sqrt(np.mean(sq_errors, axis=0))
    mean_pct_errors = np.mean(pct_errors, axis=0)

    time_b = time.time()
    print('Mean Error: ',            mean_abs_error)
    print('Mean Percentage Error: ', mean_pct_errors)
    print('RMS: ',                   rms_errors)
    print(
        'Time Taken:',
        int((time_b - time_a) / 60), 'minutes and',
        int((time_b - time_a) % 60), 'seconds.',
    )

    eval_metrics_df = pd.DataFrame({
        'Mean Error':   mean_abs_error,
        'Mean % Error': mean_pct_errors,
        'RMS':          rms_errors,
    })
    if save_training_results:
        eval_metrics_df.to_csv(model_save_dir / f'training_metrics_{model_name}.csv')

    training_config = {
        'model_name':               model_name,
        'layer_sizes':               layer_sizes,
        'dropout_rate':              DROPOUT_RATE,
        'optimizer':                 'adam',
        'loss_function':             'mean_squared_error',
        'batch_size':                BATCH_SIZE,
        'max_epochs':                MAX_EPOCHS,
        'epochs_trained':            len(train_loss_history),
        'early_stopping_patience':   EARLY_STOPPING_PATIENCE,
        'early_stopping_min_delta':  EARLY_STOPPING_MIN_DELTA,
        'train_fraction':            TRAIN_FRACTION,
        'test_fraction':             round(TEST_FRACTION - TRAIN_FRACTION, 2),
        'val_fraction':              round(1.0 - TEST_FRACTION, 2),
        'training_samples':          len(X_train),
        'test_samples':              len(X_test),
        'val_samples':               len(X_val),
        'test_loss':                 float(test_loss[0]),
        'total_parameters':          nn_model.count_params(),
        'input_features':            len(input_columns),
        'output_features':           len(target_cols),
        'input_columns':             list(input_columns),
        'output_columns':            list(target_cols),
    }
    with open(model_save_dir / f'model_config_{model_name}.json', 'w') as f:
        json.dump(training_config, f, indent=2)
