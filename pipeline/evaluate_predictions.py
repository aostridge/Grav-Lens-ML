import os
import sys
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import tensorflow as tf
tf.random.set_seed(1234)

import numpy as np
import pandas as pd
import random
import seaborn as sns
import time
from pathlib import Path
from matplotlib import pyplot as plt
from columns import (
    SOURCE_COLS, PARAM_COLS, IMAGE_COORD_COLS, NEW_COORD_COLS, DIST_RAT_COLS,
    NN_INPUT_COLS, NN_MASS_OUTPUT_COLS, NN_ELLIP_OUTPUT_COLS,
)
from config import FILE_NAME, OVERWRITE_PREDICTIONS, PREDICTION_MODE

# =============================================================================
# Constants
# =============================================================================
TRAIN_FRACTION = 0.7
TEST_FRACTION  = 0.9
BATCH_SIZE     = 1_000
VIOLIN_OFFSET  = 0.4
MASS_BINS      = [1.4, 2.8, 4.2, 5.5]

# =============================================================================
# Paths
# =============================================================================
FILE_PATH   = Path('/home/alex/Documents/Grav Lens/data')
SOURCE_PATH = 'sources'
MODEL_PATH  = 'models'
MODEL_MASS  = 'model_mass_parameter.keras'
MODEL_ELLIP = 'model_ellipticity.keras'
MODEL_BOTH  = 'model_mass_ellipticity.keras'

# =============================================================================
# Column definitions  (imported from columns.py — single source of truth)
# =============================================================================

# =============================================================================
# Prediction configuration
# =============================================================================
shuffle_data  = False
testing_only  = False

mass_input_columns   = NN_INPUT_COLS
mass_output_columns  = NN_MASS_OUTPUT_COLS
ellip_output_columns = NN_ELLIP_OUTPUT_COLS

# User configuration — edit config.py
# FILE_NAME, OVERWRITE_PREDICTIONS, and PREDICTION_MODE are imported from config.py.

save_predictions       = True
prediction_file        = (
    f'{FILE_NAME}_features_predictions' if PREDICTION_MODE == 'separate'
    else f'{FILE_NAME}_features_predictions_combined'
)
save_mass_plot         = True
mass_violin_plot_name  = f'mass_violin_plot_{FILE_NAME}_features'
save_ellip_plot        = True
ellip_violin_plot_name = f'ellipticity_violin_plot_{FILE_NAME}_features'

# =============================================================================
# Skip check
# =============================================================================
_predictions_out = FILE_PATH / MODEL_PATH / f'{prediction_file}.parquet'
if not OVERWRITE_PREDICTIONS and _predictions_out.exists():
    print(
        f"Skipping: {_predictions_out} already exists.\n"
        "Set OVERWRITE_PREDICTIONS = True in config.py, or change FILE_NAME, to regenerate."
    )
    sys.exit(0)

if PREDICTION_MODE not in ('separate', 'combined'):
    raise ValueError(f"PREDICTION_MODE must be 'separate' or 'combined', got {PREDICTION_MODE!r}")

# =============================================================================
# Load models (single load — no duplicate reload needed)
# =============================================================================
if PREDICTION_MODE == 'separate':
    mass_loaded_model  = tf.keras.models.load_model(FILE_PATH / MODEL_PATH / MODEL_MASS)
    ellip_loaded_model = tf.keras.models.load_model(FILE_PATH / MODEL_PATH / MODEL_ELLIP)
    mass_loaded_model.training  = False
    ellip_loaded_model.training = False
else:
    combined_loaded_model = tf.keras.models.load_model(FILE_PATH / MODEL_PATH / MODEL_BOTH)
    combined_loaded_model.training = False

# =============================================================================
# Load data  (direct read — chunking then concatenating is redundant)
# =============================================================================
load_cols = SOURCE_COLS + PARAM_COLS + IMAGE_COORD_COLS + NEW_COORD_COLS + DIST_RAT_COLS
data = pd.read_parquet(
    FILE_PATH / SOURCE_PATH / f'{FILE_NAME}_features.parquet',
    columns=load_cols,
)

if shuffle_data:
    shuffled_data = data.sample(frac=1, random_state=1234).reset_index(drop=True)
else:
    shuffled_data = data

n_samples       = len(shuffled_data)
train_split_idx = int(TRAIN_FRACTION * n_samples)
test_split_idx  = int(TEST_FRACTION  * n_samples)

if testing_only:
    eval_data = shuffled_data[train_split_idx:test_split_idx]
else:
    eval_data = shuffled_data

# =============================================================================
# Build TF dataset and run batch predictions
# =============================================================================
_input_np     = eval_data[mass_input_columns].to_numpy().astype(np.float32)
_mass_out_np  = eval_data[mass_output_columns].to_numpy().astype(np.float32)
_ellip_out_np = eval_data[ellip_output_columns].to_numpy().astype(np.float32)

tf_dataset = tf.data.Dataset.from_tensor_slices((_input_np, _mass_out_np, _ellip_out_np))

mass_predictions  = []
mass_abs_errors   = []
mass_pct_errors   = []
ellip_predictions = []
ellip_abs_errors  = []
ellip_pct_errors  = []

if PREDICTION_MODE == 'separate':
    @tf.function
    def predict_mass_batch(batch_input):
        return mass_loaded_model(batch_input)

    @tf.function
    def predict_ellip_batch(batch_input):
        return ellip_loaded_model(batch_input)

    for batch_in, batch_mass_out, batch_ellip_out in tf_dataset.batch(BATCH_SIZE):
        mass_prediction = predict_mass_batch(batch_in)
        mass_predictions.append(mass_prediction)
        mass_abs_errors.append(tf.abs(mass_prediction - batch_mass_out))
        mass_pct_errors.append(100 * tf.abs((mass_prediction - batch_mass_out) / batch_mass_out))

        ellip_prediction = predict_ellip_batch(batch_in)
        ellip_predictions.append(ellip_prediction)
        ellip_abs_errors.append(tf.abs(ellip_prediction - batch_ellip_out))
        ellip_pct_errors.append(100 * tf.abs((ellip_prediction - batch_ellip_out) / batch_ellip_out))

else:
    @tf.function
    def predict_combined_batch(batch_input):
        return combined_loaded_model(batch_input)

    for batch_in, batch_mass_out, batch_ellip_out in tf_dataset.batch(BATCH_SIZE):
        combined_pred    = predict_combined_batch(batch_in)
        mass_prediction  = combined_pred[:, 0:1]
        ellip_prediction = combined_pred[:, 1:2]

        mass_predictions.append(mass_prediction)
        mass_abs_errors.append(tf.abs(mass_prediction - batch_mass_out))
        mass_pct_errors.append(100 * tf.abs((mass_prediction - batch_mass_out) / batch_mass_out))

        ellip_predictions.append(ellip_prediction)
        ellip_abs_errors.append(tf.abs(ellip_prediction - batch_ellip_out))
        ellip_pct_errors.append(100 * tf.abs((ellip_prediction - batch_ellip_out) / batch_ellip_out))

mass_predictions  = tf.concat(mass_predictions,  axis=0)
mass_abs_errors   = tf.concat(mass_abs_errors,   axis=0)
mass_pct_errors   = tf.concat(mass_pct_errors,   axis=0)
ellip_predictions = tf.concat(ellip_predictions, axis=0)
ellip_abs_errors  = tf.concat(ellip_abs_errors,  axis=0)
ellip_pct_errors  = tf.concat(ellip_pct_errors,  axis=0)

predictions_df = pd.DataFrame({
    'Mass Pred':               mass_predictions.numpy().flatten(),
    'Mass Pred Error':         mass_abs_errors.numpy().flatten(),
    'Mass Pred Percent Error': mass_pct_errors.numpy().flatten(),
    'Ellip Pred':              ellip_predictions.numpy().flatten(),
    'Ellip Pred Error':        ellip_abs_errors.numpy().flatten(),
    'Ellip Pred Percent Error': ellip_pct_errors.numpy().flatten(),
})
predictions_df = predictions_df.set_index(eval_data.index)

# =============================================================================
# Save predictions
# =============================================================================
if save_predictions:
    output_df = pd.concat([shuffled_data, predictions_df], axis=1)
    output_df.to_parquet(FILE_PATH / MODEL_PATH / f'{prediction_file}.parquet', index=False)

mean_mass_error  = np.mean(np.abs(predictions_df['Mass Pred Error']))
mean_ellip_error = np.mean(np.abs(predictions_df['Ellip Pred Error']))
print(f'Mean mass error: {mean_mass_error}')
print(f'Mean ellip error: {mean_ellip_error}')

rms_mass_error  = np.sqrt(np.mean(predictions_df['Mass Pred Error'] ** 2))
rms_ellip_error = np.sqrt(np.mean(predictions_df['Ellip Pred Error'] ** 2))
print(f'\nMass RMS: {rms_mass_error}')
print(f'Ellip RMS: {rms_ellip_error}')

# =============================================================================
# Ellipticity violin plot
# =============================================================================
violin_plot_data = pd.DataFrame({
    'Mass Parameter': eval_data['Mass Parameter'].values,
    'Mass Pred':      predictions_df['Mass Pred'].values,
    'Ellipticity':    eval_data['Ellipticity'].values,
    'Ellip Pred':     predictions_df['Ellip Pred'].values,
})

subset_df = violin_plot_data[['Ellipticity', 'Ellip Pred']]
ellipticity_values = np.sort(subset_df['Ellipticity'].unique())

plt.figure(figsize=(10, 6))
sns.violinplot(
    x='Ellipticity', y='Ellip Pred', data=subset_df,
    linewidth=0.5, density_norm='width',
    palette='colorblind', hue=subset_df['Ellipticity'], legend=False,
)
plt.plot([0, 11], [0.05, 0.6], color='black', linestyle='--', linewidth=1)

# Percentiles via groupby — replaces per-value filter loop
ellip_groups = subset_df.groupby('Ellipticity')['Ellip Pred']
p2_5  = ellip_groups.quantile(0.025).loc[ellipticity_values].values
p97_5 = ellip_groups.quantile(0.975).loc[ellipticity_values].values

for i in range(len(ellipticity_values)):
    plt.hlines(p2_5[i],  i - VIOLIN_OFFSET, i + VIOLIN_OFFSET, colors='red', alpha=0.5)
    plt.hlines(p97_5[i], i - VIOLIN_OFFSET, i + VIOLIN_OFFSET, colors='red', alpha=0.5)

plt.xlabel('Ellipticity')
plt.ylabel('Ellipticity Prediction')
plt.title('Violin Plots for Each Ellipticity', fontweight='bold')
plt.tight_layout()
if save_ellip_plot:
    plt.savefig(FILE_PATH / MODEL_PATH / 'plots' / f'{ellip_violin_plot_name}.png')
plt.close()

# =============================================================================
# Mass violin plot (2×2 subplots split by mass bin)
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False, sharey=False)

conditions = [
    violin_plot_data['Mass Parameter'] <= MASS_BINS[0],
    (violin_plot_data['Mass Parameter'] > MASS_BINS[0]) & (violin_plot_data['Mass Parameter'] <= MASS_BINS[1]),
    (violin_plot_data['Mass Parameter'] > MASS_BINS[1]) & (violin_plot_data['Mass Parameter'] <= MASS_BINS[2]),
    (violin_plot_data['Mass Parameter'] > MASS_BINS[2]) & (violin_plot_data['Mass Parameter'] <= MASS_BINS[3]),
]

for condition, ax in zip(conditions, axes.flatten()):
    subset_df = violin_plot_data[condition]
    if subset_df.empty:
        subset_df = pd.DataFrame({'Mass Parameter': [0], 'Mass Pred': [0]})

    mass_values = np.sort(subset_df['Mass Parameter'].unique())
    sns.violinplot(
        x='Mass Parameter', y='Mass Pred', data=subset_df, ax=ax,
        linewidth=0.5, density_norm='width',
        palette='colorblind', hue=subset_df['Mass Parameter'], legend=False,
    )

    # Percentiles via groupby — replaces per-value filter loop
    grouped_mass = subset_df.groupby('Mass Parameter')['Mass Pred']
    perc_2_5  = grouped_mass.quantile(0.025).loc[mass_values].values
    perc_97_5 = grouped_mass.quantile(0.975).loc[mass_values].values
    for i in range(len(mass_values)):
        ax.hlines(perc_2_5[i],  i - VIOLIN_OFFSET, i + VIOLIN_OFFSET, colors='red', alpha=0.5)
        ax.hlines(perc_97_5[i], i - VIOLIN_OFFSET, i + VIOLIN_OFFSET, colors='red', alpha=0.5)

    x_min = subset_df['Mass Parameter'].min()
    x_max = subset_df['Mass Parameter'].max()
    ax.plot([0, (x_max - x_min) * 10], [x_min, x_max], color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Mass Parameter (arcseconds)')
    ax.set_ylabel('Mass Parameter Prediction (arcseconds)')
    ax.set_title(f'{x_min:.1f} < Mass < {x_max:.1f}')
    y_min, y_max = subset_df['Mass Pred'].min(), subset_df['Mass Pred'].max()
    if np.isfinite(y_min) and np.isfinite(y_max):
        ax.set_ylim(y_min - 0.05, y_max + 0.05)

plt.tight_layout()
fig.suptitle('Violin Plots for Each Mass Parameter', fontweight='bold')
if save_mass_plot:
    plt.savefig(FILE_PATH / MODEL_PATH / 'plots' / f'{mass_violin_plot_name}.png')
plt.close()

