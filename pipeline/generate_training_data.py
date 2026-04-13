from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import subprocess
import os
import math
import sys
import time
import itertools
from pathlib import Path
from scipy.optimize import minimize
from scipy.stats import truncnorm
from config import SAVE_FILE_NAME, OVERWRITE_SOURCES, USE_PROBABILITY_SAMPLING

# =============================================================================
# Constants
# =============================================================================
MASS_MIN = 0.1
MASS_MAX = 5.5          # inclusive upper bound (generate_ranges adds step/2 tolerance)
MASS_STEP = 0.1
ELLIP_MIN = 0.05
ELLIP_MAX = 0.60        # inclusive upper bound (generate_ranges adds step/2 tolerance)
ELLIP_STEP = 0.05
SOURCES_PER_MODEL = 10_000
PERCENTAGE_TO_CAUSTIC = 0.998  # 99.8% of caustic semi-major axis (paper §3.1)
NO_PROB_SCALING = 1.2          # scaling factor for discrete caustic mode only
OMITCORE_DIVISOR = 10_000
SIGMA_DIVISOR = 3.7
DIST_STEP_DIVISOR = 1_000
PERIM_SCALE = 104

# critcurves.csv column indices (outer caustic, outer crit, inner caustic, inner crit)
CAUS_OUTER_X_COL = 0
CAUS_OUTER_Y_COL = 1
CRIT_OUTER_X_COL = 2
CRIT_OUTER_Y_COL = 3
CAUS_INNER_X_COL = 4
CAUS_INNER_Y_COL = 5
CRIT_INNER_X_COL = 6
CRIT_INNER_Y_COL = 7

# =============================================================================
# Paths  (update these to match your working directory layout)
# NOTE: lensmodel binary must reside one directory above the .input/.dat/.start files
# =============================================================================
FOLDER_PATH = Path('/home/alex/Documents/Grav Lens/pipeline')
SOURCES_PATH = Path('/home/alex/Documents/Grav Lens/data/sources')
LENSMODEL_PATH = FOLDER_PATH.parent / 'lensmodel'
INPUT_FILE_PATH = FOLDER_PATH / 'SourceGridding.input'
DATA_FILE_PATH = FOLDER_PATH / 'SourceGridding.dat'
START_FILE_PATH = FOLDER_PATH / 'SourceGridding.start'
GRIDMODE = 2


# =============================================================================
# Functions
# =============================================================================

def objective(params, x, y):
    """Compute sum of squared distances between caustic points and an astroid curve.

    Args:
        params: Tuple (a, b) of astroid semi-axes.
        x: Array of x-coordinates of caustic points.
        y: Array of y-coordinates of caustic points.

    Returns:
        Scalar sum of squared residuals, or large penalty for invalid inputs.
    """
    a, b = params
    input_value = np.sign(x / a) * np.abs(x / a) ** (1 / 3)
    if np.any(input_value < -1) or np.any(input_value > 1):
        return 1e12
    predicted_y = np.sign(y) * b * (np.sin(np.arccos(input_value)) ** 3)
    return np.sum((predicted_y - y) ** 2)


def calc_ellipse_perim(a, b):
    """Compute the perimeter of an ellipse using Ramanujan's approximation.

    Args:
        a: Semi-major axis.
        b: Semi-minor axis.

    Returns:
        Approximate perimeter of the ellipse.
    """
    h = ((a - b) ** 2) / ((a + b) ** 2)
    return np.pi * (a + b) * (1 + (3 * h) / (10 + np.sqrt(4 - 3 * h)))


def isfloat(num):
    """Return True if num can be cast to float, False otherwise."""
    try:
        float(num)
        return True
    except ValueError:
        return False


def timing(duration):
    """Convert a duration in seconds to (hours, minutes, seconds).

    Args:
        duration: Elapsed time in seconds.

    Returns:
        Tuple of (int hours, int minutes, int seconds).
    """
    hours, remainder = divmod(duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    return int(hours), int(minutes), int(seconds)


def generate_ranges(param):
    """Generate a parameter range or wrap a fixed value in an array.

    Args:
        param: List of 1 element (fixed value) or 3 elements [start, stop, step].

    Returns:
        numpy array of parameter values.
    """
    if len(param) == 3:
        # Add half a step as tolerance so the stop value is included
        # (np.arange excludes its stop, causing e.g. arange(0.05, 0.60, 0.05)
        # to silently drop 0.60).
        return np.arange(param[0], param[1] + param[2] / 2, param[2])
    elif len(param) == 1:
        return param
    else:
        raise ValueError("Parameter list should contain either 1 or 3 elements.")


def read_template_lines(path):
    """Read a lensmodel config file as a list of non-blank, non-comment lines.

    Args:
        path: Path to the config file.

    Returns:
        List of stripped line strings (comments and blank lines excluded).
    """
    with open(path) as f:
        return [line.rstrip('\n') for line in f if line.strip() and not line.startswith('#')]


def write_lines(path, lines):
    """Write a list of strings as a text file, one line each.

    Args:
        path: Destination file path.
        lines: List of strings to write.
    """
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


# =============================================================================
# Configuration
# =============================================================================

headers = [
    'Source RA position', 'Source Dec position', 'Mass Parameter', 'Ellipticity',
    'Ellipticity Angle', 'Shear', 'Shear Angle',
    'Image A RA', 'Image A Dec', 'Image A Flux', 'Image A Time Del',
    'Image B RA', 'Image B Dec', 'Image B Flux', 'Image B Time Del',
    'Image C RA', 'Image C Dec', 'Image C Flux', 'Image C Time Del',
    'Image D RA', 'Image D Dec', 'Image D Flux', 'Image D Time Del',
]

np.random.seed(97634823)
# Seeds used previously:
#   7324629 — summer 2024 to 16 Nov 2024
#   1234    — before summer 2024

# User configuration — edit config.py
# SAVE_FILE_NAME, OVERWRITE_SOURCES, and USE_PROBABILITY_SAMPLING are imported
# from config.py at the top of this file.
print_lensmodel_output = False

mass_param_range  = [MASS_MIN, MASS_MAX, MASS_STEP]
ellip_range       = [ELLIP_MIN, ELLIP_MAX, ELLIP_STEP]
ellip_angle_range = [0.]   # DON'T TOUCH
shear_range       = [0.]   # DON'T TOUCH
shear_angle_range = [0.]   # DON'T TOUCH

# =============================================================================
# Setup
# =============================================================================

FOLDER_PATH.mkdir(parents=True, exist_ok=True)
SOURCES_PATH.mkdir(parents=True, exist_ok=True)

range_configs          = [mass_param_range, ellip_range, ellip_angle_range, shear_range, shear_angle_range]
param_arrays           = [generate_ranges(cfg) for cfg in range_configs]
lens_param_combinations = itertools.product(*param_arrays)

all_sources_frames  = []
all_rejected_frames = []

# Read template files once as plain text lists — avoids pandas serialise overhead
# on every one of the 672 × 3 = 2,016 per-combination writes.
input_lines = read_template_lines(INPUT_FILE_PATH)
start_lines = read_template_lines(START_FILE_PATH)
data_lines  = read_template_lines(DATA_FILE_PATH)

fi3input_path  = Path('FI3input')   # relative — lensmodel parses paths by whitespace split
fi3output_path = Path('FI3output')  # so paths with spaces in them cannot be used
lensmodel_cmd  = f'"{LENSMODEL_PATH}" "{INPUT_FILE_PATH}"'

# =============================================================================
# Main Loop: for each lens model, generate source positions inside the inner
# caustic and retrieve image data via lensmodel.
# =============================================================================

_sources_out = SOURCES_PATH / f'{SAVE_FILE_NAME}.parquet'
if not OVERWRITE_SOURCES and _sources_out.exists():
    print(
        f"Skipping: {_sources_out} already exists.\n"
        "Set OVERWRITE_SOURCES = True in config.py, or change SAVE_FILE_NAME, to regenerate."
    )
    sys.exit(0)

t0 = time.time()

for combination in lens_param_combinations:
    mass_param_val, ellip_val, ellip_angle_val, shear_val, shear_angle_val = (
        np.round(value, 3) for value in combination
    )
    print(
        f"mass_param: {mass_param_val}, ellip: {ellip_val}, "
        f"ellip_angle: {ellip_angle_val}, shear: {shear_val}, shear_angle: {shear_angle_val}"
    )

    omitcore = np.round((mass_param_val + ellip_val) / OMITCORE_DIVISOR, 5)

    # --- Modify and write .input file (plotcrit pass) ---
    input_lines[0]  = f'set omitcore={omitcore}'
    input_lines[1]  = f'gridmode {GRIDMODE}'
    input_lines[-1] = 'plotcrit critcurves.csv'
    write_lines(INPUT_FILE_PATH, input_lines)

    # --- Modify and write .start file ---
    lens_model_str = (
        f'alpha {mass_param_val} 0.0 0.0 {ellip_val} {ellip_angle_val} '
        f'{shear_val} {shear_angle_val} 0.0 0.0 1.'
    )
    start_lines[1] = lens_model_str
    write_lines(START_FILE_PATH, start_lines)

    # --- Modify and write .dat file ---
    model_str_parts = lens_model_str.split()           # split once — was split twice
    galaxy_x, galaxy_y = float(model_str_parts[2]), float(model_str_parts[3])
    data_lines[1]  = f'{galaxy_x} {galaxy_y} 0.05'
    data_lines[7]  = f'{mass_param_val + ellip_val} 0.0 1. 0.0001 1000. 0. 1000'
    data_lines[8]  = f'{mass_param_val * 5} 0.0 1. 0.0001 1000. 0. 1000'
    data_lines[9]  = f'{mass_param_val / 10} 0.0 1. 0.0001 1000. 0. 1000'
    data_lines[10] = f'{mass_param_val / 100} 0.0 1. 0.0001 1000. 0. 1000'
    write_lines(DATA_FILE_PATH, data_lines)

    # --- Run lensmodel (plotcrit) ---
    try:
        lensmodel_result = subprocess.run(lensmodel_cmd, capture_output=True, shell=True, text=True)
    except FileNotFoundError as e:
        raise RuntimeError(f"lensmodel executable not found: {lensmodel_cmd}") from e
    except Exception as e:
        raise RuntimeError(f"Error running lensmodel: {e}") from e

    if print_lensmodel_output:
        print(lensmodel_result.stdout)
    if lensmodel_result.stderr.strip():
        print("STDERR:")
        print(lensmodel_result.stderr)
        raise RuntimeError(
            f"lensmodel failed (code {lensmodel_result.returncode}): {lensmodel_result.stderr}"
        )

    # --- Read critical curves ---
    crit_curves_df = pd.read_csv(
        "critcurves.csv", header=None, skip_blank_lines=True, sep=r'\s+', comment='#'
    )
    x1 = crit_curves_df.iloc[:, CRIT_OUTER_X_COL]
    y1 = crit_curves_df.iloc[:, CRIT_OUTER_Y_COL]
    x2 = crit_curves_df.iloc[:, CRIT_INNER_X_COL]
    y2 = crit_curves_df.iloc[:, CRIT_INNER_Y_COL]
    x3 = crit_curves_df.iloc[:, CAUS_OUTER_X_COL]
    y3 = crit_curves_df.iloc[:, CAUS_OUTER_Y_COL]
    x4 = crit_curves_df.iloc[:, CAUS_INNER_X_COL]
    y4 = crit_curves_df.iloc[:, CAUS_INNER_Y_COL]

    # --- Find inner caustic boundary using vectorised diff ---
    crit_seg_count = int(len(x1) / 4)
    inner_crit_ra  = x2.to_numpy()
    inner_crit_dec = y2.to_numpy()
    point_steps    = np.sqrt(np.diff(inner_crit_ra) ** 2 + np.diff(inner_crit_dec) ** 2)
    mean_step      = np.mean(point_steps[:crit_seg_count])
    large_step_idx = np.argwhere(point_steps[crit_seg_count:] > 10 * mean_step)
    if len(large_step_idx) > 0:
        caustic_break_idx = crit_seg_count + int(large_step_idx[0][0])
    else:
        caustic_break_idx = len(inner_crit_ra) - 1

    # --- Collect unique inner caustic points ---
    outer_crit_pts  = np.column_stack((x1[:caustic_break_idx], y1[:caustic_break_idx]))
    inner_crit_pts  = np.column_stack((inner_crit_ra[:caustic_break_idx], inner_crit_dec[:caustic_break_idx]))
    crit_pts        = np.concatenate((outer_crit_pts, inner_crit_pts), axis=0)
    unique_crit_pts = np.unique(crit_pts, axis=0)
    caustic_x, caustic_y = unique_crit_pts[:, 0], unique_crit_pts[:, 1]

    # --- Fit astroid to inner caustic ---
    initial_guess = [max(abs(caustic_x)), max(abs(caustic_y))]
    bounds = ((max(abs(caustic_x)), 100), (1e-8, 100))
    astroid_fit  = minimize(objective, initial_guess, args=(caustic_x, caustic_y), bounds=bounds)
    astroid_a, astroid_b = astroid_fit.x
    caustic_axis_ratio   = astroid_a / astroid_b

    # --- Generate source positions ---
    if USE_PROBABILITY_SAMPLING:
        low     = 0
        high    = astroid_a * PERCENTAGE_TO_CAUSTIC
        mean    = astroid_a * PERCENTAGE_TO_CAUSTIC
        std_dev = (astroid_a * PERCENTAGE_TO_CAUSTIC) / SIGMA_DIVISOR
        truncated_normal = truncnorm(
            (low - mean) / std_dev, (high - mean) / std_dev, loc=mean, scale=std_dev
        )
        sample_semi_a = truncated_normal.rvs(size=SOURCES_PER_MODEL)
        sample_semi_b = sample_semi_a / caustic_axis_ratio
        random_signs  = np.random.choice([1, -1], size=SOURCES_PER_MODEL, p=[0.5, 0.5])
        source_ra  = np.random.uniform(low=-sample_semi_a, high=sample_semi_a)
        source_dec = random_signs * sample_semi_b * (
            np.cos(np.arcsin((np.abs(source_ra) / sample_semi_a) ** (1 / 3))) ** 3
        )
        source_positions = np.column_stack((source_ra, source_dec))
    else:
        current_a = astroid_a * PERCENTAGE_TO_CAUSTIC
        current_b = current_a / caustic_axis_ratio
        normal_p  = calc_ellipse_perim(current_a, current_b)
        dist_change = current_a / DIST_STEP_DIVISOR
        # Accumulate into a list then convert once — avoids O(n²) np.append in loop
        distances_list = [[dist_change, current_a, current_b, normal_p]]
        while (current_a > 0.01 * astroid_a) and (current_a - dist_change > 0):
            current_a -= dist_change
            dist_change *= NO_PROB_SCALING
            current_b = current_a / caustic_axis_ratio
            perim = calc_ellipse_perim(current_a, current_b)
            distances_list.append([dist_change, current_a, current_b, perim])
        distances = np.array(distances_list)

        # Accumulate quadrants in a list; concatenate once — avoids O(n²) realloc in loop
        quadrants = []
        for dist_val, new_a, new_b, perim in distances:
            num_sources = int(PERIM_SCALE * (perim / normal_p))
            new_x = np.linspace(-new_a, 0, num_sources)
            new_y = new_b * (np.cos(np.arcsin((abs(new_x) / new_a) ** (1 / 3))) ** 3)
            quadrants.append(np.concatenate((
                np.column_stack((new_x,  new_y)),
                np.column_stack((new_x, -new_y)),
                np.column_stack((-new_x,  new_y)),
                np.column_stack((-new_x, -new_y)),
            )))
        source_positions = np.concatenate(quadrants)

    # --- Write FI3input: count line then data in a single pass (no temp file needed) ---
    with open(fi3input_path, 'w') as f:
        f.write(f'{len(source_positions)}\n')
        np.savetxt(f, source_positions, fmt='%.10g')

    # --- Modify .input file for findimg3 pass ---
    input_lines[-1] = f'findimg3 {fi3input_path} {fi3output_path}'
    write_lines(INPUT_FILE_PATH, input_lines)

    # --- Run lensmodel (findimg3) ---
    try:
        lensmodel_result = subprocess.run(lensmodel_cmd, capture_output=True, shell=True, text=True)
    except FileNotFoundError as e:
        raise RuntimeError(f"lensmodel executable not found: {lensmodel_cmd}") from e
    except Exception as e:
        raise RuntimeError(f"Error running lensmodel: {e}") from e

    if lensmodel_result.stderr.strip():
        print("STDERR:")
        print(lensmodel_result.stderr)
        raise RuntimeError(
            f"lensmodel failed (code {lensmodel_result.returncode}): {lensmodel_result.stderr}"
        )

    # --- Parse findimg3 output ---
    # Build list of rows then convert to array once — avoids O(n²) concatenation.
    # str.split() (no args) handles any whitespace, replacing .replace('  ',' ').split(' ').
    with open(fi3output_path) as f:
        raw_lines = f.readlines()

    raw_output_rows = [
        line.split()
        for line in raw_lines
        if not line.startswith('#') and len(line) > 1
    ]
    lensmodel_output = np.array(raw_output_rows) if raw_output_rows else np.array([])

    # --- Filter for exactly 4-image systems; sort by descending magnification ---
    # Use list accumulation then single-conversion — avoids O(n²) array/DataFrame
    # reallocation that np.concatenate and pd.concat cause when called in a loop.
    quad_image_rows    = []
    rejected_image_rows = []
    image_count = 0
    for line_index in range(len(lensmodel_output)):
        if lensmodel_output[line_index][2] == '#':
            if image_count == 4:
                image_count = 0
                image_group  = lensmodel_output[line_index - 4:line_index].astype(float)
                lens_params  = [mass_param_val, ellip_val, ellip_angle_val, shear_val, shear_angle_val]
                magnifications = abs(image_group[:, 2])
                mag_sort_idx   = np.argsort(magnifications)
                source_row     = image_group[mag_sort_idx[::-1]].flatten()
                source_row     = np.concatenate((
                    lensmodel_output[line_index - 5][0:2].astype(float),
                    lens_params,
                    source_row,
                ))
                quad_image_rows.append(source_row)          # O(1) list append
            else:
                if line_index != 0:
                    rejected_image_rows.append({            # O(1) list append
                        headers[0]: lensmodel_output[line_index - (image_count + 1), 0],
                        headers[1]: lensmodel_output[line_index - (image_count + 1), 1],
                        headers[2]: mass_param_val,
                        headers[3]: ellip_val,
                        headers[4]: ellip_angle_val,
                        headers[5]: shear_val,
                        headers[6]: shear_angle_val,
                    })
                image_count = 0
        else:
            image_count += 1

    # Single O(n) conversion after the loop
    non_quad_df    = pd.DataFrame(rejected_image_rows, columns=headers[0:7])
    quad_image_data = np.stack(quad_image_rows) if quad_image_rows else np.array([])

    # --- Accumulate rejected sources (not 4 images) ---
    all_rejected_frames.append(non_quad_df)

    # --- Accumulate valid 4-image sources ---
    if quad_image_data.size != 0:
        all_sources_frames.append(pd.DataFrame(quad_image_data, columns=headers))

if all_sources_frames:
    pd.concat(all_sources_frames, ignore_index=True).to_parquet(
        SOURCES_PATH / f'{SAVE_FILE_NAME}.parquet', index=False
    )
if all_rejected_frames:
    pd.concat(all_rejected_frames, ignore_index=True).to_parquet(
        SOURCES_PATH / 'rejected_sources.parquet', index=False
    )

t1 = time.time()
hours, minutes, seconds = timing(t1 - t0)
print(f'The code took {hours} hours, {minutes} minutes and {seconds} seconds to run')
