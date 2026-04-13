import subprocess
import os
import numpy as np
import pandas as pd
import time
import math
import tensorflow as tf
from pathlib import Path
from scipy.optimize import minimize
from scipy import stats
from itertools import combinations
from columns import DIST_RAT_COLS


# =============================================================================
# Constants
# =============================================================================
DEGREES_TO_ARCSEC = 3600.0
EPSILON           = 1e-9

CHI_SQR_ACCURACY = 8
TOLERANCE        = 0.0005
TOL_COUNT        = 20
EXPIRE_LIMIT     = 10
BINARY_LENGTH    = 10
BINARY_ONES      = 4
FIXED_POSITIONS  = {7: 0, 8: 0, 9: 0}  # last 3 parameter bits always fixed to 0

# =============================================================================
# Paths & run configuration
# =============================================================================
FILE_PATH  = Path('/home/alex/Documents/Grav Lens/data/observations')
FILE_NAME  = 'quad_lens_observations.txt'

LENSMODEL_PATH   = Path('/home/alex/Documents/Grav Lens/lensmodel')
MODEL_PATH       = Path('/home/alex/Documents/Grav Lens/data/models')
MODEL_NAME       = 'quad_lens_observations'
MODEL_NAME_MASS  = 'model_mass_parameter.keras'
MODEL_NAME_ELLIP = 'model_ellipticity.keras'

# 'degrees'     — absolute sky coordinates (multiply by 3600 to convert to arcsec)
# 'arcseconds'  — relative offsets already in arcseconds (no conversion applied)
INPUT_COORDINATE_UNITS = 'arcseconds'
IMAGE_RECENTRE_SUFFIX  = ' I Recentred'
RECENTRE_START_INDEX   = 1  # slice index for recentre column lists

RECENTRE = True
ROTATE   = True
TIMEOUT  = 30

# =============================================================================
# Feature columns fed to the neural networks
# Distance ratios imported from columns.py (single source of truth).
# Rotated coordinates are specific to this script's preprocessing pipeline.
# =============================================================================
FEATURE_COLUMNS = DIST_RAT_COLS + [
    'Image A RA I Rotated', 'Image A Dec I Rotated',
    'Image B RA I Rotated', 'Image B Dec I Rotated',
    'Image C RA I Rotated', 'Image C Dec I Rotated',
    'Image D RA I Rotated', 'Image D Dec I Rotated',
]

# =============================================================================
# Helper functions
# =============================================================================

def create_folder_if_not_exists(folder_path):
    """Create directory (and parents) if it does not already exist."""
    os.makedirs(folder_path, exist_ok=True)


def dist(x1, y1, x2, y2):
    """Return Euclidean distance between two 2D points."""
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def find_angle_anti(x1, y1, x2, y2):
    """Return the anticlockwise angle (0 to 2π) from point (x1,y1) to (x2,y2)."""
    x2 -= x1
    y2 -= y1
    angle_radians = math.atan2(y2, x2)
    return (2 * math.pi + angle_radians) % (2 * math.pi)


def order_points_clockwise(points):
    """Return indices that sort 4 image positions clockwise from the topmost point.

    Args:
        points: (4, 2) array of (RA, Dec) coordinates.

    Returns:
        Array of 4 indices in clockwise order.
    """
    top_most_point_index = np.argmax(points[:, 1])
    polar_angles = np.arctan2(
        points[:, 1] - points[top_most_point_index, 1],
        points[:, 0] - points[top_most_point_index, 0],
    )
    clockwise_order_indices = np.flip(np.argsort(polar_angles))
    roll_point = np.argmin(clockwise_order_indices)
    clockwise_order_indices = np.roll(clockwise_order_indices, -roll_point)

    dist_1 = dist(points[0, 0], points[0, 1],
                  points[clockwise_order_indices[1], 0], points[clockwise_order_indices[1], 1])
    dist_2 = dist(points[0, 0], points[0, 1],
                  points[clockwise_order_indices[3], 0], points[clockwise_order_indices[3], 1])

    if abs(dist_2) < abs(dist_1):
        return clockwise_order_indices[[0, 3, 2, 1]]
    return clockwise_order_indices


def area(a, b, c):
    """Compute triangle area from side lengths using Heron's formula.

    Args:
        a, b, c: Side lengths.

    Returns:
        Triangle area. Handles near-zero floating-point negatives gracefully.
    """
    s = (a + b + c) / 2
    d = s * (s - a) * (s - b) * (s - c)
    if (d < 0) and (s - c > -EPSILON):
        d = 0
    return np.sqrt(d)


def distance_from_center(x, points):
    """Sum of squared distances from image positions to an ellipse perimeter.

    Args:
        x: Ellipse parameters [a, b, x_c, y_c, theta].
        points: (4, 2) array of image coordinates.

    Returns:
        Scalar objective value.
    """
    a, b, x_c, y_c, theta = x
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    xp = points[:, 0] - x_c
    yp = points[:, 1] - y_c
    rotated_x = cos_theta * xp + sin_theta * yp
    rotated_y = -sin_theta * xp + cos_theta * yp
    return sum(((rotated_x ** 2 / a ** 2) + (rotated_y ** 2 / b ** 2) - 1) ** 2)


def fit_ellipse(points):
    """Fit a 2D ellipse to 4 image positions via constrained SLSQP minimisation.

    Args:
        points: (4, 2) array of image coordinates.

    Returns:
        Tuple (a, b, x_c, y_c, theta) of fitted ellipse parameters.
    """
    initial_guess = [1.0, 1.0, np.mean(points[:, 0]), np.mean(points[:, 1]), 0.0]
    cons = (
        {'type': 'ineq', 'fun': lambda x: np.max(points[:, 0]) - x[2]},
        {'type': 'ineq', 'fun': lambda x: x[2] - np.min(points[:, 0])},
        {'type': 'ineq', 'fun': lambda x: np.max(points[:, 1]) - x[3]},
        {'type': 'ineq', 'fun': lambda x: x[3] - np.min(points[:, 1])},
    )
    result = minimize(
        distance_from_center, initial_guess, args=(points,),
        constraints=cons, method='SLSQP',
    )
    a, b, x_c, y_c, theta = result.x
    return a, b, x_c, y_c, theta


def compute_features(row):
    """Compute all geometric features for a single observation row.

    Combines distance ratios, sums, triangle areas, flux ratios, and
    diagonal intersection point into a single pass to avoid redundant
    distance calculations.

    Args:
        row: DataFrame row with Image A/B/C/D RA/Dec/Flux columns.

    Returns:
        pd.Series of computed feature values.
    """
    # Compute the 6 pairwise distances once — reused by ratios, sums, and areas
    d_AB = dist(row['Image A RA'], row['Image A Dec'], row['Image B RA'], row['Image B Dec'])
    d_BC = dist(row['Image B RA'], row['Image B Dec'], row['Image C RA'], row['Image C Dec'])
    d_CD = dist(row['Image C RA'], row['Image C Dec'], row['Image D RA'], row['Image D Dec'])
    d_AC = dist(row['Image A RA'], row['Image A Dec'], row['Image C RA'], row['Image C Dec'])
    d_AD = dist(row['Image A RA'], row['Image A Dec'], row['Image D RA'], row['Image D Dec'])
    d_BD = dist(row['Image B RA'], row['Image B Dec'], row['Image D RA'], row['Image D Dec'])
    total_dist = d_AB + d_BC + d_CD + d_AC + d_AD + d_BD

    # Normalised distance ratios
    ratios = {
        'Ratio AB': d_AB / total_dist,
        'Ratio BC': d_BC / total_dist,
        'Ratio CD': d_CD / total_dist,
        'Ratio AC': d_AC / total_dist,
        'Ratio AD': d_AD / total_dist,
        'Ratio BD': d_BD / total_dist,
    }

    # Adjacent distance sums
    sums = {
        'Sum AB,DA':  d_AB + d_AD,
        'Sum BC, CD': d_BC + d_CD,
    }

    # Triangle areas (Heron's formula)
    areas = {
        'Area ABD': area(d_AB, d_BD, d_AD),
        'Area ABC': area(d_AB, d_BC, d_AC),
        'Area BCD': area(d_BC, d_CD, d_BD),
        'Area ACD': area(d_AC, d_CD, d_AD),
    }

    # Normalised flux ratios
    flux = [row['Image A Flux'], row['Image B Flux'], row['Image C Flux'], row['Image D Flux']]
    total_flux = sum(abs(f) for f in flux)
    mag_ratios = {
        'Mag Ratio A': abs(flux[0]) / total_flux,
        'Mag Ratio B': abs(flux[1]) / total_flux,
        'Mag Ratio C': abs(flux[2]) / total_flux,
        'Mag Ratio D': abs(flux[3]) / total_flux,
    }

    # Diagonal intersection point and distances to it
    x1, y1 = row['Image A RA'], row['Image A Dec']
    x2, y2 = row['Image B RA'], row['Image B Dec']
    x3, y3 = row['Image C RA'], row['Image C Dec']
    x4, y4 = row['Image D RA'], row['Image D Dec']

    verticalAC = (x1 == x3)
    verticalBD = (x2 == x4)

    if verticalAC or verticalBD:
        if verticalAC and verticalBD:
            x_int, y_int = np.nan, np.nan
        elif verticalAC:
            m2 = (y4 - y2) / (x4 - x2)
            x_int = x1
            y_int = m2 * (x_int - x4) + y4
        else:
            m1 = (y3 - y1) / (x3 - x1)
            x_int = x4
            y_int = m1 * (x_int - x1) + y1
    else:
        m1 = (y3 - y1) / (x3 - x1)
        m2 = (y4 - y2) / (x4 - x2)
        if m1 == m2:
            x_int, y_int = np.nan, np.nan
        else:
            x_int = ((y2 - m2 * x2) - (y1 - m1 * x1)) / (m1 - m2)
            y_int = m1 * (x_int - x1) + y1

    intersection = {
        'Intersection Point RA':      x_int,
        'Intersection Point Dec':     y_int,
        'Distance A to Intersection': dist(x1, y1, x_int, y_int),
        'Distance B to Intersection': dist(x2, y2, x_int, y_int),
        'Distance C to Intersection': dist(x3, y3, x_int, y_int),
        'Distance D to Intersection': dist(x4, y4, x_int, y_int),
    }

    return pd.Series({**ratios, **sums, **areas, **mag_ratios, **intersection})


def fit_ellipse_and_distances(row):
    """Fit ellipse to image positions and compute distances from images to ellipse centre.

    Writes results directly to the global DataFrame df.

    Args:
        row: DataFrame row with Image A/B/C/D RA/Dec columns.
    """
    points = np.array([
        [row['Image A RA'], row['Image A Dec']],
        [row['Image B RA'], row['Image B Dec']],
        [row['Image C RA'], row['Image C Dec']],
        [row['Image D RA'], row['Image D Dec']],
    ])
    a, b, x_c, y_c, theta = fit_ellipse(points)

    # Vectorised distances to centre using numpy
    centre = np.array([x_c, y_c])
    distances = np.linalg.norm(points - centre, axis=1)

    df.at[row.name, 'Ellipse Centre RA']  = x_c
    df.at[row.name, 'Ellipse Centre Dec'] = y_c
    df.at[row.name, 'Ellipse a']          = a
    df.at[row.name, 'Ellipse b']          = b
    df.at[row.name, 'Ellipse theta']      = theta
    df.at[row.name, 'Distance A to Centre'] = distances[0]
    df.at[row.name, 'Distance B to Centre'] = distances[1]
    df.at[row.name, 'Distance C to Centre'] = distances[2]
    df.at[row.name, 'Distance D to Centre'] = distances[3]


def swap_groups_per_row(row, order):
    """Reorder image data columns in a row according to a given permutation.

    Args:
        row: DataFrame row.
        order: List of 4 indices specifying the new image ordering.
    """
    index_list = ['A', 'B', 'C', 'D']
    if 'Image A ErrorRA' in df.columns:
        columns = [
            f'Image {idx} {param}'
            for idx in index_list
            for param in ['RA', 'Dec', 'ErrorRA', 'ErrorDec', 'Flux', 'FluxError']
        ]
    else:
        columns = [
            f'Image {idx} {param}'
            for idx in index_list
            for param in ['RA', 'Dec', 'Flux', 'Time Del']
        ]
    len_col = int(len(columns) / 4)
    new_values = np.concatenate([row[columns[i * len_col:(i + 1) * len_col]].values for i in order])
    for i, col in enumerate(columns):
        df.at[row.name, col] = new_values[i]




def parse_image_set(lines):
    """Parse a block of lines from the observation text file into an image data dict.

    Args:
        lines: List of strings; first line is image label (e.g. '#J2033'), then
               4 data lines each with RA Dec ErrorRA ErrorDec Flux FluxError.

    Returns:
        Dict with keys 'Image', 'Image A RA', 'Image A Dec', etc.
    """
    image_data = {}
    image_name = lines[0].strip().replace('#', '')
    image_params = ['RA', 'Dec', 'ErrorRA', 'ErrorDec', 'Flux', 'FluxError']
    for i, line in enumerate(lines[2:], start=0):
        parts = line.split()
        for j, param in enumerate(image_params):
            image_data[f'Image {chr(65 + i)} {param}'] = float(parts[j])
    return {'Image': image_name, **image_data}


def optimize_model(row, output_path, results_dir_name, acc, tolerance, expire_limit,
                   bin_length, bin_ones, fixed_positions, image_suffix=''):
    """Iteratively optimise a lens model using stochastic binary parameter masking.

    Calls lensmodel repeatedly with randomly varied parameter flags, monitoring
    χ² convergence. Stops when χ² < acc, improvement stalls, or timeout limit
    is exceeded.

    Args:
        row: DataFrame row for the current observation.
        output_path: Path to the directory containing lens_model config files.
        results_dir_name: Name of the results subdirectory.
        acc: χ² convergence threshold.
        tolerance: Minimum relative χ² improvement to avoid stall count increment.
        expire_limit: Number of consecutive timeouts before giving up.
        bin_length: Total length of the binary parameter flag string.
        bin_ones: Maximum number of free parameters per iteration.
        fixed_positions: Dict mapping bit indices to fixed values (0 or 1).
        image_suffix: Column suffix for coordinate system to optimise in.
    """
    finished      = False
    is_first_pass = True
    start_time    = time.time()
    stall_count   = 0
    timeout_count = 0
    angle_rotated = False
    result_prefix = ''

    # Read all three config files once before the loop — eliminates 3 pd.read_csv()
    # calls per iteration (typically hundreds of iterations per observation).
    # start_file and data_file are still written each iteration so lensmodel always
    # sees fresh data; the in-memory copies remain valid after each write.
    input_file = pd.read_csv(
        f'{output_path}lens_model.input', header=None, skip_blank_lines=True, comment='#'
    )
    start_file = pd.read_csv(
        f'{output_path}lens_model.start', header=None, skip_blank_lines=True, comment='#'
    )
    data_file = pd.read_csv(
        f'{output_path}lens_model.dat', header=None, skip_blank_lines=True, comment='#'
    )

    # input_file fields are static for the entire optimisation run — set and write once.
    input_file.iloc[0, :] = [f'data {output_path}lens_model.dat']
    input_file.iloc[7, :] = [f'startup {output_path}lens_model.start']
    input_file.iloc[8, :] = ['optimize']
    input_file.to_csv(f'{output_path}lens_model.input', index=False, header=None)

    while not finished:

        if is_first_pass:
            data_file.iloc[1, :]  = f'{row[f"Ellipse Centre RA{image_suffix}"]} {row[f"Ellipse Centre Dec{image_suffix}"]} 0.05'
            data_file.iloc[7, :]  = f'{row[f"Image A RA{image_suffix}"]} {row[f"Image A Dec{image_suffix}"]} {row["Image A Flux"]} 0.0005 1000000 0. 1000'
            data_file.iloc[8, :]  = f'{row[f"Image B RA{image_suffix}"]} {row[f"Image B Dec{image_suffix}"]} {row["Image B Flux"]} 0.0005 1000000 0. 1000'
            data_file.iloc[9, :]  = f'{row[f"Image C RA{image_suffix}"]} {row[f"Image C Dec{image_suffix}"]} {row["Image C Flux"]} 0.0005 1000000 0. 1000'
            data_file.iloc[10, :] = f'{row[f"Image D RA{image_suffix}"]} {row[f"Image D Dec{image_suffix}"]} {row["Image D Flux"]} 0.0005 1000000 0. 1000'
        else:
            lens_model_str = read_best_start(f"{row['Image']}/best.start")
            model_str_parts = lens_model_str.split()
            data_file.iloc[1, :] = f'{model_str_parts[2]} {model_str_parts[3]} 0.05'
        data_file.to_csv(f'{output_path}lens_model.dat', index=False, header=None)

        if is_first_pass and (timeout_count == 0):
            lens_model_str = (
                f'alpha {row["Predicted Mass"]} {row[f"Ellipse Centre RA{image_suffix}"]} '
                f'{row[f"Ellipse Centre Dec{image_suffix}"]} {row["Predicted Ellip"]} '
                f'{math.degrees(row["Ellipticity Angle"])} 0. 0. 0. 0. 1.'
            )
            param_mask = '1 0 0 1 0 0 0 0 0 0'
            start_file.iloc[2, :] = param_mask
        else:
            same_mask = True
            while same_mask:
                param_mask = custom_binary_string(
                    length=bin_length, min_ones=1, max_ones=bin_ones,
                    fixed_positions=fixed_positions, seed=None,
                )
                if not start_file.iloc[2, :].equals(param_mask):
                    start_file.iloc[2, :] = param_mask
                    same_mask = False

        start_file.iloc[1, :] = lens_model_str
        start_file.to_csv(f'{output_path}lens_model.start', index=False, header=None)

        results_folder_path = f'{output_path}{results_dir_name}/{row["Image"]}/'
        create_folder_if_not_exists(results_folder_path)
        command = f'"{LENSMODEL_PATH}" "{output_path}lens_model.input"'

        try:
            subprocess.run(
                command, capture_output=True, text=True, shell=True,
                timeout=TIMEOUT, cwd=results_folder_path,
            )
            chi_sqr = read_chi_sqr(f"{row['Image']}/best-chi.dat")

            if not is_first_pass:
                change = tolerance * prev_chi
                if prev_chi - chi_sqr < change:
                    stall_count += 1
                else:
                    stall_count = 0
                if chi_sqr < acc:
                    finished = True
                elif stall_count >= TOL_COUNT:
                    if not angle_rotated:
                        df.at[row.name, 'Ellipticity Angle'] += math.pi / 2
                        angle_rotated = True
                        is_first_pass = True
                    else:
                        finished = True

                converted_num = binary_convert_base_10(param_mask)
                param_combo_stats[converted_num]['value'] += (prev_chi - chi_sqr)
                param_combo_stats[converted_num]['count'] += 1

                param_change_list = get_one_positions(param_mask)
                for param in param_change_list:
                    param_bit_stats[param]['value'] += (prev_chi - chi_sqr)
                    param_bit_stats[param]['count'] += 1
            else:
                is_first_pass = False

            prev_chi = chi_sqr
            timeout_count = 0

        except subprocess.TimeoutExpired:
            timeout_count += 1
            if timeout_count >= expire_limit:
                if not angle_rotated:
                    result_prefix = 'Expired '
                if not is_first_pass:
                    df.at[row.name, f'{result_prefix}Chi Sqr'] = chi_sqr
                    df.at[row.name, f'{result_prefix}Best Mass Model'] = lens_model_str
                    df.at[row.name, f'{result_prefix}Best Source Pos'] = read_source_best_data(
                        f"{row['Image']}/best-img.dat"
                    )
                else:
                    df.at[row.name, f'{result_prefix}Chi Sqr']         = -1
                    df.at[row.name, f'{result_prefix}Best Mass Model']  = 'NaN'
                    df.at[row.name, f'{result_prefix}Best Source Pos']  = 'NaN'
                if not angle_rotated:
                    is_first_pass = True
                    df.at[row.name, 'Ellipticity Angle'] += math.pi / 2
                    angle_rotated = True
                    timeout_count = 0
                else:
                    df.at[row.name, 'Subprocess Duration'] = 'NaN'
                    save_new_model_diff(row)
                    return

    df.at[row.name, 'Chi Sqr']         = chi_sqr
    df.at[row.name, 'Best Mass Model'] = lens_model_str
    df.at[row.name, 'Best Source Pos'] = read_source_best_data(f"{row['Image']}/best-img.dat")
    end_time = time.time()
    df.at[row.name, 'Subprocess Duration'] = end_time - start_time
    save_new_model_diff(row)
    print(f"{row['Image']} Chi Sqr {df.at[row.name, 'Chi Sqr']}")


def recentre_points(row, rec_from, rec_to, start_index):
    """Translate image and source coordinates to a new reference centre.

    Args:
        row: DataFrame row.
        rec_from: Column suffix of the current coordinate system.
        rec_to: Column suffix of the target coordinate system (' E Recentred' or ' I Recentred').
        start_index: Slice index into the coordinate column lists (controls which columns to shift).
    """
    if rec_to == ' E Recentred':
        name = 'Ellipse Centre'
    elif rec_to == ' I Recentred':
        name = 'Intersection Point'
    else:
        raise ValueError('rec_to must be " E Recentred" or " I Recentred".')

    ra_columns  = ['Source RA position', 'Image A RA',  'Image B RA',  'Image C RA',  'Image D RA',
                   'Ellipse Centre RA',  'Intersection Point RA'][start_index:]
    dec_columns = ['Source Dec position', 'Image A Dec', 'Image B Dec', 'Image C Dec', 'Image D Dec',
                   'Ellipse Centre Dec', 'Intersection Point Dec'][start_index:]

    df.at[row.name, f'RA Shift{rec_from}{rec_to}']  = row[f'{name} RA{rec_from}']
    df.at[row.name, f'Dec Shift{rec_from}{rec_to}'] = row[f'{name} Dec{rec_from}']

    for ra_col, dec_col in zip(ra_columns, dec_columns):
        df.at[row.name, ra_col  + rec_to] = row[ra_col  + rec_from] - row[f'{name} RA{rec_from}']
        df.at[row.name, dec_col + rec_to] = row[dec_col + rec_from] - row[f'{name} Dec{rec_from}']


def to_arcseconds(df, input_units):
    """Convert image RA/Dec columns to arcseconds if given in degrees.

    Args:
        df: DataFrame with image coordinate columns.
        input_units: 'degrees' or 'arcseconds'.

    Returns:
        DataFrame with coordinates in arcseconds.
    """
    cols = [
        'Image A RA', 'Image B RA', 'Image C RA', 'Image D RA',
        'Image A Dec', 'Image B Dec', 'Image C Dec', 'Image D Dec',
    ]
    if input_units.lower() == 'degrees':
        df[cols] = df[cols] * DEGREES_TO_ARCSEC
    elif input_units.lower() == 'arcseconds':
        pass
    else:
        raise ValueError('input_units must be "degrees" or "arcseconds".')
    return df


def find_align_angle(row, name, l):
    """Compute the ellipticity position angle from image positions around a centre point.

    Args:
        row: DataFrame row.
        name: Centre label string (e.g. 'Ellipse Centre' or 'Intersection Point').
        l: Recentred suffix label ('E' or 'I').
    """
    letters = ['A', 'B', 'C', 'D']
    # Compute sum of all anticlockwise angles then derive the mean alignment angle
    angle_total = sum(
        find_angle_anti(
            row[f'{name} RA {l} Recentred'], row[f'{name} Dec {l} Recentred'],
            row[f'Image {letter} RA {l} Recentred'], row[f'Image {letter} Dec {l} Recentred'],
        )
        for letter in letters
    )
    angle = (angle_total - math.pi) / 4
    if angle > math.pi:
        angle -= math.pi
    df.at[row.name, 'Ellipticity Angle'] = angle


def align_image_with_ellip(row, name, start_index):
    """Rotate recentred image coordinates to align with the ellipticity axis.

    Args:
        row: DataFrame row.
        name: Coordinate suffix (' E' or ' I').
        start_index: Slice index into coordinate column lists.
    """
    ra_columns  = ['Source RA position', 'Image A RA',  'Image B RA',  'Image C RA',  'Image D RA',
                   'Ellipse Centre RA',  'Intersection Point RA'][start_index:]
    dec_columns = ['Source Dec position', 'Image A Dec', 'Image B Dec', 'Image C Dec', 'Image D Dec',
                   'Ellipse Centre Dec', 'Intersection Point Dec'][start_index:]
    if f'RA Shift{name} Recentred' in row:
        ra_columns.append('RA Shift')
        dec_columns.append('Dec Shift')

    recentred_ra_columns  = [col + name + ' Recentred' for col in ra_columns]
    recentred_dec_columns = [col + name + ' Recentred' for col in dec_columns]
    rotated_ra_columns    = [col + name + ' Rotated'   for col in ra_columns]
    rotated_dec_columns   = [col + name + ' Rotated'   for col in dec_columns]

    angle_rad = -row['Ellipticity Angle']
    new_angle = row['Ellipse theta'] + angle_rad
    if new_angle > 2 * math.pi:
        new_angle -= 2 * math.pi
    df.at[row.name, f'Ellipse theta{name} Rotated']      = new_angle
    df.at[row.name, f'Ellipticity Angle{name} Rotated']  = 0

    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    for rc_ra, rc_dec, rot_ra, rot_dec in zip(
        recentred_ra_columns, recentred_dec_columns,
        rotated_ra_columns, rotated_dec_columns,
    ):
        df.at[row.name, rot_ra]  = row[rc_ra] * cos_a - row[rc_dec] * sin_a
        df.at[row.name, rot_dec] = row[rc_ra] * sin_a + row[rc_dec] * cos_a


def read_best_start(best_name):
    """Read the alpha model string from a lensmodel best.start file.

    Args:
        best_name: Relative path to best.start from the results directory.

    Returns:
        Model string (line 2 of the file).
    """
    with open(f'{FILE_PATH}/Results_{MODEL_NAME}/{best_name}', 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines[1]


def read_chi_sqr(name):
    """Read the χ² value from a lensmodel best-chi.dat file.

    Args:
        name: Relative path to best-chi.dat from the results directory.

    Returns:
        χ² as a float.
    """
    with open(f'{FILE_PATH}/Results_{MODEL_NAME}/{name}', 'r') as f:
        numbers_list = [float(s) for s in f.readline().strip().split()]
    return numbers_list[0]


def read_source_best_data(name):
    """Read the source position string from a lensmodel best-img.dat file.

    Args:
        name: Relative path to best-img.dat from the results directory.

    Returns:
        Space-separated string of numeric tokens from the first data line.
    """
    with open(f'{FILE_PATH}/Results_{MODEL_NAME}/{name}', 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    tokens = lines[0].split()
    numbers = [
        t for t in tokens
        if t.replace('.', '', 1).replace('-', '', 1).isdigit() or 'e' in t
    ]
    return ' '.join(numbers)


def generate_random_binary_string(length, max_ones):
    """Generate a random space-separated binary string with trailing '0 0 0'.

    Args:
        length: Total string length including 3 trailing zeros.
        max_ones: Maximum number of ones allowed.

    Returns:
        Space-separated binary string.
    """
    probabilities = np.arange(1, max_ones + 1, dtype=float)
    probabilities /= probabilities.sum()
    num_ones = np.random.choice(range(1, min(length - 2, max_ones) + 1), p=probabilities)
    binary_string = ''.join(['1' if i < num_ones else '0' for i in range(length - 3)])
    shuffled_string = ''.join(np.random.choice(list(binary_string), len(binary_string), replace=False))
    return ' '.join(shuffled_string) + ' 0 0 0'


def custom_binary_string(length=10, min_ones=2, max_ones=6, fixed_positions=None, seed=None):
    """Generate a binary string with a random number of ones within given constraints.

    Args:
        length: Total length of the binary string.
        min_ones: Minimum number of ones (excluding fixed ones).
        max_ones: Maximum total number of ones.
        fixed_positions: Dict mapping index → value (0 or 1) for fixed bits.
        seed: Optional random seed.

    Returns:
        Space-separated binary string.
    """
    if seed is not None:
        np.random.seed(seed)
    if fixed_positions is None:
        fixed_positions = {}

    for idx, val in fixed_positions.items():
        if idx < 0 or idx >= length:
            raise ValueError('Fixed position out of bounds')
        if val not in (0, 1):
            raise ValueError('Fixed values must be 0 or 1')

    forced_ones = sum(fixed_positions.values())
    if forced_ones > max_ones:
        raise ValueError('Too many forced 1s for given max_ones')

    total_ones = np.random.randint(min_ones, max_ones + 1)
    if total_ones < forced_ones:
        total_ones = forced_ones

    remaining_ones = total_ones - forced_ones
    result = [0] * length

    for idx, val in fixed_positions.items():
        result[idx] = val

    available_positions = [i for i in range(length) if i not in fixed_positions]
    if remaining_ones > len(available_positions):
        raise ValueError('Not enough free positions to place remaining 1s')

    chosen_positions = np.random.choice(available_positions, remaining_ones, replace=False).tolist()
    for idx in chosen_positions:
        result[idx] = 1

    return ' '.join(map(str, result))


def binary_dict(length, ones):
    """Build a tracking dictionary for all binary strings up to `ones` ones.

    Args:
        length: Number of bits.
        ones: Maximum number of ones.

    Returns:
        Dict mapping decimal integer key → {'value': 0, 'count': 0}.
    """
    result = {}
    for ones_count in range(ones + 1):
        for ones_positions in combinations(range(length), ones_count):
            binary_num = ['0'] * length
            for pos in ones_positions:
                binary_num[pos] = '1'
            decimal_num = int(''.join(binary_num), 2)
            result[decimal_num] = {'value': 0, 'count': 0}
    return dict(sorted(result.items()))


def binary_convert_base_10(binary_string):
    """Convert a space-separated binary string (minus trailing 3 zeros) to decimal.

    Args:
        binary_string: Space-separated binary string.

    Returns:
        Integer decimal value.
    """
    processed_binary = binary_string.replace(' ', '')[:-3]
    return int(processed_binary, 2)


def get_one_positions(binary_string):
    """Return 1-based indices of '1' bits (excluding trailing 3 zeros).

    Args:
        binary_string: Space-separated binary string.

    Returns:
        List of 1-based integer positions.
    """
    trimmed = binary_string.replace(' ', '')[:-3]
    return [i + 1 for i, bit in enumerate(trimmed) if bit == '1']


def read_img_data(name, full):
    """Read image position and flux data from a lensmodel image data file.

    Args:
        name: Path to the file (relative to results dir if not full, absolute if full).
        full: If True use name as absolute path; if False prepend the results directory.

    Returns:
        List of 4 lists, each containing the numeric tokens for one image row.
    """
    path = name if full else f'{FILE_PATH}/Results_{MODEL_NAME}/{name}'
    with open(path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    numbers_list = []
    for i in range(4):
        tokens = lines[7 + i].split()
        numbers = [
            t for t in tokens
            if t.replace('.', '', 1).replace('-', '', 1).isdigit() or 'e' in t
        ]
        numbers_list.append(numbers)
    return numbers_list


def calc_new_model_diff(old_nums_list, new_nums_list):
    """Compute mean image position and flux differences between two model outputs.

    Args:
        old_nums_list: List of 4 lists of numeric strings (original model).
        new_nums_list: List of 4 lists of numeric strings (optimised model).

    Returns:
        List of [image_diff_str, mag_diff_str] formatted in scientific notation.
    """
    old_arr = np.array(old_nums_list, dtype=float)
    new_arr = np.array(new_nums_list, dtype=float)
    difference = old_arr - new_arr
    image_diff = np.mean(difference[:, :2])
    mag_diff   = np.mean(difference[:, 2])
    return ['{:e}'.format(image_diff), '{:e}'.format(mag_diff)]


def save_new_model_diff(row):
    """Compute and store image/flux differences between input and optimised models.

    Args:
        row: DataFrame row for the current observation.
    """
    d1 = read_img_data(f'{FILE_PATH}/lens_model.dat', full=True)
    d2 = read_img_data(f"{row['Image']}/best-imgfit.dat", full=False)
    model_diff = calc_new_model_diff(d1, d2)
    df.at[row.name, 'Mean Image Difference'] = model_diff[0]
    df.at[row.name, 'Mean Flux Difference']  = model_diff[1]


def find_image_order(row):
    """Identify the correct image ordering (A = max flux, then clockwise).

    Args:
        row: DataFrame row with image position and flux columns.

    Returns:
        Clockwise order indices from order_points_clockwise.
    """
    fluxes = row[['Image A Flux', 'Image B Flux', 'Image C Flux', 'Image D Flux']].astype(float)
    max_flux_index = abs(fluxes).idxmax()
    image_letter = max_flux_index.split()[-2]
    letter_list = ['A', 'B', 'C', 'D']

    if image_letter != 'A':
        df.at[row.name, 'Image A RA']  = row[f'Image {image_letter} RA']
        df.at[row.name, 'Image A Dec'] = row[f'Image {image_letter} Dec']
        df.at[row.name, f'Image {image_letter} RA']  = row['Image A RA']
        df.at[row.name, f'Image {image_letter} Dec'] = row['Image A Dec']
        df.at[row.name, 'Image A Flux'] = row[f'Image {image_letter} Flux']
        df.at[row.name, f'Image {image_letter} Flux'] = row['Image A Flux']
        swap_index = letter_list.index(image_letter)
        letter_list[0] = image_letter
        letter_list[swap_index] = 'A'

    points_arr = np.asarray([
        row[[f'Image {letter_list[i]} RA', f'Image {letter_list[i]} Dec']]
        for i in range(4)
    ]).astype(float)

    return order_points_clockwise(points_arr)


# =============================================================================
# Main pipeline
# =============================================================================

results_dir_name = f'Results_{MODEL_NAME}'
if FILE_PATH.exists():
    create_folder_if_not_exists(FILE_PATH / results_dir_name)

# Read and parse observation text file
with open(FILE_PATH / FILE_NAME, 'r') as file:
    lines = [line.strip() for line in file if line.strip()]

raw_image_blocks = []
current_block    = []
for line in lines:
    if line.startswith('#'):
        if current_block:
            raw_image_blocks.append(current_block)
            current_block = []
        current_block.append(line)
    else:
        current_block.append(line)
raw_image_blocks.append(current_block)

observation_records = [parse_image_set(block) for block in raw_image_blocks]
df = pd.DataFrame(observation_records)

df = to_arcseconds(df, INPUT_COORDINATE_UNITS)

# Reorder images (Image A = max flux, then clockwise)
image_order = df.apply(lambda row: find_image_order(row), axis=1)
df.apply(lambda row: swap_groups_per_row(row, image_order[row.name]), axis=1)

# Compute all geometric features in a single apply pass
features = df.apply(compute_features, axis=1)
df[features.columns] = features

# Fit ellipse (timed separately — computationally expensive)
t_start = time.time()
df.apply(fit_ellipse_and_distances, axis=1)
print(f'Ellipse fitting took {time.time() - t_start:.2f}s')

df.apply(lambda row: recentre_points(row, '', IMAGE_RECENTRE_SUFFIX, RECENTRE_START_INDEX), axis=1)
df.apply(lambda row: find_align_angle(row, 'Ellipse Centre', 'I'), axis=1)
df.apply(lambda row: align_image_with_ellip(row, ' I', RECENTRE_START_INDEX), axis=1)

# Load trained models and predict initial parameters
mass_loaded_model  = tf.keras.models.load_model(MODEL_PATH / MODEL_NAME_MASS)
ellip_loaded_model = tf.keras.models.load_model(MODEL_PATH / MODEL_NAME_ELLIP)

mass_prediction        = np.asarray(mass_loaded_model.predict(df[FEATURE_COLUMNS], verbose=0))
ellipticity_prediction = np.asarray(ellip_loaded_model.predict([df[FEATURE_COLUMNS]], verbose=0))

# Clamp ellipticity to valid range (0, 1)
ellipticity_prediction = np.clip(ellipticity_prediction, 0.01, 0.99)

df['Predicted Mass']  = mass_prediction
df['Predicted Ellip'] = ellipticity_prediction

# Iterative lens model optimisation
param_combo_stats = binary_dict(BINARY_LENGTH - 3, BINARY_ONES)
param_bit_stats   = binary_dict(3, 3)

for index, row in df.iterrows():
    optimize_model(
        row, str(FILE_PATH) + '/', results_dir_name,
        CHI_SQR_ACCURACY, TOLERANCE, EXPIRE_LIMIT,
        BINARY_LENGTH, BINARY_ONES, FIXED_POSITIONS,
        image_suffix=IMAGE_RECENTRE_SUFFIX,
    )

# Write optimisation_results.csv once at the end
df.to_csv(FILE_PATH / results_dir_name / 'optimisation_results.csv', index=False)

print('Completed!')
