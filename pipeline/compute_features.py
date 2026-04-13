import numpy as np
import pandas as pd
import sys
import time
from pathlib import Path
from scipy.optimize import minimize
from columns import INTERSECTION_COLS
from config import FILE_NAME, OVERWRITE_FEATURES

# =============================================================================
# Constants
# =============================================================================
EPSILON = 1e-9
IMAGE_RA_COL = 7      # column index of Image A RA in the input CSV
IMAGE_DEC_COL = 8     # column index of Image A Dec in the input CSV
IMAGE_STRIDE = 4      # column stride between images (RA, Dec, Flux, TimeDel)
IMAGE_FLUX_OFFSET = 2 # offset from image RA column to flux column
COLOCATED_EPS = 1e-9  # images closer than this (arcsec) are treated as co-located

# =============================================================================
# Paths
# =============================================================================
FILE_PATH = Path('/home/alex/Documents/Grav Lens/data/sources')

# User configuration — edit config.py
# FILE_NAME and OVERWRITE_FEATURES are imported from config.py at the top of this file.

# =============================================================================
# Functions
# =============================================================================

def dist(x1, y1, x2, y2):
    """Return the Euclidean distance between two 2D points."""
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def order_points_clockwise(points):
    """Return indices that sort 4 image positions clockwise from the topmost point.

    Args:
        points: (4, 2) array of (RA, Dec) image coordinates.

    Returns:
        Array of 4 indices representing the clockwise ordering.
    """
    top_most_point_index = np.argmax(points[:, 1])
    polar_angles = np.arctan2(
        points[:, 1] - points[top_most_point_index, 1],
        points[:, 0] - points[top_most_point_index, 0],
    )
    clockwise_order_indices = np.flip(np.argsort(polar_angles))
    roll_point = np.argmin(clockwise_order_indices)
    clockwise_order_indices = np.roll(clockwise_order_indices, -roll_point)
    return clockwise_order_indices


def area(a, b, c):
    """Compute the area of a triangle from its three side lengths (Heron's formula).

    Args:
        a, b, c: Side lengths.

    Returns:
        Triangle area. Returns 0 for degenerate near-zero cases.
    """
    s = (a + b + c) / 2
    d = s * (s - a) * (s - b) * (s - c)
    if (d < 0) and (s - c > -EPSILON):
        d = 0
    return np.sqrt(d)


def area_vec(a, b, c):
    """Vectorised Heron's formula for arrays of triangle side lengths.

    Args:
        a, b, c: Arrays of side lengths.

    Returns:
        Array of triangle areas.
    """
    s = (a + b + c) / 2
    d = s * (s - a) * (s - b) * (s - c)
    d = np.where((d < 0) & (s - c > -EPSILON), 0.0, d)
    return np.sqrt(d)


def distance_from_perimeter(x, points):
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
    """Fit a 2D ellipse to 4 image positions, constraining the centre within the image bounding box.

    Args:
        points: (4, 2) array of image coordinates.

    Returns:
        Tuple (x_c, y_c) of the ellipse centre.
    """
    initial_guess = [1.0, 1.0, np.mean(points[:, 0]), np.mean(points[:, 1]), 0.0]
    cons = (
        {'type': 'ineq', 'fun': lambda x: np.max(points[:, 0]) - x[2]},
        {'type': 'ineq', 'fun': lambda x: x[2] - np.min(points[:, 0])},
        {'type': 'ineq', 'fun': lambda x: np.max(points[:, 1]) - x[3]},
        {'type': 'ineq', 'fun': lambda x: x[3] - np.min(points[:, 1])},
    )
    result = minimize(
        distance_from_perimeter, initial_guess, args=(points,),
        constraints=cons, method='trust-constr',
    )
    _, _, x_c, y_c, _ = result.x
    return x_c, y_c


def find_intersection_point(row):
    """Find intersection of diagonals AC and BD; return centred image coordinates.

    Args:
        row: DataFrame row with Image A/B/C/D RA/Dec columns.

    Returns:
        pd.Series with intersection point and intersection-centred image coordinates.
        Values are NaN if the diagonals are parallel.
    """
    x1, y1 = row['Image A RA'], row['Image A Dec']
    x2, y2 = row['Image B RA'], row['Image B Dec']
    x3, y3 = row['Image C RA'], row['Image C Dec']
    x4, y4 = row['Image D RA'], row['Image D Dec']

    verticalAC = (x1 == x3)
    verticalBD = (x2 == x4)

    if verticalAC or verticalBD:
        if verticalAC and verticalBD:
            return pd.Series({k: np.nan for k in INTERSECTION_COLS})
        elif verticalAC:
            m2 = (y4 - y2) / (x4 - x2)
            x_intersect = x1
            y_intersect = m2 * (x_intersect - x4) + y4
        else:
            m1 = (y3 - y1) / (x3 - x1)
            x_intersect = x4
            y_intersect = m1 * (x_intersect - x1) + y1
    else:
        m1 = (y3 - y1) / (x3 - x1)
        m2 = (y4 - y2) / (x4 - x2)
        if m1 == m2:
            return pd.Series({k: np.nan for k in INTERSECTION_COLS})
        x_intersect = ((y2 - m2 * x2) - (y1 - m1 * x1)) / (m1 - m2)
        y_intersect = m1 * (x_intersect - x1) + y1

    return pd.Series({
        'Intersection Point RA':  x_intersect,
        'Intersection Point Dec': y_intersect,
        'New A RA':  x1 - x_intersect,
        'New A Dec': y1 - y_intersect,
        'New B RA':  x2 - x_intersect,
        'New B Dec': y2 - y_intersect,
        'New C RA':  x3 - x_intersect,
        'New C Dec': y3 - y_intersect,
        'New D RA':  x4 - x_intersect,
        'New D Dec': y4 - y_intersect,
    })


def compute_intersection_points(df):
    """Vectorised intersection of diagonals AC and BD for all rows simultaneously.

    Replaces a row-by-row apply(find_intersection_point) call with pure numpy
    operations, giving a ~100x speedup on large datasets.

    Args:
        df: DataFrame with Image A/B/C/D RA/Dec columns.

    Returns:
        DataFrame with INTERSECTION_COLS columns, indexed to match df.
        Rows where diagonals are parallel or both vertical yield NaN.
    """
    x1, y1 = df['Image A RA'].values, df['Image A Dec'].values
    x2, y2 = df['Image B RA'].values, df['Image B Dec'].values
    x3, y3 = df['Image C RA'].values, df['Image C Dec'].values
    x4, y4 = df['Image D RA'].values, df['Image D Dec'].values

    # Slopes: NaN where the diagonal is vertical or effectively vertical.
    # Use an absolute threshold rather than exact equality — lensmodel can
    # produce RA values of ~1e-14 for on-axis images, which are physically
    # zero but pass an exact != check, giving slopes of ~1e12 that corrupt
    # the intersection and all downstream New coord columns.
    SLOPE_EPS = 1e-9
    with np.errstate(divide='ignore', invalid='ignore'):
        m1 = np.where(np.abs(x3 - x1) > SLOPE_EPS, (y3 - y1) / (x3 - x1), np.nan)
        m2 = np.where(np.abs(x4 - x2) > SLOPE_EPS, (y4 - y2) / (x4 - x2), np.nan)

    N = len(df)
    xi = np.full(N, np.nan)
    yi = np.full(N, np.nan)

    # Normal case: neither line vertical and slopes differ
    mask = np.isfinite(m1) & np.isfinite(m2) & (m1 != m2)
    xi[mask] = (
        (y2[mask] - m2[mask] * x2[mask]) - (y1[mask] - m1[mask] * x1[mask])
    ) / (m1[mask] - m2[mask])
    yi[mask] = m1[mask] * (xi[mask] - x1[mask]) + y1[mask]

    # AC vertical, BD not vertical
    mask = ~np.isfinite(m1) & np.isfinite(m2)
    xi[mask] = x1[mask]
    yi[mask] = m2[mask] * (x1[mask] - x4[mask]) + y4[mask]

    # BD vertical, AC not vertical
    mask = np.isfinite(m1) & ~np.isfinite(m2)
    xi[mask] = x4[mask]
    yi[mask] = m1[mask] * (x4[mask] - x1[mask]) + y1[mask]

    # Both vertical or parallel slopes → xi/yi remain NaN (correct behaviour)

    return pd.DataFrame({
        'Intersection Point RA':  xi,
        'Intersection Point Dec': yi,
        'New A RA':  x1 - xi, 'New A Dec': y1 - yi,
        'New B RA':  x2 - xi, 'New B Dec': y2 - yi,
        'New C RA':  x3 - xi, 'New C Dec': y3 - yi,
        'New D RA':  x4 - xi, 'New D Dec': y4 - yi,
    }, index=df.index)


# =============================================================================
# Column headers
# =============================================================================
headers = [
    'Source RA position', 'Source Dec position', 'Mass Parameter', 'Ellipticity',
    'Ellipticity Angle', 'Shear', 'Shear Angle',
    'Image A RA', 'Image A Dec', 'Image A Flux', 'Image A Time Del',
    'Image B RA', 'Image B Dec', 'Image B Flux', 'Image B Time Del',
    'Image C RA', 'Image C Dec', 'Image C Flux', 'Image C Time Del',
    'Image D RA', 'Image D Dec', 'Image D Flux', 'Image D Time Del',
    'Ratio AD', 'Ratio AC', 'Ratio AB', 'Ratio BD', 'Ratio BC', 'Ratio CD',
    'Sum AB,DA', 'Sum BC, CD',
    'Area ABD', 'Area ABC', 'Area BCD', 'Area ACD',
    'Mag Ratio A', 'Mag Ratio B', 'Mag Ratio C', 'Mag Ratio D',
    'Rotation Angle',
]

# =============================================================================
# Skip check
# =============================================================================
_features_out = FILE_PATH / f'{FILE_NAME}_features.parquet'
if not OVERWRITE_FEATURES and _features_out.exists():
    print(
        f"Skipping: {_features_out} already exists.\n"
        "Set OVERWRITE_FEATURES = True in config.py, or change FILE_NAME, to regenerate."
    )
    sys.exit(0)

# =============================================================================
# Load data
# =============================================================================
data = pd.read_parquet(FILE_PATH / f'{FILE_NAME}.parquet')
N = len(data)

# =============================================================================
# Reorder images B, C, D to run clockwise from Image A (highest flux)
# Vectorised: compute all orderings as batch numpy operations.
# =============================================================================

# Extract RA/Dec for all 4 images → shape (N, 4, 2)
image_coord_col_idx = [IMAGE_RA_COL + IMAGE_STRIDE * j + k for j in range(4) for k in range(2)]
image_coords = data.iloc[:, image_coord_col_idx].to_numpy().reshape(N, 4, 2)

# Batch order_points_clockwise: find topmost point and sort by descending polar angle
topmost_idx = np.argmax(image_coords[:, :, 1], axis=1)             # (N,)
topmost_dec = image_coords[np.arange(N), topmost_idx, 1]
topmost_ra  = image_coords[np.arange(N), topmost_idx, 0]
dy = image_coords[:, :, 1] - topmost_dec[:, np.newaxis]
dx = image_coords[:, :, 0] - topmost_ra[:, np.newaxis]
polar_angles    = np.arctan2(dy, dx)                                # (N, 4)
clockwise_order = np.fliplr(np.argsort(polar_angles, axis=1))       # (N, 4)

# Roll each row so the element with the smallest value (0) comes first
roll_offsets  = np.argmin(clockwise_order, axis=1)                  # (N,)
rolled_order  = (np.arange(4)[np.newaxis, :] + roll_offsets[:, np.newaxis]) % 4
clockwise_order = clockwise_order[np.arange(N)[:, np.newaxis], rolled_order]  # (N, 4)

# Flip order [0,1,2,3] → [0,3,2,1] where the adjacent image is closer than the opposite
p0 = image_coords[np.arange(N), clockwise_order[:, 0], :]
p1 = image_coords[np.arange(N), clockwise_order[:, 1], :]
p3 = image_coords[np.arange(N), clockwise_order[:, 3], :]
dist_to_next = np.sqrt(np.sum((p0 - p1) ** 2, axis=1))
dist_to_prev = np.sqrt(np.sum((p0 - p3) ** 2, axis=1))
should_flip  = np.abs(dist_to_prev) < np.abs(dist_to_next)
image_order  = clockwise_order.copy()
image_order[should_flip] = clockwise_order[should_flip][:, [0, 3, 2, 1]]

# Co-location fix: if would-be B (position 1) and D (position 3) are at the same
# location, the BD diagonal is degenerate. Swap C and D so the co-located pair
# sits at consecutive positions (B, C) instead of diagonal positions (B, D).
p1_final    = image_coords[np.arange(N), image_order[:, 1], :]
p3_final    = image_coords[np.arange(N), image_order[:, 3], :]
dist_b_to_d = np.sqrt(np.sum((p1_final - p3_final) ** 2, axis=1))
colocated_bd = dist_b_to_d < COLOCATED_EPS
image_order[colocated_bd] = image_order[colocated_bd][:, [0, 1, 3, 2]]

# Reorder image data in columns 7-22 (4 images × 4 cols each) → shape (N, 4, 4)
# .copy() is required: without it, to_numpy() returns a view of the underlying
# DataFrame buffer. Writing to data.iloc[:, 11:15] (Image B) would then silently
# corrupt image_data_block[:, 1, :], causing d_src_idx=1 to read stale B data
# and produce Image D = Image B (both pointing to the same overwritten slot).
image_data_block = data.iloc[:, 7:23].to_numpy().copy().reshape(N, 4, 4)

# Map clockwise-order indices to source image indices for B, C, D slots.
# images[idx-1] in the original: idx=0→D, idx=1→B, idx=2→C, idx=3→D
b_src_idx = np.where(image_order[:, 1] == 0, 3, image_order[:, 1])
c_src_idx = np.where(image_order[:, 2] == 0, 3, image_order[:, 2])
d_src_idx = np.where(image_order[:, 3] == 0, 3, image_order[:, 3])

data.iloc[:, 11:15] = image_data_block[np.arange(N), b_src_idx, :]
data.iloc[:, 15:19] = image_data_block[np.arange(N), c_src_idx, :]
data.iloc[:, 19:23] = image_data_block[np.arange(N), d_src_idx, :]

# =============================================================================
# Compute geometric features (vectorised)
# =============================================================================

A_ra  = data.iloc[:, IMAGE_RA_COL].values
A_dec = data.iloc[:, IMAGE_DEC_COL].values
B_ra  = data.iloc[:, IMAGE_RA_COL + IMAGE_STRIDE].values
B_dec = data.iloc[:, IMAGE_DEC_COL + IMAGE_STRIDE].values
C_ra  = data.iloc[:, IMAGE_RA_COL + 2 * IMAGE_STRIDE].values
C_dec = data.iloc[:, IMAGE_DEC_COL + 2 * IMAGE_STRIDE].values
D_ra  = data.iloc[:, IMAGE_RA_COL + 3 * IMAGE_STRIDE].values
D_dec = data.iloc[:, IMAGE_DEC_COL + 3 * IMAGE_STRIDE].values

# Pairwise distances (raw, used for areas and sums before normalisation)
d_AD = dist(A_ra, A_dec, D_ra, D_dec)
d_AC = dist(A_ra, A_dec, C_ra, C_dec)
d_AB = dist(A_ra, A_dec, B_ra, B_dec)
d_BD = dist(B_ra, B_dec, D_ra, D_dec)
d_BC = dist(B_ra, B_dec, C_ra, C_dec)
d_CD = dist(C_ra, C_dec, D_ra, D_dec)

# Sums computed from raw distances
sum_ABDA = d_AD + d_BD
sum_BCCD = d_AC + d_AB

# Triangle areas using Heron's formula on raw distances
area_ABD = area_vec(d_AD, d_AB, d_BD)
area_ABC = area_vec(d_AC, d_AB, d_BC)
area_BCD = area_vec(d_BD, d_BC, d_CD)
area_ACD = area_vec(d_AD, d_AC, d_CD)

# Normalised flux ratios
flux_col_idx    = [IMAGE_RA_COL + IMAGE_FLUX_OFFSET + IMAGE_STRIDE * j for j in range(4)]
flux_values     = data.iloc[:, flux_col_idx].to_numpy()
total_flux      = np.sum(np.abs(flux_values), axis=1, keepdims=True)
norm_flux_ratios = np.abs(flux_values) / total_flux  # (N, 4)

# Normalised distance ratios
pairwise_distances = np.column_stack([d_AD, d_AC, d_AB, d_BD, d_BC, d_CD])
distance_sum       = pairwise_distances.sum(axis=1, keepdims=True)
norm_dist_ratios   = pairwise_distances / distance_sum

feature_df = pd.DataFrame({
    'Ratio AD':    norm_dist_ratios[:, 0],
    'Ratio AC':    norm_dist_ratios[:, 1],
    'Ratio AB':    norm_dist_ratios[:, 2],
    'Ratio BD':    norm_dist_ratios[:, 3],
    'Ratio BC':    norm_dist_ratios[:, 4],
    'Ratio CD':    norm_dist_ratios[:, 5],
    'Sum AB,DA':   sum_ABDA,
    'Sum BC, CD':  sum_BCCD,
    'Area ABD':    area_ABD,
    'Area ABC':    area_ABC,
    'Area BCD':    area_BCD,
    'Area ACD':    area_ACD,
    'Mag Ratio A': norm_flux_ratios[:, 0],
    'Mag Ratio B': norm_flux_ratios[:, 1],
    'Mag Ratio C': norm_flux_ratios[:, 2],
    'Mag Ratio D': norm_flux_ratios[:, 3],
    'Rotation Angle': np.zeros(N),
})

# =============================================================================
# Centre images on diagonal intersection point
# Skips the intermediate _not_centred.csv write/read — data stays in memory.
# compute_intersection_points() replaces the row-by-row apply for ~100x speedup.
# =============================================================================
output_df = pd.concat([data, feature_df], axis=1)
output_df[INTERSECTION_COLS] = compute_intersection_points(output_df)

# Drop rows where the diagonal intersection is degenerate (NaN) — these arise
# from configurations where two images are co-located or diagonals are parallel,
# producing undefined or extreme intersection coordinates that corrupt NN inputs.
n_before = len(output_df)
output_df = output_df[output_df['Intersection Point RA'].notna()].reset_index(drop=True)
n_dropped = n_before - len(output_df)
if n_dropped:
    print(f"Dropped {n_dropped} rows with degenerate diagonal intersections ({n_dropped/n_before*100:.2f}%).")

output_df.to_parquet(FILE_PATH / f'{FILE_NAME}_features.parquet', index=False)
