"""Shared column-name lists for the gravitational lensing pipeline.

Import from this module in every pipeline script rather than redefining lists
locally. A single source of truth prevents cross-file ordering mismatches (the
root cause of the new_coords RA-grouped vs RA/Dec-paired ordering bug).

Usage
-----
    from columns import DIST_RAT_COLS, NEW_COORD_COLS, NN_INPUT_COLS

Scripts must be run from the `pipeline/` directory (or that directory
must be on PYTHONPATH) for the import to resolve.
"""

# =============================================================================
# Source position
# =============================================================================
SOURCE_COLS = ['Source RA position', 'Source Dec position']

# =============================================================================
# Lens model parameters
# =============================================================================
MASS_PARAM_COL   = ['Mass Parameter']
ELLIP_PARAM_COL  = ['Ellipticity']
PARAM_COLS       = [
    'Mass Parameter', 'Ellipticity', 'Ellipticity Angle', 'Shear', 'Shear Angle',
]

# =============================================================================
# Raw image coordinates (before centring)
# =============================================================================
IMAGE_COORD_COLS = [
    'Image A RA',  'Image A Dec',
    'Image B RA',  'Image B Dec',
    'Image C RA',  'Image C Dec',
    'Image D RA',  'Image D Dec',
]

# =============================================================================
# Intersection-centred coordinates (output of compute_features.py)
# Paired RA/Dec ordering matches the column layout in the _Features parquet output.
# =============================================================================
INTERSECTION_COLS = [
    'Intersection Point RA', 'Intersection Point Dec',
    'New A RA', 'New A Dec',
    'New B RA', 'New B Dec',
    'New C RA', 'New C Dec',
    'New D RA', 'New D Dec',
]

NEW_COORD_COLS = [
    'New A RA', 'New A Dec',
    'New B RA', 'New B Dec',
    'New C RA', 'New C Dec',
    'New D RA', 'New D Dec',
]

# =============================================================================
# Geometric features derived by AllInOneProcessing.py
# =============================================================================
DIST_RAT_COLS = ['Ratio AD', 'Ratio AC', 'Ratio AB', 'Ratio BD', 'Ratio BC', 'Ratio CD']
SUM_COLS      = ['Sum AB,DA', 'Sum BC, CD']
AREA_COLS     = ['Area ABD', 'Area ABC', 'Area BCD', 'Area ACD']
MAG_COLS      = ['Mag Ratio A', 'Mag Ratio B', 'Mag Ratio C', 'Mag Ratio D']

# =============================================================================
# Neural network inputs and outputs
# NN_INPUT_COLS defines the 14-feature vector fed to both mass and ellipticity
# networks. Both training (train_networks.py) and prediction
# (evaluate_predictions.py, fit_lens_model.py) must use this exact list — order matters.
# =============================================================================
NN_INPUT_COLS        = DIST_RAT_COLS + NEW_COORD_COLS  # 14 features
NN_MASS_OUTPUT_COLS  = MASS_PARAM_COL
NN_ELLIP_OUTPUT_COLS = ELLIP_PARAM_COL
