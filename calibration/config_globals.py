# Light emission can be estimated for one or more sources
# Available modes:
#   - NFSLS
#   - NZFSLS
#   - NFPLS
#   - NSLS
#   - NSLS2D
#
# Available generic models:
#   - PLS: point light source
#   - SLS: spot light source (with spread function)
#
# Availabel flags:
#   - N: normalized radiance (emission at x is fixed)
#   - Z: principal direction fixed to camera forward (SLS only)
#   - F: fixed position (known pose wrt. camera or aligned to optical center)

OPTIMIZE_LIGHT = ['NFSLS', 'FSLS', 'FSLS']
OPTIMIZE_LIGHT_INITIAL_MU = [1.35, 1.35, 1.35]  # initial spread factor for each light
OPTIMIZE_LIGHT_INITIAL_D = [[[0], [0], [1], [0]], [[0], [0], [1], [0]], [[0], [0], [1], [0]]]
OPTIMIZE_LIGHT_SHARED_PARAMS = []  # list of parameters names to share between all lights
OPTIMIZE_LIGHT_AREA_SAMPLING_LEVEL = [2, 2, 2]  # for each source. set to 0 to disable

# Bidirectional Reflectance Distribution Function (BRDF) is estimated for the
# whole calibration pattern bet but is not a useful parameter.
# Available modes:
#   - DIFFUSE: Ω → 1/π
#   - PHONG: (1 - k_s) · 1/π + k_s · |ω_i · ω_o|^n
#   - LUT: non-parametric function Ω → [0, 1]
OPTIMIZE_BRDF = 'DIFFUSE'

# Vignetting is estimated globally for the camera lens.
# Available modes:
#   - COSINE: cos(θ)^d, where θ = ∠(-ω_i, z) and d ∈ [1, ∞]
#   - LUT: non-parametric function θ → [0, 1]
#   - NONE: constant vignetting = 1.0
OPTIMIZE_VIGNETTING = 'COSINE'

# Auto-gain is estimated for each independet frame.
# Available modes:
#   - ALL: optimize value for all existing frames
#   - EXCLUDE_FIRST: fix the value of the first frame to 1.0
OPTIMIZE_GAIN = 'ALL'

# Points used for optimization
# Available modes:
#   - PATTERN: sample the points uniformly on the calibration pattern.
#       Arguments:
#         - margin: remove some sampling lines at the borders to create a margin
#                   (vertical, horizontal) or (top, horizontal, bottom) or
#                     (top, right, bottom, left)
#         - max_theta: remove some sampling points where the incedence angle of
#                      light on the pattern might cause a non-Lambertian behaviour.
#         - max_alpha: camera deformatico might be wrongly calibrated at the edges
#                      of the image, due to undersampling. we ignore these pixels.
#         - max_grid_cound: avoid oversampling some areas of the image when too
#                           many sampling points lay on the borders of the image.
#                           (n_rows, n_cols, max_count)
SAMPLING_STRATEGY = 'PATTERN'
SAMPLING_ARGUMENTS = {
                      'margin': (5, 5),
                      'max_theta': 60.0,
                      # 'max_alpha': 60.0,
                      'max_grid_count': (27, 36, 1),
                    #   'repeat': [(-0.25, -0.25), (-0.25, 0), (-0.25, +0.25),
                    #              (0, -0.25),                 (0, +0.25),
                    #              (+0.25, -0.25), (+0.25, 0), (+0.25, +0.25)]
                      'level': 2,  # for multi-level sampling (0: disable, 1: 2x, 2: 4x, 3: 8x, ...)
                    }

# Frame selection parameters
FRAME_COUNT = [2,4,8,12,8,4,2,1,1,1,1,1,1,1,1] # number of frames to use for calibration
FRAME_MAX_DISTANCE_TO_PATTERN_M = 0.015  # meters
FRAME_MIN_DISTANCE_TO_PATTERN_M = 0.003  # meters

# Image pre-processing
# Gaussian blur kernel size used to reduce noise in the images before
# processing. Set to 0 to disable.
IMREAD_GAUSSIANBLUR_KSIZE = 9

# Estimated sigma for residual normalization (see Huber loss)
HUBER_SIGMA_EST = 1.4826 * 2.2584  # 1.4826 * median err.

import os  # nopep8
import numpy as np  # nopep8

# Endoscope distal end parameters
CONFIG_PATH = os.path.dirname(os.path.realpath(__file__))
ENDOSCOPE_DISTAL_END_IMAGE = os.path.join(CONFIG_PATH, 'images/CF-H190L.png')
ENDOSCOPE_DISTAL_END_IMAGE_CENTER = (19.6185, 14.1445)
ENDOSCOPE_DISTAL_END_OUTER_DIAMETER_M = 0.0132
ENDOSCOPE_DISTAL_END_OUTER_DIAMETER_PX = 37.3565
ENDOSCOPE_LIGHT_DIAMETERS_M = [0.002279684338325, 0.002279684338325, 0.002279684338325]
ENDOSCOPE_LIGHT_CENTERS = \
    [np.array([[0.00323582241377003], [0.000553702836186474], [0.0], [1.0]]),
     np.array([[-0.00277399113942687], [0.00265685489807664], [0.0], [1.0]]),
     np.array([[-0.00127153775112765], [-0.00314024065423688], [0.0], [1.0]])]

from datetime import datetime  # nopep8
RESULTS_NAME = datetime.today().strftime("%Y%m%d.%H%M%S")


def get_all():
    return {
        'OPTIMIZE_LIGHT': OPTIMIZE_LIGHT,
        'OPTIMIZE_LIGHT_INITIAL_MU': OPTIMIZE_LIGHT_INITIAL_MU,
        'OPTIMIZE_LIGHT_INITIAL_D': OPTIMIZE_LIGHT_INITIAL_D,
        'OPTIMIZE_LIGHT_SHARED_PARAMS': OPTIMIZE_LIGHT_SHARED_PARAMS,
        'OPTIMIZE_LIGHT_AREA_SAMPLING_LEVEL': OPTIMIZE_LIGHT_AREA_SAMPLING_LEVEL,
        'OPTIMIZE_BRDF': OPTIMIZE_BRDF,
        'OPTIMIZE_VIGNETTING': OPTIMIZE_VIGNETTING,
        'OPTIMIZE_GAIN': OPTIMIZE_GAIN,
        'SAMPLING_STRATEGY': SAMPLING_STRATEGY,
        'SAMPLING_ARGUMENTS': SAMPLING_ARGUMENTS,
        'FRAME_COUNT': FRAME_COUNT,
        'FRAME_MAX_DISTANCE_TO_PATTERN_M': FRAME_MAX_DISTANCE_TO_PATTERN_M,
        'FRAME_MIN_DISTANCE_TO_PATTERN_M': FRAME_MIN_DISTANCE_TO_PATTERN_M,
        'IMREAD_GAUSSIANBLUR_KSIZE': IMREAD_GAUSSIANBLUR_KSIZE,
        'HUBER_SIGMA_EST': HUBER_SIGMA_EST,
        'CONFIG_PATH': CONFIG_PATH,
        'ENDOSCOPE_DISTAL_END_IMAGE': ENDOSCOPE_DISTAL_END_IMAGE,
        'ENDOSCOPE_DISTAL_END_IMAGE_CENTER': ENDOSCOPE_DISTAL_END_IMAGE_CENTER,
        'ENDOSCOPE_DISTAL_END_OUTER_DIAMETER_M': ENDOSCOPE_DISTAL_END_OUTER_DIAMETER_M,
        'ENDOSCOPE_DISTAL_END_OUTER_DIAMETER_PX': ENDOSCOPE_DISTAL_END_OUTER_DIAMETER_PX,
        'ENDOSCOPE_LIGHT_DIAMETERS_M': ENDOSCOPE_LIGHT_DIAMETERS_M,
        'ENDOSCOPE_LIGHT_CENTERS': ENDOSCOPE_LIGHT_CENTERS,
        'RESULTS_NAME': RESULTS_NAME,
    }
