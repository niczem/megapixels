from os.path import join

DATA_STORE = 'data'
DIR_APP = join(DATA_STORE, '')

# face recognition
FACEREC_NUM_ERS = 5

# -----------------------------------------------------------------------------
# Directory names for filtering
# -----------------------------------------------------------------------------
MIN_FACE_SIZE = 50
MAX_FACE_SIZE = 600
HAAR_SCALE = 1.1
# haar frontalface at 20-30 --> pose filter, 7 = too easy, 10 = medium, 
#   15 = selective, 20 = picky, 30 = pose filter
HAAR_OVERLAPS = 35
MIN_IMAGE_SIZE = (200, 200)
MAX_IMAGE_SIZE = (1000, 1000)
QUERY_IMAGE_WIDTH = 600
QUERY_IMAGE_HEIGHT = 600
FACEREC_NUM_JITTERS = 5

# Glow
FP_GLOW_OPTIMIZED = join(DIR_APP, 'glow/graph_optimized.pb')
FP_GLOW_UNOPTIMIZED = join(DIR_APP, 'glow/graph_unoptimized.pb')
FP_GLOW_ATTR = join(DIR_APP, 'glow/attr.npy')
FP_GLOW_X = join(DIR_APP, 'glow/x.npy')
FP_GLOW_Y = join(DIR_APP, 'glow/y.npy')
FP_GLOW_Z = join(DIR_APP, 'glow/z_manipulate.npy')
USE_OPTIMIZED = True

# flask
DIR_STATIC_LOCAL = 'app/server/static'
DIR_PROCESSED_LOCAL = join(DIR_STATIC_LOCAL, 'processed')
DIR_DATASET_LOCAL = join(DIR_STATIC_LOCAL, 'dataset')
DIR_UPLOADS_LOCAL = join(DIR_STATIC_LOCAL, 'uploads')
DIR_TEMPLATES_LOCAL = join('app/server/templates/')

DIR_UPLOADS_STATIC = 'uploads'
DIR_PROCESSED_STATIC = 'processed'
DIR_DATASET_STATIC = 'dataset'  # symlink to actual location


# -----------------------------------------------------------------------------
# Directory names for filtering
# -----------------------------------------------------------------------------
DIRNAME_GROUND_TRUTH = 'confirm'
DIRNAME_REJECTED = 'rejected'
DIRNAME_APPROVED = 'approved'
#APPROVED_EXT = 'jpg'


DEFAULT_PORT = 5000
DEFAULT_IP = '0.0.0.0'
DEFAULT_MATCHES = 3  # number of matches returned in JSON
DEFAULT_POINTS = 3  # must be odd number
# generally use 0, increasing causes weird things to happen
# but may give slightly larger crop area
DEFAULT_EXPAND = 0 
DIR_DATASET = join(DATA_STORE, 'datasets/megapixels')
DEFAULT_ENCODINGS = join(DIR_DATASET, 'ch4/encodings.json')
DEFAULT_TARGETS = join(DIR_DATASET, 'ch4/downloads/')
DEFAULT_TARGETS_SYMLINK = 'app/server/static/dataset'

import logging
LOGLEVEL = logging.INFO
LOGFILE_FORMAT = "%(levelname)s: %(message)s"
LOGFILE = None  # filepath/to/your.log file