# Path to the folder containing the paintings
import math

# Number of painters in sample set
NUM_PAINTERS = 3

# Number of paintings per painter in sample set
NUM_PAINTINGS = 100

# Percentage of data for training
PCT_TRAINING = 0.8

# Dimensions of paintings for CNN input
IMAGE_DIM = (150, 150)

# Batch size per epoch
# WARNING: Batch size must not be larger than number of validation samples!
# Otherwise, number validation samples / batch size = 0!
BATCH_SIZE = 10

# Number of epochs
NUM_EPOCH = 50

# Some static names
FOLDER_NAME_TRAINING = 'training'
FOLDER_NAME_VALIDATION = 'validation'
URL_PAINTINGS = './paintings'
URL_CNN_MODEL = './model.h5py'
URL_DATA = './data'
CLASS_MODE = 'categorical'
NUM_TRAINING = math.ceil(NUM_PAINTINGS * PCT_TRAINING)
NUM_VALIDATION = math.floor(NUM_PAINTINGS * (1 - PCT_TRAINING))
NUM_STEPS_PER_EPOCH = math.floor(NUM_TRAINING / BATCH_SIZE)
NUM_VALIDATION_STEPS = math.floor(NUM_VALIDATION / BATCH_SIZE)
