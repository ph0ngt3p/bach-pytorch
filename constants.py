import os

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
TRAINVAL_DATA_PATH = os.path.join(DATA_DIR, 'train')
LOG_DIR = os.path.join(PROJECT_DIR, 'logs')
HDF5_TRAIN_PATH = os.path.join(DATA_DIR, 'train_val_data.hdf5')
HDF5_TEST_PATH = os.path.join(DATA_DIR, 'test_data.hdf5')
RESULT_FILE = os.path.join(PROJECT_DIR, 'test_results')
