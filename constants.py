import os

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
TRAINVAL_DATA_PATH = os.path.join(DATA_DIR, 'train')
LOG_DIR = os.path.join(PROJECT_DIR, 'logs')
NPY_TRAIN_PATH = os.path.join(DATA_DIR, 'train_val.npz')
NPY_TEST_PATH = os.path.join(DATA_DIR, 'test.npz')
RESULT_FILE = os.path.join(PROJECT_DIR, 'test_results')
INPUT_SIZE=224
