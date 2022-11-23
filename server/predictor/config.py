import os
BASE_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
#URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"

SEED_SPLIT = 404
SEED_MODEL = 404
REG_MODEL_NAME = os.path.realpath(os.path.join(BASE_DIR, "models", "reg_model.sav"))

TRAIN_DATA = os.path.realpath(os.path.join(BASE_DIR, "data", "train.csv"))

DATA_DIR = os.path.realpath(os.path.join(BASE_DIR, "data"))


TARGET = "survived"

TRAINED_MODEL_DIR = './train/trained_models/'
PIPELINE_NAME = 'logistic_regression'
PIPELINE_SAVE_FILE = f'{PIPELINE_NAME}_output.pkl'