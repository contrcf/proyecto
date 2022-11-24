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

NUMERICAL_VARS = ['pclass','age','sibsp','parch','fare']
CATEGORICAL_VARS = ['sex','cabin','embarked','title']

NUMERICAL_VARS_WITH_NA = ['age','fare']
CATEGORICAL_VARS_WITH_NA = ['cabin','embarked']
NUMERICAL_NA_NOT_ALLOWED = [var for var in NUMERICAL_VARS if var not in NUMERICAL_VARS_WITH_NA]
CATEGORICAL_NA_NOT_ALLOWED = [var for var in CATEGORICAL_VARS if var not in CATEGORICAL_VARS_WITH_NA]