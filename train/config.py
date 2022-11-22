DATASETS_DIR = 'data/' 
URL = 'https://www.openml.org/data/get_csv/16826755/phpMYEkMl'
DROP_COLS = ['boat','body','home.dest','ticket','name']
RETRIEVED_DATA = 'raw-data.csv'


SEED_SPLIT = 404
TRAIN_DATA_FILE = DATASETS_DIR + 'train.csv'
TEST_DATA_FILE  = DATASETS_DIR + 'test.csv'


TARGET = 'survived'
FEATURES = ['pclass','sex','age','sibsp','parch','fare','cabin','embarked','title']
NUMERICAL_VARS = ['pclass','age','sibsp','parch','fare']
CATEGORICAL_VARS = ['sex','cabin','embarked','title']


NUMERICAL_VARS_WITH_NA = ['age','fare']
CATEGORICAL_VARS_WITH_NA = ['cabin','embarked']
NUMERICAL_NA_NOT_ALLOWED = [var for var in NUMERICAL_VARS if var not in NUMERICAL_VARS_WITH_NA]
CATEGORICAL_NA_NOT_ALLOWED = [var for var in CATEGORICAL_VARS if var not in CATEGORICAL_VARS_WITH_NA]


SEED_MODEL = 404

TRAINED_MODEL_DIR = 'trained_models/'
PIPELINE_NAME = 'logistic_regression'
PIPELINE_SAVE_FILE = f'{PIPELINE_NAME}_output.pkl'

