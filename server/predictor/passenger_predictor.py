import sys
from . import config_
import numpy as np
from pandas import pd
#from sklearn.datasets import load_passenger
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from models.models import Passenger
import joblib

sys.path.append("..")

class PassengerPredictor:

    def __init__(self):

     input_data =  pd.read_csv(config_.TRAIN_DATA_FILE)


     validated_data = input_data

     if input_data[config_.NUMERICAL_NA_NOT_ALLOWED].isnull().any().any():
        validated_data = validated_data.dropna(subset=config_.NUMERICAL_NA_NOT_ALLOWED)
        
     if input_data[config_.CATEGORICAL_NA_NOT_ALLOWED].isnull().any().any():
        validated_data = validated_data.dropna(subset=config_.CATEGORICAL_NA_NOT_ALLOWED) 
     
     self.clf = self.train_model(validated_data)

      
    def train_model(validated_data): 

     file_path = config_.TRAINED_MODEL_DIR + config_.PIPELINE_SAVE_FILE
     trained_model = joblib.load(filename=file_path)

     preds = trained_model.predict(validated_data)
     proba = trained_model.predict_proba(validated_data)

     pd.concat([validated_data.reset_index(), pd.Series(preds, name='preds'), pd.Series(pd.DataFrame(proba)[1], name='probas')], 1).head()
       
     return "regresar la prediccion"  






    def predict_passanger(self, passenger: Passenger):
        X = [passenger.Survived, passenger.Pclass, passenger.Age, passenger.SibSp,
                                                                  passenger.Parch,      
                                                                  passenger.Fare,
                                                                  passenger.Cabin, 
                                                                  passenger.Sex_female,
                                                                  passenger.Sex_male,  
                                                                  passenger.Embarked_S,
                                                                  passenger.Embarked_C,
                                                                  passenger.Embarked_Q,
                                                                  
        ]
        
        prediction = self.clf.predict_proba([X])
                        

      

        return {'probability': round(max(prediction[0]), 2)}


