import sys

import numpy as np
from pandas import pd
from sklearn.linear_model import LogisticRegression
from models.models import Passenger
import joblib
import config

sys.path.append("..")

class PassengerPredictor:

    def __init__(self):
      file_path = config.TRAINED_MODEL_DIR + config.PIPELINE_SAVE_FILE
      self.trained_model = joblib.load(filename=file_path)

    def predict_passanger(self, passenger: Passenger):
        X = [passenger.Survived, passenger.Pclass, passenger.Age, passenger.SibSp,
                                                                  passenger.Parch,      
                                                                  passenger.Fare,
                                                                  passenger.Cabin, 
                                                                  passenger.Sex,
                                                                  passenger.Embarked
                                                                  
        ]
        
        validated_data = X

        if X[config.NUMERICAL_NA_NOT_ALLOWED].isnull().any().any():
           validated_data = validated_data.dropna(subset=config.NUMERICAL_NA_NOT_ALLOWED)
        
        if X[config.CATEGORICAL_NA_NOT_ALLOWED].isnull().any().any():
           validated_data = validated_data.dropna(subset=config.CATEGORICAL_NA_NOT_ALLOWED) 
        
        preds = self.trained_model.predict(X)
        proba = self.trained_model.predict_proba(X)
                         

        return "Prediccion:"+ str(round(max(preds[0]), 2)) + " Probabilidad:" + str(round(max(proba[0]), 2))

     


