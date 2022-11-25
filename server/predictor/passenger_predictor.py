import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from server.models.models import Passenger
import joblib
import config

sys.path.append("..")

class PassengerPredictor:

    def __init__(self):

      file_path = config.TRAINED_MODEL_DIR + config.PIPELINE_SAVE_FILE
      self.trained_model = joblib.load(filename=file_path)
  
    def predict_passanger(self, passenger):
        Dic = {'survived':[0,1,0],'pclass':passenger.Pclass, 'sex':passenger.Sex,
                                                                'age':passenger.Age, 
                                                                'sibsp':passenger.SibSp,
                                                                'parch':passenger.Parch,      
                                                                'fare':passenger.Fare,
                                                                'cabin':passenger.Cabin, 
                                                                'embarked':passenger.Embarked,
                                                                'title':passenger.Title
        }
        df=pd.DataFrame.from_dict(Dic)
        X_train, X_test, y_train, y_test = train_test_split(
                                                        df.drop(config.TARGET, axis=1),
                                                        df[config.TARGET],
                                                        test_size=0.2,
                                                        random_state=404 
                                                   )

        preds = self.trained_model.predict(X_test)
        proba = self.trained_model.predict_proba(X_test)
                         
        print(preds)
        print(proba)
        return preds
        #return "Prediccion:"+ str(round(max(preds[0]), 2)) 

     


