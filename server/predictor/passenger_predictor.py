import sys
import joblib
import config
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from server.models.models import Passenger


sys.path.append("..")

class PassengerPredictor:

    def __init__(self):

      file_path = config.TRAINED_MODEL_DIR + config.PIPELINE_SAVE_FILE
      self.trained_model = joblib.load(filename=file_path)
  
    def predict_passanger(self, passenger:Passenger):
        
        prueba = {'pclass':passenger.Pclass,'Survived':passenger.Survived,'sex':passenger.Sex,'age':passenger.Age,'sibsp':passenger.SibSp,'parch':passenger.Parch,'fare':passenger.Fare,'cabin':passenger.Fare,'embarked':passenger.Embarked,'title':passenger.Title}
            
        df=pd.DataFrame(prueba,index=[1])
        preds = self.trained_model.predict(df)
        proba = self.trained_model.predict_proba(df)
                     
        return "Prediccion:"+ str(preds) + " Probabilidad:"+ str((proba))

    

