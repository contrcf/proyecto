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
        
        '''
        Dic = {'Survived':[0,1,0],'Pclass':passenger.Pclass, 'Sex':passenger.Sex,
                                                                'Age':passenger.Age, 
                                                                'Sibsp':passenger.SibSp,
                                                                'Parch':passenger.Parch,      
                                                                'Fare':passenger.Fare,
                                                                'Cabin':passenger.Cabin, 
                                                                'Embarked':passenger.Embarked,
                                                                'Title':passenger.Title
        }
        '''


      # Dic=[passenger.Survived,passenger.Pclass,passenger.Sex,passenger.Age,passenger.SibSp,passenger.Parch,passenger.Fare,passenger.Cabin,passenger.Embarked,passenger.Title]

       # df=pd.DataFrame.from_dict(Dic)
        #X_train, X_test, y_train, y_test = train_test_split(
        #                                                df.drop(config.TARGET, axis=1),
        #                                                df[config.TARGET],
        ##                                                test_size=0.2,
         #                                               random_state=404 
         #                                          )
          #proba = self.trained_model.predict_proba(X_test)
       # Pssngr=Passenger()
       # Pssngr.Survived=[0,1,0]
       # Pssngr.Pclass=[1,2,3]
       # Pssngr.Sex=['male','female','male',]
       # Pssngr.Age=[58,59,60]
       # Pssngr.SibSp=[0,0,1]
       # Pssngr.Parch=[2,2,2]
       # Pssngr.Fare=[113.27,110.10,120.20]
       # Pssngr.Cabin=['D48','D50','B09']
       # Pssngr.Embarked=['S','C','Q']
       # Pssngr.Title=['Mr','Miss','Mrs']

        prueba = {'Pclass':[1,2,3,passenger.Pclass],'Survived':[0,1,0,passenger.Survived],'sex':['male','female','male',passenger.Sex],'age':[58,59,60,passenger.Age],'sibsp':[0,0,1,passenger.SibSp],'parch':[2,2,2,passenger.Parch],'fare':[113.27,110.10,120.20,passenger.Fare],'cabin':['D48','D50','B09',passenger.Fare],'embarked':['S','C','Q',passenger.Embarked],'title':['Mr','Miss','Mrs',passenger.Title]}
     
        df=pd.DataFrame(prueba)
        
        X_train, X_test, y_train, y_test = train_test_split(
                                                        df.drop(config.TARGET, axis=1),
                                                        df[config.TARGET],
                                                        test_size=0.2,
                                                       random_state=404 
                                                  )
    
        preds = self.trained_model.predict(y_test)
        proba = self.trained_model.predict_proba(y_test)
        print(preds)
        print(proba)                 
        return "Prediccion:"+ str(preds) + " Probabilidad:"+ str((proba))

    

