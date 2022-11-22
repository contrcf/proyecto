import sys

import numpy as np
#from sklearn.datasets import load_passenger
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from models.models import Passenger

sys.path.append("..")

class PassengerPredictor:

    def __init__(self):
        self.X, self.y = load_passenger(return_X_y=True)

        

        self.clf = self.train_model()
      
    def train_model(self) -> LogisticRegression:
        return LogisticRegression(solver='lbfgs',
                                  max_iter=1000,
                                  multi_class='multinomial').fit(self.X, self.y)

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
                        

       # print({'class': self.iris_type[np.argmax(prediction)],
       #         'probability': round(max(prediction[0]), 2)})

       # return {'class': self.iris_type[np.argmax(prediction)],
       #         'probability': round(max(prediction[0]), 2)}

        return {'probability': round(max(prediction[0]), 2)}


