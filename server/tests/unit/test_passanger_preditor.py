from pathlib import Path
import json
import sys
from starlette.responses import JSONResponse
import pytest

# Add root to sys.path
# https://fortierq.github.io/python-import/
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

from server.predictor import passenger_predictor
from models.models import Passenger


def get_test_response_data():
    return ["Prediccion:[1] Probabilidad:[[0.46878898 0.53121102]]"]

def set_data_string():
    PassStr = '{"Survived": 0, "Pclass": 1, "Age": 20, "SibSp": 0, "Parch": 2, "Fare": 113.275, "Cabin": "D46", "Sex": "female", "Embarked": "c","Title": "Miss"}'  
    return PassStr

@pytest.mark.parametrize(get_test_response_data,set_data_string)
def test_response_parametrize(StrResponse, Data_string):
  
  PassengerPredictor = passenger_predictor()

  assert type( JSONResponse(PassengerPredictor.predict_passanger(Data_string)) ) == StrResponse
