import json
import logging
from fastapi import FastAPI
from starlette.responses import JSONResponse
from .train_model import GetDataTrainModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(levelname)s: %(asctime)s|%(name)s|%(message)s")
file_handler = logging.FileHandler("server.log")
file_handler.setFormatter(formatter)

app = FastAPI()

@app.get("/")
def read_root():
    logger.info("Titanic Predictor is ready to go!")
    return "Titanic Train Model is ready to go!"


@app.get("/healthcheck", status_code=200)
async def healthcheck():
    logger.info("Servers are ready to go!")
    return "Servers are ready to go!"


@app.post("/passangertrainmodel")
async def getdatatrainmodel():

    logger.debug(f"Obtaining data and training model")
    ObtainData = GetDataTrainModel 
    ObtainData.get_data()
    ObtainData.train()
    
    logger.debug(f"Model traind and stored")

    return "Model trained, ready to go!"


'''
if __name__ == "__main__":


  prueba = '{"survived":0,"pclass":1,"sex":"male","age":58.0,"sibsp":0,"parch":2,"fare":113.275,"cabin":"D48","embarked":"C","title":"Mr"}'
  
  Pssngr=Passenger
  Pssngr.Survived=1
  Pssngr.Pclass=[1,2,3]
  Pssngr.Sex=['male','female','male',]
  Pssngr.Age=[58,59,60]
  Pssngr.SibSp=[0,0,1]
  Pssngr.Parch=[2,2,2]
  Pssngr.Fare=[113.27,110.10,120.20]
  Pssngr.Cabin=['D48','D50','B09']
  Pssngr.Embarked=['S','C','Q']
  Pssngr.Title=['Mr','Miss','Mrs']

  PassPrep=PassengerPredictor()
  PassPrep.predict_passanger(Pssngr)
  
  print('DOne')

'''