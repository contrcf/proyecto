import json
import logging
from fastapi import FastAPI
from starlette.responses import JSONResponse
from train.train_model import GetDataTrainModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(levelname)s: %(asctime)s|%(name)s|%(message)s")
file_handler = logging.FileHandler("server.log")
file_handler.setFormatter(formatter)

app = FastAPI()

@app.get("/")
def read_root():
    """ """
    logger.info("Titanic Predictor is ready to go!")
    return "Titanic Train Model is ready to go!"


@app.get("/healthcheck", status_code=200)
async def healthcheck():
    logger.info("Servers are ready to go!")
    return "Servers are ready to go!"


@app.get("/passangertrainmodel")
async def getdatatrainmodel():
    logger.debug(f"Obtaining data and training model")
    ObtainData = GetDataTrainModel 
    ObtainData.get_data()
    ObtainData.train()
    logger.debug(f"Model traind and stored")
    return "Model trained, ready to go!"

