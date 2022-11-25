import json
import logging
from fastapi import FastAPI
from starlette.responses import JSONResponse

from server.models.models import Passenger
from server.predictor.passenger_predictor import PassengerPredictor as PassengerPredictor


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
    return "Titanic Predictor is ready to go!"


@app.get("/healthcheck", status_code=200)
async def healthcheck():
    logger.info("Servers are ready to go!")
    return "Servers are ready to go!"


@app.post("/passanger_predictor")
async def classify(Passenger_features: Passenger):
    logger.debug(f"Incoming passanger features to the server: {Passenger_features}")
    Passenger_Predictor = PassengerPredictor()
    response = JSONResponse(Passenger_Predictor.predict_passanger(Passenger_features))
    logger.debug(f"Outgoing classification from the server: {response}")
    return response
