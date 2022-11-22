from pydantic import BaseModel


class Passenger(BaseModel):
    Survived: float
    Pclass: float
    Age: float
    SibSp: float
    Parch: float
    Fare: float
    Cabin: float
    Sex_female: float
    Sex_male: float
    Embarked_C: float
    Embarked_Q: float
    Embarked_S: float