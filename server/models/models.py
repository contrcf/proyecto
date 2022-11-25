from pydantic import BaseModel


class Passenger(BaseModel):
    Survived: float
    Pclass: float
    Age: float
    SibSp: float
    Parch: float
    Fare: float
    Cabin: str
    Sex: str
    Embarked: str
    Title: str
    
    
    
    


