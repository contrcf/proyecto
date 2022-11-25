# calculator-fast-api
This is a calculator transformed to a API with FastAPI

# Setup
* Create a virtual environment with:
```
python3 -m venv venv
```

* Activate the virtual environment

```
source venv/bin/activate
```

# Install the other libraries
Run the following command to install the other libraries.

```
pip install -r requirements.txt
```

# Tests

````
$ pytest tests/integration/ -v
````
````
tests/integration/test_integration.py::test_integration PASSED                                                                                [ 33%]
tests/integration/test_integration.py::test_integration_parametrize[5-5-5/4-3/4-5] PASSED                                                     [ 66%]
tests/integration/test_integration.py::test_integration_parametrize[8-7/5-15-3/8-137.475] PASSED                                              [100%]
````

# Run FastAPI
Run next commando to start predictor api locally


```
Run next command to start server
uvicorn server.main:app --port 5001 --reload

Run next command to start model train.
uvicorn train.main:app --port 5002 --reload

```

# Example request to sever.
Run a quest to predict a passanger to server
```
curl 'http://localhost:5001/healthcheck' 

curl 'http://localhost:5001/passanger_predictor' -X POST -H 'Content-Type: application/json' -d '{"survived":0,"pclass":1,"sex":"male","age":58.0,"sibsp":0,"parch":2,"fare":113.275,"cabin":"D48","embarked":"C","title":"Mr"}'

{
  "Survived": 0,
  "Pclass": 1,
  "Age": 20,
  "SibSp": 0,
  "Parch": 2,
  "Fare": 113.275,
  "Cabin": "D46",
  "Sex": "female",
  "Embarked": "c",
  "Title": "Miss"
}
```

# Example request to trainer.
Run a request to train the model
```
curl 'http://localhost:5002/healthcheck' 

curl 'http://localhost:5002/passangertrainmodel' 

```