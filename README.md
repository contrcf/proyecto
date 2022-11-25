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
uvicorn train.model:app --port 5002 --reload

```

# Example request.
Run a quest to predict a passanger
```
curl 'http://localhost:8080/iris/classify_iris' -X POST -H 'Content-Type: application/json' -d '{"sepal_l": 5, "sepal_w": 2, "petal_l": 3, "petal_w": 4}'
```