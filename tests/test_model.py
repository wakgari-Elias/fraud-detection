import pandas as pd
from src.model import train_logistic_regression

def test_model_training():
    X = pd.DataFrame({"a": [0,1,0,1], "b":[1,0,1,0]})
    y = [0,1,0,1]
    model = train_logistic_regression(X,y)
    assert model is not None
