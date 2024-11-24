import pickle
from typing import List

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

with open("model/model.pkl", 'rb') as model_file:
    model = pickle.load(model_file)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/predict")
async def predict(children: int, bmi: float, age: int, smoker: str = "no"):
    print(f"Input: {smoker}, {children}, {bmi}, {age}")
    df = pd.DataFrame({
        "children": [children],
        "age": [age],
        "bmi": [bmi],
        "smoker": [smoker]
    })
    prediction = model.predict(df).tolist()

    return {"predicted": prediction}


class Prediction(BaseModel):
    age: int
    bmi: float
    children: int
    smoker: str

    class Config:
        schema_extra = {
            "example": {
                "age": 20,
                "bmi": 30.4,
                "children": 5,
                "smoker": "yes"
            }
        }


class PredictionList(BaseModel):
    data: List[Prediction]


@app.post("/predict")
async def prediction(data: Prediction):
    print(f"Input: {data}")
    df = pd.DataFrame([data.dict()])
    output = model.predict(df)[0]
    return {"predicted": output}


@app.post("/predict-multiple")
async def prediction(data: PredictionList):
    print(f"Input: {data}")
    df = pd.DataFrame(data.dict()["d"])
    output = model.predict(df).tolist()
    return {"predicted": output}
