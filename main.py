import json
import dill
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
with open('C:/Users/папа/PycharmProjects/Golikov_final_projekt/model/data/model_hits_sessions_binar.pkl', 'rb') as file:
    model = dill.load(file)


class Form(BaseModel):
    utm_source: object
    utm_medium: object
    device_category: object
    geo_city: object


class Prediction(BaseModel):
    predict: int


@app.get('/status')
def status():
    return "I'm OK"


@app.get('/version')
def version():
    return model['metadata']


@app.post('/predict', response_model=Prediction)
def predict(form: Form):
    df = pd.DataFrame.from_dict([form.dict()])
    y = model['model'].predict(df)
    y = y[0]

    return {'predict': y}
