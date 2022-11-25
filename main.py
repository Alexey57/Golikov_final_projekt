import json
import dill
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
with open('C:/Users/папа/PycharmProjects/Golikov_final_projekt/model/data/model_hits_sessions_binar.pkl', 'rb') as file:
    model = dill.load(file)


class Form(BaseModel):
    session_id: str
    hit_date: str
    hit_number: int
    hit_type: str
    event_category: str
    event_action: str
    client_id: int
    visit_date: str
    visit_time: str
    visit_number: int
    utm_source: str
    utm_medium: str
    utm_campaign: str
    utm_adcontent: str
    device_category: str
    device_brand: str
    device_screen_resolution: str
    device_browser: str
    geo_country: str
    geo_city: str


class Prediction(BaseModel):
    utm_source: str
    utm_medium: str
    device_category: str
    geo_city: str
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

    return {'city': df.geo_city, 'device': df.device_category, 'source': df.utm_source, 'pred': y}
