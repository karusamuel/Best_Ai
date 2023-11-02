from fastapi import FastAPI
import pickle
from pydantic import BaseModel

import pandas as pd

class Flower(BaseModel):
    s_p_l: float
    s_p_w: float
    p_t_l: float
    p_t_w: float


app = FastAPI()

with open('./iris/model.pkl','rb') as file:
    model = pickle.load(file)

@app.get('/')
def root():
    return{"message":"Hello Word","name":"samuel"}

@app.post('/predict')
def pred(flower: Flower):

    data = {
        "SepalLengthCm":[flower.s_p_l],
        "SepalWidthCm":[flower.s_p_w],
        "PetalLengthCm":[flower.p_t_l],
        "PetalWidthCm":[flower.p_t_w]
    }

    df = pd.DataFrame(data)
    pred = model.predict(df)

    return{"prediction":pred[0]}