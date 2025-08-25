from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import numpy as np
import pickle
import os

app = FastAPI()

# Mount static directory to serve frontend files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve the HTML file at root path "/"
@app.get("/")
def serve_index():
    return FileResponse("static/index.html")

# Load model at startup
with open("model/xgbregressor_model.pkl", "rb") as f:
    model = pickle.load(f)

class InputData(BaseModel):
    TV: float
    Radio: float
    Newspaper: float

@app.get("/predict")
def predict_get(
    TV: float = Query(...),
    Radio: float = Query(...),
    Newspaper: float = Query(...)
):
    try:
        Total_Spend = TV + Radio + Newspaper
        TV_Radio = TV * Radio
        TV_Newspaper = TV * Newspaper
        Radio_Newspaper = Radio * Newspaper
        TV_Sq = TV ** 2
        Radio_Sq = Radio ** 2
        Newspaper_Sq = Newspaper ** 2

        input_features = np.array([[
            TV, Radio, Newspaper,
            Total_Spend,
            TV_Radio, TV_Newspaper, Radio_Newspaper,
            TV_Sq, Radio_Sq, Newspaper_Sq
        ]])

        prediction = model.predict(input_features)[0]
        return {"predicted_sales": float(prediction)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict")
def predict_post(data: InputData):
    try:
        Total_Spend = data.TV + data.Radio + data.Newspaper
        TV_Radio = data.TV * data.Radio
        TV_Newspaper = data.TV * data.Newspaper
        Radio_Newspaper = data.Radio * data.Newspaper
        TV_Sq = data.TV ** 2
        Radio_Sq = data.Radio ** 2
        Newspaper_Sq = data.Newspaper ** 2

        input_features = np.array([[
            data.TV, data.Radio, data.Newspaper,
            Total_Spend,
            TV_Radio, TV_Newspaper, Radio_Newspaper,
            TV_Sq, Radio_Sq, Newspaper_Sq
        ]])

        prediction = model.predict(input_features)[0]
        return {"predicted_sales": float(prediction)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
