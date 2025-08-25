from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import pickle
import numpy as np
import os

app = FastAPI()

# Load model
model_path = "model/xgbregressor_model.pkl"
with open(model_path, "rb") as f:
    model = pickle.load(f)

@app.get("/", response_class=HTMLResponse)
async def home():
    html = """
    <html>
    <head>
        <title>Model Prediction</title>
    </head>
    <body>
        <h2>Adjust Advertising Budgets:</h2>

        <label>TV: <span id="tv_val">100</span></label><br>
        <input type="range" min="0" max="300" value="100" id="tv" oninput="updateValue()"><br><br>

        <label>Radio: <span id="radio_val">20</span></label><br>
        <input type="range" min="0" max="100" value="20" id="radio" oninput="updateValue()"><br><br>

        <label>Newspaper: <span id="news_val">30</span></label><br>
        <input type="range" min="0" max="100" value="30" id="news" oninput="updateValue()"><br><br>

        <h3>Predicted Sales: <span id="prediction">--</span></h3>

        <script>
            function updateValue() {
                document.getElementById("tv_val").innerText = document.getElementById("tv").value;
                document.getElementById("radio_val").innerText = document.getElementById("radio").value;
                document.getElementById("news_val").innerText = document.getElementById("news").value;

                fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        TV: parseFloat(document.getElementById("tv").value),
                        Radio: parseFloat(document.getElementById("radio").value),
                        Newspaper: parseFloat(document.getElementById("news").value)
                    }),
                })
                .then(response => response.json())
                .then(data => {
                    document.getElementById("prediction").innerText = data.prediction.toFixed(2);
                });
            }

            window.onload = updateValue;
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)

# Predict endpoint
from pydantic import BaseModel

class AdData(BaseModel):
    TV: float
    Radio: float
    Newspaper: float

@app.post("/predict")
def predict(data: AdData):
    input_array = np.array([[data.TV, data.Radio, data.Newspaper]])
    prediction = model.predict(input_array)[0]
    return {"prediction": prediction}
