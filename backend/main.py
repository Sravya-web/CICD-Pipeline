from fastapi import FastAPI 
from pydantic import BaseModel
from pathlib import Path
import pickle
import numpy as np

app = FastAPI(title="Linear Regression API")

# import as
# Model_PATH = path(os.getenv("MODEL_PATH", "models/model.pk1")

MODEL_PATH = "models/model.pkl"
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    raise RuntimeError(f"Failed to load model from {MODEL_PATH}: {e}")


# Input schema
class InputData(BaseModel):
    area: float
    bedrooms: int


# Prediction endpoint
@app.post("/predict")
def predict(data: InputData):
    X = np.array([[data.area, data.bedrooms]])
    prediction = model.predict(X)[0]
    return {"predicted_price": float(prediction)}
