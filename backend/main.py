from pathlib import Path
import os
import pickle
from fastapi import FastAPI

app = FastAPI()


MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/model.pkl"))
model = None

def load_model():
    global model
    if model is None:
        if not MODEL_PATH.exists():
            print(f"⚠️ Model not found at {MODEL_PATH}, skipping load (tests/CI mode)")
            return None
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
    return model

