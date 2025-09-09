from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os

app = FastAPI(title="MLOps Microservice")

class Item(BaseModel):
    text: str

# Lazy model load (placeholder: replace with your trained pipeline)
MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model.joblib")
_model = None

def get_model():
    global _model
    if _model is None and os.path.exists(MODEL_PATH):
        _model = joblib.load(MODEL_PATH)
    return _model

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(item: Item):
    model = get_model()
    if model is None:
        return {"error": "Model not found. Train and place at artifacts/model.joblib"}
    # For text models, ensure the joblib pipeline includes vectorizer/tokenizer
    pred = model.predict([item.text])[0]
    return {"prediction": str(pred)}
