from fastapi import FastAPI
from app.schemas import PredictRequest, PredictResponse
import joblib
import pandas as pd
from pathlib import Path

app = FastAPI(title="ML Production API", version="1.0")

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
pre = joblib.load(ARTIFACT_DIR / "preprocessor.pkl")
model = joblib.load(ARTIFACT_DIR / "model.pkl")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    row = pd.DataFrame([payload.model_dump()])
    Xp = pre.transform(row)
    p = float(model.predict_proba(Xp)[:, 1][0])
    return PredictResponse(probability=p, prediction=int(p >= 0.5))
