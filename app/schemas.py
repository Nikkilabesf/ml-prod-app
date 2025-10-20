from pydantic import BaseModel

class PredictRequest(BaseModel):
    age: float
    bmi: float
    avg_glucose: float
    sex: str
    smoker: str
    region: str

class PredictResponse(BaseModel):
    probability: float
    prediction: int
