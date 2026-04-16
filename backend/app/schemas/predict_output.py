from pydantic import BaseModel

class PredictOutput(BaseModel):
    position: str
    probability: float