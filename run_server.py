import mlflow.pyfunc
from fastapi import FastAPI
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = FastAPI()
model = mlflow.pyfunc.load_model("models:/random-forest-classifier-model-dev/latest")

@app.get("/health")
def health_check():
    return {"message": "OK"}

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    scaler = StandardScaler()
    df = scaler.fit_transform(df)
    return {"prediction": model.predict(df).tolist()}