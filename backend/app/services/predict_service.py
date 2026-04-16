import os
import mlflow
import pandas as pd
from sklearn.preprocessing import StandardScaler
from app.schemas.predict_input import PredictInput
from app.schemas.predict_output import PredictOutput

if os.environ.get("MLFLOW_TRACKING_URI"):
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))

model = mlflow.pyfunc.load_model("models:/random_forest_model_dev/latest")

def predict_position(input_data: PredictInput) -> PredictOutput:
    df = pd.DataFrame([input_data.model_dump()])
    scaler = StandardScaler()
    df = scaler.fit_transform(df)
    prediction = model.predict(df)
    return PredictOutput(position=prediction[0], probability=prediction[1])