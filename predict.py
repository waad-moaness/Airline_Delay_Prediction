import pickle
import numpy as np
import pandas as pd  
from fastapi import FastAPI
import uvicorn

from pydantic import BaseModel


with open('model.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)


app = FastAPI(title = 'airline_delay_prediction')


class FlightFeatures(BaseModel):
    year: int
    month: int
    arr_flights: float
    arr_cancelled: float
    arr_diverted: float
    carrier: str
    carrier_name: str
    airport: str
    airport_name: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the Airline Delay Prediction API. Go to /docs to see the API."}


@app.post('/predict')
def predict(flight: FlightFeatures):
    if pipeline is None:
        return {"error": "Model not loaded. Please check server logs."}

    data = flight.model_dump()
    
    result_df = pd.DataFrame([data])
    prediction = pipeline.predict(result_df)

    result = float(np.expm1(prediction[0]))

    return {
       "model_input": data,
       "predicted_delay_minutes": result

    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)