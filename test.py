
import pandas as pd
import numpy as np 
import requests

url = 'http://localhost:9696/predict'

test = {
    "year": 2022,
    "month": 5,
    "arr_flights": 181.0,
    "arr_cancelled": 0.0,
    "arr_diverted": 0.0,
    "carrier": "9e",
    "carrier_name": "endeavor_air_inc",
    "airport": "ags",
    "airport_name": "augusta_ga_augusta_regional_at_bush_field"
}

response = requests.post(url, json=test)

predictions = response.json()

print(f'the airline delay in minutes is : {predictions['predicted_delay_minutes']}')

