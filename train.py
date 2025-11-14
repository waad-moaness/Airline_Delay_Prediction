import pandas as pd
import numpy as np 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor 

import pickle


def load_data():

    df = pd.read_csv("data/Airline_Delay_Cause.csv",engine='python')

    df.dropna(inplace=True)

    df.reset_index(inplace=True, drop= True)

    numerical = ['year','month','arr_flights','arr_cancelled','arr_diverted']
    categorical = ['carrier', 'carrier_name', 'airport', 'airport_name']
    target = ['arr_delay']

    df = df[numerical + categorical + target].copy()

    for col in categorical: 
         df[col] = df[col].str.lower().str.replace(' ' ,'_' ).str.replace(r'[^\w\s]', '', regex=True)

    df.arr_delay = np.log1p(df.arr_delay)

    X = df.drop('arr_delay', axis= 1)
    y = df.arr_delay

    return X , y , numerical , categorical


    
def train_model(X , y , numerical , categorical):
   

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical),
            ('cat', categorical_transformer, categorical)
        ],
    )

    pipeline = Pipeline(steps=[

        ('preprocessor', preprocessor), 
        ('model', XGBRegressor(     
            learning_rate=0.1,    
            max_depth=15,
            min_child_weight=1,
            objective='reg:squarederror',
            eval_metric='rmse',
            n_jobs=-1,           
            seed=1,
            verbosity=1,

            n_estimators=100    
        ))
    ])

    pipeline.fit(X, y)

    return pipeline


def save_model(pipeline , output_file):

    with open(output_file, 'wb') as f_out:
        pickle.dump(pipeline, f_out)



X_train , y_train ,numerical , categorical = load_data()
pipeline = train_model(X_train , y_train ,numerical , categorical)
save_model(pipeline, 'model.bin')

print('Model saved to model.bin')