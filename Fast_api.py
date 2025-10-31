from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Annotated
import pickle
import pandas as pd
import numpy as np

with open("model/all_models.pkl",'rb') as f:
    models = pickle.load(f)

city_encoder = models['Ordinal_city_encoder']
lb_iqr = models['label_encoder']
std = models['standard_scaler']
knn = models['knn_model']
dtree = models['dtree_model']

app = FastAPI(title="Air Pollution Level Prediction API")


class UserInput(BaseModel):
    City: Annotated[str, Field(..., description="Enter the City name.", examples=["Rajkot"])]
    Day: Annotated[int, Field(..., gt=0, lt=32, description="Enter the Day.", examples=[4])]
    Month: Annotated[int, Field(..., gt=0, lt=13, description="Enter the Month.", examples=[2])]
    Year: Annotated[int, Field(..., gt=0, description="Enter the Year.", examples=[2018])]
    PM25: Annotated[float, Field(..., gt=0, description="Enter the PM2.5 value.", examples=[28])]
    PM10: Annotated[float, Field(..., gt=0, description="Enter the PM10 value.", examples=[30])]
    NO: Annotated[float, Field(..., gt=0, description="Enter the 'NO' value.", examples=[50])]
    NO2: Annotated[float, Field(..., gt=0, description="Enter the 'NO2' value.", examples=[50])]
    NOx: Annotated[float, Field(..., gt=0, description="Enter the 'NOx' value.", examples=[50])]
    NH3: Annotated[float, Field(..., gt=0, description="Enter the 'NH3' value.", examples=[50])]
    CO: Annotated[float, Field(..., gt=0, description="Enter the 'CO' value.", examples=[50])]
    SO2: Annotated[float, Field(..., gt=0, description="Enter the 'SO2' value.", examples=[50])]
    O3: Annotated[float, Field(..., gt=0, description="Enter the 'O3' value.", examples=[50])]
    Benzene: Annotated[float, Field(..., gt=0, description="Enter the Benzene value.", examples=[50])]
    Toluene: Annotated[float, Field(..., gt=0, description="Enter the Toluene value.", examples=[50])]
    Xylene: Annotated[float, Field(..., gt=0, description="Enter the Xylene value.", examples=[50])]
    AQI: Annotated[float, Field(..., gt=0, description="Enter the AQI value.", examples=[50])]


@app.get("/")
def root():
    return {"message": "Welcome to the Air Pollution Level Prediction API!"}

@app.post("/predict")
def predict_air_pollution(data: UserInput):
 
    df = pd.DataFrame([{
        "City": data.City,
        "Day": data.Day,
        "Month": data.Month,
        "Year": data.Year,
        "PM2.5": data.PM25,
        "PM10": data.PM10,
        "NO": data.NO,
        "NO2": data.NO2,
        "NOx": data.NOx,
        "NH3": data.NH3,
        "CO": data.CO,
        "SO2": data.SO2,
        "O3": data.O3,
        "Benzene": data.Benzene,
        "Toluene": data.Toluene,
        "Xylene": data.Xylene,
        "AQI": data.AQI
    }])

    df[["City"]] = city_encoder.transform(df[["City"]])

    #scale down all the data

    new_df = std.transform(df)

    #make Prediction using the ML Models

    knn_pred = knn.predict(new_df)[0]
    tree_pred = dtree.predict(new_df)[0]
    
    #Lets Invers Transform this Two Models 

    label_knn_pred = lb_iqr.inverse_transform([knn_pred])[0]
    label_tree_pred = lb_iqr.inverse_transform([tree_pred])[0]

    return JSONResponse(
        status_code=200,
        content={
            "KNN Model Prediction": label_knn_pred,
            "DecisionTree Model Prediction": label_tree_pred,
            "Note": "Predictions represent Air Pollution Level classification labels."
        }
    )