from pydantic import BaseModel, Field
from typing import Annotated

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
