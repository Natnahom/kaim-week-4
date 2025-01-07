from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('../notebooks/trained_model.joblib')

# Create FastAPI instance
app = FastAPI()

# Define a Pydantic model for the input data
class InputData(BaseModel):
    Store: int
    DayOfWeek: int
    Open: int
    Promo: int
    SchoolHoliday: int
    CompetitionDistance: float
    Promo2: int
    Promo2SinceWeek: int
    Promo2SinceYear: int
    Weekday: int
    IsWeekend: int
    DaysToHoliday: int
    DaysAfterHoliday: int
    BeginningOfMonth: int
    MidMonth: int
    EndOfMonth: int
    StateHoliday_a: int
    StoreType_b: int
    StoreType_c: int
    StoreType_d: int
    Assortment_b: int
    Assortment_c: int

# Define the root endpoint
@app.get("/")
# async def read_root():
#     return {"message": "Welcome to the FastAPI application!"}

# # Define the prediction endpoint
# @app.post("/predict/")
def predict(data: InputData):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([data.dict()])

    # Make predictions
    prediction = model.predict(input_df)

    # Return the prediction
    return {"predicted_sales": 1000}

# Run the app using: uvicorn app:app --reload