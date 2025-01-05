import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_data(data):
    # Handle missing values
    data['CompetitionDistance'] = data['CompetitionDistance'].fillna(data['CompetitionDistance'].median())
    
    # Convert date column to datetime
    data['Date'] = pd.to_datetime(data['Date'])
    
    # Feature extraction
    data['Weekday'] = data['Date'].dt.weekday
    data['IsWeekend'] = (data['Weekday'] >= 5).astype(int)
    
    # Replace with an actual holiday date
    holiday_date = '2023-12-25'  # Example holiday date
    data['DaysToHoliday'] = (data['Date'] - pd.to_datetime(holiday_date)).dt.days
    data['DaysAfterHoliday'] = (pd.to_datetime(holiday_date) - data['Date']).dt.days.clip(lower=0)
    
    # Create month features
    data['BeginningOfMonth'] = (data['Date'].dt.day == 1).astype(int)
    data['MidMonth'] = ((data['Date'].dt.day > 1) & (data['Date'].dt.day <= 15)).astype(int)
    data['EndOfMonth'] = (data['Date'].dt.day == data['Date'].dt.days_in_month).astype(int)
    
    # Encode categorical variables
    data = pd.get_dummies(data, columns=['StateHoliday', 'StoreType', 'Assortment'], drop_first=True)
    
    # Drop unnecessary columns
    data.drop(['Id', 'Date', 'PromoInterval', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear'], axis=1, inplace=True, errors='ignore')

    return data