import pandas as pd
import numpy as np
import logging

# Set up logging
logging.basicConfig(filename='eda_log.txt', level=logging.INFO)

def data_cleaning(sales_file, store_file):
    # Load the datasets
    train_data = pd.read_csv(sales_file)
    store_data = pd.read_csv(store_file)

    # Check for missing values
    missing_train = train_data.isnull().sum()
    missing_store = store_data.isnull().sum()
    logging.info('Checked for missing values.')

    # Merge store information into train_data
    train_data = train_data.merge(store_data, on='Store', how='left')

    # Handle missing values for 'CompetitionDistance' if it exists
    if 'CompetitionDistance' in train_data.columns:
        train_data['CompetitionDistance'] = train_data['CompetitionDistance'].fillna(train_data['CompetitionDistance'].median())
    
    # Handle missing values for 'Promo2SinceYear' if it exists
    if 'Promo2SinceYear' in train_data.columns:
        train_data['Promo2SinceYear'] = train_data['Promo2SinceYear'].fillna(0)

    # Ensure 'Sales' column exists for outlier detection
    if 'Sales' in train_data.columns:
        Q1 = train_data['Sales'].quantile(0.25)
        Q3 = train_data['Sales'].quantile(0.75)
        IQR = Q3 - Q1

        # Remove outliers
        train_data = train_data[(train_data['Sales'] >= (Q1 - 1.5 * IQR)) & (train_data['Sales'] <= (Q3 + 1.5 * IQR))]
    
    logging.info('Cleaned data by handling missing values and removing outliers.')

    return train_data, store_data  # Only return train_data