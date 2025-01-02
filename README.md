## This is week-4 project of 10 academy

## Task1: Rossmann Store Sales Data Cleaning

## Overview

This project involves cleaning and preparing the Rossmann store sales dataset for analysis. The dataset is comprised of sales data, store information, and additional features that may impact sales performance.

## Files

- `data_cleaning.py`: The main script for cleaning the data.
- `store.csv`: Contains information about the stores.
- `train.csv`: Contains sales data for training the model.
- `sample_submission.csv`: A sample submission file for predictions.

## Requirements

Ensure you have the necessary Python packages installed. You can use the following command to install them:

pip install pandas matplotlib seaborn

# Usage
Load the Data: Update the file paths in the data_cleaning.py script to point to your local copies of the train.csv and store.csv files.
Run the Data Cleaning Script: You can execute the script to clean the data.

Analyze the Cleaned Data: After cleaning, the train_data DataFrame will be ready for further analysis or modeling.

Functionality
The data_cleaning function performs the following tasks:

Loads the sales and store datasets.
Merges the datasets on the 'Store' column.
Handles missing values for specific columns.
Removes outliers from the 'Sales' column using the Interquartile Range (IQR) method.
Returns the cleaned training data.
Logging
The script includes logging to track the cleaning process and any issues that arise during execution. Check the eda_log.txt file for details.

Contributing
Feel free to fork this repository and submit pull requests if you make improvements or find bugs.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Author: Natnahom Asfaw
Date: 02/01/2025