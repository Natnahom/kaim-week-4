import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scripts.data_preprocessing import preprocess_data

def hyperparameter_tuning(processed_data, target_column='Sales', test_size=0.2, cv_folds=3):
    # Preprocess the training data
    X = processed_data.drop(target_column, axis=1)
    y = processed_data[target_column]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Create a pipeline with scaling and the model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(random_state=42))
    ])

    # Set up the parameter grid for hyperparameter tuning
    param_grid = {
        'regressor__n_estimators': [50],
        'regressor__max_depth': [None, 10],
        'regressor__min_samples_split': [2],
        'regressor__min_samples_leaf': [1],
    }

    # Perform Grid Search
    grid_search = GridSearchCV(estimator=pipeline, 
                               param_grid=param_grid, 
                               scoring='neg_mean_squared_error', 
                               cv=cv_folds,
                               verbose=2,
                               n_jobs=-1)

    # Fit the model
    grid_search.fit(X_train, y_train)

    # Get the best model
    best_model = grid_search.best_estimator_

    # Make predictions on the test set
    predictions = best_model.predict(X_test)

    # Calculate Mean Squared Error
    mse = mean_squared_error(y_test, predictions)

    # Return the best model and the mean squared error
    return best_model, mse, grid_search.best_params_
