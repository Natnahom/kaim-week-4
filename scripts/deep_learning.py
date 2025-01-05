# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from sklearn.preprocessing import MinMaxScaler

# def create_lstm_model(input_shape):
#     model = tf.keras.Sequential([
#         tf.keras.layers.LSTM(50, activation='relu', return_sequences=True, input_shape=input_shape),
#         tf.keras.layers.LSTM(50, activation='relu'),
#         tf.keras.layers.Dense(1)  # Output layer
#     ])
#     model.compile(optimizer='adam', loss='mean_squared_error')
#     return model

# # def prepare_time_series_data(data):
# #     # Ensure data is stationary and create supervised learning data
# #     # Add your time series processing logic here
# #     return X, y  # Returns features and target
import tensorflow as tf

def build_lstm_model(X):
    # Reshape data for LSTM
    X_reshaped = X.values.reshape((X.shape[0], X.shape[1], 1))
    
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(50, activation='relu', input_shape=(X_reshaped.shape[1], 1)))
    model.add(tf.keras.layers.Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model