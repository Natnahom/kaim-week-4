import tensorflow as tf

def build_lstm_model(X):
    # Reshape data for LSTM
    X_reshaped = X.values.reshape((X.shape[0], X.shape[1], 1))
    
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(50, activation='relu', input_shape=(X_reshaped.shape[1], 1)))
    model.add(tf.keras.layers.Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model