import pickle
from datetime import datetime

def serialize_model(model):
    timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    filename = f"model_{timestamp}.pkl"
    
    with open(filename, 'wb') as file:
        pickle.dump(model, file)