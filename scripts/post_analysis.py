import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure()
    plt.title("Feature Importances")
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), np.array(feature_names)[indices], rotation=90)
    plt.xlim([-1, len(importances)])
    plt.show()

def estimate_confidence_interval(predictions, confidence=0.95):
    error_margin = 1.96 * predictions.std() / np.sqrt(len(predictions))
    return predictions.mean() - error_margin, predictions.mean() + error_margin

def analyze_predictions(model, X_test, y_test):
    feature_importance = model.named_steps['regressor'].feature_importances_
    predictions = model.predict(X_test)
    
    confidence_intervals = np.array([
        predictions - 1.96 * np.std(predictions),
        predictions + 1.96 * np.std(predictions)
    ]).T
    
    return feature_importance, confidence_intervals