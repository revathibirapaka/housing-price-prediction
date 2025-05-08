from preprocess import prepare_data
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from joblib import dump
import os
from load_housing_prices import load_housing_data
import numpy as np

def train_and_save_model(X, y, model_path="models/housing_model.pkl"):
    model = LinearRegression()
    model.fit(X, y)
    print("Any NaNs in X?", np.isnan(X).any())

    # Evaluate
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    print("MSE:", mse)

    # Save the model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    dump(model, model_path)
    print(f"Model saved to {model_path}")

import logging
logging.basicConfig(level=logging.INFO)

logging.info("Model trained successfully.")

