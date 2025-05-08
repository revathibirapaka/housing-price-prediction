import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "scripts"))
from scripts.fetch_housing_prices import fetch_housing_data
from scripts.load_housing_prices import load_housing_data
from scripts.preprocess import prepare_data
from scripts.train_model import train_and_save_model
from scripts.custom_transformers import CustomAttributeAdder

def main():
    fetch_housing_data()
    df = load_housing_data()
    X, y = prepare_data(df)
    train_and_save_model(X, y)

if __name__ == "__main__":
    main()

import logging
logging.basicConfig(level=logging.INFO)

logging.info("Model trained successfully.")

