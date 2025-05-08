from pathlib import Path
import pandas as pd

def load_housing_data():
    # Compute the path two levels up from this script and then into data/raw
    data_path = Path(__file__).resolve().parent.parent / "data" / "raw" / "housing.csv"
    return pd.read_csv(data_path)
