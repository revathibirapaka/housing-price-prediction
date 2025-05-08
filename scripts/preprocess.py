import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from custom_transformers import CustomAttributeAdder  # assuming you defined it elsewhere
from load_housing_prices import load_housing_data

# Load the raw data
housing = load_housing_data()

housing_labels = housing["median_house_value"].copy()

housing = housing.drop("median_house_value", axis=1)


num_attribs = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
]

cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),  # <--- this fills in missing values
    ("std_scaler", StandardScaler()),
])


full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs)
])

X_prepared = full_pipeline.fit_transform(housing)
print("Any NaNs in X?", np.isnan(X_prepared).any())  # Add this

def prepare_data(df):
    df = load_housing_data()
    return X_prepared, housing_labels

# Custom transformer to add combined attributes
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rooms_per_household = X[:, 3] / X[:, 6]
        population_per_household = X[:, 5] / X[:, 6]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, 4] / X[:, 3]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

if __name__ == "__main__":
    from load_housing_prices import load_housing_data
    df = load_housing_data()
    X, y = prepare_data(df)
    print("Prepared data shape:", X.shape)
