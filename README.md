```
# Housing Price Prediction

This project is a modular, production-ready implementation of a machine learning pipeline for predicting housing prices based on the California Housing dataset. The goal is to demonstrate a clean architecture for data ingestion, preprocessing, training, and model serialization.

## Project Structure

```

housing\_project/
│
├── scripts/
│   ├── **init**.py
│   ├── fetch\_housing\_prices.py     # Downloads dataset and stores it locally
│   ├── load\_housing\_prices.py      # Loads the dataset into a DataFrame
│   ├── preprocess.py               # Builds and applies preprocessing pipeline
│   ├── train\_model.py              # Trains and saves the regression model
│   ├── custom\_transformers.py      # Contains custom transformers for feature engineering
│
├── models/
│   └── housing\_model.pkl           # Saved trained model
│
├── main.py                         # Orchestrates the full pipeline end-to-end
├── requirements.txt                # List of Python dependencies
└── README.md                       # Project documentation

````

## Objective

This project aims to:

- Download and load the California housing dataset.
- Clean and preprocess the data using pipelines and custom transformers.
- Train a linear regression model.
- Evaluate the model using mean squared error (MSE).
- Save the trained model to disk for later use.

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/your-username/housing_project.git
cd housing_project
````

### 2. Create and activate a virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate      # On Unix or MacOS
venv\Scripts\activate.bat     # On Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the pipeline

```bash
python main.py
```

This will:

* Download the data into a `datasets/` directory
* Load the dataset
* Apply preprocessing
* Train a model
* Save the model to the `models/` folder

## Requirements

The required libraries are listed in `requirements.txt`. Key packages include:

* numpy
* pandas
* scikit-learn
* joblib

To install them:

```bash
pip install -r requirements.txt
```

## Notes

* The model is a simple Linear Regression for demonstration.
* Missing values are handled using `SimpleImputer` within a pipeline.
* Categorical variables are encoded using `OneHotEncoder`.

## Custom Transformers

The project includes a custom transformer `CustomAttributeAdder` that adds engineered features such as:

* `rooms_per_household`
* `population_per_household`
* `bedrooms_per_room`

## Author

\Revathi Birapaka


```
