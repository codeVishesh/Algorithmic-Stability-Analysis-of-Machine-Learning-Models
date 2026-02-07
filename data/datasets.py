
import numpy as np
from sklearn.datasets import make_regression, fetch_california_housing

def load_synthetic_data(noise=0.3, random_state=42):
    X, y, coef = make_regression(
        n_samples=2000,
        n_features=15,
        n_informative=8,
        noise=noise,
        coef=True,
        random_state=random_state
    )
    return X, y, coef

def load_real_data():
    data = fetch_california_housing()
    return data.data, data.target
