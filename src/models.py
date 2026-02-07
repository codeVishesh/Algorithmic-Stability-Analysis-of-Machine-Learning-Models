
# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.svm import SVR
# from sklearn.neural_network import MLPRegressor

# def get_models():
#     return {
#         "LinearRegression": LinearRegression(),
#         "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
#         "SVR": SVR(),
#         "NeuralNetwork": MLPRegressor(hidden_layer_sizes=(50,), max_iter=1000, random_state=42)
#     }

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def get_models():
    return {
        "LinearRegression": LinearRegression(),

        "RandomForest": RandomForestRegressor(
            n_estimators=100,
            random_state=42
        ),

        "SVR": Pipeline([
            ("scaler", StandardScaler()),
            ("svr", SVR())
        ]),

        "NeuralNetwork": Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(
                hidden_layer_sizes=(50,),
                max_iter=2000,
                early_stopping=True,
                n_iter_no_change=20,
                random_state=42
            ))
        ])
    }
