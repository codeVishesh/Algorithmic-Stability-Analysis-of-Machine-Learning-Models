
# import numpy as np
# from sklearn.model_selection import train_test_split
# # from sklearn.metrics import mean_squared_error
# from sklearn.metrics import root_mean_squared_error


# from src.perturbation import perturb_data
# from src.stability import stability_score

# def run_experiment(X, y, models, n_runs=50):
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )

#     results = {}

#     for name, model in models.items():
#         predictions = []
#         rmses = []

#         for _ in range(n_runs):
#             X_p, y_p = perturb_data(X_train, y_train)
#             model.fit(X_p, y_p)
#             y_pred = model.predict(X_test)

#             predictions.append(y_pred)
#             rmses.append(mean_squared_error(y_test, y_pred, squared=False))

#         predictions = np.array(predictions)

#         results[name] = {
#             "stability": stability_score(predictions),
#             "rmse": np.mean(rmses)
#         }

#     return results

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

from src.perturbation import perturb_data
from src.stability import stability_score


def run_experiment(X, y, models, n_runs=50):
    """
    Runs algorithmic stability experiments for given models.

    Parameters
    ----------
    X : ndarray
        Feature matrix
    y : ndarray
        Target vector
    models : dict
        Dictionary of model_name -> model_object
    n_runs : int
        Number of perturbed training runs

    Returns
    -------
    results : dict
        Dictionary containing mean RMSE and stability score for each model
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = {}

    for name, model in models.items():
        predictions = []
        rmses = []

        for _ in range(n_runs):
            # Apply small perturbation to training data
            X_p, y_p = perturb_data(X_train, y_train)

            # Train model
            model.fit(X_p, y_p)

            # Predict on fixed test set
            y_pred = model.predict(X_test)

            predictions.append(y_pred)
            rmses.append(root_mean_squared_error(y_test, y_pred))

        predictions = np.array(predictions)

        results[name] = {
            "stability": stability_score(predictions),
            "rmse": np.mean(rmses)
        }

    return results

