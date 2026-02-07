
import numpy as np

def stability_score(predictions):
    # predictions shape: (n_runs, n_samples)
    return np.mean(np.var(predictions, axis=0))
