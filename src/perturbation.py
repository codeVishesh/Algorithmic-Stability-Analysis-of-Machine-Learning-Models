
import numpy as np

def perturb_data(X, y, noise_std=0.05, keep_ratio=0.95):
    X_noisy = X + np.random.normal(0, noise_std, X.shape)
    idx = np.random.choice(len(X), int(len(X) * keep_ratio), replace=False)
    return X_noisy[idx], y[idx]
