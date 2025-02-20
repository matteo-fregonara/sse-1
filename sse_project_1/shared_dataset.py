import numpy as np


def get_dataset():
    # Set seed for reproducibility
    np.random.seed(42)

    # Generate synthetic dataset
    X = np.random.randn(1000, 10).astype(np.float32)
    y = (np.sum(X[:, :3], axis=1) > 0).astype(np.float32)

    return X, y