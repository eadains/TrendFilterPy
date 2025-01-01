import numpy as np
import cvxpy as cp


def squared_error_loss(y: np.ndarray, y_hat: np.ndarray) -> cp.Expression:
    return cp.sum_squares(y - y_hat)
