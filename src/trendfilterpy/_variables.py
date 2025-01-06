from typing import Optional

import cvxpy as cp
import numpy as np
import numpy.typing as npt
from scipy.sparse import dia_matrix, spdiags


def make_D_matrix(n: int) -> dia_matrix:
    ones = np.ones(n)
    return spdiags(np.vstack([-ones, ones]), range(2), m=n - 1, n=n)


class FilterVar:
    def __init__(self, x: npt.ArrayLike, name: Optional[str] = None) -> None:
        # TODO: Do we want to check for x to be 1-dimensional? I don't expect users to be creating these so that may
        # not be necessary, but also checking won't introduce much overhead given we aren't creating these very often
        self.sort_idx = np.argsort(x)
        # np.unique guarantees that the returned values are sorted
        self.unique_vals, self.rebuild_idx = np.unique(x, return_inverse=True)
        self.D_mat = make_D_matrix(len(self.unique_vals))
        self.beta = cp.Variable(len(self.unique_vals), name=name)


class FittedFilterVar:
    def __init__(self, unique_vals: npt.ArrayLike, beta: npt.NDArray, name: Optional[str] = None) -> None:
        self.unique_vals = unique_vals
        self.beta = beta
        self.name = name

    def predict(self, x: npt.ArrayLike):
        # Our fitted function is stepwise and right continuous so we want our index to satisfy a[i-1] <= v < a[i]
        idx = np.searchsorted(self.unique_vals, x, side="right")
        # TODO: Make test to ensure that if given x value is outside of the range of observed values that the correct
        # beta element is returned: first if x is less than all observed values and last if x is greater than all

        # Then we want the beta value from i - 1 as that gives the proper value for the range that v falls in
        # If 0 then x is below smallest observed value so we want to return the first value. When x is greater than
        # largest observed value idx = len(beta) so we want to return the last element that has index 1 less than that
        idx = np.where(idx == 0, idx, idx - 1)
        return self.beta[idx]


class CatVar:
    def __init__(self, x: npt.ArrayLike, name: Optional[str] = None) -> None:
        self.unique_vals, self.rebuild_idx = np.unique(x, return_inverse=True)
        self.beta = cp.Variable(len(self.unique_vals), name=name)

    def predict(self, x: npt.ArrayLike):
        # TODO: Handle unseen categories
        idx = np.searchsorted(self.unique_vals, x, side="right")
        idx = np.where(idx == 0, idx, idx - 1)
        return self.beta[idx]


class FittedCatVar:
    def __init__(self, unique_vals: npt.ArrayLike, beta: npt.NDArray, name: Optional[str] = None) -> None:
        self.unique_vals = unique_vals
        self.beta = beta
        self.name = name

    def predict(self, x: npt.ArrayLike):
        # TODO: Handle unseen categories
        idx = np.searchsorted(self.unique_vals, x, side="right")
        idx = np.where(idx == 0, idx, idx - 1)
        return self.beta[idx]
