from typing import Literal, Optional

import cvxpy as cp
import numpy as np
import numpy.typing as npt
from scipy.sparse import dia_matrix, spdiags


def make_D_matrix(n: int) -> dia_matrix:
    ones = np.ones(n)
    return spdiags(np.vstack([-ones, ones]), range(2), m=n - 1, n=n)


def _get_stepwise_indices(unique_vals: npt.ArrayLike, x: npt.ArrayLike) -> np.ndarray:
    """Get indices for stepwise function mapping using searchsorted.

    Maps input values to indices in unique_vals array using right-continuous
    step function logic. Used for both continuous (FilterVar) and categorical
    (CatVar) variable indexing.

    Parameters
    ----------
    unique_vals : array_like
        Sorted array of unique values defining the step function breakpoints.
    x : array_like
        Input values to map to indices.

    Returns
    -------
    numpy.ndarray
        Array of indices corresponding to each value in x, where each index
        satisfies unique_vals[i-1] <= x < unique_vals[i] for the right-continuous
        step function.
    """
    idx = np.searchsorted(unique_vals, x, side="right")
    return np.where(idx == 0, idx, idx - 1)


class FilterVar:
    def __init__(self, x: npt.ArrayLike, name: Optional[str] = None) -> None:
        # TODO: Do we want to check for x to be 1-dimensional? I don't expect users to be creating these so that may
        # not be necessary, but also checking won't introduce much overhead given we aren't creating these very often
        self.sort_idx = np.argsort(x)
        # np.unique guarantees that the returned values are sorted
        self.unique_vals, self.rebuild_idx = np.unique(x, return_inverse=True)
        self.D_mat = make_D_matrix(len(self.unique_vals))
        self.beta = cp.Variable(len(self.unique_vals), name=name)

    def predict(self, x: npt.ArrayLike) -> cp.Expression:
        # Our fitted function is stepwise and right continuous so we want our index to satisfy a[i-1] <= v < a[i]
        idx = _get_stepwise_indices(self.unique_vals, x)
        return self.beta[idx]


class FittedFilterVar:
    def __init__(self, unique_vals: npt.ArrayLike, beta: npt.NDArray, name: Optional[str] = None) -> None:
        self.unique_vals = unique_vals
        self.beta = beta
        self.name = name

    def predict(self, x: npt.ArrayLike) -> np.ndarray:
        idx = _get_stepwise_indices(self.unique_vals, x)
        return self.beta[idx]


class CatVar:
    def __init__(
        self, x: npt.ArrayLike, name: Optional[str] = None, unseen_action: Literal["error", "ignore"] = "error"
    ) -> None:
        self.unique_vals, self.rebuild_idx = np.unique(x, return_inverse=True)
        self.beta = cp.Variable(len(self.unique_vals), name=name)
        self.unseen_action = unseen_action

    def predict(self, x: npt.ArrayLike) -> cp.Expression:
        x_array = np.asarray(x)

        # Check for unseen categories
        is_unseen = ~np.isin(x_array, self.unique_vals)

        if np.any(is_unseen):
            if self.unseen_action == "error":
                unseen_vals = np.unique(x_array[is_unseen])
                raise ValueError(f"Unseen categorical values: {unseen_vals}")
            elif self.unseen_action == "ignore":
                # Create selection matrix that maps each input to the corresponding beta or zero
                selection_matrix = np.zeros((len(x_array), len(self.unique_vals)))

                # For seen values, set the appropriate column to 1
                seen_mask = ~is_unseen
                if np.any(seen_mask):
                    seen_x = x_array[seen_mask]
                    idx = _get_stepwise_indices(self.unique_vals, seen_x)
                    # We need an array of the same length of seen_mask containing the desired index values
                    # for the zip below
                    idx_array = np.zeros_like(seen_mask, dtype=int)
                    idx_array[seen_mask] = idx
                    for i, (is_seen, beta_idx) in enumerate(zip(seen_mask, idx_array, strict=True)):
                        if is_seen:
                            selection_matrix[i, beta_idx] = 1

                return selection_matrix @ self.beta

        # All values are seen, use original logic
        idx = _get_stepwise_indices(self.unique_vals, x_array)
        return self.beta[idx]


class FittedCatVar:
    def __init__(
        self,
        unique_vals: npt.ArrayLike,
        beta: npt.NDArray,
        name: Optional[str] = None,
        unseen_action: Literal["error", "ignore"] = "error",
    ) -> None:
        self.unique_vals = unique_vals
        self.beta = beta
        self.name = name
        self.unseen_action = unseen_action

    def predict(self, x: npt.ArrayLike) -> np.ndarray:
        x_array = np.asarray(x)

        # Check for unseen categories
        is_unseen = ~np.isin(x_array, self.unique_vals)

        if np.any(is_unseen):
            if self.unseen_action == "error":
                unseen_vals = np.unique(x_array[is_unseen])
                raise ValueError(f"Unseen categorical values: {unseen_vals}")
            elif self.unseen_action == "ignore":
                result = np.zeros(x_array.shape[0])

                # For seen values, get their predictions
                seen_mask = ~is_unseen
                if np.any(seen_mask):
                    seen_x = x_array[seen_mask]
                    idx = _get_stepwise_indices(self.unique_vals, seen_x)
                    result[seen_mask] = self.beta[idx]

                # Any elements not set are 0 by construction
                return result

        # All values are seen
        idx = _get_stepwise_indices(self.unique_vals, x_array)
        return self.beta[idx]
