from sklearn.base import BaseEstimator, RegressorMixin
import sklearn.utils.validation as skval
import numpy.typing as npt
import numpy as np
from trendfilterpy import _losses
from trendfilterpy._variables import CatVar, FilterVar
import cvxpy as cp
from typing import Optional


class TrendFilterRegression(RegressorMixin, BaseEstimator):
    def __init__(self, lam: float = 1) -> None:
        super().__init__()
        self.lam = lam

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike, categorical_features: Optional[list[int]] = None):
        if not categorical_features:
            categorical_features = []
        X, y = skval.validate_data(self, X, y, reset=True, y_numeric=True, ensure_min_samples=2)

        # TODO: Check to make sure categorical_features contains only valid values
        # TODO: change vars_ to a private variable
        self.vars_: list[CatVar | FilterVar] = []
        for i in range(X.shape[1]):
            if i in categorical_features:
                self.vars_.append(CatVar(X[:, i]))
            else:
                self.vars_.append(FilterVar(X[:, i]))

        alpha = cp.Variable(name="alpha")
        y_hat = alpha + cp.sum([var.beta[var.rebuild_idx] for var in self.vars_])

        penalty_terms = []
        for var in self.vars_:
            if type(var) is CatVar:
                # TODO: Make sure penalty for categoricals is working/what we want?
                penalty_terms.append(cp.norm(var.beta, 2))
            elif type(var) is FilterVar:
                penalty_terms.append(cp.norm(var.D_mat @ var.beta, 1))
        penalty = cp.sum(penalty_terms)

        objective = cp.Minimize(0.5 * _losses.squared_error_loss(y, y_hat) + self.lam * penalty)
        constraints = [cp.sum(var.beta) == 0 for var in self.vars_ if type(var) is FilterVar]
        problem = cp.Problem(objective, constraints)
        results = problem.solve(solver="CLARABEL")
        # TODO: check for convergence

        for var in self.vars_:
            # Beta arrays are at this point cvxpy variables but we now have the fitted values so replace them with
            # numpy arrays to reduce memory overhead
            # TODO: We may only want to store values for the breakpoints instead of keeping a value around for every
            # unique value that we started with
            var.beta = var.beta.value
        self.fitted_values_ = y_hat.value
        self.intercept_ = alpha.value.item()

        return self

    def predict(self, X: npt.ArrayLike):
        skval.check_is_fitted(self)
        X = skval.validate_data(self, X, reset=False)

        predictions = np.full(X.shape[0], self.intercept_)
        for i in range(X.shape[1]):
            predictions += self.vars_[i].predict(X[:, i])

        return predictions
