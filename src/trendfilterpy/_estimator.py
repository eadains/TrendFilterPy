from typing import Optional

import cvxpy as cp
import numpy as np
import numpy.typing as npt
import sklearn.utils.validation as skval
from sklearn.base import BaseEstimator, RegressorMixin

from trendfilterpy import _dists, _links
from trendfilterpy._variables import CatVar, FilterVar, FittedCatVar, FittedFilterVar


class TrendFilterRegression(RegressorMixin, BaseEstimator):
    def __init__(
        self,
        dist: type[_dists.Distribution] = _dists.NormalDistribution,
        link: type[_links.LinkFunction] = _links.IdentityLink,
        lam: float = 1,
    ) -> None:
        super().__init__()
        self.lam = lam
        self.dist = dist
        self.link = link

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike, categorical_features: Optional[list[int]] = None):
        # TODO: Check for convexity of self.dist deviance method
        # TODO: Check for monotonicity of self.link method
        if not categorical_features:
            categorical_features = []

        # For some reason pylance cannot resolve any of the functions inside sklearn's validation file
        checked: tuple[npt.NDArray, npt.NDArray] = skval.validate_data(  # type: ignore
            self, X, y, reset=True, y_numeric=True, ensure_min_samples=2
        )
        X, y = checked

        # TODO: Check to make sure categorical_features contains only valid values
        vars = []
        for i in range(X.shape[1]):
            if i in categorical_features:
                vars.append(CatVar(X[:, i]))
            else:
                vars.append(FilterVar(X[:, i]))

        alpha = cp.Variable(name="alpha")
        eta = alpha + cp.sum([var.beta[var.rebuild_idx] for var in vars])
        mu = self.link().eval_inverse(eta)

        penalty_terms = []
        for var in vars:
            if type(var) is CatVar:
                # TODO: Make sure penalty for categoricals is working/what we want?
                penalty_terms.append(cp.norm(var.beta, 2))
            elif type(var) is FilterVar:
                penalty_terms.append(cp.norm(var.D_mat @ var.beta, 1))
        penalty = cp.sum(penalty_terms)

        objective = cp.Minimize(self.dist().deviance(y, mu) + self.lam * penalty)
        constraints = [cp.Zero(cp.sum(var.beta)) for var in vars if type(var) is FilterVar]
        # CVXPY Problem is annotated as List[Constraint] which is invariant, so a list of Zero contraints is not
        # considered a subtype. The type annotation for Problem should be Sequence[Constraint]
        problem = cp.Problem(objective, constraints)  # type: ignore
        results = problem.solve(solver="CLARABEL")
        # TODO: check for convergence

        self.vars_ = []
        for i, var in enumerate(vars):
            # TODO: We only want to store values for the breakpoints instead of keeping a value around for every
            # unique value that we started with
            if var.beta.value is None:
                raise ValueError(f"Beta vector for var {i} is None and is not being fitted correctly")

            if type(var) is CatVar:
                self.vars_.append(FittedCatVar(var.unique_vals, var.beta.value))
            elif type(var) is FilterVar:
                self.vars_.append(FittedFilterVar(var.unique_vals, var.beta.value))

        self.mu_ = mu.value
        self.eta_ = eta.value

        if alpha.value is None:
            raise ValueError("Intercept is None and has not been fitted correctly")
        self.intercept_ = alpha.value.item()

        return self

    def predict(self, X: npt.ArrayLike):
        skval.check_is_fitted(self)
        checked: npt.NDArray = skval.validate_data(self, X, reset=False)  # type: ignore
        X = checked

        predictions = np.full(X.shape[0], self.intercept_)
        for i in range(X.shape[1]):
            predictions += self.vars_[i].predict(X[:, i])

        return predictions
