from typing import Iterable, Optional, Sequence

import cvxpy as cp
import numpy as np
import numpy.typing as npt
import sklearn.utils.validation as skval
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import BaseCrossValidator, KFold

from trendfilterpy import _dists, _links
from trendfilterpy._variables import CatVar, FilterVar, FittedCatVar, FittedFilterVar


class TrendFilterRegression(RegressorMixin, BaseEstimator):
    def __init__(
        self, dist: Optional[_dists.Distribution] = None, link: Optional[_links.LinkFunction] = None, lam: float = 1
    ) -> None:
        super().__init__()
        self.lam = lam
        self.dist = dist
        self.link = link

    def fit(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        weights: Optional[npt.ArrayLike] = None,
        categorical_features: Optional[Sequence[int]] = None,
    ):
        dist = _dists.NormalDistribution() if self.dist is None else self.dist
        link = dist.canonical_link() if self.link is None else self.link
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
                vars.append(
                    # validate_data() sets feature_names_in_ if feature names are given by X or else deletes the
                    # attribute. The type-checker does not like this.
                    CatVar(X[:, i], name=self.feature_names_in_[i] if hasattr(self, "feature_names_in_") else None)  # type: ignore
                )
            else:
                vars.append(
                    FilterVar(X[:, i], name=self.feature_names_in_[i] if hasattr(self, "feature_names_in_") else None)  # type: ignore
                )

        alpha = cp.Variable(name="alpha")
        eta = alpha + cp.sum([var.beta[var.rebuild_idx] for var in vars])

        penalty_terms = []
        for var in vars:
            if type(var) is CatVar:
                # TODO: Make sure penalty for categoricals is working/what we want?
                penalty_terms.append(cp.norm(var.beta, 1))
            elif type(var) is FilterVar:
                penalty_terms.append(cp.norm(var.D_mat @ var.beta, 1))
        penalty = cp.sum(penalty_terms)

        weights = np.ones(X.shape[0]) if weights is None else np.asarray(weights)

        # Rescale the penalty term by the sum of sample weights so relevant scale of lambda is the same
        # regardless of size of input dataset. This is the same as taking the weighted mean deviance, but
        # is more numerically stable (fewer tiny values)
        objective = cp.Minimize(dist.deviance(y, eta, weights, link) + np.sum(weights) * self.lam * penalty)
        constraints = [cp.Zero(cp.sum(var.beta)) for var in vars if type(var) is FilterVar]
        # TODO: Set initial guess intelligently from data
        # TODO: Test solver settings to increase performance for our specific problem
        # CVXPY Problem is annotated as List[Constraint] which is invariant, so a list of Zero constraints is not
        # considered a subtype. The type annotation for Problem should be Sequence[Constraint]
        problem = cp.Problem(objective, constraints)  # type: ignore
        results = problem.solve(solver="CLARABEL", verbose=True)
        # TODO: check for convergence

        self.vars_ = []
        for i, var in enumerate(vars):
            # TODO: We only want to store values for the breakpoints instead of keeping a value around for every
            # unique value that we started with
            if var.beta.value is None:
                raise ValueError(f"Beta vector for var {i} is None and is not being fitted correctly")

            if type(var) is CatVar:
                self.vars_.append(FittedCatVar(var.unique_vals, var.beta.value, var.beta.name()))
            elif type(var) is FilterVar:
                self.vars_.append(FittedFilterVar(var.unique_vals, var.beta.value, var.beta.name()))

        self.mu_ = link.eval_inverse(eta).value
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


class TrendFilterRegressionCV(RegressorMixin, BaseEstimator):
    def __init__(
        self,
        dist: Optional[_dists.Distribution] = None,
        link: Optional[_links.LinkFunction] = None,
        lams: Sequence[float] = [0.01, 0.1, 1, 10],
    ) -> None:
        super().__init__()
        self.lams = lams
        self.dist = dist
        self.link = link

    def _make_terms(self, X: npt.NDArray, y: npt.NDArray, weights: npt.NDArray, categorical_features: Sequence[int]):
        # TODO: Check to make sure categorical_features contains only valid values
        vars = []
        for i in range(X.shape[1]):
            if i in categorical_features:
                vars.append(
                    # validate_data() sets feature_names_in_ if feature names are given by X or else deletes the
                    # attribute. The type-checker does not like this.
                    CatVar(X[:, i], name=self.feature_names_in_[i] if hasattr(self, "feature_names_in_") else None)  # type: ignore
                )
            else:
                vars.append(
                    FilterVar(X[:, i], name=self.feature_names_in_[i] if hasattr(self, "feature_names_in_") else None)  # type: ignore
                )

        alpha = cp.Variable(name="alpha")
        eta = alpha + cp.sum([var.beta[var.rebuild_idx] for var in vars])

        penalty_terms = []
        for var in vars:
            if type(var) is CatVar:
                # TODO: Make sure penalty for categoricals is working/what we want?
                penalty_terms.append(cp.norm(var.beta, 1))
            elif type(var) is FilterVar:
                penalty_terms.append(cp.norm(var.D_mat @ var.beta, 1))
        penalty = cp.sum(penalty_terms)

        return vars, alpha, eta, penalty

    def fit(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        weights: Optional[npt.ArrayLike] = None,
        categorical_features: Optional[Sequence[int]] = None,
        cv: Optional[BaseCrossValidator] = None,
    ):
        dist = _dists.NormalDistribution() if self.dist is None else self.dist
        link = dist.canonical_link() if self.link is None else self.link
        if not categorical_features:
            categorical_features = []
        cv = KFold() if cv is None else cv

        # For some reason pylance cannot resolve any of the functions inside sklearn's validation file
        checked: tuple[npt.NDArray, npt.NDArray] = skval.validate_data(  # type: ignore
            self, X, y, reset=True, y_numeric=True, ensure_min_samples=2
        )
        X, y = checked

        weights = np.ones(X.shape[0]) if weights is None else np.asarray(weights)

        # Rescale the penalty term by the sum of sample weights so relevant scale of lambda is the same
        # regardless of size of input dataset. This is the same as taking the weighted mean deviance, but
        # is more numerically stable (fewer tiny values)
        cv_results = np.zeros((cv.get_n_splits(), len(self.lams)))
        for i, (train, test) in enumerate(cv.split(X, y)):
            X_train = X[train]
            y_train = y[train]
            weights_train = weights[train]

            X_test = X[test]
            y_test = y[test]
            weights_test = weights[test]

            vars, alpha, eta, penalty = self._make_terms(X_train, y_train, weights_train, categorical_features)
            lam = cp.Parameter(name="lambda", nonneg=True)
            objective = cp.Minimize(
                dist.deviance(y_train, eta, weights_train, link) + np.sum(weights_train) * lam * penalty
            )
            constraints = [cp.Zero(cp.sum(var.beta)) for var in vars if type(var) is FilterVar]
            # TODO: Set initial guess intelligently from data
            # TODO: Test solver settings to increase performance for our specific problem
            # CVXPY Problem is annotated as List[Constraint] which is invariant, so a list of Zero constraints is not
            # considered a subtype. The type annotation for Problem should be Sequence[Constraint]
            problem = cp.Problem(objective, constraints)  # type: ignore

            fold_results = np.zeros(len(self.lams))
            for i, lam_val in enumerate(self.lams):
                lam.value = lam_val
                problem.solve(solver="CLARABEL", verbose=True)
                # TODO: check for convergence

                oos_eta = np.full(X_test.shape[0], alpha.value)
                for i in range(X_test.shape[1]):
                    oos_eta += vars[i].predict(X_test[:, i]).value

                fold_results[i] = dist.deviance(y_test, oos_eta, weights_test, link).value

            cv_results[i] = fold_results

        return cv_results

        self.vars_ = []
        for i, var in enumerate(vars):
            # TODO: We only want to store values for the breakpoints instead of keeping a value around for every
            # unique value that we started with
            if var.beta.value is None:
                raise ValueError(f"Beta vector for var {i} is None and is not being fitted correctly")

            if type(var) is CatVar:
                self.vars_.append(FittedCatVar(var.unique_vals, var.beta.value, var.beta.name()))
            elif type(var) is FilterVar:
                self.vars_.append(FittedFilterVar(var.unique_vals, var.beta.value, var.beta.name()))

        self.mu_ = link.eval_inverse(eta).value
        self.eta_ = eta.value

        if alpha.value is None:
            raise ValueError("Intercept is None and has not been fitted correctly")
        self.intercept_ = alpha.value.item()

        return self
