from typing import Optional, Sequence, Union

import cvxpy as cp
import numpy as np
import numpy.typing as npt
import sklearn.utils.validation as skval
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import BaseCrossValidator, KFold
from typing_extensions import Self

from trendfilterpy import _dists, _links
from trendfilterpy._variables import CatVar, FilterVar, FittedCatVar, FittedFilterVar


class TrendFilterRegression(RegressorMixin, BaseEstimator):
    """
    Trend filtering regression using generalized linear models with regularization.

    This class implements trend filtering regression that fits piecewise-linear functions
    to continuous variables and handles categorical variables using L1 regularization.
    The model minimizes the deviance (negative log-likelihood) of a generalized linear model
    plus an L1 penalty on the second differences (for continuous variables) or coefficients
    (for categorical variables).

    The optimization problem solved is:
        min_β Deviance(y, η(X,β)) + λ * Penalty(β)

    Where:
    - η is the linear predictor (intercept + sum of variable contributions)
    - Deviance depends on the distribution (Normal, Poisson, etc.) and link function
    - Penalty is ||D²β||₁ for continuous variables and ||β||₁ for categorical variables
    - D² is the second-difference matrix that promotes piecewise-linear solutions

    Parameters
    ----------
    dist : Distribution, optional
        The distribution family for the generalized linear model. If None, uses
        NormalDistribution (equivalent to linear regression).
    link : LinkFunction, optional
        The link function relating the linear predictor to the mean. If None,
        uses the canonical link for the specified distribution.
    lam : float, default=0.01
        Regularization parameter (λ). Higher values produce smoother fits.
        Must be non-negative.

    Attributes
    ----------
    intercept_ : float
        The fitted intercept term (α) of the model.
    vars_ : list of FittedFilterVar or FittedCatVar
        Fitted variable objects containing the estimated coefficients and unique
        values for each input feature.
    mu_ : ndarray of shape (n_samples,)
        Fitted mean values μ = g⁻¹(η) where g is the link function.
    eta_ : ndarray of shape (n_samples,)
        Fitted linear predictor values η = α + Σᵢ βᵢ(xᵢ).

    Notes
    -----
    The trend filtering penalty ||D²β||₁ encourages piecewise-linear solutions by
    penalizing the second differences of the coefficient vector. This results in
    functions that are smooth but can have breakpoints where the slope changes.

    For categorical variables, the standard L1 penalty ||β||₁ is used instead,
    which encourages sparsity in the category effects.

    The constraint Σⱼ βⱼ = 0 is applied to continuous variables to ensure
    identifiability (the intercept captures the overall level).

    Examples
    --------
    >>> import numpy as np
    >>> from trendfilterpy import TrendFilterRegression
    >>> X = np.linspace(0, 10, 100).reshape(-1, 1)
    >>> y = np.sin(X.ravel()) + 0.1 * np.random.randn(100)
    >>> model = TrendFilterRegression(lam=0.1)
    >>> model.fit(X, y)
    >>> predictions = model.predict(X)
    """

    def __init__(
        self, dist: Optional[_dists.Distribution] = None, link: Optional[_links.LinkFunction] = None, lam: float = 0.01
    ) -> None:
        """
        Initialize the TrendFilterRegression estimator.

        Parameters
        ----------
        dist : Distribution, optional
            The distribution family for the generalized linear model. If None,
            defaults to NormalDistribution (Gaussian/linear regression).
            Available distributions include Normal, Poisson, Binomial, Gamma,
            InverseGaussian, and Tweedie.
        link : LinkFunction, optional
            The link function g(μ) relating the mean μ to the linear predictor η.
            If None, uses the canonical link for the specified distribution
            (e.g., identity for Normal, log for Poisson, logit for Binomial).
        lam : float, default=0.01
            Regularization parameter controlling the strength of the penalty.
            Higher values produce smoother, more regularized fits. Must be >= 0.
            When lam=0, no penalty is applied (interpolation).
        """
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
    ) -> Self:
        """
        Fit the trend filtering regression model.

        This method solves the convex optimization problem:
            min_{α,β} Deviance(y, α + Σᵢ βᵢ(xᵢ)) + λ * Σᵢ Penalty(βᵢ)

        Where Penalty(βᵢ) is ||D²βᵢ||₁ for continuous variables (promoting
        piecewise-linear functions) and ||βᵢ||₁ for categorical variables
        (promoting sparsity).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data. Each column represents a feature, with continuous
            features treated as trend filtering variables and categorical
            features specified via categorical_features parameter.
        y : array-like of shape (n_samples,)
            Target values. Must be numeric and compatible with the chosen
            distribution (e.g., non-negative integers for Poisson).
        weights : array-like of shape (n_samples,), optional
            Sample weights. If None, all samples have equal weight.
            The penalty term is automatically scaled by the sum of weights
            to maintain consistent regularization strength across datasets.
        categorical_features : sequence of int, optional
            Indices of features to treat as categorical. These features will
            use L1 penalty instead of the trend filtering penalty. If None,
            all features are treated as continuous.

        Returns
        -------
        self : TrendFilterRegression
            Fitted estimator.

        Notes
        -----
        The optimization uses the CLARABEL solver through CVXPY. The solver
        parameters are currently fixed but may be made configurable in future
        versions.

        For continuous variables, the sum-to-zero constraint Σⱼ βⱼ = 0 is
        enforced to ensure identifiability. This means the intercept captures
        the overall level while the βⱼ coefficients represent deviations.

        Raises
        ------
        ValueError
            If the optimization fails to converge or if input validation fails.
        """
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
        problem.solve(solver="CLARABEL", verbose=True)
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

    def predict(self, X: npt.ArrayLike) -> npt.NDArray:
        """
        Predict using the fitted trend filtering model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples for prediction. Must have the same number of
            features as the training data.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values. These are the fitted values μ = g⁻¹(η)
            where g is the link function and η is the linear predictor.

        Notes
        -----
        The prediction is computed as:
            η = α + Σᵢ βᵢ(xᵢ)
            μ = g⁻¹(η)

        Where α is the intercept, βᵢ are the fitted variable functions,
        and g⁻¹ is the inverse link function.

        For values of X outside the range seen during training, the
        fitted functions extrapolate using the trend at the boundaries.
        """
        skval.check_is_fitted(self)
        checked: npt.NDArray = skval.validate_data(self, X, reset=False)  # type: ignore
        X = checked

        predictions = np.full(X.shape[0], self.intercept_)
        for i in range(X.shape[1]):
            predictions += self.vars_[i].predict(X[:, i])

        return predictions


class TrendFilterRegressionCV(RegressorMixin, BaseEstimator):
    """
    Trend filtering regression with cross-validation for hyperparameter selection.

    This class extends TrendFilterRegression by automatically selecting the optimal
    regularization parameter λ using cross-validation. It evaluates multiple λ values
    on cross-validation folds and selects the one that minimizes the out-of-sample
    deviance.

    The cross-validation procedure:
    1. Split the data into training and validation folds
    2. For each λ value and each fold:
       - Fit the model on the training portion
       - Evaluate deviance on the validation portion
    3. Select λ with the lowest average validation deviance
    4. Refit the model on the full dataset using the selected λ

    Parameters
    ----------
    dist : Distribution, optional
        The distribution family for the generalized linear model. If None, uses
        NormalDistribution (equivalent to linear regression).
    link : LinkFunction, optional
        The link function relating the linear predictor to the mean. If None,
        uses the canonical link for the specified distribution.
    lams : sequence of float, default=(0.01, 0.1, 1, 10)
        Grid of regularization parameters to evaluate via cross-validation.
        All values must be non-negative.

    Attributes
    ----------
    best_lam_ : float
        The regularization parameter selected by cross-validation.
    intercept_ : float
        The fitted intercept term using the best λ.
    vars_ : list of FittedFilterVar or FittedCatVar
        Fitted variable objects using the best λ.
    mu_ : ndarray of shape (n_samples,)
        Fitted mean values using the best λ.
    eta_ : ndarray of shape (n_samples,)
        Fitted linear predictor values using the best λ.

    Notes
    -----
    Cross-validation helps prevent overfitting by selecting λ based on
    out-of-sample performance rather than in-sample fit. This typically
    leads to better generalization to new data.

    The default λ grid spans several orders of magnitude. For best results,
    consider adjusting the grid based on the scale of your data and problem.

    Examples
    --------
    >>> import numpy as np
    >>> from trendfilterpy import TrendFilterRegressionCV
    >>> X = np.linspace(0, 10, 100).reshape(-1, 1)
    >>> y = np.sin(X.ravel()) + 0.1 * np.random.randn(100)
    >>> model = TrendFilterRegressionCV(lams=[0.01, 0.1, 1.0, 10.0])
    >>> model.fit(X, y)
    >>> print(f"Selected lambda: {model.best_lam_}")
    >>> predictions = model.predict(X)
    """

    def __init__(
        self,
        dist: Optional[_dists.Distribution] = None,
        link: Optional[_links.LinkFunction] = None,
        lams: Sequence[float] = (0.01, 0.1, 1, 10),
    ) -> None:
        """
        Initialize the TrendFilterRegressionCV estimator.

        Parameters
        ----------
        dist : Distribution, optional
            The distribution family for the generalized linear model. If None,
            defaults to NormalDistribution (Gaussian/linear regression).
        link : LinkFunction, optional
            The link function g(μ) relating the mean μ to the linear predictor η.
            If None, uses the canonical link for the specified distribution.
        lams : sequence of float, default=(0.01, 0.1, 1, 10)
            Grid of regularization parameters to evaluate via cross-validation.
            Should span a reasonable range for your problem. Smaller values
            produce more flexible fits, larger values produce smoother fits.
        """
        super().__init__()
        self.lams = lams
        self.dist = dist
        self.link = link

    def _make_terms(
        self, X: npt.NDArray, y: npt.NDArray, weights: npt.NDArray, categorical_features: Sequence[int]
    ) -> tuple[list[Union[FilterVar, CatVar]], cp.Variable, cp.Expression, cp.Expression]:
        """
        Create optimization variables and penalty terms for the CV problem.

        This helper method constructs the CVXPY variables and expressions needed
        for the trend filtering optimization, including the intercept, variable
        coefficients, linear predictor, and penalty terms.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training features.
        y : ndarray of shape (n_samples,)
            Target values (not used in this method but kept for consistency).
        weights : ndarray of shape (n_samples,)
            Sample weights (not used in this method but kept for consistency).
        categorical_features : sequence of int
            Indices of categorical features.

        Returns
        -------
        vars : list
            List of variable objects (CatVar or FilterVar) for each feature.
        alpha : cvxpy.Variable
            The intercept variable.
        eta : cvxpy.Expression
            The linear predictor η = α + Σᵢ βᵢ(xᵢ).
        penalty : cvxpy.Expression
            The total penalty term Σᵢ Penalty(βᵢ).
        """
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
        if isinstance(penalty, int):
            raise ValueError("Penalty term returned as an int, was expecting a cvxpy.Expression")

        return vars, alpha, eta, penalty

    def _make_problem(
        self,
        dist: _dists.Distribution,
        link: _links.LinkFunction,
        X: npt.NDArray,
        y: npt.NDArray,
        weights: npt.NDArray,
        categorical_features: Sequence[int],
    ) -> tuple[cp.Problem, cp.Parameter, list[Union[FilterVar, CatVar]], cp.Variable, cp.Expression]:
        """
        Create the complete CVXPY optimization problem for CV evaluation.

        This method combines the terms from _make_terms with the distribution's
        deviance function to create a complete optimization problem with a
        parameterized λ value for cross-validation.

        Parameters
        ----------
        dist : Distribution
            The distribution family for computing deviance.
        link : LinkFunction
            The link function for the GLM.
        X : ndarray of shape (n_samples, n_features)
            Training features.
        y : ndarray of shape (n_samples,)
            Target values.
        weights : ndarray of shape (n_samples,)
            Sample weights.
        categorical_features : sequence of int
            Indices of categorical features.

        Returns
        -------
        problem : cvxpy.Problem
            The complete optimization problem.
        lam : cvxpy.Parameter
            The λ parameter that can be varied for CV evaluation.
        vars : list
            Variable objects for each feature.
        alpha : cvxpy.Variable
            The intercept variable.
        eta : cvxpy.Expression
            The linear predictor.
        """
        vars, alpha, eta, penalty = self._make_terms(X, y, weights, categorical_features)

        lam = cp.Parameter(name="lambda", nonneg=True)
        objective = cp.Minimize(
            # Rescale the penalty term by the sum of sample weights so relevant scale of lambda is the same
            # regardless of size of input dataset. This is the same as taking the weighted mean deviance, but
            # is more numerically stable (fewer tiny values)
            dist.deviance(y, eta, weights, link) + np.sum(weights) * lam * penalty
        )
        constraints = [cp.Zero(cp.sum(var.beta)) for var in vars if type(var) is FilterVar]
        # TODO: Set initial guess intelligently from data
        # TODO: Test solver settings to increase performance for our specific problem
        # CVXPY Problem is annotated as List[Constraint] which is invariant, so a list of Zero constraints is not
        # considered a subtype. The type annotation for Problem should be Sequence[Constraint]
        return cp.Problem(objective, constraints), lam, vars, alpha, eta  # type: ignore

    def fit(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        weights: Optional[npt.ArrayLike] = None,
        categorical_features: Optional[Sequence[int]] = None,
        cv: Optional[BaseCrossValidator] = None,
    ) -> Self:
        """
        Fit the trend filtering model with cross-validation for λ selection.

        This method performs k-fold cross-validation to select the optimal
        regularization parameter, then refits the model on the full dataset
        using the selected λ.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        weights : array-like of shape (n_samples,), optional
            Sample weights. If None, all samples have equal weight.
        categorical_features : sequence of int, optional
            Indices of features to treat as categorical. If None,
            all features are treated as continuous.
        cv : BaseCrossValidator, optional
            Cross-validation strategy. If None, uses KFold with default settings.

        Returns
        -------
        self : TrendFilterRegressionCV
            Fitted estimator with selected hyperparameters.

        Notes
        -----
        The cross-validation process evaluates each λ in self.lams and selects
        the one with the lowest average out-of-sample deviance. This helps
        prevent overfitting and typically improves generalization.

        For computational efficiency, the optimization problem is recompiled
        for each CV fold, which can be time-consuming for large models.
        """
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

        cv_results = np.zeros((cv.get_n_splits(), len(self.lams)))
        for i, (train, test) in enumerate(cv.split(X, y)):
            print(f"CV Fold: {i+1}")
            X_train = X[train]
            y_train = y[train]
            weights_train = weights[train]

            X_test = X[test]
            y_test = y[test]
            weights_test = weights[test]

            # TODO: Refactor so we don't have to recompile the problem for each CV fold as this is quite time
            # consuming for larger models
            problem, lam, vars, alpha, _ = self._make_problem(
                dist, link, X_train, y_train, weights_train, categorical_features
            )

            fold_results = np.zeros(len(self.lams))
            for j, lam_val in enumerate(self.lams):
                lam.value = lam_val
                problem.solve(solver="CLARABEL", verbose=False)
                # TODO: check for convergence

                oos_eta = np.full(X_test.shape[0], alpha.value)
                for k in range(X_test.shape[1]):
                    # These vars are not Fitted, so this is a CVXPY expression, so we need to extract its value
                    oos_eta += vars[k].predict(X_test[:, k]).value

                print({np.sum(weights_test * (y_test - oos_eta) ** 2)})
                fold_results[j] = dist.deviance(y_test, oos_eta, weights_test, link).value

            cv_results[i] = fold_results

        mean_cv_results = np.mean(cv_results, axis=0)
        lam_min_idx = int(np.argmin(mean_cv_results))
        self.best_lam_ = self.lams[lam_min_idx]

        problem, lam, vars, alpha, eta = self._make_problem(dist, link, X, y, weights, categorical_features)
        lam.value = self.best_lam_
        problem.solve(solver="CLARABEL", verbose=False)

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

    def predict(self, X: npt.ArrayLike) -> npt.NDArray:
        """
        Predict using the fitted trend filtering model with CV-selected λ.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples for prediction. Must have the same number of
            features as the training data.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values using the model fitted with the best λ
            selected via cross-validation.

        Notes
        -----
        This method is identical to TrendFilterRegression.predict() but uses
        the model parameters fitted with the λ value selected through
        cross-validation.
        """
        skval.check_is_fitted(self)
        checked: npt.NDArray = skval.validate_data(self, X, reset=False)  # type: ignore
        X = checked

        predictions = np.full(X.shape[0], self.intercept_)
        for i in range(X.shape[1]):
            predictions += self.vars_[i].predict(X[:, i])

        return predictions
