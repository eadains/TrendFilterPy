from typing import Callable

import numpy as np
import numpy.testing as nptest
import pytest
from sklearn.model_selection import KFold
from sklearn.utils.estimator_checks import parametrize_with_checks

from trendfilterpy import TrendFilterRegression, TrendFilterRegressionCV


# TODO: Check why this fails type checking
@parametrize_with_checks([TrendFilterRegression(), TrendFilterRegressionCV()])  # type: ignore
def test_sklearn_checks(estimator: TrendFilterRegression, check: Callable) -> None:
    """Check that our estimator passes sklearn checks that valid estimators should pass."""
    check(estimator)


class TestFitNoPenalty:
    """Check model fit procedure with no penalty term, default normal distribution, and default identity link.

    This effectively tests the least squares version of the model.
    When lam=0 the fitted beta values from the model should exactly match the given y values for each unique value of x.
    """

    def test_continuous_1var_unique(self) -> None:
        """Check model with 1 continuous input variable that has no duplicate values."""
        X = np.asarray([[x] for x in range(10)])
        y = (X * 3).ravel()
        model = TrendFilterRegression(lam=0)
        model.fit(X, y)

        assert len(model.vars_) == 1
        assert model.intercept_ == pytest.approx(np.mean(y))
        # Because of zero penalty fitted beta values should be y values just recentered around the intercept
        nptest.assert_allclose(model.vars_[0].beta, y - np.mean(y), atol=1e-08)
        nptest.assert_allclose(model.mu_, y, atol=1e-08)

    def test_continuous_2var_unique(self) -> None:
        """Check model fit with 2 continuous input variables neither of which have duplicate values."""
        X = np.asarray([[x, x * 2 + 1] for x in range(10)])
        y = 5 + X[:, 0] + 3 * X[:, 1]
        model = TrendFilterRegression(lam=0)
        model.fit(X, y)

        assert len(model.vars_) == 2
        assert model.intercept_ == pytest.approx(np.mean(y))
        nptest.assert_allclose(model.vars_[0].beta + model.vars_[1].beta, y - np.mean(y), atol=1e-08)
        nptest.assert_allclose(model.mu_, y, atol=1e-08)

    def test_cont_1var_duplicates(self) -> None:
        """Check model fit with 1 continuous variable but with duplicate X values."""
        pass


class TestTrendFilterRegressionCV:
    """Test cross-validation functionality of TrendFilterRegressionCV."""

    def test_cv_basic_functionality(self) -> None:
        """Test basic CV functionality with simple data."""
        X = np.asarray([[x] for x in range(20)])
        y = X.ravel() + np.random.RandomState(42).normal(0, 0.1, 20)

        model = TrendFilterRegressionCV(lams=[0.1, 1.0, 10.0])
        model.fit(X, y)

        assert hasattr(model, "best_lam_")
        assert model.best_lam_ in [0.1, 1.0, 10.0]
        assert hasattr(model, "intercept_")
        assert hasattr(model, "vars_")
        assert len(model.vars_) == 1

        predictions = model.predict(X)
        assert predictions.shape == (20,)

    def test_cv_custom_cv_splitter(self) -> None:
        """Test CV with custom cross-validation splitter."""
        X = np.asarray([[x] for x in range(20)])
        y = X.ravel() + np.random.RandomState(42).normal(0, 0.1, 20)

        model = TrendFilterRegressionCV(lams=[0.1, 1.0])
        cv = KFold(n_splits=3, shuffle=True, random_state=42)
        model.fit(X, y, cv=cv)

        assert hasattr(model, "best_lam_")

    def test_cv_with_different_distributions(self) -> None:
        """Test CV with different distributions."""
        from trendfilterpy import _dists

        X = np.asarray([[x] for x in range(20)])
        y = np.random.RandomState(42).poisson(np.exp(X.ravel() * 0.1))

        model = TrendFilterRegressionCV(dist=_dists.PoissonDistribution(), lams=[0.1, 1.0])
        model.fit(X, y)

        assert hasattr(model, "best_lam_")
        predictions = model.predict(X)
        assert predictions.shape == (20,)

    def test_cv_empty_lams_list(self) -> None:
        """Test CV with empty lambda list should raise error."""
        X = np.asarray([[x] for x in range(10)])
        y = X.ravel()

        model = TrendFilterRegressionCV(lams=[])
        with pytest.raises((ValueError, IndexError)):
            model.fit(X, y)

    def test_cv_single_lambda(self) -> None:
        """Test CV with single lambda value."""
        X = np.asarray([[x] for x in range(10)])
        y = X.ravel() + np.random.RandomState(42).normal(0, 0.1, 10)

        model = TrendFilterRegressionCV(lams=[1.0])
        model.fit(X, y)

        assert model.best_lam_ == 1.0

    def test_cv_predict_before_fit(self) -> None:
        """Test that predict raises error before fit is called."""
        X = np.asarray([[x] for x in range(10)])

        model = TrendFilterRegressionCV()
        from sklearn.exceptions import NotFittedError

        with pytest.raises(NotFittedError):
            model.predict(X)

    def test_cv_fit_validation(self) -> None:
        """Test input validation in fit method."""
        model = TrendFilterRegressionCV()

        # Test with invalid X shape
        with pytest.raises(ValueError):
            model.fit([], [1, 2, 3])

        # Test with mismatched X and y shapes
        X = np.asarray([[1], [2], [3]])
        y = np.asarray([1, 2])
        with pytest.raises(ValueError):
            model.fit(X, y)

    def test_cv_reproducibility(self) -> None:
        """Test that CV results are reproducible with same random state."""
        X = np.asarray([[x] for x in range(20)])
        y = X.ravel() + np.random.RandomState(42).normal(0, 0.1, 20)

        cv1 = KFold(n_splits=3, shuffle=True, random_state=42)
        cv2 = KFold(n_splits=3, shuffle=True, random_state=42)

        model1 = TrendFilterRegressionCV(lams=[0.1, 1.0, 10.0])
        model2 = TrendFilterRegressionCV(lams=[0.1, 1.0, 10.0])

        model1.fit(X, y, cv=cv1)
        model2.fit(X, y, cv=cv2)

        assert model1.best_lam_ == model2.best_lam_
