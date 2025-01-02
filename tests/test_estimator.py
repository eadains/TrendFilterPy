import numpy as np
import numpy.testing as nptest
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from trendfilterpy import TrendFilterRegression


@parametrize_with_checks([TrendFilterRegression()])
def test_sklearn_checks(estimator, check) -> None:
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
