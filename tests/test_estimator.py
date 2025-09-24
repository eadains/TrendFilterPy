from typing import Callable

import numpy as np
import numpy.testing as nptest
import pytest
from sklearn.model_selection import KFold
from sklearn.utils.estimator_checks import parametrize_with_checks

from trendfilterpy import TrendFilterRegression, TrendFilterRegressionCV


@parametrize_with_checks([TrendFilterRegression(), TrendFilterRegressionCV()])  # type: ignore
def test_sklearn_checks(estimator: TrendFilterRegression | TrendFilterRegressionCV, check: Callable) -> None:
    """Check that our estimator passes sklearn checks that valid estimators should pass."""
    check(estimator)


class TestTrendFilterRegression:
    """Test cases for the TrendFilterRegression class."""

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

    def test_cont_1var_duplicates(self) -> None:
        """Check model fit with 1 continuous variable but with duplicate X values."""
        X = np.asarray([[1], [1], [2], [2], [3], [3]])
        y = np.asarray([1, 1.1, 2, 2.1, 3, 3.1])
        # Use no regularization so the model fits perfectly
        model = TrendFilterRegression(lam=0)
        model.fit(X, y)

        assert len(model.vars_) == 1
        assert model.intercept_ == pytest.approx(np.mean(y))
        # Fitted values will be the average y values of the duplicate X inputs: (1 + 1.1) / 2 = 1.05
        nptest.assert_allclose(model.mu_, np.array([1.05, 1.05, 2.05, 2.05, 3.05, 3.05]), atol=1e-08)

    def test_basic_functionality(self) -> None:
        """Test basic TrendFilterRegression functionality with simple data."""
        X = np.asarray([[x] for x in range(20)])
        y = X.ravel() + np.random.RandomState(42).normal(0, 0.1, 20)

        model = TrendFilterRegression(lam=1.0)
        model.fit(X, y)

        assert hasattr(model, "intercept_")
        assert hasattr(model, "vars_")
        assert hasattr(model, "mu_")
        assert hasattr(model, "eta_")
        assert len(model.vars_) == 1

        predictions = model.predict(X)
        assert predictions.shape == (20,)

    def test_predict_before_fit(self) -> None:
        """Test that predict raises error before fit is called."""
        X = np.asarray([[x] for x in range(10)])

        model = TrendFilterRegression()
        from sklearn.exceptions import NotFittedError

        with pytest.raises(NotFittedError):
            model.predict(X)

    def test_different_lambda_values(self) -> None:
        """Test model with different lambda values."""
        X = np.asarray([[x] for x in range(20)])
        y = X.ravel() + np.random.RandomState(42).normal(0, 0.1, 20)

        # Test with very small lambda (should be close to no penalty)
        model_small = TrendFilterRegression(lam=1e-6)
        model_small.fit(X, y)
        pred_small = model_small.predict(X)

        # Test with large lambda (should be heavily penalized)
        model_large = TrendFilterRegression(lam=100.0)
        model_large.fit(X, y)
        pred_large = model_large.predict(X)

        # Large lambda should produce smoother (less variable) predictions
        assert np.var(pred_large) < np.var(pred_small)

    def test_with_different_distribution(self) -> None:
        """Test model with non-default distribution."""
        from trendfilterpy import _dists

        X = np.asarray([[x] for x in range(20)])
        y = np.random.RandomState(42).poisson(np.exp(X.ravel() * 0.1))

        model = TrendFilterRegression(dist=_dists.PoissonDistribution(), lam=0.1)
        model.fit(X, y)

        assert hasattr(model, "intercept_")
        assert hasattr(model, "vars_")
        predictions = model.predict(X)
        assert predictions.shape == (20,)

    def test_predict_on_new_data(self) -> None:
        """Test prediction on data not seen during training."""
        X_train = np.asarray([[x] for x in range(10)])
        y_train = X_train.ravel() + np.random.RandomState(42).normal(0, 0.1, 10)

        X_test = np.asarray([[x] for x in range(10, 15)])

        model = TrendFilterRegression(lam=1.0)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        assert predictions.shape == (5,)

    def test_feature_names_handling(self) -> None:
        """Test that model handles feature names correctly."""
        import pandas as pd

        # Create DataFrame with named features
        X = pd.DataFrame({"feature1": range(10), "feature2": range(10, 20)})
        y = X["feature1"] + 0.5 * X["feature2"] + np.random.RandomState(42).normal(0, 0.1, 10)

        model = TrendFilterRegression(lam=1.0)
        model.fit(X, y)

        # Should work with both DataFrame and numpy array for prediction
        pred_df = model.predict(X)
        pred_array = model.predict(X.values)

        assert pred_df.shape == (10,)
        assert pred_array.shape == (10,)
        nptest.assert_allclose(pred_df, pred_array)


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
        assert hasattr(model, "mu_")
        assert hasattr(model, "eta_")

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
