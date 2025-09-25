import cvxpy as cp
import numpy as np
import pytest

from trendfilterpy._variables import CatVar, FilterVar, FittedCatVar, FittedFilterVar


class TestCatVar:
    """Test cases for the CatVar class."""

    def setup_method(self):
        """Setup test data for each test method."""
        self.x_train = np.array([0, 1, 2, 0, 1])
        self.x_test_seen = np.array([0, 1, 2])
        self.x_test_unseen = np.array([3, 4])
        self.x_test_mixed = np.array([0, 3, 1, 4, 2])
        self.beta_values = np.array([10.0, 20.0, 30.0])  # 0=10.0, 1=20.0, 2=30.0

    def test_predict_seen_values_error_mode(self):
        """Test prediction with only seen values in error mode."""
        cat_var = CatVar(self.x_train, unseen_action="error")
        result = cat_var.predict(self.x_test_seen)

        # Result should be a CVXPY expression
        assert isinstance(result, cp.Expression)
        # Should have same length as input
        assert result.shape == (3,)

        # Test that the expression values are correct
        cat_var.beta.value = self.beta_values
        expected_values = np.array([10.0, 20.0, 30.0])  # x_test_seen = [0, 1, 2]
        np.testing.assert_array_almost_equal(result.value, expected_values)

    def test_predict_seen_values_ignore_mode(self):
        """Test prediction with only seen values in ignore mode."""
        cat_var = CatVar(self.x_train, unseen_action="ignore")
        result = cat_var.predict(self.x_test_seen)

        # Result should be a CVXPY expression
        assert isinstance(result, cp.Expression)
        # Should have same length as input
        assert result.shape == (3,)

        # Test that the expression values are correct
        cat_var.beta.value = self.beta_values
        expected_values = np.array([10.0, 20.0, 30.0])  # x_test_seen = [0, 1, 2]
        np.testing.assert_array_almost_equal(result.value, expected_values)

    def test_predict_unseen_values_error_mode_raises_error(self):
        """Test that unseen values raise ValueError in error mode."""
        cat_var = CatVar(self.x_train, unseen_action="error")

        with pytest.raises(ValueError, match="Unseen categorical values"):
            cat_var.predict(self.x_test_unseen)

    def test_predict_mixed_values_error_mode_raises_error(self):
        """Test that mixed seen/unseen values raise ValueError in error mode."""
        cat_var = CatVar(self.x_train, unseen_action="error")

        with pytest.raises(ValueError, match="Unseen categorical values"):
            cat_var.predict(self.x_test_mixed)

    def test_predict_unseen_values_ignore_mode_returns_zeros(self):
        """Test that unseen values return zeros in ignore mode."""
        cat_var = CatVar(self.x_train, unseen_action="ignore")
        result = cat_var.predict(self.x_test_unseen)

        # Result should be a CVXPY expression
        assert isinstance(result, cp.Expression)
        # Should have same length as input
        assert result.shape == (2,)

        # Test that the expression values are correct (should be zeros for unseen values)
        cat_var.beta.value = self.beta_values
        expected_values = np.array([0.0, 0.0])  # x_test_unseen = [3, 4] (both unseen)
        np.testing.assert_array_almost_equal(result.value, expected_values)

    def test_predict_mixed_values_ignore_mode(self):
        """Test prediction with mixed seen/unseen values in ignore mode."""
        cat_var = CatVar(self.x_train, unseen_action="ignore")
        result = cat_var.predict(self.x_test_mixed)

        # Result should be a CVXPY expression
        assert isinstance(result, cp.Expression)
        # Should have same length as input
        assert result.shape == (5,)

        # Test that the expression values are correct
        cat_var.beta.value = self.beta_values
        expected_values = np.array([10.0, 0.0, 20.0, 0.0, 30.0])  # x_test_mixed = [0, 3, 1, 4, 2]
        np.testing.assert_array_almost_equal(result.value, expected_values)


class TestFilterVar:
    """Test cases for the FilterVar class."""

    def setup_method(self):
        """Setup test data for each test method."""
        # Use continuous values for FilterVar
        self.x_train = np.array([1.0, 2.5, 3.7, 1.0, 2.5])
        self.x_test_within = np.array([1.5, 2.0, 3.0])  # Values within the range
        self.x_test_below = np.array([0.5])  # Value below minimum
        self.x_test_above = np.array([4.0])  # Value above maximum
        self.x_test_mixed = np.array([0.5, 1.5, 2.8, 4.0])  # Mixed range
        self.beta_values = np.array([10.0, 20.0, 30.0])  # Values for unique points [1.0, 2.5, 3.7]

    def test_init_creates_correct_structures(self):
        """Test that FilterVar initialization creates correct data structures."""
        filter_var = FilterVar(self.x_train)

        # Should have sorted unique values
        expected_unique = np.array([1.0, 2.5, 3.7])
        np.testing.assert_array_equal(filter_var.unique_vals, expected_unique)

        # Should have CVXPY variable with correct size
        assert isinstance(filter_var.beta, cp.Variable)
        assert filter_var.beta.shape == (3,)

    def test_predict_within_range(self):
        """Test prediction with values within the training range."""
        filter_var = FilterVar(self.x_train)
        result = filter_var.predict(self.x_test_within)

        # Result should be a CVXPY expression
        assert isinstance(result, cp.Expression)
        assert result.shape == (3,)

        # Test that the expression values are correct
        filter_var.beta.value = self.beta_values
        # For stepwise right-continuous function:
        # x=1.5 falls in [1.0, 2.5) -> index 0 -> value 10.0
        # x=2.0 falls in [1.0, 2.5) -> index 0 -> value 10.0
        # x=3.0 falls in [2.5, 3.7) -> index 1 -> value 20.0
        expected_values = np.array([10.0, 10.0, 20.0])
        np.testing.assert_array_almost_equal(result.value, expected_values)

    def test_predict_below_minimum(self):
        """Test prediction with value below minimum training value."""
        filter_var = FilterVar(self.x_train)
        result = filter_var.predict(self.x_test_below)

        # Should return first beta value for values below minimum
        filter_var.beta.value = self.beta_values
        expected_values = np.array([10.0])  # Uses first value
        np.testing.assert_array_almost_equal(result.value, expected_values)

    def test_predict_above_maximum(self):
        """Test prediction with value above maximum training value."""
        filter_var = FilterVar(self.x_train)
        result = filter_var.predict(self.x_test_above)

        # Should return last beta value for values above maximum
        filter_var.beta.value = self.beta_values
        expected_values = np.array([30.0])  # Uses last value
        np.testing.assert_array_almost_equal(result.value, expected_values)

    def test_predict_mixed_range(self):
        """Test prediction with mixed range of values."""
        filter_var = FilterVar(self.x_train)
        result = filter_var.predict(self.x_test_mixed)

        filter_var.beta.value = self.beta_values
        # x=0.5 (below min) -> index 0 -> 10.0
        # x=1.5 (in [1.0, 2.5)) -> index 0 -> 10.0
        # x=2.8 (in [2.5, 3.7)) -> index 1 -> 20.0
        # x=4.0 (above max) -> index 2 -> 30.0
        expected_values = np.array([10.0, 10.0, 20.0, 30.0])
        np.testing.assert_array_almost_equal(result.value, expected_values)

    def test_predict_exact_training_values(self):
        """Test prediction with exact training values."""
        filter_var = FilterVar(self.x_train)
        result = filter_var.predict(filter_var.unique_vals)

        filter_var.beta.value = self.beta_values
        # Exact values should map to their corresponding beta values
        expected_values = np.array([10.0, 20.0, 30.0])
        np.testing.assert_array_almost_equal(result.value, expected_values)


class TestFittedFilterVar:
    """Test cases for the FittedFilterVar class."""

    def setup_method(self):
        """Setup test data for each test method."""
        self.unique_vals = np.array([1.0, 2.5, 3.7])
        self.beta = np.array([10.0, 20.0, 30.0])
        self.x_test_within = np.array([1.5, 2.0, 3.0])
        self.x_test_below = np.array([0.5])
        self.x_test_above = np.array([4.0])
        self.x_test_mixed = np.array([0.5, 1.5, 2.8, 4.0])

    def test_predict_within_range(self):
        """Test prediction with values within the range."""
        fitted_filter_var = FittedFilterVar(self.unique_vals, self.beta)
        result = fitted_filter_var.predict(self.x_test_within)

        expected_values = np.array([10.0, 10.0, 20.0])
        np.testing.assert_array_almost_equal(result, expected_values)

    def test_predict_below_minimum(self):
        """Test prediction with value below minimum."""
        fitted_filter_var = FittedFilterVar(self.unique_vals, self.beta)
        result = fitted_filter_var.predict(self.x_test_below)

        expected_values = np.array([10.0])
        np.testing.assert_array_almost_equal(result, expected_values)

    def test_predict_above_maximum(self):
        """Test prediction with value above maximum."""
        fitted_filter_var = FittedFilterVar(self.unique_vals, self.beta)
        result = fitted_filter_var.predict(self.x_test_above)

        expected_values = np.array([30.0])
        np.testing.assert_array_almost_equal(result, expected_values)

    def test_predict_mixed_range(self):
        """Test prediction with mixed range of values."""
        fitted_filter_var = FittedFilterVar(self.unique_vals, self.beta)
        result = fitted_filter_var.predict(self.x_test_mixed)

        expected_values = np.array([10.0, 10.0, 20.0, 30.0])
        np.testing.assert_array_almost_equal(result, expected_values)

    def test_predict_exact_training_values(self):
        """Test prediction with exact training values."""
        fitted_filter_var = FittedFilterVar(self.unique_vals, self.beta)
        result = fitted_filter_var.predict(self.unique_vals)

        expected_values = np.array([10.0, 20.0, 30.0])
        np.testing.assert_array_almost_equal(result, expected_values)

    def test_predict_single_value(self):
        """Test prediction with a single value."""
        fitted_filter_var = FittedFilterVar(self.unique_vals, self.beta)
        result = fitted_filter_var.predict(np.array([2.7]))

        expected_values = np.array([20.0])  # Falls in [2.5, 3.7)
        np.testing.assert_array_almost_equal(result, expected_values)

    def test_predict_empty_array(self):
        """Test prediction with empty array."""
        fitted_filter_var = FittedFilterVar(self.unique_vals, self.beta)
        result = fitted_filter_var.predict(np.array([]))

        expected_values = np.array([])
        np.testing.assert_array_equal(result, expected_values)


class TestFittedCatVar:
    """Test cases for the FittedCatVar class."""

    def setup_method(self):
        """Setup test data for each test method."""
        self.unique_vals = np.array([0, 1, 2])
        self.beta = np.array([1.0, 2.0, 3.0])
        self.x_test_seen = np.array([0, 1, 2])
        self.x_test_unseen = np.array([3, 4])
        self.x_test_mixed = np.array([0, 3, 1, 4, 2])

    def test_predict_seen_values_error_mode(self):
        """Test prediction with only seen values in error mode."""
        fitted_cat_var = FittedCatVar(self.unique_vals, self.beta, unseen_action="error")
        result = fitted_cat_var.predict(self.x_test_seen)

        # Should return correct beta values
        expected = np.array([1.0, 2.0, 3.0])  # 0=1.0, 1=2.0, 2=3.0
        np.testing.assert_array_equal(result, expected)

    def test_predict_seen_values_ignore_mode(self):
        """Test prediction with only seen values in ignore mode."""
        fitted_cat_var = FittedCatVar(self.unique_vals, self.beta, unseen_action="ignore")
        result = fitted_cat_var.predict(self.x_test_seen)

        # Should return correct beta values
        expected = np.array([1.0, 2.0, 3.0])  # 0=1.0, 1=2.0, 2=3.0
        np.testing.assert_array_equal(result, expected)

    def test_predict_unseen_values_error_mode_raises_error(self):
        """Test that unseen values raise ValueError in error mode."""
        fitted_cat_var = FittedCatVar(self.unique_vals, self.beta, unseen_action="error")

        with pytest.raises(ValueError, match="Unseen categorical values"):
            fitted_cat_var.predict(self.x_test_unseen)

    def test_predict_mixed_values_error_mode_raises_error(self):
        """Test that mixed seen/unseen values raise ValueError in error mode."""
        fitted_cat_var = FittedCatVar(self.unique_vals, self.beta, unseen_action="error")

        with pytest.raises(ValueError, match="Unseen categorical values"):
            fitted_cat_var.predict(self.x_test_mixed)

    def test_predict_unseen_values_ignore_mode_returns_zeros(self):
        """Test that unseen values return zeros in ignore mode."""
        fitted_cat_var = FittedCatVar(self.unique_vals, self.beta, unseen_action="ignore")
        result = fitted_cat_var.predict(self.x_test_unseen)

        # Should return zeros for unseen values
        expected = np.array([0.0, 0.0])  # 3=0.0, 4=0.0
        np.testing.assert_array_equal(result, expected)

    def test_predict_mixed_values_ignore_mode(self):
        """Test prediction with mixed seen/unseen values in ignore mode."""
        fitted_cat_var = FittedCatVar(self.unique_vals, self.beta, unseen_action="ignore")
        result = fitted_cat_var.predict(self.x_test_mixed)

        # Should return correct values for seen, zeros for unseen
        # Input: [0, 3, 1, 4, 2]
        expected = np.array([1.0, 0.0, 2.0, 0.0, 3.0])  # 0=1.0, 3=0.0, 1=2.0, 4=0.0, 2=3.0
        np.testing.assert_array_equal(result, expected)

    def test_predict_single_seen_value(self):
        """Test prediction with a single seen value."""
        fitted_cat_var = FittedCatVar(self.unique_vals, self.beta, unseen_action="ignore")
        result = fitted_cat_var.predict(np.array([1]))

        expected = np.array([2.0])  # 1=2.0
        np.testing.assert_array_equal(result, expected)

    def test_predict_single_unseen_value_ignore(self):
        """Test prediction with a single unseen value in ignore mode."""
        fitted_cat_var = FittedCatVar(self.unique_vals, self.beta, unseen_action="ignore")
        result = fitted_cat_var.predict(np.array([3]))

        expected = np.array([0.0])  # 3=0.0
        np.testing.assert_array_equal(result, expected)

    def test_predict_single_unseen_value_error(self):
        """Test prediction with a single unseen value in error mode."""
        fitted_cat_var = FittedCatVar(self.unique_vals, self.beta, unseen_action="error")

        with pytest.raises(ValueError, match="Unseen categorical values"):
            fitted_cat_var.predict(np.array([3]))

    def test_predict_empty_array_ignore(self):
        """Test prediction with empty array in ignore mode."""
        fitted_cat_var = FittedCatVar(self.unique_vals, self.beta, unseen_action="ignore")
        result = fitted_cat_var.predict(np.array([]))

        # Should return empty array
        expected = np.array([])
        np.testing.assert_array_equal(result, expected)

    def test_predict_all_same_unseen_value_ignore(self):
        """Test prediction with all same unseen value in ignore mode."""
        fitted_cat_var = FittedCatVar(self.unique_vals, self.beta, unseen_action="ignore")
        result = fitted_cat_var.predict(np.array([9, 9, 9]))

        expected = np.array([0.0, 0.0, 0.0])
        np.testing.assert_array_equal(result, expected)

    def test_predict_numeric_categories(self):
        """Test with numeric categorical values."""
        unique_vals = np.array([1, 2, 3])
        beta = np.array([10.0, 20.0, 30.0])
        fitted_cat_var = FittedCatVar(unique_vals, beta, unseen_action="ignore")

        # Test mixed seen/unseen numeric values
        result = fitted_cat_var.predict(np.array([1, 5, 2, 7, 3]))
        expected = np.array([10.0, 0.0, 20.0, 0.0, 30.0])  # 1=10.0, 5=0.0, 2=20.0, 7=0.0, 3=30.0
        np.testing.assert_array_equal(result, expected)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_catvar_with_duplicate_training_values(self):
        """Test CatVar with duplicate values in training data."""
        x_train = np.array([0, 0, 1, 1, 2])
        cat_var = CatVar(x_train, unseen_action="ignore")

        # Should still work correctly
        result = cat_var.predict(np.array([0, 3, 1]))
        assert isinstance(result, cp.Expression)
        assert result.shape == (3,)

    def test_fitted_catvar_with_single_category(self):
        """Test FittedCatVar with only one category."""
        unique_vals = np.array([0])
        beta = np.array([5.0])
        fitted_cat_var = FittedCatVar(unique_vals, beta, unseen_action="ignore")

        # Test with seen value
        result = fitted_cat_var.predict(np.array([0]))
        np.testing.assert_array_equal(result, np.array([5.0]))

        # Test with unseen value
        result = fitted_cat_var.predict(np.array([1]))
        np.testing.assert_array_equal(result, np.array([0.0]))
