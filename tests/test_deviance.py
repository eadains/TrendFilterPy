from typing import Callable

import cvxpy as cp
import numpy as np
import numpy.typing as npt
import pytest
import statsmodels.api as sm

from trendfilterpy._dists import (
    BinomialDistribution,
    Distribution,
    GammaDistribution,
    NormalDistribution,
    PoissonDistribution,
)
from trendfilterpy._links import IdentityLink, LinkFunction, LogitLink, LogLink

rng = np.random.RandomState(42)

DISTRIBUTIONS = [
    (
        NormalDistribution,
        sm.families.Gaussian(),
        IdentityLink,
        lambda rng, eta, link: rng.normal(link.eval_inverse(eta), 1.0),
    ),
    (PoissonDistribution, sm.families.Poisson(), LogLink, lambda rng, eta, link: rng.poisson(link.eval_inverse(eta))),
    (
        BinomialDistribution,
        sm.families.Binomial(),
        LogitLink,
        lambda rng, eta, link: rng.binomial(1, link.eval_inverse(eta)),
    ),
    (
        GammaDistribution,
        sm.families.Gamma(sm.families.links.Log()),
        LogLink,
        lambda rng, eta, link: rng.gamma(link.eval_inverse(eta) ** 2, 1 / link.eval_inverse(eta)),
    ),
]


class TestDeviance:
    """Test deviance functions against Statsmodels implementation to verify solutions are identical."""

    @pytest.mark.parametrize("dist,statsmodels_family,link,random_sampler", DISTRIBUTIONS)
    def test_glm_coefficients_match_statsmodels(
        self,
        dist: type[Distribution],
        statsmodels_family: sm.families.Family,
        link: type[LinkFunction],
        # TODO: Clarify what this callable takes as arguments
        random_sampler: Callable,
    ):
        n_samples = 1000
        n_features = 10

        # Selecting small variance here so Gamma data is well-behaved enough for statsmodels fitting algorithm
        X = rng.normal(0, 0.1, size=(n_samples, n_features))
        beta_true = rng.normal(0, 0.1, size=n_features)
        eta = X @ beta_true

        y = random_sampler(rng, eta, link())

        # Handle special case for binomial because it needs n as input
        ns = np.ones(n_samples) if dist is BinomialDistribution else None

        glm = sm.GLM(y, X, family=statsmodels_family)
        results = glm.fit()
        statsmodels_coef: npt.NDArray = results.params

        beta = cp.Variable(n_features)
        eta = X @ beta
        weights = np.ones(n_samples)

        objective = cp.Minimize(dist().deviance(y, eta, weights, link(), n=ns))

        prob = cp.Problem(objective)
        prob.solve(solver="CLARABEL")

        if beta.value is None:
            raise ValueError(f"Distribution {dist.__name__} and link {link.__name__} result in a None type beta vector")
        else:
            cvxpy_coef = beta.value

        # Compare coefficients
        np.testing.assert_allclose(
            cvxpy_coef,
            statsmodels_coef,
            rtol=1e-4,
            atol=1e-4,
            err_msg=f"Fitted coefficients differ from statsmodels for {dist.__name__} and {link.__name__}",
        )
