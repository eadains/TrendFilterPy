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
    InverseGaussianDistribution,
    NormalDistribution,
    PoissonDistribution,
    TweedieDistribution,
)
from trendfilterpy._links import IdentityLink, LinkFunction, LogitLink, LogLink, PowerLink

rng = np.random.RandomState(42)


def sim_tweedie(mu: np.ndarray, phi: float, rng: np.random.Generator, p: float = 1.5) -> np.ndarray:
    """Generate random Tweedie variables with 1<p<2 as compound Poisson-Gamma.

    Args:
        mu: Mean parameters
        phi: Dispersion parameter
        p: Power parameter, must be 1<p<2
    Returns:
        Array of Tweedie random variables
    """
    # Poisson means
    lambda_ = mu ** (2 - p) / ((2 - p) * phi)

    # Generate Poisson counts
    N = rng.poisson(lambda_)

    # Where N=0, result will be 0
    # Only generate gammas for N>0
    pos = N > 0
    result = np.zeros_like(mu)

    if np.any(pos):
        # Gamma parameters
        alpha = (2 - p) / (p - 1)  # shape
        beta = phi * (p - 1) * mu[pos] ** (p - 1)  # scale

        # Generate gamma sums where needed
        result[pos] = rng.gamma(
            shape=alpha * N[pos],  # N times shape for sum
            scale=beta,
        )

    return result


DISTRIBUTIONS = [
    (
        NormalDistribution(),
        sm.families.Gaussian(),
        IdentityLink(),
        lambda rng, eta, link: rng.normal(link.eval_inverse(eta), 1.0),
    ),
    (
        PoissonDistribution(),
        sm.families.Poisson(),
        LogLink(),
        lambda rng, eta, link: rng.poisson(link.eval_inverse(eta)),
    ),
    (
        BinomialDistribution(),
        sm.families.Binomial(),
        LogitLink(),
        lambda rng, eta, link: rng.binomial(1, link.eval_inverse(eta)),
    ),
    (
        GammaDistribution(),
        sm.families.Gamma(sm.families.links.Log()),
        LogLink(),
        lambda rng, eta, link: rng.gamma(link.eval_inverse(eta) ** 2, 1 / link.eval_inverse(eta)),
    ),
    (
        GammaDistribution(),
        sm.families.Gamma(sm.families.links.InversePower()),
        PowerLink(p=-1),
        lambda rng, eta, link: rng.gamma(link.eval_inverse(eta) ** 2, 1 / link.eval_inverse(eta)),
    ),
    (
        InverseGaussianDistribution(),
        sm.families.InverseGaussian(sm.families.links.InverseSquared()),
        PowerLink(p=-2),
        lambda rng, eta, link: rng.wald(link.eval_inverse(eta), 1),
    ),
    (
        TweedieDistribution(p=1.5),
        sm.families.Tweedie(sm.families.links.Power(power=1 - 1.5)),
        PowerLink(p=1 - 1.5),
        lambda rng, eta, link: sim_tweedie(link.eval_inverse(eta), 1, rng, p=1.5),
    ),
    (
        TweedieDistribution(p=1.5),
        sm.families.Tweedie(sm.families.links.Log()),
        LogLink(),
        lambda rng, eta, link: sim_tweedie(link.eval_inverse(eta), 1, rng, p=1.5),
    ),
]


class TestDeviance:
    """Test deviance functions against Statsmodels implementation to verify solutions are identical."""

    @pytest.mark.parametrize("dist,statsmodels_family,link,random_sampler", DISTRIBUTIONS)
    def test_glm_coefficients_match_statsmodels(
        self,
        dist: Distribution,
        statsmodels_family: sm.families.Family,
        link: LinkFunction,
        # TODO: Clarify what this callable takes as arguments
        random_sampler: Callable,
    ):
        n_samples = 1000
        n_features = 10
        # Generating strictly positive data so mu is positive for power link functions and also using uniform
        # to control numerical stability for statsmodels IRLS fitting algorithm
        X = rng.uniform(0.1, 0.5, size=(n_samples, n_features))
        beta_true = rng.uniform(0.1, 0.5, size=n_features)
        eta = X @ beta_true
        y = random_sampler(rng, eta, link)

        # Handle special case for binomial because it needs n as input
        ns = np.ones(n_samples) if type(dist) is BinomialDistribution else None

        # Statsmodels fit
        glm = sm.GLM(y, X, family=statsmodels_family)
        results = glm.fit()
        statsmodels_coef: npt.NDArray = results.params

        # CVXPY fit
        beta = cp.Variable(n_features)
        eta = X @ beta
        weights = np.ones(n_samples)

        objective = cp.Minimize(dist.deviance(y, eta, weights, link, n=ns))
        prob = cp.Problem(objective)
        prob.solve(solver="CLARABEL")

        if beta.value is None:
            raise ValueError(
                f"Distribution {type(dist).__name__} and link {type(link).__name__} result in a None type beta vector"
            )
        else:
            cvxpy_coef = beta.value

        np.testing.assert_allclose(
            cvxpy_coef,
            statsmodels_coef,
            rtol=1e-4,
            atol=1e-4,
            err_msg=f"Fitted coefficients differ from statsmodels for {type(dist).__name__} and {type(link).__name__}",
        )
