from abc import ABC, abstractmethod
from typing import Optional, TypeVar, Union

import cvxpy as cp
import numpy.typing as npt

from trendfilterpy._links import IdentityLink, LinkFunction, LogitLink, LogLink


# TODO: Implement test suite for convexity of deviance methods
class Distribution(ABC):
    @classmethod
    @abstractmethod
    # TODO: Implement canonical link in similar fashion to this
    def default_link(cls) -> type[LinkFunction]:
        """Return the default link function for this distribution."""
        pass

    @abstractmethod
    def deviance(
        self,
        y: npt.NDArray,
        eta: Union[npt.NDArray, cp.Expression],
        w: npt.NDArray,
        link: LinkFunction,
        n: Optional[npt.NDArray] = None,
    ) -> cp.Expression:
        """
        Calculate the deviance for the distribution with a specific link function.

        Args:
            y: Observed values
            eta: Linear predictor (link(mu))
            mu: Expected values
            w: Weights
            link: Link function used
            n: Number of trials (only used for certain distributions)
        """
        pass


class NormalDistribution(Distribution):
    @classmethod
    def default_link(cls) -> type[LinkFunction]:
        return IdentityLink

    def deviance(
        self,
        y: npt.NDArray,
        eta: Union[npt.NDArray, cp.Expression],
        w: npt.NDArray,
        link: LinkFunction,
        n: Optional[npt.NDArray] = None,
    ) -> cp.Expression:
        # TODO: check weights
        if isinstance(link, IdentityLink):
            # Identity link means no transform needed here
            mu = eta
            deviance = cp.sum(cp.multiply(w, (y - mu) ** 2))
        else:
            raise ValueError(f"Invalid link function used. {type(link)} was supplied but IdentityLink is expected.")

        if isinstance(deviance, cp.Expression):
            return deviance
        else:
            raise ValueError("Deviance returned int instead of cvxpy expression.")


class PoissonDistribution(Distribution):
    @classmethod
    def default_link(cls) -> type[LinkFunction]:
        return LogLink

    def deviance(
        self,
        y: npt.NDArray,
        eta: Union[npt.NDArray, cp.Expression],
        w: npt.NDArray,
        link: LinkFunction,
        n: Optional[npt.NDArray] = None,
    ) -> cp.Expression:
        # TODO: Check for that y are all non-negative integers
        # TODO: check weights
        if isinstance(link, LogLink):
            # Derived from the poisson deviance plugging in exp(eta) directly and simplifying so it is recognized as
            # convex
            # TODO: Elaborate this derivation in docstring
            deviance = -cp.multiply(y, eta) + cp.exp(eta)
            deviance = 2 * cp.sum(cp.multiply(w, deviance))
        else:
            raise ValueError(f"Invalid link function used. {type(link)} was supplied but LogLink is expected.")

        if isinstance(deviance, cp.Expression):
            return deviance
        else:
            raise ValueError("Deviance returned int instead of cvxpy expression.")


class BinomialDistribution(Distribution):
    @classmethod
    def default_link(cls) -> type[LinkFunction]:
        return LogitLink

    def deviance(
        self,
        y: npt.NDArray,
        eta: Union[npt.NDArray, cp.Expression],
        w: npt.NDArray,
        link: LinkFunction,
        n: Optional[npt.NDArray] = None,
    ) -> cp.Expression:
        # TODO: Check that y is non-negative integers and are all less than n
        # TODO: Check than n are positive integers
        # TODO: check weights
        # TODO: Is there a way to be make n non-optional for this but not for others?
        if n is None:
            raise ValueError("For binomial distribution n must be specified")

        if isinstance(link, LogitLink):
            # TODO: Check this derivation
            deviance = cp.multiply(n, cp.logistic(eta)) - cp.multiply(y, eta)
            deviance = 2 * cp.sum(cp.multiply(w, deviance))
        else:
            raise ValueError(f"Invalid link function used. {type(link)} was supplied but LogitLink is expected.")

        if isinstance(deviance, cp.Expression):
            return deviance
        else:
            raise ValueError("Deviance returned int instead of cvxpy expression.")


class GammaDistribution(Distribution):
    @classmethod
    def default_link(cls) -> type[LinkFunction]:
        # TODO: Make this the inverse link
        return LogLink

    def deviance(
        self,
        y: npt.NDArray,
        eta: Union[npt.NDArray, cp.Expression],
        w: npt.NDArray,
        link: LinkFunction,
        n: Optional[npt.NDArray] = None,
    ) -> cp.Expression:
        # TODO: Check for positivity of y
        # TODO: check weights
        if isinstance(link, LogLink):
            deviance = eta + cp.multiply(y, cp.exp(-eta))
            val = 2 * cp.sum(cp.multiply(w, deviance))
        else:
            raise ValueError(f"Invalid link function used. {type(link)} was supplied but LogLink is expected.")

        if isinstance(val, cp.Expression):
            return val
        else:
            raise ValueError("Deviance returned int instead of cvxpy expression.")


class InverseGaussianDistribution(Distribution):
    @classmethod
    def default_link(cls) -> type[LinkFunction]:
        # TODO: Make this the inverse link
        return LogLink

    def deviance(
        self,
        y: npt.NDArray,
        eta: Union[npt.NDArray, cp.Expression],
        w: npt.NDArray,
        link: LinkFunction,
        n: Optional[npt.NDArray] = None,
    ) -> cp.Expression:
        # TODO: Check for positivity of y
        # TODO: check weights
        mu = link.eval_inverse(eta)
        deviance = (y - mu) ** 2 / (mu**2 / y)
        val = cp.sum(cp.multiply(w, deviance))

        if isinstance(val, cp.Expression):
            return val
        else:
            raise ValueError("Deviance returned int instead of cvxpy expression.")
