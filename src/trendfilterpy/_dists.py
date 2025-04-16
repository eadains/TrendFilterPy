from abc import ABC, abstractmethod
from typing import Optional, Union

import cvxpy as cp
import numpy.typing as npt

from trendfilterpy._links import IdentityLink, LinkFunction, LogitLink, LogLink, PowerLink


class Distribution(ABC):
    @abstractmethod
    def canonical_link(self) -> LinkFunction:
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
    def canonical_link(self) -> LinkFunction:
        return IdentityLink()

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
            # Identity link means no transform needed for eta
            deviance = (y - eta) ** 2
        else:
            raise ValueError(f"Invalid link function used. {type(link)} was supplied but IdentityLink is expected.")

        deviance = cp.sum(cp.multiply(w, deviance))

        if isinstance(deviance, cp.Expression):
            return deviance
        else:
            raise ValueError("Deviance returned int instead of cvxpy expression.")


class PoissonDistribution(Distribution):
    def canonical_link(self) -> LinkFunction:
        return LogLink()

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
        else:
            raise ValueError(f"Invalid link function used. {type(link)} was supplied but LogLink is expected.")

        deviance = 2 * cp.sum(cp.multiply(w, deviance))

        if isinstance(deviance, cp.Expression):
            return deviance
        else:
            raise ValueError("Deviance returned int instead of cvxpy expression.")


class BinomialDistribution(Distribution):
    def canonical_link(self) -> LinkFunction:
        return LogitLink()

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
        else:
            raise ValueError(f"Invalid link function used. {type(link)} was supplied but LogitLink is expected.")

        deviance = 2 * cp.sum(cp.multiply(w, deviance))

        if isinstance(deviance, cp.Expression):
            return deviance
        else:
            raise ValueError("Deviance returned int instead of cvxpy expression.")


class GammaDistribution(Distribution):
    def canonical_link(self) -> LinkFunction:
        return PowerLink(p=-1)

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
        elif isinstance(link, PowerLink) and link.p == -1:
            deviance = -cp.log(eta) + cp.multiply(y, eta)
        else:
            raise ValueError(
                f"Invalid link function used. {type(link)} was supplied but LogLink or PowerLink with p=-1 is expected."
            )

        deviance = 2 * cp.sum(cp.multiply(w, deviance))

        if isinstance(deviance, cp.Expression):
            return deviance
        else:
            raise ValueError("Deviance returned int instead of cvxpy expression.")


class InverseGaussianDistribution(Distribution):
    def canonical_link(self) -> LinkFunction:
        return PowerLink(p=-2)

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
        if isinstance(link, PowerLink) and link.p == -2:
            deviance = cp.multiply(eta, y) - 2 * cp.power(eta, 0.5)
        else:
            raise ValueError(
                f"Invalid link function used. {type(link)} was supplied but PowerLink with p=-2 is expected."
            )

        deviance = cp.sum(cp.multiply(w, deviance))

        if isinstance(deviance, cp.Expression):
            return deviance
        else:
            raise ValueError("Deviance returned int instead of cvxpy expression.")


class TweedieDistribution(Distribution):
    def __init__(self, p: float) -> None:
        # TODO: Check p has valid value
        self.p = p

    def canonical_link(self) -> LinkFunction:
        return PowerLink(p=1 - self.p)

    def deviance(
        self,
        y: npt.NDArray,
        eta: Union[npt.NDArray, cp.Expression],
        w: npt.NDArray,
        link: LinkFunction,
        n: Optional[npt.NDArray] = None,
    ) -> cp.Expression:
        # TODO: Check for y >= 0
        # TODO: check weights
        if isinstance(link, PowerLink) and link.p == 1 - self.p:
            deviance = -cp.multiply(y, eta) / (1 - self.p) + cp.power(eta, (2 - self.p) / (1 - self.p)) / (2 - self.p)
        elif isinstance(link, LogLink):
            deviance = -cp.multiply(y, cp.exp((1 - self.p) * eta)) / (1 - self.p) + cp.exp((2 - self.p) * eta) / (
                2 - self.p
            )
        else:
            raise ValueError(
                f"Invalid link function used. {type(link)} was supplied but PowerLink(p=1-self.p) is expected."
            )

        deviance = 2 * cp.sum(cp.multiply(w, deviance))

        if isinstance(deviance, cp.Expression):
            return deviance
        else:
            raise ValueError("Deviance returned int instead of cvxpy expression.")
