from typing import TypeVar, Union

import cvxpy as cp
import numpy.typing as npt

T = TypeVar("T", bound=Union[npt.NDArray, cp.Expression])


# TODO: Implement test suite for convexity of deviance methods
class Distribution:
    def deviance(self, y: T, mu: T, w: T) -> cp.Expression:
        raise NotImplementedError


class NormalDistribution(Distribution):
    def deviance(self, y: T, mu: T, w: T) -> cp.Expression:
        val = cp.sum(cp.multiply(w, (y - mu) ** 2))
        if isinstance(val, cp.Expression):
            return val
        else:
            raise ValueError("Deviance returned int instead of cvxpy expression.")
