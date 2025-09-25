from abc import ABC, abstractmethod
from typing import TypeVar

import cvxpy as cp
import numpy as np
import numpy.typing as npt

T = TypeVar("T", npt.NDArray, cp.Expression)


class LinkFunction(ABC):
    @abstractmethod
    def eval(self, x: T) -> T:
        pass

    @abstractmethod
    def eval_inverse(self, x: T) -> T:
        pass


class IdentityLink(LinkFunction):
    def eval(self, x: T) -> T:
        return x

    def eval_inverse(self, x: T) -> T:
        return x


class LogLink(LinkFunction):
    def eval(self, x: T) -> T:
        if isinstance(x, cp.Expression):
            return cp.log(x)
        else:
            return np.log(x)

    def eval_inverse(self, x: T) -> T:
        if isinstance(x, cp.Expression):
            return cp.exp(x)
        else:
            return np.exp(x)


class LogitLink(LinkFunction):
    def eval(self, x: T) -> T:
        if isinstance(x, cp.Expression):
            return cp.log(x / (1 - x))
        else:
            return np.log(x / (1 - x))

    def eval_inverse(self, x: T) -> T:
        if isinstance(x, cp.Expression):
            return cp.inv_pos(1 + cp.exp(-x))
        else:
            return 1 / (1 + np.exp(-x))


class PowerLink(LinkFunction):
    def __init__(self, p: float) -> None:
        # TODO: Verify this is not zero
        self.p = p

    def eval(self, x: T) -> T:
        if isinstance(x, cp.Expression):
            return cp.power(x, self.p)
        else:
            return np.power(x, self.p)

    def eval_inverse(self, x: T) -> T:
        if isinstance(x, cp.Expression):
            return cp.power(x, 1 / self.p)
        else:
            return np.power(x, 1 / self.p)
