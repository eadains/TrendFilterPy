from abc import ABC, abstractmethod
from typing import TypeVar, Union

import cvxpy as cp
import numpy.typing as npt

T = TypeVar("T", bound=Union[npt.NDArray, cp.Expression])


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
        return cp.log(x)

    def eval_inverse(self, x: T) -> T:
        return cp.exp(x)


class LogitLink(LinkFunction):
    def eval(self, x: T) -> T:
        return cp.log(x / (1 - x))

    def eval_inverse(self, x: T) -> T:
        return cp.inv_pos(1 + cp.exp(-x))
