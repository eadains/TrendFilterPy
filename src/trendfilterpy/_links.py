from typing import TypeVar, Union

import cvxpy as cp
import numpy.typing as npt

T = TypeVar("T", bound=Union[npt.NDArray, cp.Expression])


class LinkFunction:
    def eval(self, x: T) -> T:
        raise NotImplementedError

    def eval_inverse(self, x: T) -> T:
        raise NotImplementedError


class IdentityLink(LinkFunction):
    def eval(self, x: T) -> T:
        return x

    def eval_inverse(self, x: T) -> T:
        return x
