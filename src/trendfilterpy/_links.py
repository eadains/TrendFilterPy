import cvxpy as cp
import numpy.typing as npt

Input = npt.NDArray | cp.Expression


class LinkFunction:
    def eval[T: Input](self, x: T) -> T:
        raise NotImplementedError

    def eval_inverse[T: Input](self, x: T) -> T:
        raise NotImplementedError


class IdentityLink(LinkFunction):
    def eval[T: Input](self, x: T) -> T:
        return x

    def eval_inverse[T: Input](self, x: T) -> T:
        return x
