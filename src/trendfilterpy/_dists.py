import cvxpy as cp
import numpy.typing as npt

Input = npt.NDArray | cp.Expression


# TODO: Implement test suite for convexity of deviance methods
class Distribution:
    def deviance(self, y: Input, mu: Input) -> cp.Expression:
        raise NotImplementedError


class NormalDistribution(Distribution):
    def deviance(self, y: Input, mu: Input) -> cp.Expression:
        return cp.sum_squares(y - mu)
