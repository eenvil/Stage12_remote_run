from abc import ABC
from typing import Sequence, Union
import torch
from torch.distributions import LogisticNormal

class Timesteps(ABC):
    """
    Timesteps base class.
    """

    def __init__(self, T: Union[int, float]):
        assert T > 0
        self._T = T

    @property
    def T(self) -> Union[int, float]:
        """
        Maximum timestep inclusive.
        int if discrete, float if continuous.
        """
        return self._T

    def is_continuous(self) -> bool:
        """
        Whether the schedule is continuous.
        """
        return isinstance(self.T, float)

class LogitNormalTrainingTimesteps(Timesteps):
    """
    Logit-Normal sampling of timesteps in [0, T].
    """

    def __init__(self, T: Union[int, float], loc: float, scale: float):
        super().__init__(T)
        self.dist = LogisticNormal(loc, scale)

    def sample(
        self,
        size: Sequence[int],
        device: torch.device = "cpu",
    ) -> torch.Tensor:
        t = self.dist.sample(size)[..., 0].to(device).mul_(self.T)
        return t if self.is_continuous() else t.round().int()
