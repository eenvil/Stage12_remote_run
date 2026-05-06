from typing import Union
import torch


class Timesteps:
    """
    Sampling timesteps.
    It defines the discretization of sampling steps.
    """

    def __init__(
        self,
        T: int,
        steps: int,
        device: torch.device = "cpu",
    ):
        self.T = T
        timesteps = torch.arange(T, -1, -(T + 1) / steps, device=device).round().int()
        self.timesteps = timesteps

    def __len__(self) -> int:
        """
        Number of sampling steps.
        """
        return len(self.timesteps)

    def __getitem__(self, idx: Union[int, torch.IntTensor]) -> torch.Tensor:
        return self.timesteps[idx]

    def index(self, t: torch.Tensor) -> torch.Tensor:
        """
        Find index by t.
        Return index of the same shape as t.
        Index is -1 if t not found in timesteps.
        """
        i, j = t.reshape(-1, 1).eq(self.timesteps).nonzero(as_tuple=True)
        idx = torch.full_like(t, fill_value=-1, dtype=torch.int)
        idx.view(-1)[i] = j.int()
        return idx