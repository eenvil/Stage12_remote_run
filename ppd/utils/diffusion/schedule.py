"""
Linear interpolation schedule (lerp).
"""

from typing import Tuple, Union
import torch
from enum import Enum


class LinearSchedule:
    """
    Linear interpolation schedule (lerp) is proposed by flow matching and rectified flow.
    It leads to straighter probability flow theoretically. It is also used by Stable Diffusion 3.
        
        x_t = (1 - t) * x_0 + t * x_T

    """

    def __init__(self, T: Union[int, float] = 1.0):
        self.T = T

    def forward(self, x_0: torch.Tensor, x_T: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Diffusion forward function.
        """
        t = t[(...,) + (None,) * (x_0.ndim - t.ndim)] if t.ndim < x_0.ndim else t
        return (1 - t / self.T) * x_0 + (t / self.T) * x_T

    def convert_from_pred(
        self, pred: torch.Tensor, pred_type: 'velocity', x_t: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert from velocity prediction. Return predicted x_0 and x_T.
        """
        t = t[(...,) + (None,) * (x_t.ndim - t.ndim)] if t.ndim < x_t.ndim else t
        A_t = 1 - t / self.T
        B_t = t / self.T
        
        # pred_type = 'velocity'
        pred_x_0 = x_t - B_t * pred
        pred_x_T = x_t + A_t * pred

        return pred_x_0, pred_x_T

    def convert_to_pred(
        self, x_0: torch.Tensor, x_T: torch.Tensor, t: torch.Tensor, pred_type: 'velocity'
    ) -> torch.FloatTensor:
        """
        Convert to velocity prediction target given x_0 and x_T.
        Predict velocity dx/dt based on the lerp schedule (x_T - x_0).
        Proposed by rectified flow (https://arxiv.org/abs/2209.03003)
        """
        # pred_type = 'velocity'
        return x_T - x_0
