import torch
from enum import Enum
from ppd.utils.diffusion.timesteps import Timesteps
from ppd.utils.diffusion.schedule import LinearSchedule


class EulerSampler:
    """
    The Euler method is the simplest ODE solver.
    """

    def __init__(
        self,
        schedule: LinearSchedule,
        timesteps: Timesteps,
        prediction_type: 'velocity',
    ):
        self.schedule = schedule
        self.timesteps = timesteps
        self.prediction_type = prediction_type


    def step(
        self,
        pred: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Step to the next timestep.
        """
        return self.step_to(pred, x_t, t, self.get_next_timestep(t), **kwargs)

    def step_to(
        self,
        pred: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        s: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Steps from x_t at timestep t to x_s at timestep s. Returns x_s.
        """
        t = t[(...,) + (None,) * (x_t.ndim - t.ndim)] if t.ndim < x_t.ndim else t
        s = s[(...,) + (None,) * (x_t.ndim - s.ndim)] if s.ndim < x_t.ndim else s
        T = self.schedule.T
        # Step from x_t to x_s.
        pred_x_0, pred_x_T = self.schedule.convert_from_pred(pred, self.prediction_type, x_t, t)
        pred_x_s = self.schedule.forward(pred_x_0, pred_x_T, s.clamp(0, T))
        # Clamp x_s to x_0 and x_T if s is out of bound.
        pred_x_s = pred_x_s.where(s >= 0, pred_x_0)
        pred_x_s = pred_x_s.where(s <= T, pred_x_T)
        return pred_x_s

    def get_next_timestep(
        self,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get the next sample timestep.
        Support multiple different timesteps t in a batch.
        If no more steps, return out of bound value -1 or T+1.
        """
        T = self.timesteps.T
        steps = len(self.timesteps)
        curr_idx = self.timesteps.index(t)
        next_idx = curr_idx + 1

        s = self.timesteps[next_idx.clamp_max(steps - 1)]
        s = s.where(next_idx < steps, -1)
        return s