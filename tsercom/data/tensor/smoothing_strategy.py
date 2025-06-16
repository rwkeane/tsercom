import abc
import torch


class SmoothingStrategy(abc.ABC):
    """
    Abstract base class for tensor data smoothing strategies.
    """

    @abc.abstractmethod
    def interpolate_series(
        self,
        timestamps: torch.Tensor,
        values: torch.Tensor,
        required_timestamps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Interpolates values for a series of required timestamps based on keyframes.

        Args:
            timestamps: torch.Tensor: A 1D tensor of Unix timestamps (float64), sorted.
            values: torch.Tensor: A 1D tensor of corresponding values (float32).
            required_timestamps: torch.Tensor: A 1D tensor of Unix timestamps (float64)
                               for which values are needed.

        Returns:
            torch.Tensor: A 1D tensor of interpolated values (float32), same size as
                          `required_timestamps`. Contains `float('nan')` where
                          interpolation is not possible.
        """
