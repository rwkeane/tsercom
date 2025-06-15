"""Module defining strategies for tensor smoothing and interpolation."""

import abc
import torch


class SmoothingStrategy(abc.ABC):
    """
    Abstract base class for tensor smoothing strategies.
    """

    @abc.abstractmethod
    def interpolate(
        self,
        start_tensor: torch.Tensor,
        end_tensor: torch.Tensor,
        ratio: float,
    ) -> torch.Tensor:
        """Calculates an interpolated tensor between two keyframe tensors.

        Args:
            start_tensor: The starting tensor (keyframe).
            end_tensor: The ending tensor (keyframe).
            ratio: The interpolation ratio (0.0 for start_tensor, 1.0 for end_tensor).

        Returns:
            The interpolated tensor.
        """
        # This method must be implemented by subclasses.


class LinearInterpolationStrategy(SmoothingStrategy):
    """
    A concrete strategy that implements linear interpolation.
    """

    def interpolate(
        self,
        start_tensor: torch.Tensor,
        end_tensor: torch.Tensor,
        ratio: float,
    ) -> torch.Tensor:
        """Calculates an interpolated tensor using linear interpolation.

        Args:
            start_tensor: The starting tensor (keyframe).
            end_tensor: The ending tensor (keyframe).
            ratio: The interpolation ratio (0.0 for start_tensor, 1.0 for end_tensor).

        Returns:
            The interpolated tensor: start_tensor + (end_tensor - start_tensor) * ratio.
        """
        return start_tensor + (end_tensor - start_tensor) * ratio
