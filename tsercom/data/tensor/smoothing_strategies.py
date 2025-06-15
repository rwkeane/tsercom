"""Defines strategies for interpolating tensor data."""
import abc
import torch


class SmoothingStrategy(abc.ABC):
    """Abstract base class for defining tensor interpolation strategies."""

    @abc.abstractmethod
    def interpolate(
        self,
        start_tensor: torch.Tensor,
        end_tensor: torch.Tensor,
        ratio: float,
    ) -> torch.Tensor:
        """Calculates an interpolated tensor between two keyframe tensors."""
        # Pass is removed as per pylint W0107


class LinearInterpolationStrategy(SmoothingStrategy):
    """Implements a linear interpolation strategy for tensors."""

    def interpolate(
        self,
        start_tensor: torch.Tensor,
        end_tensor: torch.Tensor,
        ratio: float,
    ) -> torch.Tensor:
        """Calculates a linearly interpolated tensor."""
        return start_tensor + (end_tensor - start_tensor) * ratio
