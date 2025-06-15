"""Defines strategies for interpolating between tensor keyframes."""

import abc
import torch


class SmoothingStrategy(abc.ABC):
    """Abstract base class for a tensor smoothing/interpolation strategy."""

    @abc.abstractmethod
    def interpolate(
        self,
        start_tensor: torch.Tensor,
        end_tensor: torch.Tensor,
        ratio: float,
    ) -> torch.Tensor:
        """Calculates an interpolated tensor between two keyframe tensors."""
        # This is an abstract method, so it should not have a pass statement.
        # If a default implementation is desired, it should be provided,
        # otherwise, it's up to subclasses to implement.
        # For Pylint's unnecessary-pass, we remove it if it's truly abstract.


class LinearInterpolationStrategy(SmoothingStrategy):
    """Performs linear interpolation between two tensors."""

    def interpolate(
        self,
        start_tensor: torch.Tensor,
        end_tensor: torch.Tensor,
        ratio: float,
    ) -> torch.Tensor:
        """Performs linear interpolation between start_tensor and end_tensor."""
        return start_tensor + (end_tensor - start_tensor) * ratio
