"""Defines the abstract base class for tensor data smoothing strategies."""

from abc import ABC, abstractmethod

import torch


class SmoothingStrategy(ABC):
    """Abstract base class for defining strategies to smooth or interpolate tensor data."""

    @abstractmethod
    def interpolate_series(
        self,
        timestamps: torch.Tensor,  # float tensor, e.g., Unix timestamps
        values: torch.Tensor,  # float tensor
        required_timestamps: torch.Tensor,  # float tensor
    ) -> torch.Tensor:
        """Interpolate or smooth a series of timestamped tensor values.

        Args:
            timestamps: A 1D torch.Tensor of numerical timestamps, sorted ascending.
            values: A 1D torch.Tensor of corresponding values. Must be the same
                    length as `timestamps`.
            required_timestamps: A 1D torch.Tensor of numerical timestamps for which
                                 values are needed.

        Returns:
            A 1D torch.Tensor of interpolated/smoothed values, corresponding to
            each `required_timestamp`.

        """
        raise NotImplementedError
