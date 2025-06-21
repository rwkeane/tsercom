import abc

import torch


class SmoothingStrategy(abc.ABC):
    """Abstract base class for tensor data smoothing strategies.
    """

    @abc.abstractmethod
    def interpolate_series(
        self,
        timestamps: torch.Tensor,
        values: torch.Tensor,
        required_timestamps: torch.Tensor,
    ) -> torch.Tensor:
        """Interpolates values for a series of required timestamps based on keyframes.

        Timestamps are expected to be numerical (e.g., Unix timestamps as float).

        Args:
            timestamps: A 1D torch.Tensor of numerical timestamps, sorted ascending.
            values: A 1D torch.Tensor of corresponding values. Must be the same
                    length as `timestamps`.
            required_timestamps: A 1D torch.Tensor of numerical timestamps for which
                                 values are needed.

        Returns:
            A 1D torch.Tensor of interpolated values, corresponding to each
            `required_timestamp`.

        """
