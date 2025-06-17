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
            timestamps: A 1D tensor of sorted keyframe timestamps (e.g., float64 POSIX).
            values: A 1D tensor of keyframe values corresponding to 'timestamps'.
                    Must be the same length as 'timestamps'.
            required_timestamps: A 1D tensor of timestamps for which values are needed.
                                 These should also be sorted for optimal performance,
                                 though not strictly required by all implementations.

        Returns:
            A 1D tensor of interpolated values, corresponding to each required_timestamp.
            Values for timestamps outside the keyframe range may be extrapolated
            based on the specific strategy. If no keyframes are provided,
            the behavior is strategy-dependent (e.g., return NaNs or fill_values).
        """
