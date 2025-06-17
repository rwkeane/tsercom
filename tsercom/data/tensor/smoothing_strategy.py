import abc
import torch  # Added torch

# TODO(jules): Consider if we need to support different dtypes for timestamps or values
# in the future, potentially through generics if Python versions allow, or overloads.
# For now, assuming timestamps are float64 (Unix timestamps) and values are float32/float64.


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
            timestamps: A 1D tensor of sorted keyframe timestamps (e.g., Unix timestamps).
                        Expected to be of dtype float64.
            values: A 1D tensor of keyframe values corresponding to the timestamps.
                    Shape should be (N,) or (N, D) for D-dimensional values.
            required_timestamps: A 1D tensor of sorted timestamps for which values are needed.
                                 Expected to be of dtype float64.

        Returns:
            A tensor of interpolated values corresponding to each required_timestamp.
            The shape will match the shape of `values` for the value dimensions,
            e.g., (M,) if values are (N,), or (M, D) if values are (N, D),
            where M is the number of required_timestamps.
        """
        pass  # Python style: use pass for abstract methods body
