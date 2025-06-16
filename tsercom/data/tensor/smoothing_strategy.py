import abc
from datetime import datetime
from typing import (
    List,
    Tuple,
    Union,
)  # Added Union based on LinearInterpolationStrategy return type


class SmoothingStrategy(abc.ABC):
    """
    Abstract base class for tensor data smoothing strategies.
    """

    @abc.abstractmethod
    def interpolate_series(
        self,
        keyframes: List[Tuple[datetime, float]],
        required_timestamps: List[datetime],
    ) -> List[
        Union[float, None]
    ]:  # Matched return type with concrete implementation
        """
        Interpolates values for a series of required timestamps based on keyframes.

        Args:
            keyframes: A time-sorted list of (timestamp, value) tuples.
            required_timestamps: A list of datetime objects for which values are needed.

        Returns:
            A list of interpolated values (float) or None.
        """
