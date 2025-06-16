"""
Provides the abstract base class for smoothing strategies and common types.
"""

import abc
from datetime import datetime
from typing import List, Tuple, Union

# Using Union for float/int, but torch usually uses float for tensor values
Numeric = Union[float, int]


class SmoothingStrategy(abc.ABC):
    """
    Abstract base class for defining smoothing strategies.

    A smoothing strategy is responsible for taking a series of timestamped
    keyframes for a single data point and generating interpolated values
    for a list of required timestamps.
    """

    @abc.abstractmethod
    def interpolate_series(
        self,
        keyframes: List[Tuple[datetime, Numeric]],
        required_timestamps: List[datetime],
    ) -> List[Numeric]:
        """
        Interpolates a series of keyframes to generate values at required timestamps.

        Args:
            keyframes: A list of (timestamp, value) tuples, sorted chronologically.
                       It's assumed that this list contains the historical data points
                       for a single index/metric.
            required_timestamps: A list of timestamps for which interpolated values
                                 are needed. These should also be sorted.

        Returns:
            A list of interpolated values, corresponding one-to-one with the
            `required_timestamps`. If interpolation cannot be performed for a
            given required timestamp (e.g., no keyframes available, or timestamp
            is outside a reasonable extrapolation range), the strategy should
            define how to handle this (e.g., return a default value, NaN, or
            the value of the nearest keyframe).
        """
        pass
