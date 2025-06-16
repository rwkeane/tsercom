"""Strategies for smoothing tensor data over time."""

import abc
from typing import List, Tuple
from datetime import datetime


class SmoothingStrategy(abc.ABC):
    """
    Abstract base class for defining tensor data smoothing strategies.

    The core idea is that each index in a tensor has its own independent
    timeline of keyframes. This strategy is applied per-index.
    """

    @abc.abstractmethod
    def interpolate_series(
        self,
        keyframes: List[Tuple[datetime, float]],
        required_timestamps: List[datetime],
    ) -> List[float]:
        """
        Calculates interpolated values for a series of required timestamps.

        This method operates on the keyframe history of a single index.

        Args:
            keyframes: A list of (timestamp, value) tuples, sorted by timestamp.
                       These are the known real data points for a single index.
            required_timestamps: A list of timestamps for which interpolated
                                 values are needed. These must also be sorted.

        Returns:
            A list of float values corresponding to the interpolated values
            at each of the required_timestamps.
        """
        # This method must be implemented by subclasses.
