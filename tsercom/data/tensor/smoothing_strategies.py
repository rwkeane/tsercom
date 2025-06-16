import abc
import bisect
from datetime import datetime
from typing import List, Tuple, Union


class SmoothingStrategy(abc.ABC):
    """
    Abstract base class for tensor data smoothing strategies.

    This class defines the interface for different interpolation algorithms
    that can be used by the SmoothedTensorDemuxer.
    """

    @abc.abstractmethod
    def interpolate_series(
        self,
        keyframes: List[Tuple[datetime, float]],
        required_timestamps: List[datetime],
    ) -> List[Union[float, None]]:
        """
        Interpolates values for a series of required timestamps based on keyframes.

        This method should be implemented by concrete strategies to provide
        specific interpolation logic for a single index's data.

        Args:
            keyframes: A time-sorted list of (timestamp, value) tuples representing
                       the known data points for a single tensor index.
            required_timestamps: A list of datetime objects for which interpolated
                                 values are needed. These timestamps should also be
                                 sorted chronologically.

        Returns:
            A list of interpolated values (float) or None if interpolation is not
            possible for a given required timestamp. The list corresponds to the
            order of `required_timestamps`.
        """


class LinearInterpolationStrategy(SmoothingStrategy):
    """
    Implements linear interpolation for a series of data points.
    """

    def interpolate_series(
        self,
        keyframes: List[Tuple[datetime, float]],
        required_timestamps: List[datetime],
    ) -> List[Union[float, None]]:
        """
        Performs linear interpolation for a given set of keyframes and required timestamps.

        For each required timestamp, this method finds the two keyframes that
        bracket it and performs linear interpolation.

        - If a required timestamp is before the first keyframe, the value of the
          first keyframe is returned.
        - If a required timestamp is after the last keyframe, the value of the
          last keyframe is returned.
        - If a required timestamp exactly matches a keyframe, the value of that
          keyframe is returned.
        - If there are no keyframes, all interpolated values will be None.
        - If there is only one keyframe, its value is returned for all required
          timestamps.

        Args:
            keyframes: A time-sorted list of (timestamp, value) tuples.
            required_timestamps: A list of datetime objects for which values are needed.
                                 It's assumed these are sorted.

        Returns:
            A list of interpolated float values or None, corresponding to each
            required timestamp.
        """
        if not keyframes:
            return [None] * len(required_timestamps)

        if len(keyframes) == 1:
            return [keyframes[0][1]] * len(required_timestamps)

        # Extract timestamps and values for easier processing
        key_times, key_values = zip(*keyframes)

        interpolated_results: List[Union[float, None]] = []

        for ts_req in required_timestamps:
            # Handle cases where ts_req is outside the range of key_times
            if ts_req <= key_times[0]:
                interpolated_results.append(key_values[0])
                continue
            if ts_req >= key_times[-1]:
                interpolated_results.append(key_values[-1])
                continue

            # Find the insertion point for ts_req in key_times
            # bisect_left will find the index idx such that all key_times[j] for j < idx are < ts_req,
            # and all key_times[j] for j >= idx are >= ts_req.
            idx = bisect.bisect_left(key_times, ts_req)

            # If ts_req matches a keyframe time exactly
            if key_times[idx] == ts_req:
                interpolated_results.append(key_values[idx])
                continue

            # At this point, key_times[idx-1] < ts_req < key_times[idx]
            # These are t1, v1 and t2, v2 for interpolation
            t1, v1 = key_times[idx - 1], key_values[idx - 1]
            t2, v2 = key_times[idx], key_values[idx]

            # Perform linear interpolation
            # Proportion of time elapsed between t1 and t2
            if (
                t2 - t1
            ).total_seconds() == 0:  # Should not happen if timestamps are unique and sorted
                interpolated_value = v1  # Or v2, if t1 and t2 are identical, take the earlier one's value
            else:
                proportion = (ts_req - t1).total_seconds() / (
                    t2 - t1
                ).total_seconds()
                interpolated_value = v1 + proportion * (v2 - v1)

            interpolated_results.append(interpolated_value)

        return interpolated_results
