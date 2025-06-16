import bisect
from datetime import datetime
from typing import List, Tuple, Union

from tsercom.data.tensor.smoothing_strategy import (
    SmoothingStrategy,
)  # Import ABC from its new file name


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

        - If a required timestamp is before the first keyframe, the value of the
          first keyframe is returned (extrapolation).
        - If a required timestamp is after the last keyframe, the value of the
          last keyframe is returned (extrapolation).
        - If a required timestamp exactly matches a keyframe, the value of that
          keyframe is returned.
        - If there are no keyframes, all interpolated values will be None.
        - If there is only one keyframe, its value is returned for all required
          timestamps.

        Args:
            keyframes: A time-sorted list of (timestamp, value) tuples.
            required_timestamps: A list of datetime objects for which values are needed.
                                 It's assumed these are sorted for efficiency, though the
                                 logic here processes them one by one independently.

        Returns:
            A list of interpolated float values or None, corresponding to each
            required timestamp.
        """
        if not keyframes:
            return [None] * len(required_timestamps)

        if len(keyframes) == 1:
            return [keyframes[0][1]] * len(required_timestamps)

        key_times, key_values = zip(*keyframes)
        interpolated_results: List[Union[float, None]] = []

        for ts_req in required_timestamps:
            if ts_req <= key_times[0]:
                interpolated_results.append(key_values[0])
                continue
            if ts_req >= key_times[-1]:
                interpolated_results.append(key_values[-1])
                continue

            idx = bisect.bisect_left(key_times, ts_req)

            if key_times[idx] == ts_req:
                interpolated_results.append(key_values[idx])
                continue

            t1, v1 = key_times[idx - 1], key_values[idx - 1]
            t2, v2 = key_times[idx], key_values[idx]

            if (t2 - t1).total_seconds() == 0:
                interpolated_value = v1
            else:
                proportion = (ts_req - t1).total_seconds() / (
                    t2 - t1
                ).total_seconds()
                interpolated_value = v1 + proportion * (v2 - v1)

            interpolated_results.append(interpolated_value)

        return interpolated_results
