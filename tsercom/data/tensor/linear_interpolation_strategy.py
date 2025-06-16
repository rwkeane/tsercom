"""
Provides the LinearInterpolationStrategy class for smoothing tensor data.
"""

import bisect
from datetime import datetime
from typing import List, Tuple  # Numeric was Union[float, int]

# Assuming Numeric and SmoothingStrategy will be imported correctly
from tsercom.data.tensor.smoothing_strategy import SmoothingStrategy, Numeric


class LinearInterpolationStrategy(SmoothingStrategy):
    """
    A smoothing strategy that implements linear interpolation.

    For each required timestamp, it finds the two keyframes that bracket it
    and performs linear interpolation.
    - If a required timestamp is before the first keyframe, the value of the
      first keyframe is used (forward fill/hold).
    - If a required timestamp is after the last keyframe, the value of the
      last keyframe is used (backward fill/hold).
    - If a required timestamp exactly matches a keyframe, that keyframe's
      value is used.
    - If no keyframes are provided, it will raise a ValueError, as interpolation
      is not possible.
    """

    def interpolate_series(
        self,
        keyframes: List[Tuple[datetime, Numeric]],
        required_timestamps: List[datetime],
    ) -> List[Numeric]:
        """
        Interpolates a series using linear interpolation.

        Args:
            keyframes: Sorted list of (timestamp, value) keyframes.
            required_timestamps: Sorted list of timestamps to interpolate for.

        Returns:
            List of interpolated values.

        Raises:
            ValueError: If `keyframes` is empty and `required_timestamps` is not.
        """
        if not required_timestamps:
            return []

        if not keyframes:
            raise ValueError(
                "Cannot interpolate with no keyframes provided "
                "when required_timestamps is not empty."
            )

        results: List[Numeric] = []
        keyframe_timestamps = [kf[0] for kf in keyframes]
        keyframe_values = [kf[1] for kf in keyframes]

        for req_ts in required_timestamps:
            if req_ts < keyframe_timestamps[0]:
                results.append(keyframe_values[0])
                continue

            if req_ts > keyframe_timestamps[-1]:
                results.append(keyframe_values[-1])
                continue

            idx_t2 = bisect.bisect_left(keyframe_timestamps, req_ts)

            if (
                idx_t2 < len(keyframe_timestamps)
                and keyframe_timestamps[idx_t2] == req_ts
            ):
                results.append(keyframe_values[idx_t2])
            else:
                if idx_t2 == 0:
                    results.append(
                        keyframe_values[0]
                    )  # Should be covered by exact match or before first
                    continue

                t1 = keyframe_timestamps[idx_t2 - 1]
                v1 = keyframe_values[idx_t2 - 1]
                t2 = keyframe_timestamps[idx_t2]
                v2 = keyframe_values[idx_t2]

                time_delta_total = (t2 - t1).total_seconds()

                if time_delta_total == 0:
                    results.append(v1)
                else:
                    time_ratio = (
                        req_ts - t1
                    ).total_seconds() / time_delta_total
                    interpolated_value = v1 + (v2 - v1) * time_ratio
                    results.append(interpolated_value)
        return results
