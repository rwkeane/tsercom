"""Linear interpolation strategy for tensor data."""

from datetime import datetime
from typing import List, Tuple

from tsercom.data.tensor.smoothing_strategy import SmoothingStrategy

class LinearInterpolationStrategy(SmoothingStrategy):
    """
    A smoothing strategy that uses linear interpolation.

    For a given index, this strategy finds the two closest keyframes
    (one before, one after) for each required timestamp and interpolates
    linearly between them. If a required timestamp is outside the range
    of known keyframes, it will extrapolate using the two nearest points
    (or return the single keyframe's value if only one exists).
    """

    def interpolate_series(
        self,
        keyframes: List[Tuple[datetime, float]],
        required_timestamps: List[datetime],
    ) -> List[float]:
        """
        Calculates linearly interpolated values for a series of required timestamps.

        Args:
            keyframes: A list of (timestamp, value) tuples, sorted by timestamp,
                       for a single index.
            required_timestamps: A list of sorted timestamps for which values are needed.

        Returns:
            A list of float values for each of the required_timestamps.
        """
        if not keyframes:
            return [float("nan")] * len(required_timestamps)

        if len(keyframes) == 1:
            return [keyframes[0][1]] * len(required_timestamps)

        results: List[float] = []
        kf_timestamps = [kf[0] for kf in keyframes]
        kf_values = [kf[1] for kf in keyframes]

        for t_req_dt in required_timestamps:
            t_req = t_req_dt.timestamp()

            surround_idx = -1
            for i, kf_ts_dt in enumerate(kf_timestamps):
                if kf_ts_dt.timestamp() >= t_req:
                    surround_idx = i
                    break

            t1_dt: datetime
            v1: float
            t2_dt: datetime
            v2: float

            if kf_timestamps[0].timestamp() >= t_req :
                if len(keyframes) > 1 :
                    t1_dt, v1 = keyframes[0]
                    t2_dt, v2 = keyframes[1]
                else:
                    results.append(kf_values[0])
                    continue
            elif surround_idx == -1:
                t1_dt, v1 = keyframes[-2]
                t2_dt, v2 = keyframes[-1]
            elif kf_timestamps[surround_idx].timestamp() == t_req:
                results.append(kf_values[surround_idx])
                continue
            else:
                t1_dt, v1 = keyframes[surround_idx - 1]
                t2_dt, v2 = keyframes[surround_idx]

            t1 = t1_dt.timestamp()
            t2 = t2_dt.timestamp()

            if t1 == t2:
                results.append(v1)
                continue

            interpolated_value = v1 + (t_req - t1) * (v2 - v1) / (t2 - t1)
            results.append(interpolated_value)

        return results
