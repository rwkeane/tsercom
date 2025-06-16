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
            # No data to interpolate from, return NaNs or raise error?
            # For now, let's assume we might want a default value or handle upstream.
            # For safety in calculations, returning NaN is often better than zero.
            return [float("nan")] * len(required_timestamps)

        if len(keyframes) == 1:
            # Only one keyframe, so all interpolated values are that keyframe's value.
            return [keyframes[0][1]] * len(required_timestamps)

        results: List[float] = []
        kf_timestamps = [kf[0] for kf in keyframes]
        kf_values = [kf[1] for kf in keyframes]

        for t_req_dt in required_timestamps:
            t_req = (
                t_req_dt.timestamp()
            )  # Convert datetime to float timestamp for easier comparison

            # Find surround_idx: index of the first keyframe >= t_req
            surround_idx = -1
            for i, kf_ts_dt in enumerate(kf_timestamps):
                if kf_ts_dt.timestamp() >= t_req:
                    surround_idx = i
                    break

            if surround_idx == 0:
                # t_req is before or at the very first keyframe
                # Extrapolate using the first two keyframes (or just use first if only one before)
                # or simply clamp to the first keyframe's value if t_req <= first keyframe
                if t_req <= kf_timestamps[0].timestamp():
                    results.append(kf_values[0])
                    continue
                # This case should ideally not be hit if surround_idx logic is correct for t_req > first keyframe
                # but as a fallback, this means t_req is between kf[0] and kf[1]
                # This will be handled by the main interpolation block if surround_idx is correctly found as 1
                # For safety, let's consider what happens if surround_idx is 0 but t_req > kf_timestamps[0]
                # This implies t_req is actually between kf_timestamps[0] and kf_timestamps[1] (if kf_timestamps[1] exists)
                # The logic below expects prev_idx and next_idx
                # Let's refine the prev/next logic

            # Determine previous and next keyframes for interpolation
            if surround_idx == -1:
                # t_req is after all known keyframes (extrapolation)
                # Use the last two keyframes
                t1_dt, v1 = keyframes[-2]
                t2_dt, v2 = keyframes[-1]
            elif surround_idx == 0:
                # t_req is before or at the first keyframe (extrapolation or direct hit)
                # If direct hit:
                if kf_timestamps[surround_idx].timestamp() == t_req:
                    results.append(kf_values[surround_idx])
                    continue
                # If before: Use first two to extrapolate "backwards", or just clamp to first value
                # For simplicity, clamping to the first value if t_req < first keyframe
                results.append(kf_values[0])
                continue
            else:
                # t_req is between two keyframes (interpolation) or a direct hit
                # If direct hit:
                if kf_timestamps[surround_idx].timestamp() == t_req:
                    results.append(kf_values[surround_idx])
                    continue
                # If t_req is between keyframes[surround_idx-1] and keyframes[surround_idx]
                t1_dt, v1 = keyframes[surround_idx - 1]
                t2_dt, v2 = keyframes[surround_idx]

            t1 = t1_dt.timestamp()
            t2 = t2_dt.timestamp()

            if (
                t1 == t2
            ):  # Should not happen if keyframes are distinct by time and sorted
                results.append(v1)  # Or v2, they are at the same time
                continue

            # Linear interpolation: v = v1 + (t - t1) * (v2 - v1) / (t2 - t1)
            interpolated_value = v1 + (t_req - t1) * (v2 - v1) / (t2 - t1)
            results.append(interpolated_value)

        return results
