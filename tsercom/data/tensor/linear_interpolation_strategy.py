import torch
from tsercom.data.tensor.smoothing_strategy import SmoothingStrategy


class LinearInterpolationStrategy(SmoothingStrategy):
    """
    Implements linear interpolation for a series of data points using torch.Tensor.
    """

    def interpolate_series(
        self,
        timestamps: torch.Tensor,  # float64, Unix timestamps
        values: torch.Tensor,  # float, shape (N,) or (N, D)
        required_timestamps: torch.Tensor,  # float64, Unix timestamps
    ) -> torch.Tensor:
        """
        Performs tensor-based linear interpolation.

        Handles edge cases:
        - No keyframes: Returns tensor of NaNs for required_timestamps.
        - Single keyframe: Extrapolates that keyframe's value.
        - Required timestamps outside keyframe range: Extrapolates using boundary keyframes.
        - Exact matches with keyframe timestamps: Uses the value of the keyframe found by 'left'-side search.
        Args:
            timestamps: A 1D tensor of sorted keyframe timestamps (Unix epoch, float64).
                        Shape: (N,).
            values: A tensor of keyframe values. Can be 1D (N,) or 2D (N, D) for
                    multi-dimensional values. Assumed to be float.
            required_timestamps: A 1D tensor of sorted timestamps for which values
                                 are needed (Unix epoch, float64). Shape: (M,).

        Returns:
            A tensor of interpolated values. Shape will be (M,) if values are 1D,
            or (M, D) if values are 2D.
        """
        num_req_ts = required_timestamps.shape[0]
        if num_req_ts == 0:
            return (
                torch.empty(
                    (0, *values.shape[1:]),
                    dtype=values.dtype,
                    device=values.device,
                )
                if values.ndim > 1
                else torch.empty(0, dtype=values.dtype, device=values.device)
            )

        if timestamps.numel() == 0:
            value_shape = (
                (num_req_ts, values.shape[1])
                if values.ndim == 2
                else (num_req_ts,)
            )
            return torch.full(
                value_shape,
                float("nan"),
                dtype=values.dtype,
                device=values.device,
            )

        if timestamps.numel() == 1:
            return (
                values.expand(num_req_ts, *values.shape[1:])
                if values.ndim > 1
                else values.expand(num_req_ts)
            )

        # Promote types for calculation precision
        calc_timestamps = timestamps.to(dtype=torch.float64)
        calc_required_timestamps = required_timestamps.to(dtype=torch.float64)
        original_value_dtype = values.dtype
        calc_values = values.to(dtype=torch.float64)

        # Initialize result tensor
        if calc_values.ndim == 1:
            interpolated_values = torch.empty_like(
                calc_required_timestamps, dtype=torch.float64
            )
        else:  # calc_values.ndim == 2
            interpolated_values = torch.empty(
                (num_req_ts, calc_values.shape[1]),
                dtype=torch.float64,
                device=calc_values.device,
            )

        # Find insertion points using 'left' side, similar to bisect_left
        indices_left = torch.searchsorted(
            calc_timestamps, calc_required_timestamps, right=False
        )

        # --- Handle exact matches ---
        # Mask for required_timestamps that exactly match a keyframe timestamp
        exact_match_mask = torch.zeros_like(
            calc_required_timestamps, dtype=torch.bool
        )
        # Consider only indices within bounds [0, N-1] for exact matches
        valid_indices_for_exact_match = indices_left < calc_timestamps.shape[0]

        if torch.any(valid_indices_for_exact_match):
            # Get the keyframe timestamps and required timestamps only for valid indices
            kf_ts_at_indices_left = calc_timestamps[
                indices_left[valid_indices_for_exact_match]
            ]
            req_ts_for_exact_check = calc_required_timestamps[
                valid_indices_for_exact_match
            ]

            # Check for equality
            sub_exact_match_found = (
                kf_ts_at_indices_left == req_ts_for_exact_check
            )

            # Update the main exact_match_mask
            exact_match_mask[valid_indices_for_exact_match] = (
                sub_exact_match_found
            )

            if torch.any(sub_exact_match_found):
                # Assign values for exact matches
                # indices_left[exact_match_mask] might seem redundant but ensures correct broadcasting/indexing
                # if exact_match_mask was created differently. Here it's a direct application.
                interpolated_values[exact_match_mask] = calc_values[
                    indices_left[exact_match_mask]
                ]

        # --- Handle extrapolations (for non-exact matches) ---
        # Left extrapolation: required_timestamps < first keyframe timestamp
        left_extrapolation_mask = (indices_left == 0) & (~exact_match_mask)
        if torch.any(left_extrapolation_mask):
            interpolated_values[left_extrapolation_mask] = calc_values[0]

        # Right extrapolation: required_timestamps > last keyframe timestamp (not >=, as exact match to last is handled)
        # indices_left will be N for these.
        right_extrapolation_mask = (
            indices_left == calc_timestamps.shape[0]
        ) & (~exact_match_mask)
        if torch.any(right_extrapolation_mask):
            interpolated_values[right_extrapolation_mask] = calc_values[-1]

        # --- Handle interpolations (for non-exact matches within bounds) ---
        # Candidates are 0 < indices_left < N.
        # Actual interpolation happens if not an exact match.
        interpolation_candidate_mask = (indices_left > 0) & (
            indices_left < calc_timestamps.shape[0]
        )
        interpolation_mask = interpolation_candidate_mask & (~exact_match_mask)

        if torch.any(interpolation_mask):
            # Subset of required_timestamps that need interpolation
            interp_req_ts = calc_required_timestamps[interpolation_mask]
            # Corresponding indices in the original keyframe timestamps array
            interp_indices_in_kf = indices_left[interpolation_mask]

            # Keyframes for interpolation: (t1, v1) is at [interp_indices_in_kf - 1]
            #                             (t2, v2) is at [interp_indices_in_kf]
            t1 = calc_timestamps[interp_indices_in_kf - 1]
            v1 = calc_values[interp_indices_in_kf - 1]
            t2 = calc_timestamps[interp_indices_in_kf]
            v2 = calc_values[interp_indices_in_kf]

            denominator = t2 - t1

            # Initialize proportion tensor for the elements being interpolated
            proportion = torch.zeros_like(interp_req_ts, dtype=torch.float64)

            # Mask for valid (non-zero) denominators for safe division
            # Using a small epsilon for float comparison to avoid division by true zero or near-zero.
            valid_denominator_mask = torch.abs(denominator) > 1e-9

            # Calculate proportion only where denominator is valid
            if torch.any(
                valid_denominator_mask
            ):  # Corrected closing parenthesis
                proportion[valid_denominator_mask] = (
                    interp_req_ts[valid_denominator_mask]
                    - t1[valid_denominator_mask]
                ) / denominator[valid_denominator_mask]

            # For zero/tiny denominator (t1 approx t2), result should be v1.
            # This is achieved if proportion for these cases is 0.
            proportion[~valid_denominator_mask] = 0.0
            # (If v1 and v2 are different at the same t1=t2, this means we take v1)

            # Perform interpolation
            if calc_values.ndim == 1:  # v1, v2 are 1D
                interpolated_segment = v1 + proportion * (v2 - v1)
            else:  # v1, v2 are 2D, proportion needs broadcasting: (k,) to (k,1)
                interpolated_segment = v1 + proportion.unsqueeze(-1) * (
                    v2 - v1
                )

            # Assign calculated values to the correct slice of interpolated_values
            interpolated_values[interpolation_mask] = interpolated_segment

        return interpolated_values.to(original_value_dtype)
