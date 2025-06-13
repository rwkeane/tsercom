# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-arguments

"""
Provides the SmoothedTensorDemuxer class, extending TensorDemuxer to output
a smoothed, linearly interpolated stream of tensor data. It stores real keyframes
received via on_update_received and generates interpolated tensor values
periodically in a worker task.
"""

import asyncio
import bisect
import datetime
import logging  # Added logging
from typing import List, Tuple, Optional

import torch

from tsercom.data.tensor.tensor_demuxer import TensorDemuxer  # Absolute import


class SmoothedTensorDemuxer(TensorDemuxer):
    """
    Extends TensorDemuxer to provide a linearly interpolated stream of torch.Tensor data.
    """

    def __init__(
        self,
        client: TensorDemuxer.Client,  # Client for the smoothed output
        tensor_length: int,
        smoothing_period_seconds: float = 1.0,
        data_timeout_seconds: Optional[float] = None,  # For super().__init__
    ):
        super().__init__(
            client=client,  # Pass the same client to base.
            tensor_length=tensor_length,
            **(
                {}
                if data_timeout_seconds is None
                else {"data_timeout_seconds": data_timeout_seconds}
            ),
        )

        if smoothing_period_seconds <= 0:
            raise ValueError("Smoothing period must be positive.")

        self._sm_client: TensorDemuxer.Client = (
            client  # Client for smoothed data
        )
        # self._tensor_length is available from base via property
        self._smoothing_period_seconds: float = smoothing_period_seconds

        # _keyframes stores (timestamp, full_real_tensor_value)
        self._keyframes: List[Tuple[datetime.datetime, torch.Tensor]] = []
        # _keyframe_explicit_updates[i] stores the explicit updates for _keyframes[i]
        self._keyframe_explicit_updates: List[List[Tuple[int, float]]] = []

        self._keyframe_lock = (
            asyncio.Lock()
        )  # Protects _keyframes and _keyframe_explicit_updates

        # Tracks the timestamp of the last synthetic tensor emitted for an index
        # For SmoothedTensorDemuxer, it's simpler as it doesn't manage multiple client indices.
        # This will track the last emission for its single smoothed output stream.
        self._last_synthetic_emitted_at: Optional[datetime.datetime] = None

        self._interpolation_loop_task: Optional[asyncio.Task[None]] = (
            None  # Typed Task
        )
        self._stop_event = (
            asyncio.Event()
        )  # Used to signal the interpolation loop to stop

    async def start(self) -> None:  # Added return type
        """Starts the interpolation loop task if not already running."""
        if (
            self._interpolation_loop_task is None
            or self._interpolation_loop_task.done()
        ):
            self._stop_event.clear()
            self._interpolation_loop_task = asyncio.create_task(
                self._interpolation_worker()
            )
            # Ensure the task is "awaited" or managed if start() is called multiple times
            # or if the object is destroyed without calling close().
            # asyncio.create_task returns a Task object that can be cancelled.

    async def close(self) -> None:  # Added return type
        """
        Stops the interpolation loop and cleans up resources.
        This method should be called to ensure graceful shutdown.
        """
        self._stop_event.set()
        if (
            self._interpolation_loop_task
            and not self._interpolation_loop_task.done()
        ):
            try:
                # Give the task a moment to stop gracefully
                await asyncio.wait_for(
                    self._interpolation_loop_task,
                    timeout=self._smoothing_period_seconds + 0.5,
                )
            except asyncio.TimeoutError:
                self._interpolation_loop_task.cancel()  # Force cancel if it doesn't stop
                try:
                    await self._interpolation_loop_task  # Await cancellation
                except asyncio.CancelledError:
                    pass  # Expected
            except asyncio.CancelledError:
                pass  # Expected if already cancelled
            except Exception:  # pylint: disable=broad-except
                # Log other exceptions during task shutdown
                # print(f"Error during interpolation_loop_task shutdown: {e}") # Consider logging
                pass
        self._interpolation_loop_task = None

        # If TensorDemuxer had an async close method, we'd call it:
        # if hasattr(super(), 'close') and asyncio.iscoroutinefunction(super().close):
        #    await super().close()
        # elif hasattr(super(), 'close'):
        #    super().close()

    async def _interpolation_worker(self) -> None:  # Added return type
        """
        Periodically generates and emits interpolated tensors.
        Periodically generates and emits interpolated tensors."""
        try:
            while not self._stop_event.is_set():
                # Determine sleep duration at the end of the loop iteration
                # Default to a short poll if no work, or responsive poll if work done.
                sleep_duration_after_iteration = (
                    min(self._smoothing_period_seconds / 2.0, 0.1)
                    if self._smoothing_period_seconds > 0
                    else 0.1
                )

                await self._keyframe_lock.acquire()
                try:
                    if len(self._keyframes) < 2:
                        # Not enough data, current iteration cannot interpolate.
                        # Lock will be released in finally. Sleep duration remains default.
                        pass  # Go to finally, then sleep
                    else:
                        first_kf_ts = self._keyframes[0][0]
                        last_kf_ts = self._keyframes[-1][0]

                        if self._last_synthetic_emitted_at is None:
                            # Start first synthetic point one period after the first real point
                            next_emission_timestamp = (
                                first_kf_ts
                                + datetime.timedelta(
                                    seconds=self._smoothing_period_seconds
                                )
                            )
                        else:
                            next_emission_timestamp = (
                                self._last_synthetic_emitted_at
                                + datetime.timedelta(
                                    seconds=self._smoothing_period_seconds
                                )
                            )

                        # If last emission was reset due to out-of-order data, next_emission_timestamp might be too early
                        if next_emission_timestamp < first_kf_ts:
                            next_emission_timestamp = (
                                first_kf_ts
                                + datetime.timedelta(
                                    seconds=self._smoothing_period_seconds
                                )
                            )

                        # If we've interpolated past all known real data points
                        if next_emission_timestamp > last_kf_ts:
                            # Caught up to real data. Lock released in finally. Sleep duration default.
                            pass  # Go to finally
                        else:
                            # Find t1: the last keyframe whose timestamp is <= next_emission_timestamp
                            idx_t1 = (
                                bisect.bisect_right(
                                    self._keyframes,
                                    next_emission_timestamp,
                                    key=lambda kf: kf[0],
                                )
                                - 1
                            )

                            if idx_t1 < 0:
                                logging.debug(
                                    "SmoothedDemuxer: next_emission_timestamp %s is before the first keyframe %s.",
                                    next_emission_timestamp,
                                    first_kf_ts,
                                )
                                # Should be caught by prior checks, but as safeguard. Lock released in finally.
                            else:
                                t1, v1 = self._keyframes[idx_t1]

                                # If next_emission_timestamp lands exactly on a real keyframe (t1),
                                # advance it to be one smoothing period after this real keyframe.
                                if t1 == next_emission_timestamp:
                                    next_emission_timestamp = t1 + datetime.timedelta(
                                        seconds=self._smoothing_period_seconds
                                    )
                                    if (
                                        next_emission_timestamp >= last_kf_ts
                                    ):  # Use >= to include case where it lands on last_kf_ts
                                        # Advanced past or onto last keyframe, nothing more to do this cycle.
                                        pass  # Go to finally
                                    else:  # Continue with potentially new next_emission_timestamp
                                        # Re-find idx_t1 for the *new* next_emission_timestamp
                                        idx_t1 = (
                                            bisect.bisect_right(
                                                self._keyframes,
                                                next_emission_timestamp,
                                                key=lambda kf: kf[0],
                                            )
                                            - 1
                                        )
                                        if (
                                            idx_t1 < 0
                                        ):  # Should not happen if next_emission_timestamp was advanced from a valid t1
                                            # pass # Go to finally: W0107 unnecessary-pass
                                            pass  # Retaining pass as it's a valid no-op placeholder in this complex conditional.
                                        else:
                                            t1, v1 = self._keyframes[idx_t1]

                                # Find t2: the first keyframe whose timestamp is > t1
                                # This relies on idx_t1 being valid and correctly found for the (potentially updated) next_emission_timestamp
                                if (
                                    idx_t1 >= 0
                                ):  # Ensure idx_t1 is valid before trying to find idx_t2
                                    idx_t2 = idx_t1 + 1
                                    if idx_t2 < len(self._keyframes):
                                        t2, v2 = self._keyframes[idx_t2]

                                        if not (
                                            t1 < next_emission_timestamp < t2
                                        ):
                                            logging.debug(
                                                "SmoothedDemuxer: next_emission_timestamp %s not strictly between t1=%s and t2=%s for interpolation.",
                                                next_emission_timestamp,
                                                t1,
                                                t2,
                                            )
                                        elif (t2 - t1).total_seconds() == 0:
                                            logging.warning(
                                                "SmoothedDemuxer: t1=%s and t2=%s have identical timestamps, skipping interpolation.",
                                                t1,
                                                t2,
                                            )
                                        else:
                                            # Perform interpolation
                                            time_ratio = (
                                                next_emission_timestamp - t1
                                            ).total_seconds() / (
                                                t2 - t1
                                            ).total_seconds()
                                            time_ratio = max(
                                                0.0, min(time_ratio, 1.0)
                                            )
                                            v_synthetic = (
                                                v1 + (v2 - v1) * time_ratio
                                            )

                                            current_synthetic_value = (
                                                v_synthetic.clone()
                                            )
                                            current_emission_ts = (
                                                next_emission_timestamp
                                            )

                                            self._keyframe_lock.release()  # Release before async client call

                                            await self._sm_client.on_tensor_changed(
                                                current_synthetic_value,
                                                current_emission_ts,
                                            )

                                            await self._keyframe_lock.acquire()  # Re-acquire for the state update
                                            self._last_synthetic_emitted_at = (
                                                current_emission_ts
                                            )

                                            sleep_duration_after_iteration = (
                                                min(
                                                    self._smoothing_period_seconds
                                                    / 10.0,
                                                    0.05,
                                                )
                                                if self._smoothing_period_seconds
                                                > 0
                                                else 0.05
                                            )
                                    else:
                                        # idx_t2 is out of bounds, no t2 found
                                        logging.debug(
                                            "SmoothedDemuxer: No keyframe t2 found after t1=%s at index %s.",
                                            t1,
                                            idx_t1,
                                        )
                                else:
                                    # idx_t1 was < 0, already handled by an earlier 'pass to finally'
                                    # Retaining pass as it's a valid no-op placeholder in this complex conditional.
                                    pass
                except Exception as e:  # pylint: disable=broad-except
                    logging.error(
                        "SmoothedDemuxer: Error in interpolation worker: %s",
                        e,
                        exc_info=True,
                    )
                    sleep_duration_after_iteration = (
                        self._smoothing_period_seconds
                        if self._smoothing_period_seconds > 0
                        else 1.0
                    )
                finally:
                    if self._keyframe_lock.locked():
                        self._keyframe_lock.release()

                await asyncio.sleep(sleep_duration_after_iteration)

        except asyncio.CancelledError:
            logging.debug(
                "SmoothedDemuxer: Interpolation worker task was cancelled."
            )
        except Exception as e:  # pylint: disable=broad-except
            logging.error(
                "SmoothedDemuxer: Interpolation worker crashed: %s",
                e,
                exc_info=True,
            )
        finally:
            if self._keyframe_lock.locked():
                self._keyframe_lock.release()

    async def on_update_received(
        self, tensor_index: int, value: float, timestamp: datetime.datetime
    ) -> None:
        """
        Receives a partial update for a tensor at a specific timestamp.

        This overridden method is responsible for:
        1. Maintaining its own set of "real" tensor keyframes (`self._keyframes`).
           This involves logic similar to the base TensorDemuxer: finding the
           correct timeslice, applying the partial update to reconstruct the
           full tensor for that timestamp, and handling cascades if an
        out-of-order
           update modifies an existing real keyframe.
        2. After updating its real keyframes, it should ensure the interpolation
           logic is aware that new real data is available.

        This method will NOT call `super().on_update_received()` because the
        SmoothedTensorDemuxer is providing its own smoothed stream and managing
        its own real keyframes independently of the base class's `_tensor_states`.
        (Implementation to be added in a subsequent step)
        """
        async with self._keyframe_lock:
            # 1. Validate tensor_index (using property from base class)
            if not (0 <= tensor_index < self._tensor_length):
                logging.warning(
                    "SmoothedTensorDemuxer: Invalid tensor_index %s for tensor_length %s. Update ignored.",
                    tensor_index,
                    self._tensor_length,
                )
                return

            # 2. Find insertion point for the timestamp
            # self._keyframes stores (timestamp, tensor_value)
            # We use key=lambda x_tuple: x_tuple[0] to sort/search by timestamp
            insertion_point = bisect.bisect_left(
                self._keyframes, timestamp, key=lambda x_tuple: x_tuple[0]
            )

            # 3. Determine if an entry for this timestamp already exists
            is_new_timestamp_entry = True
            if (
                insertion_point < len(self._keyframes)
                and self._keyframes[insertion_point][0] == timestamp
            ):
                is_new_timestamp_entry = False

            tensor_value_changed = False

            # 4. Determine initial tensor state and explicit updates list
            if is_new_timestamp_entry:
                if insertion_point == 0:
                    # Earliest timestamp, start with a zero tensor
                    current_tensor = torch.zeros(
                        self._tensor_length, dtype=torch.float32
                    )
                else:
                    # Inherit from the chronological predecessor's tensor state
                    current_tensor = self._keyframes[insertion_point - 1][
                        1
                    ].clone()
                current_explicit_updates: List[Tuple[int, float]] = []
            else:
                # Existing timestamp, get current tensor and explicit updates
                _existing_ts, existing_tensor_val = self._keyframes[
                    insertion_point
                ]
                current_tensor = existing_tensor_val.clone()
                current_explicit_updates = list(
                    self._keyframe_explicit_updates[insertion_point]
                )  # Clone list

            # 5. Apply the update (tensor_index, value) to current_tensor
            if current_tensor[tensor_index].item() != value:
                current_tensor[tensor_index] = value
                tensor_value_changed = True

            # 6. Update current_explicit_updates
            found_in_explicit = False
            for i, (idx, _val) in enumerate(current_explicit_updates):
                if idx == tensor_index:
                    if current_explicit_updates[i][1] != value:
                        current_explicit_updates[i] = (tensor_index, value)
                        # tensor_value_changed is already true if current_tensor changed.
                        # If current_tensor didn't change but explicit update did, that's also a change.
                        tensor_value_changed = True
                    found_in_explicit = True
                    break
            if not found_in_explicit:
                current_explicit_updates.append((tensor_index, value))
                tensor_value_changed = True  # New explicit update always means a change in how this keyframe is defined

            # 7. Store the updated/new keyframe
            if is_new_timestamp_entry:
                self._keyframes.insert(
                    insertion_point, (timestamp, current_tensor)
                )
                self._keyframe_explicit_updates.insert(
                    insertion_point, current_explicit_updates
                )
            else:
                # Only update if the tensor value actually changed.
                # If only explicit_updates list changed definition but result is same tensor,
                # tensor_value_changed would reflect that.
                if (
                    tensor_value_changed
                ):  # Or check against self._keyframes[insertion_point][1]
                    self._keyframes[insertion_point] = (
                        timestamp,
                        current_tensor,
                    )
                # Always update explicit updates list if it could have changed
                self._keyframe_explicit_updates[insertion_point] = (
                    current_explicit_updates
                )

            # 8. Cascading update for subsequent keyframes
            needs_cascade = tensor_value_changed
            if (
                is_new_timestamp_entry
                and insertion_point < len(self._keyframes) - 1
            ):
                # If a new entry is inserted and it's not at the very end,
                # it becomes a new predecessor for the one previously at insertion_point,
                # so a cascade is needed.
                needs_cascade = True

            if needs_cascade:
                # Start cascade from the point of change (insertion_point)
                # The loop goes up to len(self._keyframes) - 1 because we process idx_next_kf
                for idx_kf_being_updated in range(
                    insertion_point, len(self._keyframes)
                ):
                    if (
                        idx_kf_being_updated == 0
                    ):  # First keyframe has no predecessor to inherit from for cascade
                        # If the very first keyframe was changed, subsequent ones might need update if they inherited.
                        # This is handled by the loop structure starting from insertion_point.
                        # If insertion_point is 0, the first kf is updated.
                        # Then the loop continues to idx_kf_being_updated = 0.
                        # The 'predecessor_tensor' for the *next* one (idx 1) will be keyframes[0].
                        pass  # Nothing to do for the first element itself in terms of *its* predecessor

                    if idx_kf_being_updated + 1 < len(self._keyframes):
                        # This is the keyframe we are potentially re-calculating
                        idx_next_kf = idx_kf_being_updated + 1

                        next_kf_timestamp, old_next_kf_tensor_val = (
                            self._keyframes[idx_next_kf]
                        )
                        next_kf_explicit_updates = (
                            self._keyframe_explicit_updates[idx_next_kf]
                        )

                        # The predecessor for next_kf is the one at idx_kf_being_updated (which might have just changed)
                        predecessor_tensor = self._keyframes[
                            idx_kf_being_updated
                        ][1]
                        new_next_kf_tensor_val = predecessor_tensor.clone()

                        for idx_update, val_update in next_kf_explicit_updates:
                            if (
                                0 <= idx_update < self._tensor_length
                            ):  # Should always be true if validated on input
                                new_next_kf_tensor_val[idx_update] = val_update

                        if not torch.equal(
                            new_next_kf_tensor_val, old_next_kf_tensor_val
                        ):
                            self._keyframes[idx_next_kf] = (
                                next_kf_timestamp,
                                new_next_kf_tensor_val,
                            )
                            # Continue cascade: next iteration will use this updated keyframe as predecessor
                        else:
                            # If this keyframe's tensor doesn't change, subsequent ones won't either from this path
                            break
                    else:  # idx_kf_being_updated is the last keyframe, no 'next' to cascade to.
                        break

            # 9. Reset _last_synthetic_emitted_at for interpolation recalibration
            # If value changed, or if a new keyframe was inserted anywhere but the end.
            should_reset_emission_tracker = tensor_value_changed or (
                is_new_timestamp_entry
                and insertion_point < len(self._keyframes) - 1
            )

            if should_reset_emission_tracker:
                if insertion_point == 0:  # Change at the very beginning
                    self._last_synthetic_emitted_at = None
                else:
                    # Reset to the timestamp of the keyframe just before the change/insertion
                    # This ensures interpolation re-evaluates from that point.
                    self._last_synthetic_emitted_at = self._keyframes[
                        insertion_point - 1
                    ][0]

    async def get_tensor_at_timestamp(
        self, timestamp: datetime.datetime
    ) -> Optional[torch.Tensor]:
        """
        Retrieves a REAL keyframe tensor if one exists for the exact timestamp.

        This method provides access to the "real" data points that the
        SmoothedTensorDemuxer is using as a basis for interpolation.
        It does not return interpolated values. For interpolated values,
        the client should listen to `on_tensor_changed` from this class.
        """
        async with self._keyframe_lock:
            # Find the timestamp in our list of real keyframes
            # _keyframes stores (timestamp, tensor), so key is x[0]
            i = bisect.bisect_left(
                self._keyframes, timestamp, key=lambda x_tuple: x_tuple[0]
            )
            if (
                i != len(self._keyframes)
                and self._keyframes[i][0] == timestamp
            ):
                return self._keyframes[i][
                    1
                ].clone()  # Return a clone of the real tensor
        return None
