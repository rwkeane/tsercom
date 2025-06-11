"""
Provides the SmoothedTensorDemuxer class for generating smoothed tensor data
based on incoming real keyframes.
"""

import asyncio
import bisect
import datetime
import logging  # Standard library imports first
from typing import (
    List,
    Tuple,
    Optional,
)  # AsyncGenerator might be adjusted later

import torch  # Third-party imports
from tsercom.data.tensor.tensor_demuxer import (
    TensorDemuxer,
)  # First-party imports


# Configure basic logging for demonstration; in a real app, this would be more structured
logging.basicConfig(level=logging.INFO)


class SmoothedTensorDemuxer(
    TensorDemuxer
):  # pylint: disable=too-many-instance-attributes
    """Extends TensorDemuxer to provide smoothed tensor data streams.

    This class receives real-time tensor updates, which are first processed by
    the base `TensorDemuxer` to form discrete "keyframe" tensors at specific
    timestamps. `SmoothedTensorDemuxer` then uses these keyframes to generate
    a continuous stream of interpolated tensors at a regular frequency defined
    by `smoothing_period_seconds`.

    This is useful for applications that require smooth, continuous tensor data
    even when the underlying data arrives sporadically or out of order.

    Attributes:
        _actual_downstream_client (TensorDemuxer.Client): The client that will receive
            the smoothed tensor data.
        _base_client_adapter (_BaseProcessorClient): An internal adapter that receives
            keyframe tensors from the base TensorDemuxer.
        _smoothing_period_seconds (float): The time interval, in seconds, at which
            smoothed (interpolated) tensors are generated and emitted.
        _keyframes (List[Tuple[datetime.datetime, torch.Tensor]]): A sorted list
            of (timestamp, tensor) tuples representing the real keyframes received.
        _keyframe_lock (asyncio.Lock): A lock to protect access to `_keyframes`.
        _last_synthetic_emitted_at (Optional[datetime.datetime]): The timestamp of
            the last synthetically generated tensor that was emitted. This is used
            to determine the timestamp of the next synthetic tensor.
        _interpolation_loop_task (asyncio.Task): The background task that runs
            the interpolation worker loop.
        _loop (asyncio.AbstractEventLoop): The event loop on which the interpolation
            task is scheduled.
    """

    class _BaseProcessorClient(TensorDemuxer.Client):
        """Internal client adapter to receive keyframe tensors from TensorDemuxer.

        This class acts as the client for the base `TensorDemuxer`. When the base
        demuxer reconstructs a tensor (a keyframe), it calls `on_tensor_changed`
        on this adapter, which then passes the keyframe to the
        `SmoothedTensorDemuxer` for processing.
        """

        def __init__(self, outer_instance: "SmoothedTensorDemuxer"):
            """Initializes the _BaseProcessorClient.

            Args:
                outer_instance: A reference to the outer SmoothedTensorDemuxer instance.
            """
            self._outer_instance = outer_instance

        async def on_tensor_changed(
            self, tensor: torch.Tensor, timestamp: datetime.datetime
        ) -> None:
            """Handles a new keyframe tensor from the base TensorDemuxer.

            This method is called by the base `TensorDemuxer` when a complete
            tensor for a given timestamp (a keyframe) is available. It passes
            this keyframe to the outer `SmoothedTensorDemuxer`'s
            `_handle_real_keyframe` method.

            Args:
                tensor: The reconstructed keyframe tensor.
                timestamp: The timestamp of the keyframe tensor.
            """
            await self._outer_instance._handle_real_keyframe(timestamp, tensor)

    def __init__(  # pylint: disable=too-many-arguments
        self,
        client: TensorDemuxer.Client,
        tensor_length: int,
        smoothing_period_seconds: float = 1.0,
        data_timeout_seconds: float = 60.0,
    ):
        """Initializes the SmoothedTensorDemuxer.

        Args:
            client: The client to notify of smoothed tensor changes.
            tensor_length: The expected length of the tensors.
            smoothing_period_seconds: The target interval for emitting smoothed
                tensors. Must be positive.
            data_timeout_seconds: How long the base TensorDemuxer should keep
                data for a specific timestamp before it's considered stale.

        Raises:
            ValueError: If smoothing_period_seconds is not positive.
        """
        self._actual_downstream_client = client
        self._base_client_adapter = SmoothedTensorDemuxer._BaseProcessorClient(
            self
        )
        # Note: The call to super().__init__ using self._base_client_adapter means that
        # the TensorDemuxer's _cleanup_old_data will be based on updates received
        # by the base class, which are the "real" keyframes.
        super().__init__(
            self._base_client_adapter,
            tensor_length,
            data_timeout_seconds=data_timeout_seconds,
        )

        if smoothing_period_seconds <= 0:
            raise ValueError("Smoothing period must be positive.")
        self._smoothing_period_seconds = smoothing_period_seconds

        self._keyframes: List[Tuple[datetime.datetime, torch.Tensor]] = []
        self._keyframe_lock: asyncio.Lock = asyncio.Lock()
        self._last_synthetic_emitted_at: Optional[datetime.datetime] = None

        self._interpolation_loop_task = asyncio.create_task(
            self._interpolation_worker()
        )
        # It's good practice to store a reference to the loop for task management
        self._loop = asyncio.get_event_loop()

    async def _handle_real_keyframe(
        self, timestamp: datetime.datetime, tensor: torch.Tensor
    ) -> None:
        """Processes and stores a new keyframe tensor.

        This method is called by the `_BaseProcessorClient` when a new keyframe
        is received from the base `TensorDemuxer`. It acquires a lock, inserts
        or updates the keyframe in the sorted `_keyframes` list, and determines
        if the `_last_synthetic_emitted_at` timestamp needs to be reset to
        trigger recalculation of synthetic tensors.

        Recalculation is triggered if:
        - A keyframe is inserted at the beginning of the sequence.
        - An out-of-order keyframe arrives that is earlier than the last emitted
          synthetic point.
        - The system has just received enough keyframes (>=2) to start interpolation
          for the first time and `_last_synthetic_emitted_at` was None (due to
          `is_new_insertion` on the second keyframe making `keyframe_updated_or_inserted_out_of_order` true).

        Args:
            timestamp: The timestamp of the new keyframe.
            tensor: The tensor data of the new keyframe.
        """
        async with self._keyframe_lock:
            new_keyframe = (timestamp, tensor.clone())

            # Search for the timestamp in the list of keyframes.
            # The key function tells bisect_left to compare new_keyframe[0] (the timestamp)
            # with the timestamps of the existing keyframes by extracting their first element.
            insert_idx = bisect.bisect_left(
                self._keyframes, new_keyframe[0], key=lambda kf: kf[0]
            )

            keyframe_updated_or_inserted_out_of_order = False
            is_new_insertion = False
            old_len_keyframes = len(
                self._keyframes
            )  # Track length before modification

            if (
                insert_idx < len(self._keyframes)
                and self._keyframes[insert_idx][0] == timestamp
            ):
                # Update existing keyframe
                if not torch.equal(
                    self._keyframes[insert_idx][1], new_keyframe[1]
                ):
                    self._keyframes[insert_idx] = new_keyframe
                    keyframe_updated_or_inserted_out_of_order = True
            else:
                # Insert new keyframe
                self._keyframes.insert(insert_idx, new_keyframe)
                is_new_insertion = True
                # An insertion is "out of order" for recalculation if it's not at the very end
                if insert_idx < len(self._keyframes) - 1:
                    keyframe_updated_or_inserted_out_of_order = True

            # If we just added the second keyframe, and it was a new insertion,
            # consider it a scenario that needs to re-evaluate emission point.
            if (
                is_new_insertion
                and old_len_keyframes < 2
                and len(self._keyframes) >= 2
            ):
                keyframe_updated_or_inserted_out_of_order = True

            # Determine if recalculation trigger is needed
            # This happens if:
            # 1. An existing keyframe was updated.
            # 2. A new keyframe was inserted somewhere not at the end (captured by keyframe_updated_or_inserted_out_of_order).
            # 3. The timestamp of the new/updated keyframe is older than the last synthetic emission.
            #    (This covers cases where a very old keyframe arrives late).

            # Restructure conditions to help Mypy with type narrowing for _last_synthetic_emitted_at
            lsea = self._last_synthetic_emitted_at

            cond2_reset = False
            if lsea is not None and is_new_insertion: # Swapped order for R1716
                if timestamp < lsea:
                    cond2_reset = True

            cond3_reset = False
            # For R1716, ensure 'lsea is not None' is evaluated early if other conditions depend on 'lsea' being non-None.
            if lsea is not None and not is_new_insertion and keyframe_updated_or_inserted_out_of_order:
                if timestamp <= lsea:
                    cond3_reset = True

            should_reset_emission_point = (
                keyframe_updated_or_inserted_out_of_order
                or cond2_reset
                or cond3_reset
            ) # Ensure no trailing whitespace on line 220

            if should_reset_emission_point:
                current_lsea = self._last_synthetic_emitted_at

                if insert_idx == 0:
                    # Case 1: Change at the very beginning (new first keyframe).
                    self._last_synthetic_emitted_at = None
                elif current_lsea is None:
                    # Case 2: LSEA was None.
                    if len(self._keyframes) >= 2: # And now we have enough points to interpolate. Ensure no trailing whitespace on line 240
                        # Keep LSEA as None so worker emits first_kf_ts.
                        self._last_synthetic_emitted_at = None
                    # else: (len(_keyframes) < 2 and current_lsea is None and insert_idx != 0)
                    # _last_synthetic_emitted_at remains None. No action needed.
                elif timestamp < current_lsea: # current_lsea is not None here
                    # Case 3: Out-of-order update before the last point we emitted.
                    self._last_synthetic_emitted_at = None
                elif insert_idx > 0: # current_lsea is not None, and timestamp >= current_lsea
                    # Case 4: In-order update, or out-of-order but still after current emission point.
                    # Reset to the keyframe prior to the change.
                    self._last_synthetic_emitted_at = self._keyframes[insert_idx - 1][0]
                # Fall-through: current_lsea is not None, timestamp >= current_lsea, and insert_idx is 0.
                # This case is covered by the first `if insert_idx == 0`.
                # If no condition met (e.g. update to last keyframe, LSEA was already prior), LSEA doesn't change.

    async def on_update_received(
        self, tensor_index: int, value: float, timestamp: datetime.datetime
    ) -> None:
        """Handles raw tensor updates from the upstream provider.

        This method overrides the base class method to simply pass through
        the raw updates. The base `TensorDemuxer` will process these updates,
        form keyframes, and then notify this `SmoothedTensorDemuxer` via the
        `_BaseProcessorClient` adapter.

        Args:
            tensor_index: The index in the tensor that is being updated.
            value: The new float value for the tensor element.
            timestamp: The timestamp of this update.
        """
        # This method is called by the ultimate upstream provider of raw updates.
        # It delegates to the TensorDemuxer base class, which will eventually call
        # our _BaseProcessorClient.on_tensor_changed with a "real" tensor.
        await super().on_update_received(tensor_index, value, timestamp)

    async def _interpolation_worker(self) -> None:
        """The main background worker loop for generating interpolated tensors.

        This loop periodically:
        1. Checks if there are enough keyframes (at least 2) to perform interpolation.
        2. Determines the timestamp for the next synthetic tensor based on
           `_last_synthetic_emitted_at` and `_smoothing_period_seconds`.
        3. If the next synthetic timestamp is within the range of known keyframes
           and is genuinely after the last emitted point (or if it's the first point):
            a. Finds the two keyframes that bracket the synthetic timestamp.
            b. Performs linear interpolation between these two keyframes.
            c. If the synthetic timestamp exactly matches a keyframe, that keyframe's
               tensor is used directly.
            d. Emits the resulting tensor to the `_actual_downstream_client`.
            e. Updates `_last_synthetic_emitted_at`.
        4. Sleeps for a short duration if work was done, or a longer duration
           (a fraction of the smoothing period) if no work was done.

        The loop continues until the task is cancelled (e.g., by the `close` method).
        It includes error handling for `asyncio.CancelledError` (expected on shutdown)
        and other unexpected exceptions.
        """
        try:
            while True:
                processed_in_iteration = False
                async with self._keyframe_lock:
                    if len(self._keyframes) < 2:
                        # Not enough data to interpolate
                        pass  # Will proceed to sleep outside lock
                    else:
                        first_kf_ts = self._keyframes[0][0]
                        last_kf_ts = self._keyframes[-1][0]

                        if self._last_synthetic_emitted_at is None:
                            next_synthetic_ts = first_kf_ts
                        else:
                            next_synthetic_ts = (
                                self._last_synthetic_emitted_at
                                + datetime.timedelta(
                                    seconds=self._smoothing_period_seconds
                                )
                            )

                        # R1731: Consider using 'next_synthetic_ts = max(next_synthetic_ts, first_kf_ts)'
                        next_synthetic_ts = max(next_synthetic_ts, first_kf_ts)

                        if next_synthetic_ts <= last_kf_ts and (
                            self._last_synthetic_emitted_at is None
                            or next_synthetic_ts
                            > self._last_synthetic_emitted_at
                        ):
                            kf2_idx = bisect.bisect_right(
                                self._keyframes,
                                next_synthetic_ts,
                                key=lambda x: x[0],
                            )
                            kf1_idx = kf2_idx - 1

                            if kf1_idx < 0:
                                # This case should ideally be prevented by next_synthetic_ts >= first_kf_ts logic.
                                # If reached, log it and skip to avoid error.
                                logging.warning(
                                    "kf1_idx < 0 unexpected: ns_ts=%s, first_kf_ts=%s",
                                    next_synthetic_ts,
                                    first_kf_ts,
                                )
                                # W0107: Unnecessary pass statement removed
                            else:
                                t1, v1 = self._keyframes[kf1_idx]
                                v_interpolated: Optional[torch.Tensor] = None

                                if t1 == next_synthetic_ts:
                                    v_interpolated = v1.clone()
                                elif kf2_idx < len(self._keyframes):
                                    t2, v2 = self._keyframes[kf2_idx]
                                    if t2 > t1:
                                        time_diff_total_seconds = (
                                            t2 - t1
                                        ).total_seconds()
                                        if (
                                            time_diff_total_seconds > 1e-9
                                        ):  # Avoid division by zero for very close timestamps
                                            ratio = (
                                                next_synthetic_ts - t1
                                            ).total_seconds() / time_diff_total_seconds
                                            # Clamp ratio to [0, 1] to prevent extrapolation due to potential float inaccuracies
                                            ratio = max(0.0, min(1.0, ratio))
                                            v_interpolated = (
                                                v1 + (v2 - v1) * ratio
                                            )
                                        else:
                                            v_interpolated = (
                                                v1.clone()
                                            )  # or v2.clone(), if t1 and t2 are effectively same
                                    else:
                                        v_interpolated = v1.clone()
                                # If kf2_idx is out of bounds, it means next_synthetic_ts is after the last keyframe's time (t1),
                                # but next_synthetic_ts <= last_kf_ts was true. This implies next_synthetic_ts == last_kf_ts.
                                # This case should be covered by t1 == next_synthetic_ts if t1 is the last keyframe.
                                # If t1 is not the last keyframe, but kf2_idx is out of bounds, it's an issue.
                                # However, bisect_right should give len(self._keyframes) if next_synthetic_ts is >= last_kf_ts.
                                # The outer check `next_synthetic_ts <= last_kf_ts` should handle this.
                                # If v_interpolated is still None here, it might mean next_synthetic_ts == last_kf_ts and it wasn't handled by t1 == next_synthetic_ts.
                                # This can happen if kf1_idx points to the second to last, and next_synthetic_ts is exactly last_kf_ts.
                                # In this specific scenario, kf2_idx would be len(self._keyframes), so we can't use self._keyframes[kf2_idx].
                                # We should use the last keyframe directly.
                                elif (
                                    next_synthetic_ts == last_kf_ts
                                ):  # Explicitly check if target is the last keyframe
                                    v_interpolated = self._keyframes[-1][
                                        1
                                    ].clone()

                                if v_interpolated is not None:
                                    if self._actual_downstream_client:
                                        await self._actual_downstream_client.on_tensor_changed(
                                            v_interpolated.clone(),
                                            next_synthetic_ts,
                                        )
                                    self._last_synthetic_emitted_at = (
                                        next_synthetic_ts
                                    )
                                    processed_in_iteration = True

                if processed_in_iteration:
                    await asyncio.sleep(0.001)
                else:
                    await asyncio.sleep(
                        max(0.01, self._smoothing_period_seconds / 3.0)
                    )  # Ensure sleep is at least 10ms

        except asyncio.CancelledError:
            # W1203: Use lazy % formatting in logging functions
            logging.info("Interpolation worker for %s cancelled.", self)
            raise
        except Exception as e: # pylint: disable=broad-except # Keep for worker resilience
            # W1203: Use lazy % formatting in logging functions
            logging.error(
                "Unexpected error in interpolation worker: %s",
                e,
                exc_info=True,
            )
            raise

    async def close(self) -> None:
        """
        Gracefully shuts down the SmoothedTensorDemuxer.

        Cancels the background interpolation worker task and waits for it to complete.
        """
        if (
            self._interpolation_loop_task
            and not self._interpolation_loop_task.done()
        ):
            self._interpolation_loop_task.cancel()
            try:
                await self._interpolation_loop_task
            except asyncio.CancelledError:
                # W1203: Use lazy % formatting in logging functions
                logging.info("Interpolation loop task successfully cancelled.")
            except Exception as e: # pylint: disable=broad-except # Shutdown should be robust
                # W1203: Use lazy % formatting in logging functions
                logging.error(
                    "Error during interpolation task shutdown: %s", e
                )
        # If TensorDemuxer base class had an async close method:
        # if hasattr(super(), 'close') and asyncio.iscoroutinefunction(super().close):
        #     await super().close()
        # elif hasattr(super(), 'close'): # For a synchronous close
        #     super().close()

    # TODO: Consider adding a keyframe cleanup mechanism in _interpolation_worker
    # or a separate task to prevent self._keyframes from growing indefinitely if
    # the base TensorDemuxer's cleanup doesn't implicitly manage this for keyframes.
    # The base class cleanup is based on its own _latest_update_timestamp, which might
    # be far ahead of the keyframes being stored here if updates are sparse.
    # For now, keyframes are kept based on the TensorDemuxer's data_timeout_seconds
    # relative to the LATEST update the TensorDemuxer base has seen. If SmoothedTensorDemuxer
    # needs its own, more aggressive keyframe pruning, it'd be an addition here.
