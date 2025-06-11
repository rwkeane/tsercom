"""
Provides the SmoothedTensorDemuxer class for generating smoothed tensor data
based on incoming real keyframes.
"""
import asyncio
import bisect
import datetime
import logging # Standard library imports first
from typing import (
    List,
    Tuple,
    Optional,
)  # AsyncGenerator might be adjusted later

import torch # Third-party imports
from tsercom.data.tensor.tensor_demuxer import TensorDemuxer # First-party imports


# Configure basic logging for demonstration; in a real app, this would be more structured
logging.basicConfig(level=logging.INFO)


class SmoothedTensorDemuxer(TensorDemuxer):
    """
    Extends TensorDemuxer to provide smoothed tensor data.

    It receives real tensor states (keyframes) from the underlying TensorDemuxer
    via an adapter and then generates interpolated ("smoothed") tensor states
    at a regular interval defined by `smoothing_period_seconds`.
    """

    class _BaseProcessorClient(TensorDemuxer.Client):
        """
        Internal client adapter to receive keyframes from the base TensorDemuxer.
        """
        def __init__(self, outer_instance: "SmoothedTensorDemuxer"):
            self._outer_instance = outer_instance

        async def on_tensor_changed(
            self, tensor: torch.Tensor, timestamp: datetime.datetime
        ) -> None:
            await self._outer_instance._handle_real_keyframe(timestamp, tensor)

    def __init__(
        self,
        client: TensorDemuxer.Client,
        tensor_length: int,
        smoothing_period_seconds: float = 1.0,
        data_timeout_seconds: float = 60.0,
    ):
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
            old_len_keyframes = len(self._keyframes) # Track length before modification

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
            if is_new_insertion and old_len_keyframes < 2 and len(self._keyframes) >= 2:
                keyframe_updated_or_inserted_out_of_order = True


            # Determine if recalculation trigger is needed
            # This happens if:
            # 1. An existing keyframe was updated.
            # 2. A new keyframe was inserted somewhere not at the end.
            # 3. The timestamp of the new/updated keyframe is older than the last synthetic emission.
            #    (This covers cases where a very old keyframe arrives late).
            should_reset_emission_point = (
                keyframe_updated_or_inserted_out_of_order
                or (
                    is_new_insertion
                    and (
                        self._last_synthetic_emitted_at
                        and timestamp < self._last_synthetic_emitted_at
                    )
                )
                or (
                    not is_new_insertion
                    and keyframe_updated_or_inserted_out_of_order
                    and (
                        self._last_synthetic_emitted_at
                        and timestamp <= self._last_synthetic_emitted_at
                    )
                )
            )

            if should_reset_emission_point:
                initial_lsea_was_none = self._last_synthetic_emitted_at is None

                if insert_idx == 0 or \
                   (not initial_lsea_was_none and timestamp < self._last_synthetic_emitted_at):
                    # Case 1: Change at the very beginning (new first keyframe).
                    # Case 2: Out-of-order update before the last point we emitted.
                    self._last_synthetic_emitted_at = None
                elif initial_lsea_was_none and len(self._keyframes) >= 2:
                    # Case 3: We hadn't emitted anything (_LSEA was None), and now we have enough
                    # keyframes to start. Keep _LSEA as None so worker emits first_kf_ts.
                    # This handles the scenario where `is_new_insertion` might be False for the
                    # final component of the second keyframe, but _LSEA was still None.
                    self._last_synthetic_emitted_at = None
                elif insert_idx > 0:
                    # Case 4: In-order update, or out-of-order but still after current emission point.
                    # Reset to the keyframe prior to the change to force re-evaluation from there.
                    self._last_synthetic_emitted_at = self._keyframes[insert_idx - 1][0]
                # Fall-through: If none of these specific reset conditions are met but
                # should_reset_emission_point was true, _LSEA might not change.
                # This could happen if e.g. the tensor value of the *last* keyframe changes,
                # and _LSEA was already pointing to the one before it. No change needed to _LSEA.
                # In this case, self._last_synthetic_emitted_at might not need to change,
                # or if it does (e.g. an update to the keyframe it was based on),
                # the current logic of resetting to [insert_idx-1][0] is generally fine.
                # However, consider if the updated keyframe *was* _last_synthetic_emitted_at.
                # If keyframe at self._last_synthetic_emitted_at is *updated*,
                # insert_idx would point to it. Then _last_synthetic_emitted_at becomes keyframes[insert_idx-1][0].
                # This is correct for re-evaluating from that point.

                # If the list becomes very small (e.g. < 2 keyframes after deletions/cleanup not shown here),
                # _last_synthetic_emitted_at = None might also be appropriate.
                # Current logic assumes _cleanup_old_data in base class handles TensorDemuxer's states,
                # and SmoothedTensorDemuxer might need its own keyframe cleanup eventually.
                # For now, this reset logic is focused on reacting to new/updated data.

    async def on_update_received(
        self, tensor_index: int, value: float, timestamp: datetime.datetime
    ) -> None:
        # This method is called by the ultimate upstream provider of raw updates.
        # It delegates to the TensorDemuxer base class, which will eventually call
        # our _BaseProcessorClient.on_tensor_changed with a "real" tensor.
        await super().on_update_received(tensor_index, value, timestamp)

    async def _interpolation_worker(self) -> None:
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
                "Unexpected error in interpolation worker: %s", e, exc_info=True
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
                logging.error("Error during interpolation task shutdown: %s", e)
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
