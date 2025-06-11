import abc
import asyncio
import datetime
import logging  # Added for Pylint fix
from typing import Optional

import torch  # Third-party

# TORCH_TYPING_PLACEHOLDER # Import torchtyping if available, otherwise skip.
# Try to import torchtyping, but don't fail if it's not there.
# This is a common pattern if a library is optional.
try:
    from torchtyping import TensorType  # type: ignore[import-not-found]
except ImportError:
    # Define a placeholder if torchtyping is not available
    # This allows the code to run, but type checking for tensor dimensions will not be as rich.
    # Users would need to install torchtyping for full benefits.
    # For this exercise, we'll make it a simple generic type.
    TensorType = torch.Tensor


class SmoothedTensorProvider:
    class Client(abc.ABC):
        @abc.abstractmethod
        async def on_smoothed_tensor(
            self, tensor: torch.Tensor, timestamp: datetime.datetime
        ) -> None:
            """Callback for when a new smoothed tensor is available.

            Args:
                tensor: The interpolated tensor.
                timestamp: The timestamp of the interpolated tensor.
            """

    def __init__(
        self,
        client: "SmoothedTensorProvider.Client",
        smoothing_period_seconds: float = 1.0,
    ):
        """Initializes the SmoothedTensorProvider.

        Args:
            client: The client to send smoothed tensors to.
            smoothing_period_seconds: The time interval for interpolated tensor outputs.
                Must be greater than zero.
        """
        if smoothing_period_seconds <= 0:
            raise ValueError("smoothing_period_seconds must be positive.")

        self._client: SmoothedTensorProvider.Client = client
        self._smoothing_period_seconds: float = smoothing_period_seconds

        self._last_timestamp_1: Optional[datetime.datetime] = None
        self._last_tensor_1: Optional[TensorType] = None
        self._last_timestamp_2: Optional[datetime.datetime] = None
        self._last_tensor_2: Optional[TensorType] = None

        self._interpolation_task: Optional[asyncio.Task[None]] = None
        self._lock = (
            asyncio.Lock()
        )  # To protect state variables if needed, though on_tensor_changed is async

    async def on_tensor_changed(
        self, tensor: TensorType, timestamp: datetime.datetime
    ) -> None:
        """Receives a new tensor data point.

        This method conforms to the TensorDemuxer.Client interface.

        Args:
            tensor: The new tensor.
            timestamp: The timestamp of the new tensor.
        """
        async with self._lock:
            if self._last_timestamp_1 is None or self._last_tensor_1 is None:
                self._last_timestamp_1 = timestamp
                self._last_tensor_1 = tensor
                return

            if (
                self._last_timestamp_2 is not None
                and self._last_tensor_2 is not None
            ):
                self._last_timestamp_1 = self._last_timestamp_2
                self._last_tensor_1 = self._last_tensor_2

            self._last_timestamp_2 = timestamp
            self._last_tensor_2 = tensor

            # Ensure timestamps are monotonic if necessary, or handle out-of-order data.
            # For now, assuming timestamps are generally increasing.
            if self._last_timestamp_1 >= self._last_timestamp_2:
                # Option 1: Log a warning and ignore the new point if timestamps are not strictly increasing.
                # Option 2: Reset and treat the new point as the first point.
                # For simplicity, let's log and make the new point the first one, effectively resetting.
                # print(f"Warning: New timestamp {timestamp} is not after {self._last_timestamp_1}. Resetting first point.")
                self._last_timestamp_1 = timestamp
                self._last_tensor_1 = tensor
                self._last_timestamp_2 = None
                self._last_tensor_2 = None
                if (
                    self._interpolation_task
                    and not self._interpolation_task.done()
                ):
                    self._interpolation_task.cancel()
                return

            if (
                self._interpolation_task
                and not self._interpolation_task.done()
            ):
                self._interpolation_task.cancel()
                try:
                    await self._interpolation_task  # Wait for cancellation to complete
                except asyncio.CancelledError:
                    pass  # Expected

            # Ensure all required data is present before starting task
            # Check only for timestamps; tensors are guaranteed if timestamps are present.
            if self._last_timestamp_1 and self._last_timestamp_2:
                self._interpolation_task = asyncio.create_task(
                    self._perform_interpolation(
                        self._last_timestamp_1,
                        self._last_tensor_1.clone(),  # Use .clone() for tensors
                        self._last_timestamp_2,
                        self._last_tensor_2.clone(),  # Use .clone() for tensors
                    )
                )

    async def _perform_interpolation(
        self,
        ts1: datetime.datetime,
        t1: TensorType,
        ts2: datetime.datetime,
        t2: TensorType,
    ) -> None:
        """Performs linear interpolation between two tensor data points."""
        try:
            total_duration_seconds = (ts2 - ts1).total_seconds()

            # If the duration is zero or negative, or too small, no interpolation
            if total_duration_seconds <= 1e-9:  # Using a small epsilon
                return

            # It should be strictly after ts1
            current_target_timestamp = ts1 + datetime.timedelta(
                seconds=self._smoothing_period_seconds
            )

            while current_target_timestamp < ts2:
                # Check for cancellation at the beginning of each iteration
                await asyncio.sleep(
                    0
                )  # Yield control, allowing cancellation to be processed

                time_since_ts1_seconds = (
                    current_target_timestamp - ts1
                ).total_seconds()

                alpha = time_since_ts1_seconds / total_duration_seconds

                # Clamp alpha just in case of floating point issues, though < ts2 should prevent alpha >= 1
                alpha = max(0.0, min(alpha, 1.0))

                interpolated_tensor = t1 + alpha * (t2 - t1)

                await self._client.on_smoothed_tensor(
                    interpolated_tensor, current_target_timestamp
                )

                current_target_timestamp += datetime.timedelta(
                    seconds=self._smoothing_period_seconds
                )

        except asyncio.CancelledError:  # pylint: disable=try-except-raise
            # print("Interpolation task cancelled.") # Optional: for debugging
            raise  # Re-raise to ensure the task is properly cancelled
        except Exception as e:  # pylint: disable=broad-exception-caught
            # print(f"Error during interpolation: {e}") # Optional: for logging
            # Depending on desired robustness, might want to handle specific exceptions
            logging.error("Error during interpolation: %s", e, exc_info=True)
            # Or re-raise if the error should propagate

    async def cancel(self) -> None:
        """Cancels any ongoing interpolation task."""
        async with self._lock:
            if (
                self._interpolation_task
                and not self._interpolation_task.done()
            ):
                self._interpolation_task.cancel()
                try:
                    await self._interpolation_task
                except asyncio.CancelledError:
                    pass  # Expected
                self._interpolation_task = None
