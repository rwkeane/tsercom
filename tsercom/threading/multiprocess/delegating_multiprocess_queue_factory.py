"""Defines a factory for queues that eagerly creates multiple queue types
and dynamically delegates to one based on the first item sent.

The DelegatingMultiprocessQueueFactory eagerly creates two sets of queues:
1. One pair using DefaultMultiprocessQueueFactory.
2. A second pair using TorchMultiprocessQueueFactory (if PyTorch is available).

The DelegatingMultiprocessQueueSink inspects the first item sent.
If it's a torch.Tensor, it sends a coordination message on the default queue
and then uses the torch queue for all subsequent items. Otherwise, it sends a
different coordination message on the default queue and uses the default queue
for all items.

The DelegatingMultiprocessQueueSource listens on the default queue for the
coordination message and then switches to the appropriate underlying queue.
"""

import queue  # For queue.Empty
from typing import TypeVar, Generic, Optional, Tuple, Literal

from tsercom.threading.multiprocess.default_multiprocess_queue_factory import (
    DefaultMultiprocessQueueFactory,
)
from tsercom.threading.multiprocess.torch_multiprocess_queue_factory import (
    TorchMultiprocessQueueFactory,
)
from tsercom.threading.multiprocess.multiprocess_queue_sink import (
    MultiprocessQueueSink,
)
from tsercom.threading.multiprocess.multiprocess_queue_source import (
    MultiprocessQueueSource,
)
from tsercom.threading.multiprocess.multiprocess_queue_factory import (
    MultiprocessQueueFactory,
)

_TORCH_AVAILABLE: bool
_torch_tensor_type: Optional[type] = None

try:
    import torch

    # Keep torch_mp_imported if needed by TorchMultiprocessQueueFactory,
    # but it's not directly used in *this* file's logic anymore for manager decisions.
    # import torch.multiprocessing as torch_mp_imported # Not directly used here anymore for manager
    _TORCH_AVAILABLE = True
    _torch_tensor_type = torch.Tensor
except ImportError:
    _TORCH_AVAILABLE = False


_QueueItemType = TypeVar("_QueueItemType")

# Coordination messages
USE_TORCH_QUEUE_MSG = "USE_TORCH"
USE_DEFAULT_QUEUE_MSG = "USE_DEFAULT"
SelectedQueueType = Literal["torch", "default"]


# pylint: disable=W0231 # __init__ not called for base class if not appropriate
class DelegatingMultiprocessQueueSink(MultiprocessQueueSink[_QueueItemType]):
    """
    A multiprocessing queue sink that, on the first 'put' operation,
    selects an underlying queue (Torch or default) based on the item type.
    It sends a coordination message on the default queue before using the
    selected queue.
    """

    def __init__(
        self,
        default_sink: MultiprocessQueueSink[_QueueItemType],
        torch_sink: Optional[MultiprocessQueueSink[_QueueItemType]],
        # The coordination messages are strings, so the default_sink must be able to handle them.
        # We assume _QueueItemType can be a string for this coordination message.
        # If _QueueItemType is strictly not a string, this would need adjustment.
        # For now, the coordination channel (default_sink) is typed with _QueueItemType.
        # A more robust solution might use a dedicated control queue for strings.
    ):
        """
        Initializes the DelegatingMultiprocessQueueSink.

        Args:
            default_sink: The sink for the default multiprocessing queue.
                         This queue is also used for the initial coordination message.
            torch_sink: The sink for the PyTorch multiprocessing queue, if available.
        """
        self.__default_sink = default_sink
        self.__torch_sink = torch_sink
        self.__selected_queue_type: Optional[SelectedQueueType] = None
        self.__closed_flag = False

    def _select_queue_and_send_coordination(
        self, item: _QueueItemType
    ) -> None:
        """Selects queue based on item type and sends coordination message."""
        actual_data = getattr(item, "data", item)

        if (
            _TORCH_AVAILABLE
            and self.__torch_sink is not None
            and _torch_tensor_type
            is not None  # Make sure torch.Tensor was imported
            and isinstance(actual_data, _torch_tensor_type)
        ):
            # Using type: ignore as put_nowait expects _QueueItemType, but we're sending a string.
            # This implies _QueueItemType must be compatible with str for coordination.
            self.__default_sink.put_nowait(USE_TORCH_QUEUE_MSG)  # type: ignore
            self.__selected_queue_type = "torch"
        else:
            self.__default_sink.put_nowait(USE_DEFAULT_QUEUE_MSG)  # type: ignore
            self.__selected_queue_type = "default"

    def put_blocking(
        self, obj: _QueueItemType, timeout: Optional[float] = None
    ) -> bool:
        """Puts an item into the selected queue, blocking if necessary."""
        if self.__closed_flag:
            raise RuntimeError("Sink closed.")

        if self.__selected_queue_type is None:
            self._select_queue_and_send_coordination(obj)
            # The first item still needs to be put after coordination.

        if self.__selected_queue_type == "torch":
            if (
                self.__torch_sink is None
            ):  # Should not happen if selected_queue_type is "torch"
                raise RuntimeError("Torch sink selected but not available.")
            return self.__torch_sink.put_blocking(obj, timeout=timeout)
        # Default to default_sink if not torch (covers "default" or if torch somehow became None)
        return self.__default_sink.put_blocking(obj, timeout=timeout)

    def put_nowait(self, obj: _QueueItemType) -> bool:
        """Puts an item into the selected queue without blocking."""
        if self.__closed_flag:
            raise RuntimeError("Sink closed.")

        if self.__selected_queue_type is None:
            self._select_queue_and_send_coordination(obj)
            # The first item still needs to be put after coordination.

        if self.__selected_queue_type == "torch":
            if self.__torch_sink is None:  # Should not happen
                raise RuntimeError(
                    "Torch sink selected but not available for put_nowait."
                )
            return self.__torch_sink.put_nowait(obj)
        return self.__default_sink.put_nowait(obj)

    def close(self) -> None:
        """Marks the sink as closed. Does not close underlying sinks here,
        as they are managed by the factory that created them."""
        self.__closed_flag = True
        # Optionally, could call close on underlying sinks if ownership is here,
        # but typically factory handles lifecycle of underlying queues.
        # self.__default_sink.close()
        # if self.__torch_sink:
        #     self.__torch_sink.close()

    @property
    def closed(self) -> bool:
        """Returns True if the sink is closed, False otherwise."""
        return self.__closed_flag


# pylint: disable=W0231
class DelegatingMultiprocessQueueSource(
    MultiprocessQueueSource[_QueueItemType]
):
    """
    A multiprocessing queue source that first reads a coordination message
    from the default queue to determine which underlying queue (Torch or default)
    to use for all subsequent 'get' operations.
    """

    def __init__(
        self,
        default_source: MultiprocessQueueSource[_QueueItemType],
        torch_source: Optional[MultiprocessQueueSource[_QueueItemType]],
    ):
        """
        Initializes the DelegatingMultiprocessQueueSource.

        Args:
            default_source: The source for the default multiprocessing queue.
                            This queue is also used for the initial coordination message.
            torch_source: The source for the PyTorch multiprocessing queue, if available.
        """
        self.__default_source = default_source
        self.__torch_source = torch_source
        self.__underlying_source_to_use: Optional[
            MultiprocessQueueSource[_QueueItemType]
        ] = None

    def _receive_coordination_and_select_source(
        self, timeout: Optional[float] = None
    ) -> None:
        """Receives coordination message and sets the appropriate underlying source."""
        if self.__underlying_source_to_use is not None:
            return

        try:
            # Blocking get for the coordination message from the default queue.
            # This message is expected to be a string.
            coordination_msg = self.__default_source.get_blocking(
                timeout=timeout
            )
        except queue.Empty:  # Raised by get_blocking on timeout
            # Propagate timeout if coordination message itself times out
            raise

        if coordination_msg == USE_TORCH_QUEUE_MSG:
            if self.__torch_source is None:
                raise RuntimeError(
                    "Received USE_TORCH coordination, but no torch source is available."
                )
            self.__underlying_source_to_use = self.__torch_source
        elif coordination_msg == USE_DEFAULT_QUEUE_MSG:
            self.__underlying_source_to_use = self.__default_source
        else:
            # This case should ideally not happen if the sink sends valid messages.
            raise RuntimeError(
                f"Invalid coordination message received: {coordination_msg}"
            )

    def get_blocking(
        self, timeout: Optional[float] = None
    ) -> Optional[_QueueItemType]:
        """Gets an item from the selected queue, blocking if necessary."""
        if self.__underlying_source_to_use is None:
            try:
                self._receive_coordination_and_select_source(timeout=timeout)
            except queue.Empty:  # Timeout receiving coordination message
                return None  # Consistent with how get_blocking typically signals timeout

        if self.__underlying_source_to_use is None:
            # This should not be reached if _receive_coordination_and_select_source succeeded
            # or if it timed out (which should have returned None above).
            raise RuntimeError(
                "Underlying source not selected after coordination attempt."
            )

        # Now get the actual data item from the selected queue
        return self.__underlying_source_to_use.get_blocking(timeout=timeout)

    def get_or_none(self) -> Optional[_QueueItemType]:
        """Gets an item from the selected queue if available, otherwise returns None."""
        if self.__underlying_source_to_use is None:
            try:
                # Use a short timeout or non-blocking for coordination if this method is non-blocking.
                # If get_blocking on default_source can return None immediately if empty, that's better.
                # Assuming get_blocking with timeout=0 or a very small value for check.
                self._receive_coordination_and_select_source(
                    timeout=0.001
                )  # Small timeout for check
            except queue.Empty:  # Timeout or empty during coordination check
                return None  # Can't determine queue, so no item.

        if self.__underlying_source_to_use is None:
            # Still no source selected (e.g. coordination timed out above, or not yet sent)
            return None

        # Now get the actual data item from the selected queue
        return self.__underlying_source_to_use.get_or_none()


class DelegatingMultiprocessQueueFactory(
    MultiprocessQueueFactory[_QueueItemType], Generic[_QueueItemType]
):
    """
    A factory that eagerly creates both default and PyTorch-specific queues
    (if PyTorch is available) and provides delegating sink/source wrappers.
    The sink/source pair uses a coordination mechanism on the default queue
    to decide which underlying queue to use for actual data transfer.
    """

    def __init__(self) -> None:
        """Initializes the DelegatingMultiprocessQueueFactory."""
        super().__init__()
        # Each factory (default and torch) will manage its own manager if needed.
        # This top-level factory doesn't need its own manager instance directly.
        self.__default_factory: MultiprocessQueueFactory[_QueueItemType] = (
            DefaultMultiprocessQueueFactory()
        )
        self.__torch_factory: Optional[
            MultiprocessQueueFactory[_QueueItemType]
        ] = None

        if _TORCH_AVAILABLE:
            # Assuming TorchMultiprocessQueueFactory does not need a manager passed at init
            # or handles it internally, similar to DefaultMultiprocessQueueFactory.
            self.__torch_factory = TorchMultiprocessQueueFactory()

        # Create the queues eagerly
        self.__default_sink_internal: MultiprocessQueueSink[_QueueItemType]
        self.__default_source_internal: MultiprocessQueueSource[_QueueItemType]
        (
            self.__default_sink_internal,
            self.__default_source_internal,
        ) = self.__default_factory.create_queues()

        self.__torch_sink_internal: Optional[
            MultiprocessQueueSink[_QueueItemType]
        ] = None
        self.__torch_source_internal: Optional[
            MultiprocessQueueSource[_QueueItemType]
        ] = None

        if self.__torch_factory:
            (
                self.__torch_sink_internal,
                self.__torch_source_internal,
            ) = self.__torch_factory.create_queues()

    def create_queues(
        self,
    ) -> Tuple[
        MultiprocessQueueSink[_QueueItemType],
        MultiprocessQueueSource[_QueueItemType],
    ]:
        """
        Creates a pair of delegating queue sink and source.

        These wrappers will use the eagerly created default and Torch queues.
        """
        # The sink and source wrappers are new instances each time,
        # but they operate on the same underlying eagerly created queues.
        sink = DelegatingMultiprocessQueueSink[_QueueItemType](
            default_sink=self.__default_sink_internal,
            torch_sink=self.__torch_sink_internal,
        )
        source = DelegatingMultiprocessQueueSource[_QueueItemType](
            default_source=self.__default_source_internal,
            torch_source=self.__torch_source_internal,
        )
        return sink, source

    def shutdown(self) -> None:
        """Shuts down the underlying queue factories."""
        if hasattr(self.__default_factory, "shutdown"):
            self.__default_factory.shutdown()
        if self.__torch_factory and hasattr(self.__torch_factory, "shutdown"):
            self.__torch_factory.shutdown()


# Remove old helper, no longer used with eager creation.
# def is_torch_available() -> bool:
#    return _TORCH_AVAILABLE

# Remove old constants, not used in the new design
# INITIALIZED_KEY = "initialized"
# REAL_QUEUE_SOURCE_REF_KEY = "real_queue_source_ref"
# REAL_QUEUE_SINK_REF_KEY = "real_queue_sink_ref"
