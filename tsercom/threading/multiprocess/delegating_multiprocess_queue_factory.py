"""Defines a factory for queues that dynamically delegate queue creation.

The DelegatingMultiprocessQueueFactory creates a default IPC queue and,
if PyTorch is available, a Torch IPC queue. The DelegatingMultiprocessQueueSink
sends a coordination message ("USE_TORCH" or "USE_DEFAULT") over the default
queue to inform the DelegatingMultiprocessQueueSource which queue will be used
for actual data transmission. The first data item is then sent over the
selected queue.
"""

import multiprocessing
import queue  # For queue.Empty exception
from typing import TypeVar, Generic, Optional, Tuple, Any
from types import ModuleType

# Absolute imports for tsercom modules
from tsercom.common.constants import (
    DEFAULT_MAX_QUEUE_SIZE,
)  # If used by this module
from tsercom.common.delegates import MethodDelegate  # For on_close event
from tsercom.common.exceptions import (
    QueueTimeoutError,
)  # If specific exception needed

# from tsercom.common.messages import Message, MessageType # Not directly used by this refactor
from tsercom.common.protocols import (
    MultiprocessQueueItemProtocol,
)  # For QueueItemType bound

# from tsercom.common.utils.check import is_not_none # Not directly used by this refactor

from tsercom.threading.multiprocess.default_multiprocess_queue_factory import (
    DefaultMultiprocessQueueFactory,
    # DefaultMultiprocessQueueSink, # Not directly instantiated by Delegating classes
    # DefaultMultiprocessQueueSource, # Not directly instantiated by Delegating classes
)
from tsercom.threading.multiprocess.torch_multiprocess_queue_factory import (
    TorchMultiprocessQueueFactory,
    # TorchMultiprocessQueueSink, # Not directly instantiated by Delegating classes
    # TorchMultiprocessQueueSource, # Not directly instantiated by Delegating classes
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


_torch_mp_module: Optional[ModuleType] = None
_torch_available: bool

try:
    import torch
    import torch.multiprocessing as torch_mp_imported

    _torch_mp_module = torch_mp_imported
    _torch_available = True
except ImportError:
    _torch_available = False
    # torch import itself might be needed for torch.Tensor check later
    # if _torch_available is True but torch wasn't imported at top level.
    # However, the structure implies if _torch_available is false, torch.Tensor won't be relevant.
    # Let's ensure torch is imported if _torch_available is true, for isinstance checks.
if _torch_available and "torch" not in globals():
    import torch  # Ensure torch is available for type checking if _torch_available


QueueItemType = TypeVar("QueueItemType", bound=MultiprocessQueueItemProtocol)


def is_torch_available() -> bool:
    """Checks if PyTorch and its multiprocessing extensions are available."""
    return _torch_available and _torch_mp_module is not None


# Global constants for shared dictionary keys are removed as shared dict is removed.


class DelegatingMultiprocessQueueSink(MultiprocessQueueSink[QueueItemType]):
    """
    A sink that delegates to either a default or Torch queue.
    The decision is made upon the first item being sent and communicated
    to the source via the default queue.
    """

    def __init__(
        self,
        default_queue_sink: MultiprocessQueueSink[QueueItemType],
        torch_queue_sink: Optional[MultiprocessQueueSink[QueueItemType]],
    ):
        """
        Initializes the DelegatingMultiprocessQueueSink.

        Args:
            default_queue_sink: The sink for the default IPC queue (coordination and data).
            torch_queue_sink: The sink for the Torch IPC queue (data, if chosen).
        """
        # max_queue_size is implicitly handled by the provided sinks.
        # Initialize with a sensible default or derive from one of the sinks if necessary.
        super().__init__(
            default_queue_sink.max_queue_size if default_queue_sink else 0
        )
        self.__default_queue_sink = default_queue_sink
        self.__torch_queue_sink = torch_queue_sink
        self.__selected_queue_sink: Optional[
            MultiprocessQueueSink[QueueItemType]
        ] = None
        self.__coordination_sent: bool = False
        self.__closed_flag = False
        self.on_close = MethodDelegate[[]]()  # type: ignore[attr-defined]

    def __perform_initial_put(
        self,
        obj: QueueItemType,
        timeout: Optional[float] = None,
        use_nowait: bool = False,
    ) -> bool:
        """
        Performs the initial put operation, sending a coordination message
        and then the actual object to the appropriate queue.
        Sets self.__selected_queue_sink and self.__coordination_sent.
        """
        actual_data = getattr(obj, "data", obj)
        use_torch_path = False
        if (
            self.__torch_queue_sink is not None
            and is_torch_available()  # Checks both torch and _torch_mp_module
            and isinstance(actual_data, torch.Tensor)
        ):
            use_torch_path = True

        put_method_default_coord = (
            self.__default_queue_sink.put_nowait
            if use_nowait
            else self.__default_queue_sink.put_blocking
        )
        coordination_message = "USE_TORCH" if use_torch_path else "USE_DEFAULT"
        # Ensure coordination_message is queue-friendly (e.g. string)
        coord_args = (
            (coordination_message,)
            if use_nowait
            else (coordination_message, timeout)
        )

        if not put_method_default_coord(*coord_args):
            return False  # Failed to send coordination message

        if use_torch_path:
            # This assertion is valid because use_torch_path requires __torch_queue_sink is not None
            assert self.__torch_queue_sink is not None
            self.__selected_queue_sink = self.__torch_queue_sink
            put_method_selected = (
                self.__selected_queue_sink.put_nowait
                if use_nowait
                else self.__selected_queue_sink.put_blocking
            )
            args_selected = (obj,) if use_nowait else (obj, timeout)
            if not put_method_selected(*args_selected):
                # Problematic state: coordination sent, data failed.
                return False
        else:
            self.__selected_queue_sink = self.__default_queue_sink
            # Data is sent to the same default queue after the coordination message.
            put_method_default_data = (
                self.__default_queue_sink.put_nowait
                if use_nowait
                else self.__default_queue_sink.put_blocking
            )
            args_default_data = (obj,) if use_nowait else (obj, timeout)
            if not put_method_default_data(*args_default_data):
                # Problematic state: coordination (USE_DEFAULT) sent, data failed.
                return False

        self.__coordination_sent = True
        return True

    def put_blocking(
        self, obj: QueueItemType, timeout: Optional[float] = None
    ) -> bool:
        if self.__closed_flag:
            raise RuntimeError(
                "Cannot put item on a closed DelegatingMultiprocessQueueSink."
            )
        if not self.__coordination_sent:
            return self.__perform_initial_put(
                obj, timeout=timeout, use_nowait=False
            )

        assert (
            self.__selected_queue_sink is not None
        ), "Coordination sent but no queue selected."
        return self.__selected_queue_sink.put_blocking(obj, timeout=timeout)

    def put_nowait(self, obj: QueueItemType) -> bool:
        if self.__closed_flag:
            raise RuntimeError(
                "Cannot put item on a closed DelegatingMultiprocessQueueSink."
            )
        if not self.__coordination_sent:
            return self.__perform_initial_put(obj, use_nowait=True)

        assert (
            self.__selected_queue_sink is not None
        ), "Coordination sent but no queue selected."
        return self.__selected_queue_sink.put_nowait(obj)

    def close(self) -> None:
        if not self.__closed_flag:
            self.__closed_flag = True
            self.__default_queue_sink.close()
            if self.__torch_queue_sink is not None:
                self.__torch_queue_sink.close()
            self.on_close.invoke()

    @property
    def closed(self) -> bool:
        return self.__closed_flag

    @property
    def max_queue_size(self) -> int:
        # Return the max_queue_size of the underlying default sink,
        # as it's always present.
        return self.__default_queue_sink.max_queue_size


class DelegatingMultiprocessQueueSource(
    MultiprocessQueueSource[QueueItemType]
):
    """
    A source that delegates to either a default or Torch queue
    based on a coordination message received from the default queue.
    """

    def __init__(
        self,
        default_queue_source: MultiprocessQueueSource[QueueItemType],
        torch_queue_source: Optional[MultiprocessQueueSource[QueueItemType]],
    ):
        super().__init__(
            default_queue_source.max_queue_size if default_queue_source else 0
        )
        self.__default_queue_source = default_queue_source
        self.__torch_queue_source = torch_queue_source
        self.__selected_queue_source: Optional[
            MultiprocessQueueSource[QueueItemType]
        ] = None
        self._closed_flag = (
            False  # Renaming from __closed_flag for consistency if needed
        )
        self.on_close = MethodDelegate[[]]()  # type: ignore[attr-defined]

    def __receive_coordination_message(
        self, timeout: Optional[float] = None, use_get_or_none: bool = False
    ) -> bool:
        coordination_msg: Optional[Any] = None
        try:
            if use_get_or_none:
                coordination_msg = self.__default_queue_source.get_or_none()
            else:
                coordination_msg = self.__default_queue_source.get_blocking(
                    timeout=timeout
                )
        except (
            queue.Empty
        ):  # This is the standard exception for timeout with get_blocking
            return False
        except (
            QueueTimeoutError
        ):  # Custom exception, ensure it's handled if raised by underlying
            return False

        if coordination_msg is None:
            return False

        if not isinstance(coordination_msg, str):
            raise RuntimeError(
                f"Received unexpected coordination data type: {type(coordination_msg)}. Expected str."
            )

        if coordination_msg == "USE_TORCH":
            if self.__torch_queue_source is None:
                raise RuntimeError(
                    "Coordination specified Torch path, but no Torch queue source is configured."
                )
            self.__selected_queue_source = self.__torch_queue_source
        elif coordination_msg == "USE_DEFAULT":
            self.__selected_queue_source = self.__default_queue_source
        else:
            raise RuntimeError(
                f"Invalid coordination message: {coordination_msg}"
            )

        return True

    def get_blocking(
        self, timeout: Optional[float] = None
    ) -> Optional[QueueItemType]:
        if self.closed and self.empty():  # Properties used here
            return None

        if self.__selected_queue_source is None:
            if not self.__receive_coordination_message(
                timeout=timeout, use_get_or_none=False
            ):
                # Coordination failed (timeout or no message)
                if (
                    self.closed
                ):  # Check if closed after attempting coordination
                    return None
                # If coordination timed out, this is effectively a timeout for get_blocking itself.
                # Re-raise QueueTimeoutError or allow specific timeout exception from underlying queue
                raise QueueTimeoutError(
                    "Timeout waiting for coordination message or data."
                )

        assert (
            self.__selected_queue_source is not None
        ), "Selected queue source not set after coordination."

        try:
            item = self.__selected_queue_source.get_blocking(timeout=timeout)
            if item is None and self.closed and self.empty():
                return None
            return item
        except queue.Empty:  # Handle standard timeout exception
            if self.closed and self.empty():
                return None
            raise QueueTimeoutError(
                "Timeout getting item from selected queue."
            )  # Re-wrap or raise directly
        except QueueTimeoutError:  # Handle custom timeout exception
            if self.closed and self.empty():
                return None
            raise

    def get_or_none(self) -> Optional[QueueItemType]:
        if self.closed and self.empty():
            return None

        if self.__selected_queue_source is None:
            if not self.__receive_coordination_message(use_get_or_none=True):
                return None  # No coordination message available non-blockingly

        if self.__selected_queue_source is None:  # Still no selected source
            return None

        item = self.__selected_queue_source.get_or_none()
        if item is None and self.closed and self.empty():
            return None
        return item

    def close(self) -> None:
        if not self._closed_flag:
            self._closed_flag = True
            self.__default_queue_source.close()
            if self.__torch_queue_source is not None:
                self.__torch_queue_source.close()
            self.on_close.invoke()

    @property
    def closed(self) -> bool:
        if self._closed_flag:
            return True
        # More robust: if selected, defer to it. If not, check default.
        if self.__selected_queue_source:
            return (
                self.__selected_queue_source.closed
                and self.__selected_queue_source.empty()
            )
        if (
            self.__default_queue_source.closed
            and self.__default_queue_source.empty()
        ):
            # No coordination possible and default is drained
            return True
        return False

    @property
    def empty(self) -> bool:
        if self.__selected_queue_source:
            return self.__selected_queue_source.empty()
        # Before coordination, emptiness refers to the default queue (coordination channel)
        return self.__default_queue_source.empty()

    @property
    def max_queue_size(self) -> int:
        return self.__default_queue_source.max_queue_size

    def __len__(self) -> int:
        if self.__selected_queue_source:
            return len(self.__selected_queue_source)
        return len(
            self.__default_queue_source
        )  # Length of coordination queue before selection

    def __iter__(self) -> "DelegatingMultiprocessQueueSource[QueueItemType]":
        return self

    def __next__(self) -> QueueItemType:
        if self.__selected_queue_source is None:
            # Blocking wait for coordination for iterator
            if not self.__receive_coordination_message(
                timeout=None, use_get_or_none=False
            ):
                raise StopIteration("Coordination failed or queue closed.")

        assert (
            self.__selected_queue_source is not None
        ), "Iterator: Selected source None after coord."

        while True:  # Loop to ensure StopIteration on true end-of-queue
            if self.closed and self.empty():  # Check overall status
                raise StopIteration
            try:
                # Attempt to get item, ideally blocks indefinitely if timeout=None
                item = self.__selected_queue_source.get_blocking(
                    timeout=0.1
                )  # Short poll for responsiveness
                if item is not None:
                    return item
                # If item is None, and not closed/empty yet, means underlying get_blocking timed out
                # but queue is not declared finished by `closed` and `empty` properties.
                # Continue loop to re-evaluate `closed` and `empty` and try again.
                if self.closed and self.empty():  # Re-check after get attempt
                    raise StopIteration

            except (
                queue.Empty,
                QueueTimeoutError,
            ):  # Timeout from get_blocking
                if self.closed and self.empty():
                    raise StopIteration
                # If just a timeout but not end of stream, continue polling.
                # A short sleep can be added here if cpu usage is a concern.
                # time.sleep(0.001)
            # Any other exception will propagate and stop iteration.


class DelegatingMultiprocessQueueFactory(
    MultiprocessQueueFactory[QueueItemType], Generic[QueueItemType]
):
    """
    A factory that creates pairs of DelegatingMultiprocessQueueSink and
    DelegatingMultiprocessQueueSource. It sets up default and, if available,
    Torch IPC queues, and the delegating sink/source coordinate their usage.
    """

    def __init__(
        self, max_queue_size: int = DEFAULT_MAX_QUEUE_SIZE
    ) -> None:  # Added max_queue_size
        """Initializes the DelegatingMultiprocessQueueFactory."""
        super().__init__(max_queue_size)  # Pass max_queue_size to parent
        # Manager instances are now created per factory call if needed, not stored in self.

    # shutdown() method is removed as the factory no longer owns a persistent manager.
    # __get_manager() method is removed for the same reason.

    def create_queues(
        self,
    ) -> Tuple[
        MultiprocessQueueSink[QueueItemType],
        MultiprocessQueueSource[QueueItemType],
    ]:
        """
        Creates a pair of delegating queue sink and source.
        """
        # 1. Instantiate DefaultMultiprocessQueueFactory
        # Default factory needs its own manager
        default_manager = multiprocessing.Manager()
        default_factory = DefaultMultiprocessQueueFactory[QueueItemType](
            max_queue_size=self.max_queue_size, manager=default_manager
        )
        default_sink, default_source = default_factory.create_queues()

        torch_sink: Optional[MultiprocessQueueSink[QueueItemType]] = None
        torch_source: Optional[MultiprocessQueueSource[QueueItemType]] = None

        if is_torch_available():  # Check using the function
            try:
                # Torch factory needs its own torch manager
                assert (
                    _torch_mp_module is not None
                )  # Should be true if is_torch_available
                torch_manager = _torch_mp_module.Manager()
                torch_factory = TorchMultiprocessQueueFactory[QueueItemType](
                    max_queue_size=self.max_queue_size, manager=torch_manager
                )
                torch_sink, torch_source = torch_factory.create_queues()
            except (
                ImportError
            ):  # Should ideally not happen if is_torch_available is robust
                pass  # torch_sink, torch_source remain None
            except Exception:  # Catch other errors during torch queue creation
                # Log this? For now, fallback to default path.
                pass  # torch_sink, torch_source remain None

        delegating_sink = DelegatingMultiprocessQueueSink[QueueItemType](
            default_queue_sink=default_sink, torch_queue_sink=torch_sink
        )
        delegating_source = DelegatingMultiprocessQueueSource[QueueItemType](
            default_queue_source=default_source,
            torch_queue_source=torch_source,
        )
        return delegating_sink, delegating_source
