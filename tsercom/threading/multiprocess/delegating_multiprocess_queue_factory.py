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
from typing import (
    TypeVar,
    Optional,
    Tuple,
    Any,
    cast,
    TypeAlias,
)
from types import ModuleType

from tsercom.threading.multiprocess.multiprocess_queue_sink import (
    MultiprocessQueueSink,
)
from tsercom.threading.multiprocess.multiprocess_queue_source import (
    MultiprocessQueueSource,
)
from tsercom.threading.multiprocess.multiprocess_queue_factory import (
    MultiprocessQueueFactory,
)

# TorchMultiprocessQueueFactory import is not needed for this approach
# from tsercom.threading.multiprocess.torch_multiprocess_queue_factory import (
#     TorchMultiprocessQueueFactory,
# )


DEFAULT_MAX_QUEUE_SIZE = 0

_torch_mp_module: Optional[ModuleType] = None
_torch_available: bool
_torch_tensor_type: Optional[TypeAlias] = None

try:
    import torch
    import torch.multiprocessing as torch_mp_imported

    _torch_mp_module = torch_mp_imported
    _torch_available = True
    _torch_tensor_type = torch.Tensor
except ImportError:
    _torch_available = False

QueueItemType = TypeVar("QueueItemType")


def is_torch_available() -> bool:
    """Checks if PyTorch and its multiprocessing extensions are available."""
    return _torch_available and _torch_mp_module is not None


COORDINATION_USE_TORCH = "USE_TORCH"
COORDINATION_USE_DEFAULT = "USE_DEFAULT"


class DelegatingMultiprocessQueueSink(MultiprocessQueueSink[QueueItemType]):
    def __init__(
        self,
        default_queue_sink: MultiprocessQueueSink[QueueItemType],
        torch_queue_sink: Optional[MultiprocessQueueSink[QueueItemType]],
    ):
        super().__init__(default_queue_sink.underlying_queue)
        self.__default_queue_sink = default_queue_sink
        self.__torch_queue_sink = torch_queue_sink
        self.__selected_queue_sink: Optional[
            MultiprocessQueueSink[QueueItemType]
        ] = None
        self.__coordination_sent: bool = False
        self._closed_flag = False
        # self.on_close = MethodDelegate[[]]()

    def __perform_initial_put(
        self,
        obj: QueueItemType,
        timeout: Optional[float] = None,
        use_nowait: bool = False,
    ) -> bool:
        actual_data = getattr(obj, "data", obj)
        use_torch_path = False
        if (
            self.__torch_queue_sink is not None
            and is_torch_available()
            and _torch_tensor_type is not None
            and isinstance(actual_data, _torch_tensor_type)
        ):
            use_torch_path = True

        coordination_message: Any = (
            COORDINATION_USE_TORCH
            if use_torch_path
            else COORDINATION_USE_DEFAULT
        )

        if use_nowait:
            if not self.__default_queue_sink.put_nowait(coordination_message):
                return False
        else:
            if not self.__default_queue_sink.put_blocking(
                coordination_message, timeout=timeout
            ):
                return False

        if use_torch_path:
            assert self.__torch_queue_sink is not None
            self.__selected_queue_sink = self.__torch_queue_sink
        else:
            self.__selected_queue_sink = self.__default_queue_sink

        assert self.__selected_queue_sink is not None
        if use_nowait:
            if not self.__selected_queue_sink.put_nowait(obj):
                return False
        else:
            if not self.__selected_queue_sink.put_blocking(
                obj, timeout=timeout
            ):
                return False
        self.__coordination_sent = True
        return True

    def put_blocking(
        self, obj: QueueItemType, timeout: Optional[float] = None
    ) -> bool:
        if self._closed_flag:
            raise RuntimeError("Sink closed.")
        if not self.__coordination_sent:
            return self.__perform_initial_put(
                obj, timeout=timeout, use_nowait=False
            )
        assert (
            self.__selected_queue_sink is not None
        ), "Selected sink is None despite coordination."
        return self.__selected_queue_sink.put_blocking(obj, timeout=timeout)

    def put_nowait(self, obj: QueueItemType) -> bool:
        if self._closed_flag:
            raise RuntimeError("Sink closed.")
        if not self.__coordination_sent:
            return self.__perform_initial_put(obj, use_nowait=True)
        assert (
            self.__selected_queue_sink is not None
        ), "Selected sink is None despite coordination."
        return self.__selected_queue_sink.put_nowait(obj)

    def close(self) -> None:
        if not self._closed_flag:
            self._closed_flag = True
            if self.__default_queue_sink:
                self.__default_queue_sink.close()
            if self.__torch_queue_sink:
                self.__torch_queue_sink.close()
            # if hasattr(self, "on_close") and isinstance(
            #     self.on_close, MethodDelegate
            # ):
            #     self.on_close.invoke()

    @property
    def closed(self) -> bool:
        return self._closed_flag

    @property
    def max_queue_size(self) -> int:
        if self.__default_queue_sink and hasattr(
            self.__default_queue_sink, "max_queue_size"
        ):
            return cast(
                int, getattr(self.__default_queue_sink, "max_queue_size")
            )
        return 0


class DelegatingMultiprocessQueueSource(
    MultiprocessQueueSource[QueueItemType]
):
    def __init__(
        self,
        default_queue_source: MultiprocessQueueSource[QueueItemType],
        torch_queue_source: Optional[MultiprocessQueueSource[QueueItemType]],
    ):
        super().__init__(default_queue_source.underlying_queue)
        self.__default_queue_source = default_queue_source
        self.__torch_queue_source = torch_queue_source
        self.__selected_queue_source: Optional[
            MultiprocessQueueSource[QueueItemType]
        ] = None
        self._closed_flag = False
        # self.on_close = MethodDelegate[[]]()

    def __receive_coordination_message(
        self, timeout: Optional[float] = None, use_get_or_none: bool = False
    ) -> bool:
        coordination_msg: Any = None
        try:
            if use_get_or_none:
                coordination_msg = self.__default_queue_source.get_or_none()
            else:
                coordination_msg = self.__default_queue_source.get_blocking(
                    timeout=timeout
                )
        except queue.Empty:
            return False

        if coordination_msg is None:
            return False
        if not isinstance(coordination_msg, str):
            raise RuntimeError(
                f"Invalid coordination message type: {type(coordination_msg)}"
            )

        if coordination_msg == COORDINATION_USE_TORCH:
            if not self.__torch_queue_source:
                raise RuntimeError(
                    "Torch path requested but no torch source configured."
                )
            self.__selected_queue_source = self.__torch_queue_source
        elif coordination_msg == COORDINATION_USE_DEFAULT:
            self.__selected_queue_source = self.__default_queue_source
        else:
            raise RuntimeError(
                f"Invalid coordination message: {coordination_msg}"
            )
        return True

    def get_blocking(
        self, timeout: Optional[float] = None
    ) -> Optional[QueueItemType]:
        if self.closed and self.empty:
            return None
        if not self.__selected_queue_source:
            if not self.__receive_coordination_message(timeout=timeout):
                if self.closed and self.empty:
                    return None
                raise queue.Empty("Timeout waiting for coordination or data.")

        assert (
            self.__selected_queue_source is not None
        ), "Selected source is None after coordination attempt."
        try:
            return self.__selected_queue_source.get_blocking(timeout=timeout)
        except queue.Empty:
            if self.closed and self.empty:
                return None
            raise

    def get_or_none(self) -> Optional[QueueItemType]:
        if self.closed and self.empty:
            return None
        if not self.__selected_queue_source:
            if not self.__receive_coordination_message(use_get_or_none=True):
                return None
        if not self.__selected_queue_source:
            return None
        return self.__selected_queue_source.get_or_none()

    def close(self) -> None:
        if not self._closed_flag:
            self._closed_flag = True
            if self.__default_queue_source:
                self.__default_queue_source.close()
            if self.__torch_queue_source:
                self.__torch_queue_source.close()
            # if hasattr(self, "on_close") and isinstance(
            #     self.on_close, MethodDelegate
            # ):
            #     self.on_close.invoke()

    @property
    def closed(self) -> bool:
        if self._closed_flag:
            return True
        if self.__selected_queue_source:
            return self.__selected_queue_source.closed
        if self.__default_queue_source and self.__default_queue_source.closed:
            return True
        return False

    @property
    def empty(self) -> bool:
        if not self.__selected_queue_source:
            if self.__default_queue_source:
                return self.__default_queue_source.empty
            return True
        return self.__selected_queue_source.empty

    def __len__(self) -> int:
        if self.__selected_queue_source:
            return len(self.__selected_queue_source)
        if self.__default_queue_source:
            return len(self.__default_queue_source)
        return 0

    @property
    def max_queue_size(self) -> int:
        if self.__default_queue_source and hasattr(
            self.__default_queue_source, "max_queue_size"
        ):
            return cast(
                int, getattr(self.__default_queue_source, "max_queue_size", 0)
            )
        return 0

    def __iter__(self) -> "DelegatingMultiprocessQueueSource[QueueItemType]":
        return self

    def __next__(self) -> QueueItemType:
        if not self.__selected_queue_source:
            if not self._DelegatingMultiprocessQueueSource__receive_coordination_message(  # type: ignore [attr-defined]
                timeout=None, use_get_or_none=False
            ):
                raise StopIteration("Coordination failed or queue closed.")

        assert (
            self.__selected_queue_source is not None
        ), "Iterator: Selected source None after coord."

        while True:
            if self.closed and self.empty:
                raise StopIteration
            try:
                item = self.__selected_queue_source.get_blocking(timeout=0.1)
                if item is not None:
                    return item
                if self.closed and self.empty:
                    raise StopIteration
            except queue.Empty:
                if self.closed and self.empty:
                    raise StopIteration


class DelegatingMultiprocessQueueFactory(
    MultiprocessQueueFactory[QueueItemType]
):
    def __init__(
        self,
        max_queue_size: int = DEFAULT_MAX_QUEUE_SIZE,
    ) -> None:
        super().__init__()
        self.max_queue_size = max_queue_size

    def create_queues(
        self,
    ) -> Tuple[
        DelegatingMultiprocessQueueSink[QueueItemType],
        DelegatingMultiprocessQueueSource[QueueItemType],
    ]:
        max_size_for_mp_queue = (
            self.max_queue_size if self.max_queue_size > 0 else -1
        )

        default_manager = multiprocessing.Manager()
        actual_default_queue = default_manager.Queue(
            maxsize=max_size_for_mp_queue
        )
        default_sink = MultiprocessQueueSink[QueueItemType](
            actual_default_queue
        )
        default_source = MultiprocessQueueSource[QueueItemType](
            actual_default_queue
        )

        torch_sink_final: Optional[MultiprocessQueueSink[QueueItemType]] = None
        torch_source_final: Optional[
            MultiprocessQueueSource[QueueItemType]
        ] = None

        if is_torch_available():
            assert _torch_mp_module is not None
            actual_torch_queue = None
            if _torch_mp_module:
                try:
                    # Explicitly use 'fork' context for PyTorch queues
                    torch_fork_ctx = _torch_mp_module.get_context("fork")
                    actual_torch_queue = torch_fork_ctx.Queue(
                        maxsize=(
                            self.max_queue_size
                            if self.max_queue_size > 0
                            else -1
                        )
                    )
                except Exception:
                    # If context/queue creation fails, sink/source remain None.
                    # This allows the program to potentially proceed with default queues only.
                    # Ideally, log this failure.
                    pass

            if actual_torch_queue is not None:
                torch_sink_final = MultiprocessQueueSink[QueueItemType](
                    actual_torch_queue
                )
                torch_source_final = MultiprocessQueueSource[QueueItemType](
                    actual_torch_queue
                )

        delegating_sink = DelegatingMultiprocessQueueSink[QueueItemType](
            default_sink, torch_sink_final
        )
        delegating_source = DelegatingMultiprocessQueueSource[QueueItemType](
            default_source, torch_source_final
        )
        return delegating_sink, delegating_source
