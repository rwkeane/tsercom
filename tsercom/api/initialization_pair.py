import dataclasses
from concurrent.futures import Future
from typing import Generic, TypeVar
from tsercom.api.runtime_handle import RuntimeHandle
from tsercom.runtime.runtime_initializer import RuntimeInitializer

# Type variables for generic typing
TDataType = TypeVar("TDataType")
TEventType = TypeVar("TEventType")


@dataclasses.dataclass
class InitializationPair(Generic[TDataType, TEventType]):
    """A container holding a future for a RuntimeHandle and its initializer.

    This class pairs a `RuntimeInitializer` with the `Future` that will eventually
    hold the `RuntimeHandle` created by that initializer. This is useful for
    managing asynchronous initialization of runtimes.
    """

    # Original __init__ order: handle_future: Future[RuntimeHandle[TDataType, TEventType]], initializer: RuntimeInitializer[TDataType, TEventType]
    handle_future: Future[RuntimeHandle[TDataType, TEventType]]
    initializer: RuntimeInitializer[TDataType, TEventType]
