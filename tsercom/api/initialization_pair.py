import dataclasses
from concurrent.futures import Future
from typing import Generic, TypeVar

from tsercom.api.runtime_handle import RuntimeHandle
from tsercom.runtime.runtime_initializer import RuntimeInitializer
from tsercom.data.exposed_data import ExposedData


TDataType = TypeVar("TDataType", bound=ExposedData)
TEventType = TypeVar("TEventType")


@dataclasses.dataclass
class InitializationPair(Generic[TDataType, TEventType]):
    """A container holding a future for a RuntimeHandle and its initializer."""

    handle_future: Future[RuntimeHandle[TDataType, TEventType]]
    initializer: RuntimeInitializer[TDataType, TEventType]
