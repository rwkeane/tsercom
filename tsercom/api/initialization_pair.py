"""Defines data structures for runtime initialization pairs."""

import dataclasses
from concurrent.futures import Future
from typing import Generic, TypeVar

from tsercom.api.runtime_handle import RuntimeHandle
from tsercom.runtime.runtime_initializer import RuntimeInitializer
from tsercom.data.exposed_data import ExposedData


DataTypeT = TypeVar("DataTypeT", bound=ExposedData)
EventTypeT = TypeVar("EventTypeT")


@dataclasses.dataclass
class InitializationPair(Generic[DataTypeT, EventTypeT]):
    """A container holding a future for a RuntimeHandle and its initializer."""

    handle_future: Future[RuntimeHandle[DataTypeT, EventTypeT]]
    initializer: RuntimeInitializer[DataTypeT, EventTypeT]
