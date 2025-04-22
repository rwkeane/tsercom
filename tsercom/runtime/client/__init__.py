from typing import TypeVar
from tsercom.data.exposed_data import ExposedData
from tsercom.runtime.running_runtime import RunningRuntime

from tsercom.runtime.client.client_runtime_initializer import (
    ClientRuntimeInitializer,
)
from tsercom.runtime.client.client_runtime_manager import ClientRuntimeManager


TDataType = TypeVar("TDataType", bound=ExposedData)
TEventType = TypeVar("TEventType")
ClientRunningRuntime = RunningRuntime[
    TDataType, TEventType, ClientRuntimeInitializer
]

__all__ = [
    "ClientRuntimeInitializer",
    "ClientRuntimeManager",
    "ClientRunningRuntime",
]
