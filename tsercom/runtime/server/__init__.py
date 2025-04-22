from typing import TypeVar
from tsercom.data.exposed_data import ExposedData
from tsercom.runtime.running_runtime import RunningRuntime

from tsercom.runtime.server.server_runtime_initializer import (
    ServerRuntimeInitializer,
)
from tsercom.runtime.server.server_runtime_manager import ServerRuntimeManager

TDataType = TypeVar("TDataType", bound=ExposedData)
TEventType = TypeVar("TEventType")
ServerRunningRuntime = RunningRuntime[
    TDataType, TEventType, ServerRuntimeInitializer
]

__all__ = [
    "ServerRuntimeInitializer",
    "ServerRuntimeManager",
    "ServerRunningRuntime",
]
