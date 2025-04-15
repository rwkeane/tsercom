from typing import Any

from tsercom.runtime.runtime_manager import RuntimeManager
from tsercom.runtime.server.server_remote_main import server_remote_main
from tsercom.runtime.server.server_runtime_initializer import (
    ServerRuntimeInitializer,
)


class ServerRuntimeManager(RuntimeManager[ServerRuntimeInitializer[Any, Any]]):
    def __init__(self):
        super().__init__(out_of_process_main=server_remote_main)
