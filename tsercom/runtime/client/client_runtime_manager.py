from typing import Any

from tsercom.runtime.client.client_remote_main import client_remote_main
from tsercom.runtime.client.client_runtime_initializer import ClientRuntimeInitializer
from tsercom.runtime.runtime_manager import RuntimeManager


class ClientRuntimeManager(RuntimeManager[ClientRuntimeInitializer[Any, Any]]):
    def __init__(self):
        super().__init__(out_of_process_main = client_remote_main)