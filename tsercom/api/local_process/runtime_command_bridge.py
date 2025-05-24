"""Defines the RuntimeCommandBridge for relaying commands to a Runtime instance."""

from threading import Lock
from typing import Optional

from tsercom.api.runtime_command import RuntimeCommand
from tsercom.runtime.runtime import Runtime
from tsercom.threading.aio.aio_utils import run_on_event_loop
from tsercom.threading.atomic import Atomic


class RuntimeCommandBridge:
    """A bridge to asynchronously relay start/stop commands to a Runtime.

    This class handles the scenario where commands (like start or stop) might
    be issued before the actual Runtime instance is available. It stores the
    last command and executes it once the Runtime is set.
    """
    def __init__(self) -> None:
        """Initializes the RuntimeCommandBridge."""
        # Stores the last command received if the runtime is not yet set.
        self.__state: Atomic[Optional[RuntimeCommand]] = Atomic[Optional[RuntimeCommand]](None)

        # The Runtime instance, to be set later.
        self.__runtime: Optional[Runtime] = None
        # Mutex to protect access to the __runtime attribute.
        self.__runtime_mutex: Lock = Lock()

    def set_runtime(self, runtime: Runtime) -> None:
        """Sets the Runtime instance and executes any pending command.

        This method should only be called once. If a command (start/stop) was
        issued before the runtime was set, that command is executed now.

        Args:
            runtime: The Runtime instance to be managed by this bridge.

        Raises:
            RuntimeError: If the runtime has already been set.
        """
        with self.__runtime_mutex:
            # Ensure runtime is set only once.
            if self.__runtime is not None:
                raise RuntimeError("Runtime has already been set and cannot be changed.")
            self.__runtime = runtime

            # Check for and execute any pending command.
            pending_command = self.__state.get()
            if pending_command is None:
                return
            elif pending_command == RuntimeCommand.kStart:
                # Asynchronously start the runtime.
                run_on_event_loop(runtime.start_async)
            elif pending_command == RuntimeCommand.kStop:
                # Asynchronously stop the runtime.
                run_on_event_loop(runtime.stop)
            
            # Clear the pending command after execution.
            self.__state.set(None)


    def start(self) -> None:
        """Requests the Runtime to start.

        If the Runtime instance is already set, it's started asynchronously.
        Otherwise, the start command is stored and executed when the
        Runtime is set.
        """
        with self.__runtime_mutex:
            if self.__runtime is None:
                # Runtime not yet available, store the command.
                self.__state.set(RuntimeCommand.kStart)
            else:
                # Runtime available, start it directly.
                run_on_event_loop(self.__runtime.start_async)

    def stop(self) -> None:
        """Requests the Runtime to stop.

        If the Runtime instance is already set, it's stopped asynchronously.
        Otherwise, the stop command is stored and executed when the
        Runtime is set.
        """
        with self.__runtime_mutex:
            if self.__runtime is None:
                # Runtime not yet available, store the command.
                self.__state.set(RuntimeCommand.kStop)
            else:
                # Runtime available, stop it directly.
                run_on_event_loop(self.__runtime.stop)
