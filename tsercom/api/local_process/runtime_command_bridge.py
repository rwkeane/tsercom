"""RuntimeCommandBridge for relaying commands to a Runtime instance."""

import concurrent.futures
from functools import partial
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
        self.__state: Atomic[Optional[RuntimeCommand]] = Atomic[
            Optional[RuntimeCommand]
        ](None)

        self.__runtime: Optional[Runtime] = None
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
                raise RuntimeError(
                    "Runtime has already been set and cannot be changed."
                )
            self.__runtime = runtime

            pending_command = self.__state.get()
            if pending_command is None:
                return
            if pending_command == RuntimeCommand.START:
                run_on_event_loop(runtime.start_async)
            elif (
                pending_command == RuntimeCommand.STOP
            ):  # Still elif as it's mutually exclusive to START
                # Execute stop and wait for it to complete.
                future = run_on_event_loop(partial(runtime.stop, None))
                try:
                    future.result(timeout=5.0)
                except concurrent.futures.TimeoutError:
                    # Log or handle timeout if needed
                    pass  # Or raise an error, log, etc.

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
                self.__state.set(RuntimeCommand.START)
            else:
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
                self.__state.set(RuntimeCommand.STOP)
            else:
                # Execute stop and wait for it to complete.
                future = run_on_event_loop(partial(self.__runtime.stop, None))
                try:
                    future.result(timeout=5.0)
                except concurrent.futures.TimeoutError:
                    # Log or handle timeout if needed
                    pass  # Or raise an error, log, etc.
