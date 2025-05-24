"""Defines commands that can be issued to a runtime environment."""

from enum import Enum


class RuntimeCommand(Enum):
    """Enumerates the basic commands that can be sent to a runtime.

    These commands control the lifecycle of a runtime instance.
    """
    # Command to start the runtime.
    kStart = 1
    # Command to stop the runtime.
    kStop = 2
