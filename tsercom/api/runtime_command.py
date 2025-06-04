"""Defines commands that can be issued to a runtime environment."""

from enum import Enum


class RuntimeCommand(Enum):
    """Enumerates the basic commands that can be sent to a runtime.

    These commands control the lifecycle of a runtime instance.
    """

    START = 1
    STOP = 2
