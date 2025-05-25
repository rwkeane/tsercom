from abc import ABC, abstractmethod


class Stopable(ABC):
    """
    An abstract base class for objects that can be started and stopped.

    This class defines a common interface for managing the lifecycle of
    services or components that have a distinct running state and need a
    mechanism to be cleanly shut down.
    """

    @abstractmethod
    async def stop(self) -> None:
        """
        Stops this instance from running.
        """
        pass
