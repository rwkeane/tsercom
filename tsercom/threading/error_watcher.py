from abc import ABC, abstractmethod


# Defines an interface for objects that can monitor for and report exceptions.
class ErrorWatcher(ABC):
    """
    Abstract base class for error watching.

    Subclasses should implement the logic to monitor for exceptions
    and report them.
    """

    @abstractmethod
    def run_until_exception(self) -> None:
        """
        Runs until an exception is seen, at which point it will be thrown.

        This method is intended to be blocking. It should only return
        if an exception occurs that needs to be propagated.

        Raises:
            Exception: Any exception encountered during the monitoring process.
        """
        pass
