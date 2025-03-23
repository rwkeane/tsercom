from abc import ABC, abstractmethod


class ErrorWatcher(ABC):
    @abstractmethod
    def run_until_exception(self) -> None:
        """
        Runs until an exception is seen, at which point it will be thrown.
        """
        pass
