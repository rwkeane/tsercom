from abc import ABC, abstractmethod


class Stopable(ABC):
    @abstractmethod
    async def stop(self) -> None:
        """
        Stops this instance from running.
        """
        pass
