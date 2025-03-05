from abc import ABC, abstractmethod


class Stopable(ABC):
    @abstractmethod
    async def stop(self) -> None:
        pass
