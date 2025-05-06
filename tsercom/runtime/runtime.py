from abc import ABC, abstractmethod

from tsercom.util.stopable import Stopable


class Runtime(Stopable, ABC):
    """
    Specifies an instance that starts up and then later can be stopped.
    """

    @abstractmethod
    async def start_async(self):
        """
        Starts running this instance.
        """
        pass