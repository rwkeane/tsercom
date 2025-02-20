from abc import abstractmethod
from typing import Optional


class ClientReconnectionManager:
    @abstractmethod
    async def _on_disconnect(self, error : Optional[Exception] = None):
        pass