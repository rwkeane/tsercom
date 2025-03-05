import asyncio

from tsercom.caller_id.caller_identifier import CallerIdentifier


class CallerIdentifierWaiter:
    def __init__(self) -> None:
        self.__caller_id: CallerIdentifier | None = None

        self.__barrier = asyncio.Event()

    async def get_caller_id_async(self) -> CallerIdentifier:
        await self.__barrier.wait()
        return self.__caller_id  # type: ignore

    async def has_id(self) -> bool:
        return self.__caller_id is None

    async def set_caller_id(self, caller_id: CallerIdentifier) -> None:
        assert self.__caller_id is None
        self.__caller_id = caller_id
        self.__barrier.set()
