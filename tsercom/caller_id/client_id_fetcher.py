import asyncio

from tsercom.caller_id.proto import GetIdRequest, GetIdResponse
from tsercom.caller_id.caller_identifier import CallerIdentifier


class ClientIdFetcher:
    """
    This class provides a simple wrapper around the GetId() call that is
    supported by a number of different gRPC Services, allowing for a single
    GetId() call to be lazy-loaded and used as needed.
    """

    def __init__(self, stub) -> None:  # type: ignore
        self.__stub = stub

        self.__id: CallerIdentifier | None = None
        self.__lock = asyncio.Lock()

    async def get_id_async(self) -> CallerIdentifier:
        async with self.__lock:
            if self.__id is None:
                id = await self.__stub.GetId(GetIdRequest())
                assert isinstance(id, GetIdResponse)
                self.__id = CallerIdentifier.try_parse(id.id)
                assert self.__id is not None

            return self.__id
