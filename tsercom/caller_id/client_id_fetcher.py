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
                grpc_response = await self.__stub.GetId(GetIdRequest())
                if not isinstance(grpc_response, GetIdResponse):
                    raise TypeError(
                        f"Expected GetIdResponse from stub.GetId, but got {type(grpc_response).__name__}."
                    )
                self.__id = CallerIdentifier.try_parse(grpc_response.id)
                if self.__id is None:
                    raise ValueError(
                        f"Failed to parse CallerIdentifier from GetIdResponse. Received ID: {grpc_response.id}"
                    )
            return self.__id
