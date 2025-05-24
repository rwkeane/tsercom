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

    async def get_id_async(self) -> CallerIdentifier | None: # Return type can be None
        try:
            async with self.__lock:
                if self.__id is None:
                    # Make the RPC call
                    id_response = await self.__stub.GetId(GetIdRequest()) # type: ignore
                    assert isinstance(id_response, GetIdResponse)
                    # try_parse can return None if the id string is invalid
                    self.__id = CallerIdentifier.try_parse(id_response.id.id)
                    # If parsing fails (self.__id is None), self.__id will be None,
                    # and that will be returned.
                return self.__id
        except Exception: # pylint: disable=broad-except
            # TODO: Log this error
            # print(f"Error fetching client ID: {e}")
            return None
