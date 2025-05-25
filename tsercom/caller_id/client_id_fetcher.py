"""Defines ClientIdFetcher for lazily fetching a CallerIdentifier via gRPC."""

import asyncio
from typing import Any, Optional

from tsercom.caller_id.proto import GetIdRequest, GetIdResponse
from tsercom.caller_id.caller_identifier import CallerIdentifier


class ClientIdFetcher:
    """Fetches and caches a client ID (CallerIdentifier) from a gRPC service.

    This class provides a method `get_id_async` to retrieve a `CallerIdentifier`.
    The first call to this method makes a gRPC `GetId` request to the provided
    stub and caches the result. Subsequent calls return the cached ID.
    Access is thread-safe using an asyncio.Lock.
    """

    def __init__(self, stub: Any) -> None:
        """Initializes the ClientIdFetcher.

        Args:
            stub: The gRPC stub that provides the `GetId` method.
                  Expected to have a method like `async def GetId(GetIdRequest) -> GetIdResponse`.
        """
        self.__stub: Any = stub
        self.__id: Optional[CallerIdentifier] = None
        # Lock to ensure thread-safe lazy initialization of the ID.
        self.__lock = asyncio.Lock()

    async def get_id_async(self) -> Optional[CallerIdentifier]:
        """Lazily fetches and returns the client ID.

        The first time this method is called, it makes a gRPC call to the
        `GetId` method of the stub provided during initialization. The ID is
        then cached. Subsequent calls return the cached ID. If the gRPC call
        fails or the returned ID is invalid, None is returned.

        Returns:
            The `CallerIdentifier` if successfully fetched and parsed,
            otherwise `None`.

        Raises:
            Exception: Any exception that might occur during the gRPC call
                       or subsequent processing if not caught by the broad
                       exception handler within the method. The method attempts
                       to catch common issues and return None.
        """
        try:
            # Ensure only one coroutine attempts to fetch the ID at a time.
            async with self.__lock:
                # If the ID hasn't been fetched yet (lazy loading).
                if self.__id is None:
                    id_response: GetIdResponse = await self.__stub.GetId(GetIdRequest())
                    
                    # Ensure the response is of the expected type.
                    assert isinstance(id_response, GetIdResponse), \
                        f"Expected GetIdResponse, got {type(id_response)}"
                    
                    # CallerIdentifier.try_parse can return None if the string is invalid.
                    if id_response.id and id_response.id.id:
                        self.__id = CallerIdentifier.try_parse(id_response.id.id)
                    else:
                        # Handle cases where response or its nested id is missing.
                        self.__id = None 
                        # Optionally log a warning here if this case is unexpected.
                
                # This could be None if fetching or parsing failed.
                return self.__id
        except Exception as e: # pylint: disable=broad-except
            # Broad exception catch for any issues during RPC call or processing.
            # In a production system, more specific error handling and logging would be preferred.
            print(f"Error fetching client ID: {e}") # Basic error logging.
            return None
