from typing import Callable, Optional

import grpc

from tsercom.caller_id.proto import GetIdRequest, GetIdResponse
from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.rpc.connection.client_reconnection_handler import (
    ClientReconnectionManager,
)


class AsyncGetIdServer:
    """
    This class defines the GetId() gRPC Method, as required by a number of
    different gRPC Services.
    """

    def __init__(  # type: ignore
        self,
        on_id_created: Optional[Callable[[CallerIdentifier], None]] = None,
        on_disconnect_handler: Optional[ClientReconnectionManager] = None,
        *args,
        **kwargs,
    ) -> None:
        self.__callback = on_id_created
        self.__on_disconnect_handler = on_disconnect_handler

        super().__init__(*args, **kwargs)

    async def GetId(self, request: GetIdRequest, context : grpc.aio.ServicerContext) -> GetIdResponse:
        id = CallerIdentifier()
        if self.__callback is not None:
            self.__callback(id)
        try:
            return GetIdResponse(id=id.to_grpc_type())
        except Exception as e:
            if self.__on_disconnect_handler is not None:
                await self.__on_disconnect_handler._on_disconnect(e)
            if isinstance(e, AssertionError):
                raise
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION)
            return GetIdResponse()
