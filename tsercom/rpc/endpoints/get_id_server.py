from typing import Callable, Optional

from tsercom.caller_id.proto import GetIdRequest, GetIdResponse
from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.rpc.connection.client_reconnection_handler import ClientReconnectionManager


class AsyncGetIdServer:
    """
    This class defines the GetId() gRPC Method, as required by a number of
    different gRPC Services.
    """
    def __init__(
            self,
            on_id_created : Optional[Callable[[CallerIdentifier], None]] = None,
            on_disconnect_handler : Optional[ClientReconnectionManager] = None,
            *args,
            **kwargs):
        self.__callback = on_id_created
        self.__on_disconnect_handler = on_disconnect_handler

        super().__init__(*args, **kwargs)
    
    async def GetId(self, request : GetIdRequest, context):
        id = CallerIdentifier()
        if not self.__callback is None:
            self.__callback(id)
        try:
            return GetIdResponse(id = id.to_grpc_type())
        except Exception as e:
            if not self.__on_disconnect_handler is None:
                self.__on_disconnect_handler(e)
            if isinstance(e, AssertionError):
                raise