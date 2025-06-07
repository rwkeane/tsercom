"""Provides an asynchronous gRPC server component for the GetId method."""

import logging
from typing import Callable, Optional

import grpc

from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.caller_id.proto import GetIdRequest, GetIdResponse
from tsercom.rpc.connection.client_reconnection_handler import (
    ClientReconnectionManager,
)


class AsyncGetIdServer:
    """Implements the GetId() gRPC method asynchronously.

    This server component can be part of various gRPC services that need to
    provide a unique CallerIdentifier to clients. It allows optional callbacks
    for when an ID is created and when a disconnection occurs.
    """

    def __init__(
        self,
        on_id_created: Optional[Callable[[CallerIdentifier], None]] = None,
        on_disconnect_handler: Optional[ClientReconnectionManager] = None,
    ) -> None:
        """Initializes the AsyncGetIdServer.

        Args:
            on_id_created: Optional callback invoked with the new CallerIdentifier
                           when it's generated.
            on_disconnect_handler: Optional ClientReconnectionManager to be
                                   notified on disconnection errors.
        """
        self.__callback = on_id_created
        self.__on_disconnect_handler = on_disconnect_handler
        # Since this class doesn't explicitly inherit from another,
        # super() refers to object. object.__init__ takes no arguments.
        super().__init__()

    async def GetId(
        self, request: GetIdRequest, context: grpc.aio.ServicerContext
    ) -> GetIdResponse:
        """Handles the GetId gRPC request.

        Generates a new CallerIdentifier, optionally invokes a callback with it,
        and returns it in a GetIdResponse. Handles exceptions and potential
        disconnection notifications.

        Args:
            request: The GetIdRequest message.
            context: The gRPC servicer context.

        Returns:
            A GetIdResponse containing the new CallerIdentifier.
        """
        assert isinstance(request, GetIdRequest), "Invalid request type"
        new_id = (
            CallerIdentifier.random()
        )  # Renamed from 'id' to avoid shadowing built-in
        if self.__callback is not None:
            self.__callback(new_id)

        try:
            return GetIdResponse(id=new_id.to_grpc_type())
        except Exception as e:
            if self.__on_disconnect_handler is not None:
                await self.__on_disconnect_handler._on_disconnect(e)
            if isinstance(e, AssertionError):
                raise
            logging.error(
                f"Error during GetId processing for context {context.peer() if context else 'Unknown'}: {e}",
                exc_info=True,
            )
            await context.abort(
                grpc.StatusCode.INTERNAL,
                "An internal error occurred while generating an ID.",
            )
            raise e
