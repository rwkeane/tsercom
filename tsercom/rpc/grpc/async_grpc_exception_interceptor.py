from typing import Awaitable, Callable
import grpc

from tsercom.threading.thread_watcher import ThreadWatcher

# NOTE: This class is in the |threading| directory to avoid weird circular
# dependencies having |threading| -> |rpc| -> |threading|.


class AsyncGrpcExceptionInterceptor(grpc.aio.ServerInterceptor):
    """
    A gRPC interceptor that handles exceptions in async server methods and
    forwards them to a provided callback.

    NOTE: This class is _REALLY_ reaching into the internals of GRPC's
    experimental APIs, so it's possible that a future update will break it, but
    unlikely given how closely these methods already mirror the non-async
    versions. If that happens, this class may need to be reworked.
    """
    def __init__(self, watcher : ThreadWatcher):
        self.__error_cb = watcher.on_exception_seen
        
        super().__init__()

    async def intercept_service(
        self,
        continuation: Callable[[grpc.HandlerCallDetails],
                                Awaitable[grpc.RpcMethodHandler]],
        handler_call_details: grpc.HandlerCallDetails,
    ) -> grpc.RpcMethodHandler:
        """
        Intercepts the RPC call, catching exceptions and invoking the callback.
        """

        # Call the continuation to get the RPC method handler.
        handler : grpc.RpcMethodHandler = \
                await continuation(handler_call_details)

        # If there's no handler, it means this RPC is not implemented.
        # Let gRPC handle it.
        if handler is None:
            return None

        # Wrap each of the handler's methods (unary_unary, unary_stream, etc.)
        # to catch exceptions that occur within them.
        if handler.unary_unary is not None:
            handler = handler._replace(unary_unary = self._wrap_unary_unary(
                    handler.unary_unary, handler_call_details))
        if handler.unary_stream is not None:
            handler = handler._replace(unary_stream = self._wrap_unary_stream(
                    handler.unary_stream, handler_call_details))
        if handler.stream_unary is not None:
            handler = handler._replace(stream_unary = self._wrap_stream_unary(
                    handler.stream_unary, handler_call_details))
        if handler.stream_stream is not None:
            handler = handler._replace(stream_stream = self._wrap_stream_stream(
                    handler.stream_stream, handler_call_details))

        return handler

    def _wrap_unary_unary(self, method, method_name):
        """Wraps a unary-unary RPC method."""
        async def wrapper(request, context):
            try:
                return await method(request, context)
            except Exception as e:
                await self._handle_exception(e, method_name, context)
            except Warning as e:
                await self._handle_exception(e, method_name, context)

        return wrapper

    def _wrap_unary_stream(self, method, method_name):
        """Wraps a unary-stream RPC method."""
        async def wrapper(request, context):
            try:
                async for response in method(request, context):
                    yield response
            except Exception as e:
                await self._handle_exception(e, method_name, context)
            except Warning as e:
                await self._handle_exception(e, method_name, context)

        return wrapper

    def _wrap_stream_unary(self, method, method_name):
        """Wraps a stream-unary RPC method."""
        async def wrapper(request_iterator, context):
            try:
                return await method(request_iterator, context)
            except Exception as e:
                await self._handle_exception(e, method_name, context)
            except Warning as e:
                await self._handle_exception(e, method_name, context)

        return wrapper

    def _wrap_stream_stream(self, method, method_name):
        """Wraps a stream-stream RPC method."""
        async def wrapper(request_iterator, context):
            try:
                async for response in method(request_iterator, context):
                    yield response
            except Exception as e:
                await self._handle_exception(e, method_name, context)
            except Warning as e:
                await self._handle_exception(e, method_name, context)

        return wrapper

    async def _handle_exception(self,
                                e: Exception,
                                method_name: str,
                                context: grpc.aio.ServicerContext):
        """Handles exceptions raised by RPC methods."""
        if issubclass(type(e), StopAsyncIteration):
            raise e
        if isinstance(e, AssertionError):
            raise e
        
        self.__error_cb(e)
        await context.abort(grpc.StatusCode.UNKNOWN, f"Exception: {e}")