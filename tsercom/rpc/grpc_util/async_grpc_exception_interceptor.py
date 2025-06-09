"""Provides an asynchronous gRPC server interceptor for centralized exception handling."""

from typing import Awaitable, Callable

import grpc
import grpc.aio  # Explicitly import grpc.aio

from tsercom.threading.thread_watcher import ThreadWatcher


class AsyncGrpcExceptionInterceptor(grpc.aio.ServerInterceptor):  # type: ignore[misc]
    """
    A gRPC interceptor that handles exceptions in async server methods and
    forwards them to a provided callback.

    NOTE: This class is _REALLY_ reaching into the internals of GRPC's
    experimental APIs, so it's possible that a future update will break it, but
    unlikely given how closely these methods already mirror the non-async
    versions. If that happens, this class may need to be reworked.
    """

    def __init__(self, watcher: ThreadWatcher):
        """Initializes the AsyncGrpcExceptionInterceptor.

        Args:
            watcher: A ThreadWatcher instance to report exceptions to.
        """
        self.__error_cb = watcher.on_exception_seen

        super().__init__()

    async def intercept_service(
        self,
        continuation: Callable[
            [grpc.HandlerCallDetails], Awaitable[grpc.RpcMethodHandler]
        ],
        handler_call_details: grpc.HandlerCallDetails,
    ) -> grpc.RpcMethodHandler:
        """
        Intercepts the RPC call, catching exceptions and invoking the callback.
        """

        handler: grpc.RpcMethodHandler = await continuation(
            handler_call_details
        )

        # If there's no handler, it means this RPC is not implemented.
        # Let gRPC handle it.
        if handler is None:
            return None

        if handler.unary_unary is not None:
            handler = handler._replace(
                unary_unary=self._wrap_unary_unary(
                    handler.unary_unary, handler_call_details
                )
            )
        if handler.unary_stream is not None:
            handler = handler._replace(
                unary_stream=self._wrap_unary_stream(
                    handler.unary_stream, handler_call_details
                )
            )
        if handler.stream_unary is not None:
            handler = handler._replace(
                stream_unary=self._wrap_stream_unary(
                    handler.stream_unary, handler_call_details
                )
            )
        if handler.stream_stream is not None:
            handler = handler._replace(
                stream_stream=self._wrap_stream_stream(
                    handler.stream_stream, handler_call_details
                )
            )

        return handler

    def _wrap_unary_unary(
        self,
        method: Callable[
            [object, grpc.aio.ServicerContext], Awaitable[object]
        ],
        method_name: grpc.HandlerCallDetails,
    ) -> Callable[[object, grpc.aio.ServicerContext], Awaitable[object]]:
        """Wraps a unary-unary RPC method to provide exception handling."""

        async def wrapper(
            request: object, context: grpc.aio.ServicerContext
        ) -> Awaitable[object]:
            try:
                return await method(request, context)  # type: ignore[return-value]
            except Exception as e:
                await self._handle_exception(e, method_name, context)
                raise  # Make it clear this path does not return normally
            except (
                Warning
            ) as e:  # PEP 8: E722 do not use bare 'except' -> but this is 'except Warning'
                await self._handle_exception(e, method_name, context)
                raise  # Make it clear this path does not return normally

        return wrapper

    def _wrap_unary_stream(
        self,
        method: Callable[
            [object, grpc.aio.ServicerContext],
            Awaitable[object],  # Actual handler might be an async generator
        ],
        method_name: grpc.HandlerCallDetails,
    ) -> Callable[
        [object, grpc.aio.ServicerContext], Awaitable[object]
    ]:  # Wrapper returns an Awaitable
        """Wraps a unary-stream RPC method to provide exception handling."""

        async def wrapper(request: object, context: grpc.aio.ServicerContext) -> Awaitable[object]:  # type: ignore
            try:
                # The original method for unary-stream is expected to be an async generator.
                # However, the type hint from grpc.RpcMethodHandler is Awaitable[object].
                # We iterate over it as if it's an async generator.
                async for response in method(request, context):  # type: ignore[attr-defined]
                    yield response
            except Exception as e:
                await self._handle_exception(e, method_name, context)
                raise  # Make it clear this path does not return normally
            except Warning as e:  # PEP 8: E722 do not use bare 'except'
                await self._handle_exception(e, method_name, context)
                raise  # Make it clear this path does not return normally

        return wrapper

    def _wrap_stream_unary(
        self,
        method: Callable[
            [object, grpc.aio.ServicerContext],
            Awaitable[object],  # Actual handler takes an async iterator
        ],
        method_name: grpc.HandlerCallDetails,
    ) -> Callable[[object, grpc.aio.ServicerContext], Awaitable[object]]:
        """Wraps a stream-unary RPC method to provide exception handling."""

        async def wrapper(
            request_iterator: object, context: grpc.aio.ServicerContext
        ) -> Awaitable[object]:  # Removed type: ignore from line 140
            try:
                # The original method for stream-unary expects an async iterator.
                return await method(request_iterator, context)  # type: ignore[return-value]
            except Exception as e:
                await self._handle_exception(e, method_name, context)
                raise  # Make it clear this path does not return normally
            except Warning as e:  # PEP 8: E722 do not use bare 'except'
                await self._handle_exception(e, method_name, context)
                raise  # Make it clear this path does not return normally

        return wrapper

    def _wrap_stream_stream(
        self,
        method: Callable[
            [object, grpc.aio.ServicerContext],
            Awaitable[
                object
            ],  # Actual handler is an async gen taking async iter
        ],
        method_name: grpc.HandlerCallDetails,
    ) -> Callable[
        [object, grpc.aio.ServicerContext], Awaitable[object]
    ]:  # Wrapper returns an Awaitable
        """Wraps a stream-stream RPC method to provide exception handling."""

        async def wrapper(request_iterator: object, context: grpc.aio.ServicerContext) -> Awaitable[object]:  # type: ignore
            try:
                # The original method for stream-stream is an async generator
                # that takes an async iterator.
                async for response in method(request_iterator, context):  # type: ignore[attr-defined]
                    yield response
            except Exception as e:
                await self._handle_exception(e, method_name, context)
                raise  # Make it clear this path does not return normally
            except Warning as e:  # PEP 8: E722 do not use bare 'except'
                await self._handle_exception(e, method_name, context)
                raise  # Make it clear this path does not return normally

        return wrapper

    async def _handle_exception(
        self, e: Exception, method_name: str, context: grpc.aio.ServicerContext
    ) -> None:
        """Handles exceptions raised by RPC methods."""
        if issubclass(type(e), StopAsyncIteration):
            raise e
        if isinstance(e, AssertionError):
            raise e

        self.__error_cb(e)
        await context.abort(grpc.StatusCode.UNKNOWN, f"Exception: {e}")
