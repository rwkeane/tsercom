"""Provides utility functions for handling gRPC errors, status codes, and retry logic."""

from __future__ import annotations

import asyncio
import random

import grpc  # Keep grpc import at global scope
from google.rpc.status_pb2 import Status


def get_grpc_status_code(error: Exception) -> grpc.StatusCode | None:
    """Extracts the gRPC status code from a gRPC exception.

    Args:
        error: The exception, potentially a gRPC error.

    Returns:
        The `grpc.StatusCode` if the error is a gRPC error and a status code
        can be extracted, otherwise `None`.
    """
    from grpc_status import rpc_status

    if isinstance(error, grpc.aio.AioRpcError):
        return error.code()

    if not issubclass(type(error), grpc.RpcError):
        return None

    # Retrieve status for general RpcError
    status: Status = rpc_status.from_call(error)
    if status is None:
        return None

    return status.code


def is_server_unavailable_error(error: Exception) -> bool:
    """Checks if an exception indicates a gRPC server unavailable error.

    This includes `UNAVAILABLE` and `DEADLINE_EXCEEDED` status codes,
    and `StopAsyncIteration` which can occur during stream termination.

    Args:
        error: The exception to check.

    Returns:
        True if the error indicates server unavailability, False otherwise.
    """
    if issubclass(type(error), StopAsyncIteration):
        return True

    status_code = get_grpc_status_code(error)
    unavailable_status = grpc.StatusCode.UNAVAILABLE
    deadline_exceeded_status = grpc.StatusCode.DEADLINE_EXCEEDED
    return status_code in (
        unavailable_status,
        deadline_exceeded_status,
    )


def is_grpc_error(error: Exception) -> bool:
    """Checks if the given exception is a gRPC-related error.

    Args:
        error: The exception to check.

    Returns:
        True if the exception has an associated gRPC status code, False otherwise.
    """
    return get_grpc_status_code(error) is not None


async def delay_before_retry() -> None:
    """Asynchronously waits for a random duration before a retry attempt.

    The delay is uniformly distributed between 4 and 8 seconds.
    """
    delay = random.uniform(4, 8)
    await asyncio.sleep(delay)
