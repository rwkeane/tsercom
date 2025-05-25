from __future__ import annotations
import asyncio
from google.rpc.status_pb2 import Status
import grpc # Keep grpc import at global scope
import random


def get_grpc_status_code(error: Exception) -> grpc.StatusCode | None:
    """
    Returns the gRPC Status code associated with this |error| if one exists, and
    None in all other cases.
    """
    from grpc_status import rpc_status
    if issubclass(type(error), grpc.aio.AioRpcError):
        return error.code()  # type: ignore

    if not issubclass(type(error), grpc.RpcError):
        return None

    status: Status = rpc_status.from_call(error)
    if status is None:
        return None

    return status.code


def is_server_unavailable_error(error: Exception) -> bool:
    """
    Returns True if |error| is associated with UNAVAILABLE or DEADLINE_EXCEEDED
    gRPC Status codes, and False otherwise.
    """
    if issubclass(type(error), StopAsyncIteration):
        return True

    status_code = get_grpc_status_code(error)
    print("FOUND STATUS CODE", status_code)
    unavailable_status = grpc.StatusCode.UNAVAILABLE
    deadline_exceeded_status = grpc.StatusCode.DEADLINE_EXCEEDED
    return status_code in (
        unavailable_status,
        deadline_exceeded_status,
    )


def is_grpc_error(error: Exception) -> bool:
    return get_grpc_status_code(error) is not None


async def delay_before_retry() -> None:
    """
    Delays between 4 and 8 seconds.
    """
    delay = random.uniform(4, 8)
    await asyncio.sleep(delay)
