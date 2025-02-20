import asyncio
from google.rpc.status_pb2 import Status
from grpc_status import rpc_status
import grpc
import random
    
def get_grpc_status_code(error : Exception) -> grpc.StatusCode | None:
    """
    Returns the gRPC Status code associated with this |error| if one exists, and
    None in all other cases.
    """
    if issubclass(type(error), grpc.aio.AioRpcError):
        return error.code()
    
    if not issubclass(type(error), grpc.RpcError):
        return
    
    status : Status = rpc_status.from_call(error)
    if status is None:
        return None
    
    return status.code

def is_server_unavailable_error(error : Exception) -> bool:
    """
    Returns True if |error| is associated with UNAVAILABLE or DEADLINE_EXCEEDED
    gRPC Status codes, and False otherwise.
    """
    if issubclass(type(error), StopAsyncIteration):
        return True
    
    status_code = get_grpc_status_code(error)
    print("FOUND STATUS CODE", status_code)
    return status_code in (grpc.StatusCode.UNAVAILABLE,
                           grpc.StatusCode.DEADLINE_EXCEEDED)

def is_grpc_error(error : Exception):
    return not get_grpc_status_code(error) is None 

async def delay_before_retry():
    """
    Delays between 4 and 8 seconds.
    """
    delay = random.uniform(4, 8)
    await asyncio.sleep(delay)