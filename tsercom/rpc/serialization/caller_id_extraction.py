from typing import AsyncIterator, Callable, Optional, Tuple, TypeVar
import grpc

from tsercom.caller_id.proto import CallerId as GrpcCallerId
from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.util.is_running_tracker import IsRunningTracker


TCallType = TypeVar("TCallType")
async def extract_id_from_first_call(
        iterator : AsyncIterator[TCallType],
        is_running : Optional[IsRunningTracker] = None,
        context : Optional[grpc.aio.ServicerContext] = None,
        extractor : Optional[Callable[[TCallType], GrpcCallerId]] = None,
        validate_against : Optional[CallerIdentifier] = None) \
                -> Tuple[CallerIdentifier | None, TCallType | None]:
    
    """
    Extracts the CallerIdentifier for the next available instance received from
    |iterator|, returning both the  CallerId and the call itself if the method
    succeeds.
    
    |is_running| is the IsRunningTracker which should be watched, so that this
    call is cancelled if the calling instance stops running.
    |context| (if provided) is used to respond to the caller if an invalid
    CallerId is given.
    |extractor| is used to get the CallerId from the provided |call|. If not
    provided, defaults to checking the id parameter.
    |validate_against| is the expected CallerId, if such a value exists.
    """
    # Clean up input.
    if not is_running is None: 
        iterator = await is_running.create_stoppable_iterator(iterator)

    if extractor is None:
        extractor = lambda x: x.id

    # Extract the first call, without throwing if a StopAsyncIteration is
    # thrown.
    first_response : TCallType = None
    try:
        async for result in iterator:
            first_response = result
            if not is_running is None and not is_running.get():
                return None, None
            
            if first_response is None:
                print("ERROR! None received from remote!")
            break
    except Exception as e:
        print("Hit exception", e)
        if not context is None:
            await context.abort(grpc.StatusCode.CANCELLED,
                                "Error processing first call!")
        # TODO: Maybe this should check the exception code and split behavior?
        raise e
    
    # If it exited before the first response, return.
    if first_response is None:
        print("ERROR: First call never received!")
        if not context is None:
            await context.abort(grpc.StatusCode.CANCELLED,
                                "First call never received!")
        return None, first_response
    
    id = await extract_id_from_call(
            first_response, context, extractor, validate_against)
    return id, first_response

async def extract_id_from_call(
        call : TCallType,
        context : Optional[grpc.aio.ServicerContext] = None,
        extractor : Optional[Callable[[TCallType], GrpcCallerId]] = None,
        validate_against : Optional[CallerIdentifier] = None) \
                -> CallerIdentifier | None:
    """
    Extracts the CallerIdentifier associated with |call|, returning the CallerId
    if the method succeeds.
    
    |context| (if provided) is used to respond to the caller if an invalid
    CallerId is given.
    |extractor| is used to get the CallerId from the provided |call|. If not
    provided, defaults to checking the id parameter.
    |validate_against| is the expected CallerId, if such a value exists.
    """
    if extractor is None:
        extractor = lambda x: x.id
    
    # If the id can't be extracted, return.
    if extractor(call) is None:
        print("ERROR: Missing CallerID!")
        if not context is None:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT,
                                "Missing CallerID")
        return None
    
    # If the id is malformed, return.
    extracted = extractor(call)
    caller_id = CallerIdentifier.try_parse(extracted)
    if caller_id is None:
        print("ERROR: Invalid CallerID Received!")
        if not context is None:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT,
                                "Invalid CallerID received")
        return None
    
    # Validate it if needed.
    if not validate_against is None and caller_id != validate_against:
        print("ERROR: Invalid CallerID received!")
        if not context is None:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT,
                                "Invalid CallerID received")
        return None
    
    return caller_id