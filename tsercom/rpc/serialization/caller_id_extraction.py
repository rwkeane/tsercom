from typing import AsyncIterator, Callable, Optional, Tuple, TypeVar
import grpc
import logging

from tsercom.caller_id.proto import CallerId as GrpcCallerId
from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.util.is_running_tracker import IsRunningTracker

TCallType = TypeVar("TCallType")


async def extract_id_from_first_call(
    iterator: AsyncIterator[TCallType],
    is_running: Optional[IsRunningTracker] = None,
    context: Optional[grpc.aio.ServicerContext] = None,
    extractor: Optional[Callable[[TCallType], GrpcCallerId]] = None,
    validate_against: Optional[CallerIdentifier] = None,
) -> Tuple[CallerIdentifier | None, TCallType | None]:
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
    if is_running is not None:
        iterator = await is_running.create_stoppable_iterator(iterator)

    if extractor is None:

        def extractor(x: TCallType) -> GrpcCallerId:
            return x.id  # type: ignore

    # Extract the first call, without throwing if a StopAsyncIteration is
    # thrown.
    first_response: TCallType = None  # type: ignore
    try:
        async for result in iterator:
            first_response = result
            if is_running is not None and not is_running.get():
                return None, None

            if first_response is None:
                logging.error("None received from remote while expecting first call in iterator.")
            break
    except Exception as e:
        # TODO: Maybe this should check the exception code and split behavior?
        if isinstance(e, grpc.RpcError):
            logging.error(
                f"RpcError while iterating for first call: code={e.code()}, details='{e.details()}'. Original exception: {e}",
                exc_info=True,
            )
        else:
            logging.error(
                f"Non-RpcError exception while iterating for first call: {e}",
                exc_info=True,
            )
        
        if context is not None:
            await context.abort(
                grpc.StatusCode.CANCELLED, "Error processing first call!"
            )
        raise e

    # If it exited before the first response, return.
    if first_response is None:
        logging.error("First call never received from iterator.")
        if context is not None:
            await context.abort(
                grpc.StatusCode.CANCELLED, "First call never received!"
            )
        return None, first_response

    id = await extract_id_from_call(
        first_response, context, extractor, validate_against
    )
    return id, first_response


async def extract_id_from_call(
    call: TCallType,
    context: Optional[grpc.aio.ServicerContext] = None,
    extractor: Optional[Callable[[TCallType], GrpcCallerId]] = None,
    validate_against: Optional[CallerIdentifier] = None,
) -> CallerIdentifier | None:
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

        def extractor(x: TCallType) -> GrpcCallerId:
            return x.id  # type: ignore

    # If the id can't be extracted, return.
    if extractor(call) is None:
        logging.error("Missing CallerID in call object.")
        if context is not None:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT, "Missing CallerID"
            )
        return None

    # If the id is malformed, return.
    extracted = extractor(call)
    caller_id = CallerIdentifier.try_parse(extracted)
    if caller_id is None:
        logging.error(f"Invalid CallerID format received: {extracted}")
        if context is not None:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT, "Invalid CallerID received"
            )
        return None

    # Validate it if needed.
    if validate_against is not None and caller_id != validate_against:
        logging.error(f"Invalid CallerID received. Expected {validate_against}, got {caller_id}.")
        if context is not None:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT, "Invalid CallerID received"
            )
        return None

    return caller_id
