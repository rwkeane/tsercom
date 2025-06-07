import pytest
import datetime
from unittest.mock import AsyncMock, MagicMock
import grpc  # For grpc.StatusCode
from grpc.aio import ServicerContext
from typing import TypeVar, Generic, List, AsyncIterator, Optional


from tsercom.runtime.endpoint_data_processor import (
    EndpointDataProcessor,
    ServerTimestamp,
)
from tsercom.caller_id.caller_identifier import CallerIdentifier

# Define generic types for testing, can be simple types like object or str
DataTypeT = TypeVar("DataTypeT")
EventTypeT = TypeVar("EventTypeT")


class HelperConcreteTestProcessor(
    Generic[DataTypeT, EventTypeT],
    EndpointDataProcessor[DataTypeT, EventTypeT],
):
    __test__ = False  # Prevent pytest from collecting this class

    def __init__(self, caller_id: CallerIdentifier):
        super().__init__(caller_id)
        # Mocks will be initialized in setup_mocks

    def setup_mocks(self, mocker):
        self._process_data_mock = mocker.AsyncMock(name="_process_data_impl")
        self.desynchronize_mock = mocker.AsyncMock(name="desynchronize_impl")
        self.deregister_caller_mock = mocker.AsyncMock(
            name="deregister_caller_impl"
        )
        # __aiter__ returns an AsyncIterator, which then has __anext__
        # For this test, it's not directly used by process_data, so a simple mock is fine.
        # We'll make the __aiter__ return an object that has a mocked __anext__.
        mock_async_iterator = MagicMock()  # Removed spec=AsyncIterator
        mock_async_iterator.__anext__ = AsyncMock(
            side_effect=StopAsyncIteration()
        )
        self.__aiter_return_value = (
            mock_async_iterator  # Store it to be returned by __aiter__
        )

    async def desynchronize(
        self,
        timestamp: ServerTimestamp,
        context: Optional[grpc.aio.ServicerContext] = None,
    ) -> Optional[datetime.datetime]:
        return await self.desynchronize_mock(timestamp, context)

    async def deregister_caller(self) -> None:
        await self.deregister_caller_mock()  # pragma: no cover (not tested here)

    async def _process_data(
        self, data: DataTypeT, timestamp: datetime.datetime
    ) -> None:
        await self._process_data_mock(data, timestamp)

    def __aiter__(self) -> AsyncIterator[List[EventTypeT]]:
        return self.__aiter_return_value


@pytest.fixture
def test_data():
    return "sample_data"


@pytest.fixture
def mock_caller_id():
    return CallerIdentifier.random()


@pytest.fixture
def processor(mock_caller_id, mocker):
    # Use object as a simple placeholder for DataTypeT and EventTypeT if not specified
    proc = HelperConcreteTestProcessor[object, object](mock_caller_id)
    proc.setup_mocks(mocker)
    return proc


@pytest.mark.asyncio
async def test_process_data_timestamp_none(processor, test_data, mocker):
    fixed_now = datetime.datetime.now(datetime.timezone.utc)

    # Patch datetime.now within the tsercom.runtime.endpoint_data_processor module
    mock_dt = mocker.patch("tsercom.runtime.endpoint_data_processor.datetime")
    mock_dt.now.return_value = fixed_now
    mock_dt.timezone.utc = datetime.timezone.utc  # Ensure utc is available

    await processor.process_data(test_data, timestamp=None)

    mock_dt.now.assert_called_once_with(datetime.timezone.utc)
    processor._process_data_mock.assert_awaited_once_with(test_data, fixed_now)


@pytest.mark.asyncio
async def test_process_data_timestamp_is_datetime(processor, test_data):
    fixed_dt = datetime.datetime.now(datetime.timezone.utc)
    await processor.process_data(test_data, timestamp=fixed_dt)
    processor._process_data_mock.assert_awaited_once_with(test_data, fixed_dt)


@pytest.mark.asyncio
async def test_process_data_server_timestamp_desync_success(
    processor, test_data, mocker
):
    mock_server_ts = mocker.MagicMock(spec=ServerTimestamp)
    desynchronized_dt = datetime.datetime.now(datetime.timezone.utc)
    processor.desynchronize_mock.return_value = desynchronized_dt

    await processor.process_data(
        test_data, timestamp=mock_server_ts, context=None
    )  # Added context explicitly for clarity

    processor.desynchronize_mock.assert_awaited_once_with(mock_server_ts, None)
    processor._process_data_mock.assert_awaited_once_with(
        test_data, desynchronized_dt
    )


@pytest.mark.asyncio
async def test_process_data_server_timestamp_desync_none_no_context(
    processor, test_data, mocker
):
    mock_server_ts = mocker.MagicMock(spec=ServerTimestamp)
    processor.desynchronize_mock.return_value = None

    # No logger is used directly in EndpointDataProcessor.process_data for this case
    # Logging might occur within desynchronize_mock if it were a real method that logged.

    await processor.process_data(
        test_data, timestamp=mock_server_ts, context=None
    )

    processor.desynchronize_mock.assert_awaited_once_with(mock_server_ts, None)
    processor._process_data_mock.assert_not_awaited()
    # No direct logging to assert here from process_data itself for this path


@pytest.mark.asyncio
async def test_process_data_server_timestamp_desync_none_with_context(
    processor, test_data, mocker
):
    mock_server_ts = mocker.MagicMock(spec=ServerTimestamp)
    processor.desynchronize_mock.return_value = None
    mock_grpc_context = mocker.AsyncMock(spec=ServicerContext)

    await processor.process_data(
        test_data, timestamp=mock_server_ts, context=mock_grpc_context
    )

    processor.desynchronize_mock.assert_awaited_once_with(
        mock_server_ts, mock_grpc_context
    )
    mock_grpc_context.abort.assert_awaited_once_with(
        grpc.StatusCode.INVALID_ARGUMENT, "Invalid ServerTimestamp Provided"
    )
    processor._process_data_mock.assert_not_awaited()
