"""Unit tests for the ClientDisconnectionRetrier."""

import asyncio
import pytest
import pytest_asyncio
import logging
from functools import partial
from unittest.mock import AsyncMock

# SUT
from tsercom.rpc.connection.client_disconnection_retrier import (
    ClientDisconnectionRetrier,
)
from tsercom.util.stopable import Stopable
from tsercom.threading.thread_watcher import ThreadWatcher

# Module to patch for aio_utils
import tsercom.rpc.connection.client_disconnection_retrier as retrier_module_to_patch


# Helper Test Classes
class MockConnectable(Stopable):
    def __init__(self, name="mock_instance"):
        self.name = name
        self.stopped = False
        self.stop_call_count = 0

    async def stop(self) -> None:
        self.stopped = True
        self.stop_call_count += 1

    def __repr__(self):
        return f"<MockConnectable {self.name}>"


class _TestableRetrier(ClientDisconnectionRetrier[MockConnectable]):
    __test__ = False

    def set_on_connect(
        self, mock_connect_func: AsyncMock
    ):  # This line might cause issues after import removal
        self.connect_func = mock_connect_func

    async def _connect(self) -> MockConnectable:
        instance = await self.connect_func()
        return instance


# Pytest Fixtures
@pytest.fixture
def mock_watcher(mocker):
    return mocker.MagicMock(spec=ThreadWatcher)


@pytest.fixture
def mock_safe_disconnection_handler(mocker):
    # Making it async as the SUT might await it
    return mocker.AsyncMock(name="safe_disconnection_handler")


@pytest.fixture
def mock_delay_func(mocker):
    return mocker.AsyncMock(name="delay_func")


@pytest.fixture
def mock_is_grpc_error_func(mocker):
    return mocker.MagicMock(name="is_grpc_error_func")


@pytest.fixture
def mock_is_server_unavailable_error_func(mocker):
    return mocker.MagicMock(name="is_server_unavailable_error_func")


@pytest.fixture
def mock_connect_func(mocker):
    return mocker.AsyncMock(name="connect_func")


@pytest_asyncio.fixture
async def current_event_loop():
    return asyncio.get_running_loop()


@pytest_asyncio.fixture
async def mock_aio_utils(mocker, monkeypatch, current_event_loop):
    mock_get_loop = mocker.MagicMock(return_value=current_event_loop)
    mock_is_on_loop = mocker.MagicMock(
        return_value=True
    )  # Default to True for most tests

    # This mock needs to handle partials correctly, similar to discoverable_grpc_endpoint_connector
    def simplified_run_on_loop_side_effect(
        func_to_run, loop_arg, *args, **kwargs
    ):
        # Use current_event_loop (which is loop_instance from the outer scope) as default
        resolved_loop = loop_arg if loop_arg else current_event_loop
        if isinstance(func_to_run, partial):
            # If it's a partial, it might be an async method.
            # The SUT uses it for self.stop and self._on_disconnect
            # These are async. We need to schedule them.
            asyncio.ensure_future(func_to_run(), loop=resolved_loop)
        else:  # Direct call for simple functions/coroutines
            asyncio.ensure_future(
                func_to_run(*args, **kwargs),
                loop=resolved_loop,
            )

        f = asyncio.Future()
        f.set_result(None)  # SUT doesn't await this future.
        return f

    mock_run_on_loop = mocker.MagicMock(
        side_effect=simplified_run_on_loop_side_effect
    )

    monkeypatch.setattr(
        retrier_module_to_patch, "get_running_loop_or_none", mock_get_loop
    )
    monkeypatch.setattr(
        retrier_module_to_patch, "is_running_on_event_loop", mock_is_on_loop
    )
    monkeypatch.setattr(
        retrier_module_to_patch, "run_on_event_loop", mock_run_on_loop
    )

    return {
        "get_running_loop_or_none": mock_get_loop,
        "is_running_on_event_loop": mock_is_on_loop,
        "run_on_event_loop": mock_run_on_loop,
    }


@pytest.mark.asyncio
class TestClientDisconnectionRetrier:

    async def test_initial_connection_success(
        self,
        mock_watcher,
        mock_connect_func,
        current_event_loop,
        mock_aio_utils,
    ):
        loop = current_event_loop
        _mock_aio_utils = mock_aio_utils

        mock_instance = MockConnectable("instance1")
        mock_connect_func.return_value = mock_instance

        retrier = _TestableRetrier(watcher=mock_watcher, event_loop=loop)
        retrier.set_on_connect(mock_connect_func)

        assert await retrier.start() is True
        mock_connect_func.assert_called_once()
        assert retrier._ClientDisconnectionRetrier__instance is mock_instance
        mock_watcher.on_exception_seen.assert_not_called()

    async def test_initial_connection_server_unavailable(
        self,
        mock_watcher,
        mock_connect_func,
        mock_is_server_unavailable_error_func,
        current_event_loop,
        mock_aio_utils,
    ):
        loop = current_event_loop
        _mock_aio_utils = mock_aio_utils

        test_error = ConnectionRefusedError("Server unavailable")
        mock_connect_func.side_effect = test_error
        mock_is_server_unavailable_error_func.return_value = True

        retrier = _TestableRetrier(
            watcher=mock_watcher,
            is_server_unavailable_error_func=mock_is_server_unavailable_error_func,
            event_loop=loop,
        )
        retrier.set_on_connect(mock_connect_func)

        assert await retrier.start() is False
        mock_connect_func.assert_called_once()
        mock_is_server_unavailable_error_func.assert_called_once_with(
            test_error
        )
        assert retrier._ClientDisconnectionRetrier__instance is None
        mock_watcher.on_exception_seen.assert_not_called()

    async def test_initial_connection_other_error(
        self,
        mock_watcher,
        mock_connect_func,
        mock_is_server_unavailable_error_func,
        current_event_loop,
        mock_aio_utils,
    ):
        loop = current_event_loop
        _mock_aio_utils = mock_aio_utils

        test_error = ValueError("Some other connection error")
        mock_connect_func.side_effect = test_error
        mock_is_server_unavailable_error_func.return_value = False

        retrier = _TestableRetrier(
            watcher=mock_watcher,
            is_server_unavailable_error_func=mock_is_server_unavailable_error_func,
            event_loop=loop,
        )
        retrier.set_on_connect(mock_connect_func)

        with pytest.raises(ValueError, match="Some other connection error"):
            await retrier.start()

        mock_connect_func.assert_called_once()
        mock_is_server_unavailable_error_func.assert_called_once_with(
            test_error
        )
        assert retrier._ClientDisconnectionRetrier__instance is None
        mock_watcher.on_exception_seen.assert_not_called()

    async def test_stop_method_basic(
        self,
        mock_watcher,
        mock_connect_func,
        current_event_loop,
        mock_aio_utils,
    ):
        loop = current_event_loop
        _mock_aio_utils = mock_aio_utils

        mock_instance = MockConnectable("instance_to_stop")
        mock_connect_func.return_value = mock_instance

        retrier = _TestableRetrier(watcher=mock_watcher, event_loop=loop)
        retrier.set_on_connect(mock_connect_func)
        await retrier.start()
        assert retrier._ClientDisconnectionRetrier__instance is mock_instance

        await retrier.stop()

        assert mock_instance.stopped is True
        assert mock_instance.stop_call_count == 1
        assert retrier._ClientDisconnectionRetrier__instance is None

    async def test_disconnect_and_reconnect_success(
        self,
        mock_watcher,
        mock_connect_func,
        mock_delay_func,
        mock_is_server_unavailable_error_func,
        current_event_loop,
        mock_aio_utils,
    ):
        loop = current_event_loop
        _mock_aio_utils = mock_aio_utils

        instance1 = MockConnectable("instance1")
        mock_connect_func.return_value = instance1

        retrier = _TestableRetrier(
            watcher=mock_watcher,
            delay_before_retry_func=mock_delay_func,
            is_server_unavailable_error_func=mock_is_server_unavailable_error_func,
            event_loop=loop,
            max_retries=3,
        )
        retrier.set_on_connect(mock_connect_func)
        await retrier.start()
        assert retrier._ClientDisconnectionRetrier__instance is instance1
        mock_connect_func.reset_mock()

        disconnect_error = ConnectionAbortedError("Server connection lost")
        mock_is_server_unavailable_error_func.return_value = True

        instance2 = MockConnectable("instance2")
        mock_connect_func.side_effect = [instance2]

        await retrier._on_disconnect(disconnect_error)
        await asyncio.sleep(0)

        mock_is_server_unavailable_error_func.assert_called_with(
            disconnect_error
        )
        assert instance1.stopped is True, "Original instance should be stopped"

        mock_delay_func.assert_called_once()

        mock_connect_func.assert_called_once()

        assert (
            retrier._ClientDisconnectionRetrier__instance is instance2
        ), "Should have reconnected to new instance"
        mock_watcher.on_exception_seen.assert_not_called()

    async def test_disconnect_reaches_max_retries(
        self,
        mock_watcher,
        mock_connect_func,
        mock_delay_func,
        mock_is_server_unavailable_error_func,
        current_event_loop,
        mock_aio_utils,
        caplog,
    ):
        loop = current_event_loop
        _mock_aio_utils = mock_aio_utils

        caplog.set_level(logging.INFO)
        instance1 = MockConnectable("instance_max_retry")
        mock_connect_func.return_value = instance1
        max_retries = 2

        retrier = _TestableRetrier(
            watcher=mock_watcher,
            delay_before_retry_func=mock_delay_func,
            is_server_unavailable_error_func=mock_is_server_unavailable_error_func,
            event_loop=loop,
            max_retries=max_retries,
        )
        retrier.set_on_connect(mock_connect_func)
        await retrier.start()
        mock_connect_func.reset_mock()

        disconnect_error = ConnectionRefusedError(
            "Server unavailable persistently"
        )
        mock_is_server_unavailable_error_func.return_value = True

        retry_connect_error = ConnectionRefusedError("Still unavailable")
        mock_connect_func.side_effect = [retry_connect_error] * max_retries

        await retrier._on_disconnect(disconnect_error)
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        assert instance1.stopped is True
        assert mock_delay_func.call_count == max_retries
        assert mock_connect_func.call_count == max_retries

        assert (
            retrier._ClientDisconnectionRetrier__instance is None
        ), "Instance should be None after max retries"
        assert (
            f"Max retries ({max_retries}) reached. Stopping reconnection attempts."
            in caplog.text
        )
        mock_watcher.on_exception_seen.assert_not_called()
        _mock_aio_utils["run_on_event_loop"].assert_not_called()

    async def test_stop_method_when_not_on_event_loop(
        self,
        mock_watcher,
        mock_connect_func,
        current_event_loop,
        mock_aio_utils,
    ):
        loop = current_event_loop
        _mock_aio_utils = mock_aio_utils

        mock_instance = MockConnectable("instance_to_stop_off_loop")
        mock_connect_func.return_value = mock_instance

        is_on_loop_mock = _mock_aio_utils["is_running_on_event_loop"]
        is_on_loop_mock.side_effect = [False, True]

        retrier = _TestableRetrier(
            watcher=mock_watcher,
            event_loop=loop,
        )
        retrier.set_on_connect(mock_connect_func)
        await retrier.start()

        await retrier.stop()

        _mock_aio_utils["run_on_event_loop"].assert_called_once()
        call_args = _mock_aio_utils["run_on_event_loop"].call_args[0]
        assert call_args[0] == retrier.stop
        assert call_args[1] == loop

        await asyncio.sleep(0)

        assert mock_instance.stopped is True
        assert mock_instance.stop_call_count == 1
        assert retrier._ClientDisconnectionRetrier__instance is None
