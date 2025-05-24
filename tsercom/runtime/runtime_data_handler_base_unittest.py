"""Tests for RuntimeDataHandlerBase."""

import pytest
from unittest import mock
import grpc.aio # For ServicerContext

from tsercom.runtime.runtime_data_handler_base import RuntimeDataHandlerBase
from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.data.annotated_instance import AnnotatedInstance # For type hints
from tsercom.data.serializable_annotated_instance import SerializableAnnotatedInstance # For type hints
from tsercom.data.remote_data_reader import RemoteDataReader
from tsercom.threading.async_poller import AsyncPoller
from tsercom.runtime.endpoint_data_processor import EndpointDataProcessor


# Minimal concrete implementation for testing
class ConcreteRuntimeDataHandler(RuntimeDataHandlerBase[str, str]):
    def __init__(self, data_reader, event_source):
        super().__init__(data_reader, event_source)
        self._register_caller_mock = mock.Mock(spec=self._register_caller)
        self._unregister_caller_mock = mock.Mock(spec=self._unregister_caller)
        self._try_get_caller_id_mock = mock.Mock(spec=self._try_get_caller_id)

    def _register_caller(
        self, caller_id: CallerIdentifier, endpoint: str, port: int
    ) -> EndpointDataProcessor:
        return self._register_caller_mock(caller_id, endpoint, port)

    def _unregister_caller(self, caller_id: CallerIdentifier) -> bool:
        return self._unregister_caller_mock(caller_id)

    def _try_get_caller_id(
        self, endpoint: str, port: int
    ) -> CallerIdentifier | None:
        return self._try_get_caller_id_mock(endpoint, port)


@pytest.fixture
def mock_data_reader_fixture():
    return mock.Mock(spec=RemoteDataReader[AnnotatedInstance[str]])

@pytest.fixture
def mock_event_source_fixture():
    return mock.Mock(spec=AsyncPoller[SerializableAnnotatedInstance[str]])

@pytest.fixture
def mock_context_fixture():
    return mock.Mock(spec=grpc.aio.ServicerContext)

@pytest.fixture
def handler_fixture(mock_data_reader_fixture, mock_event_source_fixture):
    return ConcreteRuntimeDataHandler(mock_data_reader_fixture, mock_event_source_fixture)

@pytest.fixture
def mock_endpoint_processor_fixture():
    return mock.Mock(spec=EndpointDataProcessor)


class TestRuntimeDataHandlerBaseRegisterCaller:
    """Tests for the register_caller method of RuntimeDataHandlerBase."""

    def test_register_caller_with_endpoint_port_success(
        self, handler_fixture, mock_endpoint_processor_fixture
    ):
        caller_id = CallerIdentifier.random()
        endpoint = "127.0.0.1"
        port = 8080
        handler_fixture._register_caller_mock.return_value = mock_endpoint_processor_fixture

        result = handler_fixture.register_caller(
            caller_id, endpoint=endpoint, port=port
        )

        handler_fixture._register_caller_mock.assert_called_once_with(
            caller_id, endpoint, port
        )
        assert result == mock_endpoint_processor_fixture

    @mock.patch("tsercom.runtime.runtime_data_handler_base.get_client_ip")
    @mock.patch("tsercom.runtime.runtime_data_handler_base.get_client_port")
    def test_register_caller_with_context_success(
        self,
        mock_get_port,
        mock_get_ip,
        handler_fixture,
        mock_context_fixture,
        mock_endpoint_processor_fixture,
    ):
        caller_id = CallerIdentifier.random()
        expected_ip = "192.168.0.1"
        expected_port = 1234
        mock_get_ip.return_value = expected_ip
        mock_get_port.return_value = expected_port
        handler_fixture._register_caller_mock.return_value = mock_endpoint_processor_fixture

        result = handler_fixture.register_caller(
            caller_id, context=mock_context_fixture
        )

        mock_get_ip.assert_called_once_with(mock_context_fixture)
        mock_get_port.assert_called_once_with(mock_context_fixture)
        handler_fixture._register_caller_mock.assert_called_once_with(
            caller_id, expected_ip, expected_port
        )
        assert result == mock_endpoint_processor_fixture

    @mock.patch("tsercom.runtime.runtime_data_handler_base.get_client_ip")
    @mock.patch("tsercom.runtime.runtime_data_handler_base.get_client_port")
    def test_register_caller_with_context_ip_none_returns_none(
        self, mock_get_port, mock_get_ip, handler_fixture, mock_context_fixture
    ):
        caller_id = CallerIdentifier.random()
        mock_get_ip.return_value = None # Simulate IP not found
        # mock_get_port can return anything or not be called, outcome should be None

        result = handler_fixture.register_caller(
            caller_id, context=mock_context_fixture
        )

        mock_get_ip.assert_called_once_with(mock_context_fixture)
        # Depending on implementation, get_client_port might not be called if IP is None.
        # If it is called, its return value doesn't prevent returning None here.
        handler_fixture._register_caller_mock.assert_not_called()
        assert result is None

    @mock.patch("tsercom.runtime.runtime_data_handler_base.get_client_ip")
    @mock.patch("tsercom.runtime.runtime_data_handler_base.get_client_port")
    def test_register_caller_with_context_port_none_raises_value_error(
        self, mock_get_port, mock_get_ip, handler_fixture, mock_context_fixture
    ):
        caller_id = CallerIdentifier.random()
        expected_ip = "192.168.0.1"
        mock_get_ip.return_value = expected_ip
        mock_get_port.return_value = None # Simulate port not found

        with pytest.raises(ValueError) as excinfo:
            handler_fixture.register_caller(
                caller_id, context=mock_context_fixture
            )
        
        assert f"Could not determine client port from context for endpoint {expected_ip}" in str(excinfo.value)
        mock_get_ip.assert_called_once_with(mock_context_fixture)
        mock_get_port.assert_called_once_with(mock_context_fixture)
        handler_fixture._register_caller_mock.assert_not_called()

    # Argument validation tests (already implemented in RuntimeDataHandlerBase by previous subtask)
    def test_register_caller_mutex_args_endpoint_context(self, handler_fixture, mock_context_fixture):
        """Test providing both endpoint and context raises ValueError."""
        caller_id = CallerIdentifier.random()
        with pytest.raises(ValueError) as excinfo:
            handler_fixture.register_caller(caller_id, endpoint="1.2.3.4", port=123, context=mock_context_fixture)
        assert "Exactly one of 'endpoint'/'port' combination or 'context' must be provided" in str(excinfo.value)

    def test_register_caller_mutex_args_none(self, handler_fixture):
        """Test providing neither endpoint/port nor context raises ValueError."""
        caller_id = CallerIdentifier.random()
        with pytest.raises(ValueError) as excinfo:
            handler_fixture.register_caller(caller_id) # No endpoint, port or context
        assert "Exactly one of 'endpoint'/'port' combination or 'context' must be provided" in str(excinfo.value)

    def test_register_caller_endpoint_without_port(self, handler_fixture):
        """Test providing endpoint without port raises ValueError."""
        caller_id = CallerIdentifier.random()
        with pytest.raises(ValueError) as excinfo:
            handler_fixture.register_caller(caller_id, endpoint="1.2.3.4") # Port is None
        assert "If 'endpoint' is provided, 'port' must also be provided" in str(excinfo.value)
        
    def test_register_caller_port_without_endpoint(self, handler_fixture):
        """Test providing port without endpoint raises ValueError."""
        caller_id = CallerIdentifier.random()
        with pytest.raises(ValueError) as excinfo:
            handler_fixture.register_caller(caller_id, port=1234) # Endpoint is None
        assert "If 'endpoint' is provided, 'port' must also be provided" in str(excinfo.value)

    @mock.patch("tsercom.runtime.runtime_data_handler_base.get_client_ip")
    @mock.patch("tsercom.runtime.runtime_data_handler_base.get_client_port")
    def test_register_caller_context_is_not_servicer_context_raises_type_error(
        self, mock_get_port, mock_get_ip, handler_fixture
    ):
        """Test that if context is not None, it must be a ServicerContext."""
        caller_id = CallerIdentifier.random()
        not_a_servicer_context = object() # Some other object type

        # These can return valid values, the type check for context should happen before.
        mock_get_ip.return_value = "1.2.3.4"
        mock_get_port.return_value = 1234

        with pytest.raises(TypeError) as excinfo:
            handler_fixture.register_caller(caller_id, context=not_a_servicer_context)
        
        assert "Expected context to be an instance of grpc.aio.ServicerContext" in str(excinfo.value)
        mock_get_ip.assert_not_called() # Should fail before trying to use context
        mock_get_port.assert_not_called()
        handler_fixture._register_caller_mock.assert_not_called()
