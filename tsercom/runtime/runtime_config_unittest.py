"""Tests for RuntimeConfig."""

import pytest
from unittest import mock

from tsercom.runtime.runtime_config import RuntimeConfig, ServiceType
from tsercom.data.remote_data_aggregator import RemoteDataAggregator


class TestRuntimeConfig:
    """Tests for the RuntimeConfig class."""

    def test_init_with_client_string(self):
        """Test initialization with service_type='Client'."""
        config = RuntimeConfig(service_type="Client")
        assert config.is_client()
        assert not config.is_server()
        # Accessing private __service_type for exact enum check
        assert config._RuntimeConfig__service_type == ServiceType.CLIENT

    def test_init_with_server_string(self):
        """Test initialization with service_type='Server'."""
        config = RuntimeConfig(service_type="Server")
        assert not config.is_client()
        assert config.is_server()
        assert config._RuntimeConfig__service_type == ServiceType.SERVER

    def test_init_with_client_enum(self):
        """Test initialization with ServiceType.CLIENT."""
        config = RuntimeConfig(service_type=ServiceType.CLIENT)
        assert config.is_client()
        assert not config.is_server()
        assert config._RuntimeConfig__service_type == ServiceType.CLIENT

    def test_init_with_server_enum(self):
        """Test initialization with ServiceType.SERVER."""
        config = RuntimeConfig(service_type=ServiceType.SERVER)
        assert not config.is_client()
        assert config.is_server()
        assert config._RuntimeConfig__service_type == ServiceType.SERVER

    def test_init_copy_constructor_client(self):
        """Test initialization by copying from another Client RuntimeConfig."""
        mock_aggregator = mock.Mock(spec=RemoteDataAggregator)
        original_config = RuntimeConfig(
            service_type="Client",
            timeout_seconds=30,
            data_aggregator_client=mock_aggregator,
        )

        copied_config = RuntimeConfig(
            other_config=original_config
        )  # Use other_config parameter

        assert copied_config.is_client()
        assert not copied_config.is_server()
        assert copied_config._RuntimeConfig__service_type == ServiceType.CLIENT
        assert copied_config.timeout_seconds == 30
        assert copied_config.data_aggregator_client == mock_aggregator

    def test_init_copy_constructor_server(self):
        """Test initialization by copying from another Server RuntimeConfig."""
        mock_aggregator_server = mock.Mock(spec=RemoteDataAggregator)
        original_config = RuntimeConfig(
            service_type="Server",
            timeout_seconds=45,
            data_aggregator_client=mock_aggregator_server,
        )

        copied_config = RuntimeConfig(
            other_config=original_config
        )  # Use other_config parameter

        assert not copied_config.is_client()
        assert copied_config.is_server()
        assert copied_config._RuntimeConfig__service_type == ServiceType.SERVER
        assert copied_config.timeout_seconds == 45
        # Based on implementation, data_aggregator_client is copied regardless of service type
        assert copied_config.data_aggregator_client == mock_aggregator_server

    def test_default_values(self):
        """Test default timeout_seconds and data_aggregator_client."""
        config_client = RuntimeConfig(service_type="Client")
        assert config_client.timeout_seconds == 60
        assert config_client.data_aggregator_client is None

        config_server = RuntimeConfig(service_type="Server")
        assert config_server.timeout_seconds == 60
        assert config_server.data_aggregator_client is None

    def test_custom_timeout_seconds(self):
        """Test providing a custom timeout_seconds value."""
        config = RuntimeConfig(service_type="Client", timeout_seconds=120)
        assert config.timeout_seconds == 120

    def test_custom_data_aggregator_client_for_client(self):
        """Test providing a custom data_aggregator_client for Client type."""
        mock_aggregator = mock.Mock(spec=RemoteDataAggregator)
        config = RuntimeConfig(
            service_type="Client", data_aggregator_client=mock_aggregator
        )
        assert config.data_aggregator_client == mock_aggregator

    def test_custom_data_aggregator_client_for_server(self):
        """Test providing a custom data_aggregator_client for Server type."""
        mock_aggregator = mock.Mock(spec=RemoteDataAggregator)
        # Based on implementation, this should be set for server as well if provided.
        config = RuntimeConfig(
            service_type="Server", data_aggregator_client=mock_aggregator
        )
        assert config.data_aggregator_client == mock_aggregator

    def test_invalid_service_type_string(self):
        """Test initialization with an invalid service_type string raises ValueError."""
        invalid_type = "InvalidType"
        with pytest.raises(ValueError) as excinfo:
            RuntimeConfig(service_type=invalid_type)
        assert (
            f"Invalid service_type string: '{invalid_type}'. Must be 'Client' or 'Server'."
            == str(excinfo.value)
        )

    def test_data_aggregator_client_property(self):
        """Test the data_aggregator_client property returns the correct object."""
        mock_aggregator = mock.Mock(spec=RemoteDataAggregator)
        # Set via constructor to test property getter
        config = RuntimeConfig(
            service_type="Client", data_aggregator_client=mock_aggregator
        )
        assert config.data_aggregator_client == mock_aggregator

        config_none = RuntimeConfig(service_type="Client")
        assert config_none.data_aggregator_client is None

    def test_timeout_seconds_property(self):
        """Test the timeout_seconds property returns the correct value."""
        config = RuntimeConfig(service_type="Client", timeout_seconds=90)
        assert config.timeout_seconds == 90

        config_default = RuntimeConfig(
            service_type="Server"
        )  # Default timeout
        assert config_default.timeout_seconds == 60

    def test_is_client_is_server_methods(self):
        """Test is_client and is_server methods thoroughly."""
        client_config = RuntimeConfig(service_type=ServiceType.CLIENT)
        assert client_config.is_client() is True
        assert client_config.is_server() is False

        server_config = RuntimeConfig(service_type=ServiceType.SERVER)
        assert server_config.is_client() is False
        assert server_config.is_server() is True

    def test_constructor_assertion_service_type_vs_other_config(self):
        """Test ValueError when an invalid combination of service_type or other_config is provided."""
        # This string is part of the actual error message raised by RuntimeConfig
        expected_message_part = (
            "Exactly one of 'service_type' or 'other_config' must be provided"
        )

        with pytest.raises(ValueError, match=expected_message_part):
            RuntimeConfig(
                service_type="Client",
                other_config=RuntimeConfig(service_type="Server"),
            )

        with pytest.raises(ValueError, match=expected_message_part):
            RuntimeConfig()  # Neither provided


# Removed final backticks that caused syntax error.
