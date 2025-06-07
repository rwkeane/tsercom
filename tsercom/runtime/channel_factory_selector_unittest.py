import pytest

# Import the module to be spied upon
import tsercom.runtime.channel_factory_selector as cfs_module
from tsercom.runtime.channel_factory_selector import ChannelFactorySelector
from tsercom.rpc.grpc_util.channel_auth_config import (
    BaseChannelAuthConfig,
    InsecureChannelConfig,
    ServerCAChannelConfig,
    PinnedServerChannelConfig,
    ClientAuthChannelConfig,
)
from tsercom.rpc.grpc_util.transport.insecure_grpc_channel_factory import (
    InsecureGrpcChannelFactory,
)
from tsercom.rpc.grpc_util.transport.server_auth_grpc_channel_factory import (
    ServerAuthGrpcChannelFactory,
)
from tsercom.rpc.grpc_util.transport.pinned_server_auth_grpc_channel_factory import (
    PinnedServerAuthGrpcChannelFactory,
)
from tsercom.rpc.grpc_util.transport.client_auth_grpc_channel_factory import (
    ClientAuthGrpcChannelFactory,
)


# --- Test Class for _read_file_content ---
class TestReadFileContent:
    def test_read_file_content_path_none(self):
        selector = ChannelFactorySelector()
        assert selector._read_file_content(None) is None

    def test_read_file_content_success(self, mocker):
        selector = ChannelFactorySelector()
        mocker.patch(
            "builtins.open", mocker.mock_open(read_data=b"test_content")
        )
        content = selector._read_file_content("dummy_path.pem")
        assert content == b"test_content"

    def test_read_file_content_io_error(self, mocker):
        selector = ChannelFactorySelector()
        mocker.patch("builtins.open", side_effect=IOError("File not found"))
        with pytest.raises(IOError, match="File not found"):
            selector._read_file_content("dummy_path.pem")

    def test_read_file_content_empty_path(self):
        selector = ChannelFactorySelector()
        # Assuming empty path behaves like None or raises an error handled by caller
        # Based on current _read_file_content, it would try to open ""
        # Let's assume it should return None or be caught by a check before calling open
        # For now, let's test its current direct behavior if path is not None but empty
        with pytest.raises(
            IOError
        ):  # os.open("") typically raises FileNotFoundError or similar
            selector._read_file_content("")


# --- Test Class for create_factory ---
class TestCreateFactory:
    @pytest.fixture
    def selector(self):
        return ChannelFactorySelector()

    def test_create_factory_no_auth_config(self, selector):
        factory = selector.create_factory(None)
        assert isinstance(factory, InsecureGrpcChannelFactory)

    def test_create_factory_insecure_config_type(self, selector):
        auth_config = InsecureChannelConfig()
        factory = selector.create_factory(auth_config)
        assert isinstance(factory, InsecureGrpcChannelFactory)

    # ServerCAChannelConfig Tests
    def test_create_factory_server_ca_success(self, selector, mocker):
        mock_read_content = mocker.patch.object(
            selector, "_read_file_content", return_value=b"ca_cert_content"
        )
        # Spy on the __init__ method of the class using the imported module alias
        spy_server_auth_factory_init = mocker.spy(
            cfs_module.ServerAuthGrpcChannelFactory,  # Use alias
            "__init__",
        )

        auth_config = ServerCAChannelConfig(
            server_ca_cert_path="fake_ca.pem",
            server_hostname_override="host.example.com",
        )
        factory = selector.create_factory(auth_config)

        assert isinstance(factory, ServerAuthGrpcChannelFactory)
        mock_read_content.assert_called_once_with("fake_ca.pem")
        # Assert __init__ was called with expected args (excluding self)
        spy_server_auth_factory_init.assert_called_once_with(
            mocker.ANY,  # self instance
            root_ca_cert_pem=b"ca_cert_content",
            server_hostname_override="host.example.com",
        )

    def test_create_factory_server_ca_read_fails(self, selector, mocker):
        mocker.patch.object(selector, "_read_file_content", return_value=None)
        auth_config = ServerCAChannelConfig(server_ca_cert_path="fake_ca.pem")
        with pytest.raises(
            ValueError, match="Failed to read server_ca_cert_path: fake_ca.pem"
        ):
            selector.create_factory(auth_config)

    # PinnedServerChannelConfig Tests
    def test_create_factory_pinned_server_success(self, selector, mocker):
        mock_read_content = mocker.patch.object(
            selector, "_read_file_content", return_value=b"pinned_cert_content"
        )
        spy_pinned_factory_init = mocker.spy(
            cfs_module.PinnedServerAuthGrpcChannelFactory,  # Use alias
            "__init__",
        )

        auth_config = PinnedServerChannelConfig(
            pinned_server_cert_path="fake_pinned.pem",
            server_hostname_override="pinned.example.com",
        )
        factory = selector.create_factory(auth_config)

        assert isinstance(factory, PinnedServerAuthGrpcChannelFactory)
        mock_read_content.assert_called_once_with("fake_pinned.pem")
        spy_pinned_factory_init.assert_called_once_with(
            mocker.ANY,  # self
            expected_server_cert_pem=b"pinned_cert_content",  # Name from PinnedServerAuthGrpcChannelFactory constructor
            server_hostname_override="pinned.example.com",
        )

    def test_create_factory_pinned_server_read_fails(self, selector, mocker):
        mocker.patch.object(selector, "_read_file_content", return_value=None)
        auth_config = PinnedServerChannelConfig(
            pinned_server_cert_path="fake_pinned.pem"
        )
        with pytest.raises(
            ValueError,
            match="Failed to read pinned_server_cert_path: fake_pinned.pem",
        ):
            selector.create_factory(auth_config)

    # ClientAuthChannelConfig Tests
    def test_create_factory_client_auth_success(self, selector, mocker):
        def read_side_effect(path):
            if path == "c.pem":
                return b"client_cert"
            if path == "k.pem":
                return b"client_key"
            # server_ca_cert_path is not a field in ClientAuthChannelConfig
            # if path == "ca.pem":
            #     return b"ca_cert_opt"
            return None

        mock_read_content = mocker.patch.object(
            selector, "_read_file_content", side_effect=read_side_effect
        )
        spy_client_auth_factory_init = mocker.spy(
            cfs_module.ClientAuthGrpcChannelFactory,  # Use alias
            "__init__",
        )

        auth_config = ClientAuthChannelConfig(
            client_cert_path="c.pem",
            client_key_path="k.pem",
            # server_ca_cert_path="ca.pem", # Removed
            server_hostname_override="override.host",
        )
        factory = selector.create_factory(auth_config)

        assert isinstance(factory, ClientAuthGrpcChannelFactory)
        mock_read_content.assert_any_call("c.pem")
        mock_read_content.assert_any_call("k.pem")
        # mock_read_content.assert_any_call("ca.pem") # No longer called
        spy_client_auth_factory_init.assert_called_once_with(
            mocker.ANY,  # self
            client_key_pem=b"client_key",
            client_cert_pem=b"client_cert",
            root_ca_cert_pem=None,  # SUT passes None as ClientAuthChannelConfig has no CA
            server_hostname_override="override.host",
        )

    def test_create_factory_client_auth_cert_read_fails(
        self, selector, mocker
    ):
        def read_side_effect(path):
            if path == "c.pem":
                return None
            if path == "k.pem":
                return b"client_key"
            return None

        mocker.patch.object(
            selector, "_read_file_content", side_effect=read_side_effect
        )
        auth_config = ClientAuthChannelConfig(
            client_cert_path="c.pem", client_key_path="k.pem"
        )
        with pytest.raises(
            ValueError,
            match="Failed to read client_cert_path or client_key_path for tls_client_auth",
        ):
            selector.create_factory(auth_config)

    def test_create_factory_client_auth_key_read_fails(self, selector, mocker):
        def read_side_effect(path):
            if path == "c.pem":
                return b"client_cert"
            if path == "k.pem":
                return None
            return None

        mocker.patch.object(
            selector, "_read_file_content", side_effect=read_side_effect
        )
        auth_config = ClientAuthChannelConfig(
            client_cert_path="c.pem", client_key_path="k.pem"
        )
        with pytest.raises(
            ValueError,
            match="Failed to read client_cert_path or client_key_path for tls_client_auth",
        ):
            selector.create_factory(auth_config)

    def test_create_factory_client_auth_no_optional_ca(self, selector, mocker):
        # Test ClientAuthChannelConfig when optional server_ca_cert_path is None
        def read_side_effect(path):
            if path == "c.pem":
                return b"client_cert"
            if path == "k.pem":
                return b"client_key"
            return None  # Should not be called for ca.pem

        mocker.patch.object(  # Removed assignment to mock_read_content
            selector, "_read_file_content", side_effect=read_side_effect
        )
        spy_client_auth_factory_init = mocker.spy(
            cfs_module.ClientAuthGrpcChannelFactory, "__init__"  # Use alias
        )
        # ClientAuthChannelConfig does not take server_ca_cert_path.
        # server_hostname_override defaults to None if not provided.
        auth_config = ClientAuthChannelConfig(
            client_cert_path="c.pem", client_key_path="k.pem"
        )
        factory = selector.create_factory(auth_config)
        assert isinstance(factory, ClientAuthGrpcChannelFactory)
        spy_client_auth_factory_init.assert_called_once_with(
            mocker.ANY,  # self
            client_key_pem=b"client_key",
            client_cert_pem=b"client_cert",
            root_ca_cert_pem=None,  # SUT hardcodes this to None for ClientAuthGrpcChannelFactory
            server_hostname_override=None,  # Default for ClientAuthChannelConfig
        )

    # Unknown Config Type Test
    def test_create_factory_unknown_config_type(self, selector):
        class UnknownConfig(BaseChannelAuthConfig):
            pass

        auth_config = UnknownConfig()
        with pytest.raises(
            ValueError, match="Unknown or unsupported ChannelAuthConfig type"
        ):
            selector.create_factory(auth_config)
