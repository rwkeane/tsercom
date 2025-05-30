# tsercom/runtime/channel_factory_selector_unittest.py
import pytest
from unittest import mock  # Pytest-mock uses unittest.mock

from tsercom.config.grpc_channel_config import (
    GrpcChannelFactoryConfig,
)
from tsercom.runtime.channel_factory_selector import ChannelFactorySelector
from tsercom.rpc.grpc_util.transport.insecure_grpc_channel_factory import (
    InsecureGrpcChannelFactory,
)
from tsercom.rpc.grpc_util.transport.client_auth_grpc_channel_factory import (
    ClientAuthGrpcChannelFactory,
)
from tsercom.rpc.grpc_util.transport.pinned_server_auth_grpc_channel_factory import (
    PinnedServerAuthGrpcChannelFactory,
)
from tsercom.rpc.grpc_util.transport.server_auth_grpc_channel_factory import (
    ServerAuthGrpcChannelFactory,
)


@pytest.fixture
def selector() -> ChannelFactorySelector:
    return ChannelFactorySelector()


def test_create_factory_from_config_none_config(
    selector: ChannelFactorySelector,
):
    """Test that InsecureGrpcChannelFactory is returned for None config."""
    factory = selector.create_factory_from_config(None)
    assert isinstance(factory, InsecureGrpcChannelFactory)


def test_create_factory_insecure(selector: ChannelFactorySelector):
    """Test creation of InsecureGrpcChannelFactory."""
    config = GrpcChannelFactoryConfig(factory_type="insecure")
    factory = selector.create_factory_from_config(config)
    assert isinstance(factory, InsecureGrpcChannelFactory)


@mock.patch("tsercom.runtime.channel_factory_selector.os.path.exists")
@mock.patch(
    "builtins.open", new_callable=mock.mock_open, read_data=b"cert_data"
)
def test_load_credential_path(
    mock_file_open, mock_path_exists, selector: ChannelFactorySelector
):
    mock_path_exists.return_value = True
    credential = selector._load_credential("fake/path/to/cert.pem")
    assert credential == b"cert_data"
    mock_path_exists.assert_called_with("fake/path/to/cert.pem")
    mock_file_open.assert_called_with("fake/path/to/cert.pem", "rb")


@mock.patch("tsercom.runtime.channel_factory_selector.os.path.exists")
def test_load_credential_pem_string(
    mock_path_exists, selector: ChannelFactorySelector
):
    mock_path_exists.return_value = False  # Ensure it's not treated as a path
    pem_string = "-----BEGIN CERTIFICATE-----\nTEST\n-----END CERTIFICATE-----"
    credential = selector._load_credential(pem_string)
    assert credential == pem_string


def test_load_credential_none(selector: ChannelFactorySelector):
    assert selector._load_credential(None) is None


@mock.patch(
    "tsercom.runtime.channel_factory_selector.os.path.exists",
    return_value=True,
)
@mock.patch("builtins.open", side_effect=IOError("File not found"))
def test_load_credential_path_read_error(
    mock_open, mock_exists, selector: ChannelFactorySelector
):
    with pytest.raises(
        ValueError, match="Could not read credential file: fake/path.pem"
    ):
        selector._load_credential("fake/path.pem")


# Test ClientAuth Factory
@mock.patch.object(ChannelFactorySelector, "_load_credential")
def test_create_client_auth_factory(
    mock_load_credential, selector: ChannelFactorySelector
):
    mock_load_credential.side_effect = lambda x: (
        f"{x}_content".encode("utf-8") if x else None
    )

    config = GrpcChannelFactoryConfig(
        factory_type="client_auth",
        client_cert_pem_or_path="client_cert",
        client_key_pem_or_path="client_key",
        root_ca_cert_pem_or_path="root_ca",
        server_hostname_override="override.host.com",
    )
    factory = selector.create_factory_from_config(config)
    assert isinstance(factory, ClientAuthGrpcChannelFactory)
    assert factory.client_cert_pem_bytes == b"client_cert_content"
    assert factory.client_key_pem_bytes == b"client_key_content"
    assert factory.root_ca_cert_pem_bytes == b"root_ca_content"
    assert factory.server_hostname_override == "override.host.com"

    expected_calls = [
        mock.call("client_cert"),
        mock.call("client_key"),
        mock.call("root_ca"),
    ]
    mock_load_credential.assert_has_calls(expected_calls, any_order=True)


def test_create_client_auth_factory_missing_creds(
    selector: ChannelFactorySelector,
):
    config = GrpcChannelFactoryConfig(
        factory_type="client_auth", client_cert_pem_or_path="cert"
    )
    # Missing client_key
    with pytest.raises(
        ValueError,
        match="ClientAuth factory requires client_cert_pem_or_path and client_key_pem_or_path.",
    ):
        selector.create_factory_from_config(config)


# Test PinnedServerAuth Factory
@mock.patch.object(ChannelFactorySelector, "_load_credential")
def test_create_pinned_server_auth_factory(
    mock_load_credential, selector: ChannelFactorySelector
):
    mock_load_credential.return_value = b"expected_server_cert_content"
    config = GrpcChannelFactoryConfig(
        factory_type="pinned_server_auth",
        expected_server_cert_pem_or_path="server_cert",
        server_hostname_override="pinned.host.com",
    )
    factory = selector.create_factory_from_config(config)
    assert isinstance(factory, PinnedServerAuthGrpcChannelFactory)
    assert factory.expected_server_cert_pem == b"expected_server_cert_content"
    assert factory.server_hostname_override == "pinned.host.com"
    mock_load_credential.assert_called_once_with("server_cert")


def test_create_pinned_server_auth_factory_missing_creds(
    selector: ChannelFactorySelector,
):
    config = GrpcChannelFactoryConfig(factory_type="pinned_server_auth")
    with pytest.raises(
        ValueError,
        match="PinnedServerAuth factory requires expected_server_cert_pem_or_path.",
    ):
        selector.create_factory_from_config(config)


# Test ServerAuth Factory
@mock.patch.object(ChannelFactorySelector, "_load_credential")
def test_create_server_auth_factory(
    mock_load_credential, selector: ChannelFactorySelector
):
    mock_load_credential.return_value = b"root_ca_content"
    config = GrpcChannelFactoryConfig(
        factory_type="server_auth",
        root_ca_cert_pem_or_path="root_ca",
        server_hostname_override="auth.host.com",
    )
    factory = selector.create_factory_from_config(config)
    assert isinstance(factory, ServerAuthGrpcChannelFactory)
    assert factory.root_ca_cert_pem == b"root_ca_content"
    assert factory.server_hostname_override == "auth.host.com"
    mock_load_credential.assert_called_once_with("root_ca")


def test_create_server_auth_factory_missing_creds(
    selector: ChannelFactorySelector,
):
    config = GrpcChannelFactoryConfig(factory_type="server_auth")
    with pytest.raises(
        ValueError,
        match="ServerAuth factory requires root_ca_cert_pem_or_path.",
    ):
        selector.create_factory_from_config(config)


def test_create_factory_unknown_type(selector: ChannelFactorySelector):
    config = GrpcChannelFactoryConfig(factory_type="unknown_type")  # type: ignore
    with pytest.raises(
        ValueError, match="Unknown GrpcChannelFactoryType: unknown_type"
    ):
        selector.create_factory_from_config(config)
