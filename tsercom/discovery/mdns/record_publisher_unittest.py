import pytest
from unittest.mock import AsyncMock, patch

from zeroconf import ServiceInfo, IPVersion
from zeroconf.asyncio import AsyncZeroconf

from tsercom.discovery.mdns.record_publisher import RecordPublisher
from tsercom.util.ip import get_all_addresses # Assuming this is still used by RP

@pytest.mark.asyncio
async def test_publish_and_close_with_owned_zc_closes_zc(mocker):
    """Test that RecordPublisher closes its owned AsyncZeroconf instance."""

    # Mock AsyncZeroconf that will be created by RecordPublisher
    mock_owned_zc_instance = AsyncMock(spec=AsyncZeroconf)

    # Patch the AsyncZeroconf constructor within the module where RecordPublisher will call it
    with patch("tsercom.discovery.mdns.record_publisher.AsyncZeroconf", return_value=mock_owned_zc_instance) as mock_zc_constructor:
        publisher = RecordPublisher(
            name="TestService",
            type_="_testtype",
            port=1234,
            properties={b"key": b"value"},
            zc_instance=None  # Explicitly request an owned instance
        )

        # Simulate getting addresses if needed for ServiceInfo creation
        with patch("tsercom.discovery.mdns.record_publisher.get_all_addresses", return_value=[b"\x7f\x00\x00\x01"]): # 127.0.0.1
            await publisher.publish()

        mock_zc_constructor.assert_called_once_with(ip_version=IPVersion.V4Only)
        assert publisher._zc is mock_owned_zc_instance # Check it's using the mocked instance

        # ServiceInfo should be created and passed to async_register_service
        mock_owned_zc_instance.async_register_service.assert_awaited_once()
        # Get the ServiceInfo instance passed to async_register_service
        # args_list = mock_owned_zc_instance.async_register_service.call_args_list
        # service_info_arg = args_list[0][0][0] # First arg of first call
        # assert isinstance(service_info_arg, ServiceInfo)

        await publisher.close()

        # Check that unregister and close were called on the owned instance
        mock_owned_zc_instance.async_unregister_service.assert_awaited_once() # With the same service_info_arg
        mock_owned_zc_instance.async_close.assert_awaited_once()

@pytest.mark.asyncio
async def test_publish_and_close_with_shared_zc_does_not_close_shared_zc(mocker):
    """Test that RecordPublisher does not close a shared AsyncZeroconf instance."""

    mock_shared_zc_instance = AsyncMock(spec=AsyncZeroconf)
    # mock_shared_zc_instance.async_close = AsyncMock() # Already part of AsyncMock spec if autospecced

    publisher = RecordPublisher(
        name="TestSharedService",
        type_="_testsharedtype",
        port=5678,
        properties={b"shared": b"true"},
        zc_instance=mock_shared_zc_instance # Pass the shared instance
    )

    # Simulate getting addresses
    with patch("tsercom.discovery.mdns.record_publisher.get_all_addresses", return_value=[b"\x7f\x00\x00\x01"]):
        await publisher.publish()

    assert publisher._zc is mock_shared_zc_instance
    mock_shared_zc_instance.async_register_service.assert_awaited_once()
    # args_list_shared = mock_shared_zc_instance.async_register_service.call_args_list
    # service_info_arg_shared = args_list_shared[0][0][0]
    # assert isinstance(service_info_arg_shared, ServiceInfo)

    await publisher.close()

    # Check that unregister was called, but close was NOT called on the shared instance
    mock_shared_zc_instance.async_unregister_service.assert_awaited_once() # With service_info_arg_shared
    mock_shared_zc_instance.async_close.assert_not_called()
