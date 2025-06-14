import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from zeroconf.asyncio import AsyncZeroconf, AsyncServiceBrowser

# Assuming MdnsListener.Client is needed for RecordListener's constructor
from tsercom.discovery.mdns.mdns_listener import MdnsListener
from tsercom.discovery.mdns.record_listener import RecordListener
from typing import Optional # Added for Optional type hint

# A simple mock for MdnsListener.Client
class MockMdnsClient(MdnsListener.Client):
    async def _on_service_added(self, name: str, port: int, addresses: list[bytes], txt_record: dict[bytes, bytes | None]) -> None:
        pass
    async def _on_service_removed(self, name: str, service_type: str, record_listener_uuid: str) -> None:
        pass

@pytest.mark.asyncio
async def test_close_with_owned_zc_closes_zc(mocker):
    """Test that RecordListener closes its owned AsyncZeroconf instance."""

    mock_client = MockMdnsClient()
    mock_owned_zc_instance = AsyncMock(spec=AsyncZeroconf)
    # Mock the .zeroconf attribute to return another mock, as it's accessed by AsyncServiceBrowser
    mock_owned_zc_instance.zeroconf = MagicMock()

    mock_service_browser_instance = AsyncMock(spec=AsyncServiceBrowser)

    # Patch AsyncZeroconf constructor and AsyncServiceBrowser constructor
    with patch("tsercom.discovery.mdns.record_listener.AsyncZeroconf", return_value=mock_owned_zc_instance) as mock_zc_constructor, \
         patch("tsercom.discovery.mdns.record_listener.AsyncServiceBrowser", return_value=mock_service_browser_instance) as mock_browser_constructor:

        listener = RecordListener(
            client=mock_client,
            service_type="_testowned._tcp.local.",
            zc_instance=None  # Request owned instance
        )

        # __init__ should have created the AsyncZeroconf if zc_instance is None
        mock_zc_constructor.assert_called_once()
        assert listener._RecordListener__mdns is mock_owned_zc_instance

        await listener.start()
        # start() should have created the AsyncServiceBrowser
        mock_browser_constructor.assert_called_once_with(
            mock_owned_zc_instance.zeroconf,
            ["_testowned._tcp.local."],
            listener=listener
        )

        await listener.close()

        # Browser's cancel method should be called
        # AsyncServiceBrowser itself doesn't have a public cancel method in the same way sync ServiceBrowser does.
        # Its tasks are cancelled when AsyncZeroconf is closed.
        # The RecordListener.close() method sets self.__browser = None.
        # We can check if the underlying zeroconf's browser_close was called if it's part of the mock,
        # or rely on the fact that async_close on the zc instance should handle it.
        # For now, let's focus on zc.async_close().

        mock_owned_zc_instance.async_close.assert_awaited_once()

@pytest.mark.asyncio
async def test_close_with_shared_zc_does_not_close_shared_zc(mocker):
    """Test that RecordListener does not close a shared AsyncZeroconf instance."""
    mock_client = MockMdnsClient()
    mock_shared_zc_instance = AsyncMock(spec=AsyncZeroconf)
    # Mock the .zeroconf attribute
    mock_shared_zc_instance.zeroconf = MagicMock()
    # mock_shared_zc_instance.async_close = AsyncMock() # Already part of spec

    mock_service_browser_instance = AsyncMock(spec=AsyncServiceBrowser)

    with patch("tsercom.discovery.mdns.record_listener.AsyncServiceBrowser", return_value=mock_service_browser_instance) as mock_browser_constructor:
        listener = RecordListener(
            client=mock_client,
            service_type="_testshared._tcp.local.",
            zc_instance=mock_shared_zc_instance # Pass shared instance
        )

        assert listener._RecordListener__mdns is mock_shared_zc_instance

        await listener.start()
        mock_browser_constructor.assert_called_once_with(
            mock_shared_zc_instance.zeroconf,
            ["_testshared._tcp.local."],
            listener=listener
        )

        await listener.close()

        # Assert shared instance's async_close was NOT called
        mock_shared_zc_instance.async_close.assert_not_called()
