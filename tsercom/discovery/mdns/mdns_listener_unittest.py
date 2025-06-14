from zeroconf import Zeroconf
from tsercom.discovery.mdns.mdns_listener import MdnsListener

# Use an alias for MdnsListener.Client for clarity in type hints
from tsercom.discovery.mdns.mdns_listener import (
    MdnsListener as IMdnsListenerClientProto,
)


import pytest


# A mock client for the listener, conforming to MdnsListener.Client interface
class MockMdnsClient(IMdnsListenerClientProto.Client):
    async def _on_service_added(
        self,
        name: str,
        port: int,
        addresses: list[bytes],
        txt_record: dict[bytes, bytes | None],
    ) -> None:
        pass

    async def _on_service_removed(
        self, name: str, service_type: str, record_listener_uuid: str
    ) -> None:
        pass


class FaultyCustomListener(MdnsListener):
    def __init__(
        self, client: IMdnsListenerClientProto.Client, service_type: str
    ):
        # Calling superclass __init__ for MdnsListener.
        super().__init__()
        self._client = client
        self._service_type = service_type

    # Required ABC methods for instantiation
    async def start(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def add_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        pass

    async def remove_service(
        self, zc: Zeroconf, type_: str, name: str
    ) -> None:
        pass

    async def update_service(
        self, zc: Zeroconf, type_: str, name: str
    ) -> None:
        pass


@pytest.mark.asyncio
async def test_custom_listener_instantiation_failure() -> None:
    mock_client = MockMdnsClient()
    service_type = "_test_service._tcp.local."

    listener = FaultyCustomListener(mock_client, service_type)
    # If FaultyCustomListener.start() was called here, it would need to be awaited.
    # For this test, just instantiation is checked.
    assert listener is not None
