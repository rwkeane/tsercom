import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import socket
from typing import Generic, TypeVar, Callable, List, Dict, Optional, cast, Any
import asyncio  # For asyncio.sleep if needed, though mock usually suffices

from tsercom.discovery.mdns.mdns_listener import MdnsListener
from tsercom.discovery.mdns.instance_listener import (
    InstanceListener,
    ServiceInfo,
    ServiceInfoT,
)
from tsercom.discovery.mdns.record_listener import (
    RecordListener,
)  # For default factory test

# FakeMdnsListener is defined below in this file.


# Helper function for IP conversion
def str_to_ip_bytes(ip_str: str) -> bytes:
    """Converts an IPv4 string to bytes."""
    return socket.inet_aton(ip_str)


class FakeMdnsListener(MdnsListener):
    """
    A fake MdnsListener for testing InstanceListener.
    This fake listener is intended to be returned by a factory function
    passed to InstanceListener's constructor.
    """

    def __init__(self, client: MdnsListener.Client, service_type: str):
        super().__init__()
        self.client: MdnsListener.Client = client
        self.service_type: str = service_type
        self.update_service_calls: List[Dict[str, Any]] = []
        self.remove_service_calls: List[Dict[str, Any]] = []
        self.add_service_calls: List[Dict[str, Any]] = []

    def start(self) -> None:
        pass

    def update_service(self, zc: Any, type_: str, name: str) -> None:
        self.update_service_calls.append(
            {"zc": zc, "type_": type_, "name": name}
        )

    def remove_service(self, zc: Any, type_: str, name: str) -> None:
        self.remove_service_calls.append(
            {"zc": zc, "type_": type_, "name": name}
        )
        # Simulate calling the client's _on_service_removed if it exists
        if hasattr(self.client, "_on_service_removed") and callable(
            getattr(self.client, "_on_service_removed")
        ):
            # Assuming RecordListener would pass its UUID, mock or use a fixed one for tests
            mock_uuid = "fake-record-listener-uuid"
            self.client._on_service_removed(name, type_, mock_uuid)

    def add_service(self, zc: Any, type_: str, name: str) -> None:
        self.add_service_calls.append({"zc": zc, "type_": type_, "name": name})

    def simulate_service_added(
        self,
        name: str,
        port: int,
        addresses: List[bytes],
        txt_record: Dict[bytes, bytes | None],
    ) -> None:
        if self.client:
            self.client._on_service_added(name, port, addresses, txt_record)
        else:
            raise RuntimeError("FakeMdnsListener.client is not set.")

    def clear_simulation_history(self) -> None:
        self.update_service_calls = []
        self.remove_service_calls = []
        self.add_service_calls = []


# Generic type variable for use with FakeInstanceListenerClient
FClientServiceInfo = TypeVar("FClientServiceInfo", bound=ServiceInfo)


# InstanceListener.Client is now Generic[TServiceInfo], so FakeInstanceListenerClient
# just needs to inherit from it with its specific type var.
class FakeInstanceListenerClient(
    InstanceListener.Client, Generic[FClientServiceInfo]
):
    def __init__(self):
        # _on_service_added_mock is an AsyncMock to spy on the _on_service_added method.
        self._on_service_added_mock: AsyncMock = AsyncMock()
        self._on_service_removed_mock: AsyncMock = (
            AsyncMock()
        )  # Mock for removal
        # For manual tracking if preferred.
        self.received_services: List[FClientServiceInfo] = []
        self.removed_service_names: List[str] = (
            []
        )  # To store names of removed services
        self.added_event: Optional[asyncio.Event] = (
            None  # Event for added services
        )
        self.removed_event: Optional[asyncio.Event] = (
            None  # Event for removed services
        )

    # This is the concrete implementation of the abstract method.
    async def _on_service_added(
        self, connection_info: FClientServiceInfo
    ) -> None:
        """Concrete implementation of the abstract method.
        It calls the AsyncMock to record the call and allow assertions.
        """
        self.received_services.append(
            connection_info
        )  # Optional: direct recording
        await self._on_service_added_mock(connection_info)
        if self.added_event:
            self.added_event.set()

    async def _on_service_removed(self, service_name: str) -> None:
        """Concrete implementation of the abstract method for service removal."""
        self.removed_service_names.append(service_name)
        # Also remove from received_services if present by mDNS name
        self.received_services = [
            s for s in self.received_services if s.mdns_name != service_name
        ]
        await self._on_service_removed_mock(service_name)
        if self.removed_event:
            self.removed_event.set()

    # Kept for potential direct configuration if needed, though direct call to mock is primary.
    # def configure_mock_to_record(self):
    #     """Configures the AsyncMock to call our recording method."""
    #     # This might not be needed if _on_service_added directly calls the mock.
    #     pass

    def get_last_call_info(self) -> Optional[FClientServiceInfo]:
        if self._on_service_added_mock.call_args_list:
            # call_args is a tuple (args, kwargs), we want args[0]
            return self._on_service_added_mock.call_args_list[-1][0][0]
        return None

    def get_received_service_info(self) -> Optional[FClientServiceInfo]:
        """Helper to get the ServiceInfo from the last call to the AsyncMock."""
        if self._on_service_added_mock.call_count > 0:
            # Args are passed as a tuple, connection_info is the first arg
            return self._on_service_added_mock.call_args[0][0]
        return None

    def clear_calls(self):
        self._on_service_added_mock.reset_mock()
        self._on_service_removed_mock.reset_mock()
        self.received_services = []
        self.removed_service_names = []
        if self.added_event:
            self.added_event.clear()
        if self.removed_event:
            self.removed_event.clear()


class TestInstanceListener:
    """Unit tests for the InstanceListener class using factory-based injection."""

    SERVICE_TYPE = "_test_service._tcp.local."
    captured_fake_mdns_listener: Optional[FakeMdnsListener] = (
        None  # Class variable to capture the fake listener
    )

    def setup_method(self):
        """Set up common test objects for each test method."""
        self.mock_il_client = FakeInstanceListenerClient[ServiceInfo]()

        # Reset captured listener for each test
        TestInstanceListener.captured_fake_mdns_listener = None

        # This factory will be called by InstanceListener.__init__
        def fake_mdns_listener_factory(
            listener_client: MdnsListener.Client, service_type_arg: str
        ) -> FakeMdnsListener:
            # listener_client will be the InstanceListener instance itself
            # service_type_arg will be self.SERVICE_TYPE
            fake_listener = FakeMdnsListener(
                client=listener_client, service_type=service_type_arg
            )
            TestInstanceListener.captured_fake_mdns_listener = (
                fake_listener  # Capture the instance
            )
            return fake_listener

        self.factory_under_test = fake_mdns_listener_factory

        # Patch run_on_event_loop to schedule the coroutine produced by func_obj()
        def mock_run_on_event_loop_schedule_side_effect(func_obj):
            # func_obj is the partial. Calling it produces the coroutine.
            coro = func_obj()
            # Schedule the coroutine on the running event loop provided by pytest-asyncio.
            # The test itself might need a slight await asyncio.sleep(0) to ensure the task runs.
            return asyncio.create_task(coro)

        self.run_on_event_loop_patcher = patch(
            "tsercom.discovery.mdns.instance_listener.run_on_event_loop",
            side_effect=mock_run_on_event_loop_schedule_side_effect,
        )
        self.mock_run_on_event_loop = self.run_on_event_loop_patcher.start()

    def teardown_method(self):
        """Clean up after each test method."""
        self.run_on_event_loop_patcher.stop()

    def test_init_successful_with_factory(self):
        """Test successful initialization of InstanceListener with a factory."""
        instance_listener = InstanceListener[ServiceInfo](
            client=self.mock_il_client,
            service_type=self.SERVICE_TYPE,
            mdns_listener_factory=self.factory_under_test,
        )
        assert (
            TestInstanceListener.captured_fake_mdns_listener is not None
        ), "Factory was not called or did not capture listener"
        # Check that the client of the FakeMdnsListener is the InstanceListener instance
        assert (
            TestInstanceListener.captured_fake_mdns_listener.client
            == instance_listener
        )
        assert (
            TestInstanceListener.captured_fake_mdns_listener.service_type
            == self.SERVICE_TYPE
        )
        assert (
            instance_listener._InstanceListener__client == self.mock_il_client
        )
        assert (
            instance_listener._InstanceListener__listener
            == TestInstanceListener.captured_fake_mdns_listener
        )

    def test_init_successful_default_factory(self):
        """Test __init__ uses RecordListener by default."""
        # The global run_on_event_loop_patcher should generally be fine.
        # If RecordListener's __init__ itself uses run_on_event_loop in a specific
        # way that conflicts, this test might need further refinement for that patch.
        # For now, assume the default patcher is okay or RecordListener is not affected in __init__.
        instance_listener = InstanceListener[ServiceInfo](
            client=self.mock_il_client,
            service_type=self.SERVICE_TYPE,
            # No mdns_listener_factory provided
        )
        assert isinstance(
            instance_listener._InstanceListener__listener, RecordListener
        )

    def test_init_invalid_client_none(self):
        """Test __init__ with client=None raises ValueError."""
        with pytest.raises(
            ValueError, match="Client cannot be None for InstanceListener."
        ):
            InstanceListener(
                client=None,  # type: ignore
                service_type=self.SERVICE_TYPE,
                mdns_listener_factory=self.factory_under_test,
            )

    def test_init_invalid_client_type(self):
        """Test __init__ with invalid client type raises TypeError."""
        with pytest.raises(
            TypeError,
            match=r"Client must be InstanceListener\.Client, got \w+\.",
        ):  # Use regex for type name
            InstanceListener(
                client=MagicMock(),  # type: ignore
                service_type=self.SERVICE_TYPE,
                mdns_listener_factory=self.factory_under_test,
            )

    def test_init_invalid_service_type(self):
        """Test __init__ with invalid service_type raises TypeError."""
        with pytest.raises(
            TypeError, match=r"service_type must be str, got \w+\."
        ):  # Use regex for type name
            InstanceListener(
                client=self.mock_il_client,
                service_type=123,  # type: ignore
                mdns_listener_factory=self.factory_under_test,
            )

    @pytest.mark.asyncio
    async def test_on_service_added_success(self):
        """Test _on_service_added successfully processes a service and notifies client."""
        instance_listener = InstanceListener[ServiceInfo](
            client=self.mock_il_client,
            service_type=self.SERVICE_TYPE,
            mdns_listener_factory=self.factory_under_test,
        )
        assert TestInstanceListener.captured_fake_mdns_listener is not None

        record_name = "TestServiceInstance"
        port = 8080
        ip_str = "192.168.1.100"
        ip_bytes = str_to_ip_bytes(ip_str)
        txt_record = {b"name": b"My Readable Name", b"version": b"1.0"}

        TestInstanceListener.captured_fake_mdns_listener.simulate_service_added(
            record_name, port, [ip_bytes], txt_record
        )

        # Allow the created task to run
        await asyncio.sleep(0)
        self.mock_il_client._on_service_added_mock.assert_called_once()

        # Get the ServiceInfo object passed to the mock
        received_info = self.mock_il_client.get_received_service_info()
        assert received_info is not None
        assert isinstance(received_info, ServiceInfo)
        assert received_info.name == "My Readable Name"
        assert received_info.port == port
        assert received_info.addresses == [ip_str]
        assert received_info.mdns_name == record_name

    @pytest.mark.asyncio
    async def test_on_service_added_no_ip_addresses(self):
        """Test _on_service_added with no IP addresses does not notify client."""
        instance_listener = InstanceListener[ServiceInfo](
            client=self.mock_il_client,
            service_type=self.SERVICE_TYPE,
            mdns_listener_factory=self.factory_under_test,
        )
        assert TestInstanceListener.captured_fake_mdns_listener is not None
        TestInstanceListener.captured_fake_mdns_listener.simulate_service_added(
            "NoIPService", 8080, [], {b"name": b"No IP"}
        )
        await asyncio.sleep(0)  # Allow event loop to process potential tasks
        self.mock_il_client._on_service_added_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_on_service_added_ip_conversion_failure_all_invalid(self):
        """Test _on_service_added with all invalid IP addresses does not notify client."""
        instance_listener = InstanceListener[ServiceInfo](
            client=self.mock_il_client,
            service_type=self.SERVICE_TYPE,
            mdns_listener_factory=self.factory_under_test,
        )
        assert TestInstanceListener.captured_fake_mdns_listener is not None
        invalid_ip_bytes = b"this is not an ip"
        with patch(
            "socket.inet_ntoa", side_effect=socket.error("Invalid IP format")
        ):
            TestInstanceListener.captured_fake_mdns_listener.simulate_service_added(
                "InvalidIPService",
                8080,
                [invalid_ip_bytes],
                {b"name": b"Invalid IP"},
            )
        await asyncio.sleep(0)  # Allow event loop to process potential tasks
        self.mock_il_client._on_service_added_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_on_service_added_ip_conversion_failure_some_invalid(self):
        """Test _on_service_added with some invalid IP addresses processes valid ones."""
        instance_listener = InstanceListener[ServiceInfo](
            client=self.mock_il_client,
            service_type=self.SERVICE_TYPE,
            mdns_listener_factory=self.factory_under_test,
        )
        assert TestInstanceListener.captured_fake_mdns_listener is not None

        valid_ip_str = "192.168.1.101"
        valid_ip_bytes = str_to_ip_bytes(valid_ip_str)
        invalid_ip_bytes = b"badip"

        def mock_inet_ntoa(data):
            if data == valid_ip_bytes:
                return valid_ip_str
            raise socket.error("Invalid IP")

        with patch("socket.inet_ntoa", side_effect=mock_inet_ntoa):
            TestInstanceListener.captured_fake_mdns_listener.simulate_service_added(
                "MixedIPService",
                8080,
                [invalid_ip_bytes, valid_ip_bytes],
                {b"name": b"Mixed IP"},
            )

        await asyncio.sleep(0)  # Allow event loop to process potential tasks
        self.mock_il_client._on_service_added_mock.assert_called_once()
        received_info = self.mock_il_client.get_received_service_info()
        assert received_info is not None
        assert received_info.addresses == [valid_ip_str]

    @pytest.mark.asyncio
    async def test_on_service_added_txt_name_decoding_error(self):
        """Test _on_service_added with TXT name decoding error falls back to record_name."""
        instance_listener = InstanceListener[ServiceInfo](
            client=self.mock_il_client,
            service_type=self.SERVICE_TYPE,
            mdns_listener_factory=self.factory_under_test,
        )
        assert TestInstanceListener.captured_fake_mdns_listener is not None

        record_name = "UTF8ErrorService"
        ip_bytes = str_to_ip_bytes("192.168.1.102")
        txt_record_invalid_utf8 = {b"name": b"\xff\xfe"}  # Invalid UTF-8

        TestInstanceListener.captured_fake_mdns_listener.simulate_service_added(
            record_name, 8080, [ip_bytes], txt_record_invalid_utf8
        )

        await asyncio.sleep(0)  # Allow event loop to process potential tasks
        self.mock_il_client._on_service_added_mock.assert_called_once()
        received_info = self.mock_il_client.get_received_service_info()
        assert received_info is not None
        assert received_info.name == record_name

    @pytest.mark.asyncio
    async def test_on_service_added_txt_name_missing(self):
        """Test _on_service_added with no 'name' in TXT record falls back to record_name."""
        instance_listener = InstanceListener[ServiceInfo](
            client=self.mock_il_client,
            service_type=self.SERVICE_TYPE,
            mdns_listener_factory=self.factory_under_test,
        )
        assert TestInstanceListener.captured_fake_mdns_listener is not None

        record_name = "NoNameService"
        ip_bytes = str_to_ip_bytes("192.168.1.103")
        txt_record_no_name = {b"other_key": b"other_value"}

        TestInstanceListener.captured_fake_mdns_listener.simulate_service_added(
            record_name, 8080, [ip_bytes], txt_record_no_name
        )

        await asyncio.sleep(0)  # Allow event loop to process potential tasks
        self.mock_il_client._on_service_added_mock.assert_called_once()
        received_info = self.mock_il_client.get_received_service_info()
        assert received_info is not None
        assert received_info.name == record_name  # Ensured this is .name


# Ensure no trailing characters or syntax errors exist beyond this point.
