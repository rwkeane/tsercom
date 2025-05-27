import asyncio
import socket
import functools # Not strictly needed for new tests, but was in old file
from typing import Dict, List, Optional

import pytest
from pytest_mock import MockerFixture

from tsercom.discovery.mdns.instance_listener import InstanceListener, TServiceInfo
# ConcreteRecordListener needed for FakeRecordListener's client type hint
from tsercom.discovery.mdns.record_listener import RecordListener as ConcreteRecordListener
from tsercom.discovery.mdns.protocols import RecordListenerProtocol
from tsercom.discovery.service_info import ServiceInfo
from tsercom.threading.aio.global_event_loop import (
    set_tsercom_event_loop_to_current_thread,
    clear_tsercom_event_loop,
    is_global_event_loop_set,
)

# --- Helper Mocks (already defined, keep them) ---
def mock_run_on_event_loop_force_execute(func_partial, event_loop=None, *args, **kwargs):
    coro = func_partial()
    target_loop = event_loop if event_loop else asyncio.get_running_loop()
    result = target_loop.run_until_complete(coro)
    future_result = target_loop.create_future()
    future_result.set_result(result)
    return future_result

def realistic_inet_ntoa_mock(packed_ip_bytes):
    if not isinstance(packed_ip_bytes, bytes):
        raise TypeError(f"inet_ntoa: a bytes-like object is required, not '{type(packed_ip_bytes).__name__}'")
    if packed_ip_bytes == b"fail_marker":
        raise socket.error("Simulated conversion failure")
    if len(packed_ip_bytes) == 4:
        return ".".join(map(str, packed_ip_bytes))
    raise ValueError(f"Invalid input to inet_ntoa mock (not 4 bytes and not fail_marker): {packed_ip_bytes!r}")

# --- FakeRecordListener (already defined, keep it) ---
class FakeRecordListener(RecordListenerProtocol):
    def __init__(self, client: ConcreteRecordListener.Client, service_type: str):
        self.client: ConcreteRecordListener.Client = client
        self.service_type: str = service_type

    def simulate_service_added(
        self, record_name: str, port: int, addresses: List[bytes],
        txt_record: Dict[bytes, bytes | None]
    ) -> None:
        # InstanceListener is the client to FakeRecordListener, so self.client is an InstanceListener.
        # InstanceListener._on_service_added is the method that RecordListener would call.
        self.client._on_service_added(record_name, port, addresses, txt_record)

# --- Pytest Fixtures ---
@pytest.fixture
def mock_instance_listener_client(mocker: MockerFixture) -> InstanceListener.Client:
    # This is the client that InstanceListener will notify about discovered services.
    # It adheres to the InstanceListener.Client ABC.
    client = mocker.create_autospec(InstanceListener.Client, instance=True, spec_set=True)
    return client

@pytest.fixture
def managed_event_loop():
    if not is_global_event_loop_set():
        set_tsercom_event_loop_to_current_thread()
    yield
    clear_tsercom_event_loop()

# --- Test Class ---
@pytest.mark.usefixtures("managed_event_loop")
class TestInstanceListenerWithFake:

    @pytest.fixture
    def fake_record_listener(self, mocker: MockerFixture) -> FakeRecordListener:
        # Create a dummy client for the FakeRecordListener for now.
        # The actual client (the InstanceListener instance) will be set in each test.
        # This is a bit awkward; ideally, FakeRecordListener wouldn't need a client in __init__
        # if its purpose is to be controlled by the test and simulate calls *to* its client.
        # However, its `simulate_service_added` calls `self.client._on_service_added`.
        # An alternative: tests construct FakeRecordListener after creating InstanceListener.
        # For now, let's make a dummy client that FakeRecordListener can hold.
        dummy_rl_client = mocker.Mock(spec=ConcreteRecordListener.Client)
        return FakeRecordListener(dummy_rl_client, "_test_type._tcp.local.")

    def test_init_success(
        self,
        mock_instance_listener_client: InstanceListener.Client,
        fake_record_listener: FakeRecordListener # Using the simpler fixture for now
    ):
        listener = InstanceListener[ServiceInfo](mock_instance_listener_client, fake_record_listener)
        assert listener._InstanceListener__client == mock_instance_listener_client
        assert listener._InstanceListener__listener == fake_record_listener

    def test_init_raises_value_error_on_none_client(
        self, fake_record_listener: FakeRecordListener
    ):
        with pytest.raises(ValueError, match="Client argument cannot be None"):
            InstanceListener[ServiceInfo](None, fake_record_listener) # type: ignore

    def test_init_raises_type_error_on_invalid_client(
        self, fake_record_listener: FakeRecordListener
    ):
        class NotAClient:
            pass
        with pytest.raises(TypeError, match="Client must be an instance of InstanceListener.Client"):
            InstanceListener[ServiceInfo](NotAClient(), fake_record_listener) # type: ignore

    @pytest.mark.asyncio
    async def test_service_added_with_txt_name(
        self, mocker: MockerFixture, mock_instance_listener_client: InstanceListener.Client
    ):
        mock_run_on_event_loop = mocker.patch('tsercom.threading.aio.aio_utils.run_on_event_loop', side_effect=mock_run_on_event_loop_force_execute)
        mock_inet_ntoa = mocker.patch("socket.inet_ntoa", side_effect=realistic_inet_ntoa_mock)

        # Create InstanceListener and FakeRecordListener, wiring them up
        # Here, 'listener' (InstanceListener) is the client for 'fake_rl'
        listener = InstanceListener[ServiceInfo](mock_instance_listener_client, None) # type: ignore
        fake_rl = FakeRecordListener(client=listener, service_type="_test._tcp.local.")
        listener._InstanceListener__listener = fake_rl # Set the listener manually after construction

        spy_convert_service_info = mocker.spy(listener, "_convert_service_info")

        record_name = "TestService.local."
        port = 1234
        addresses_bytes = [b'\xc0\xa8\x01\x01'] # 192.168.1.1
        expected_ips = ["192.168.1.1"]
        txt_record = {b"name": b"Friendly"}

        fake_rl.simulate_service_added(record_name, port, addresses_bytes, txt_record)
        await asyncio.sleep(0)

        mock_run_on_event_loop.assert_called_once()
        mock_instance_listener_client._on_service_added.assert_called_once() # type: ignore
        
        received_info = mock_instance_listener_client._on_service_added.call_args[0][0] # type: ignore
        assert isinstance(received_info, ServiceInfo)
        assert received_info.name == "Friendly"
        assert received_info.port == port
        assert sorted(received_info.addresses) == sorted(expected_ips)
        assert received_info.mdns_name == record_name
        
        spy_convert_service_info.assert_called_once_with(mocker.ANY, txt_record)
        assert received_info is spy_convert_service_info.spy_return


    @pytest.mark.asyncio
    async def test_service_added_without_txt_name(
        self, mocker: MockerFixture, mock_instance_listener_client: InstanceListener.Client
    ):
        mock_run_on_event_loop = mocker.patch('tsercom.threading.aio.aio_utils.run_on_event_loop', side_effect=mock_run_on_event_loop_force_execute)
        mock_inet_ntoa = mocker.patch("socket.inet_ntoa", side_effect=realistic_inet_ntoa_mock)

        listener = InstanceListener[ServiceInfo](mock_instance_listener_client, None) # type: ignore
        fake_rl = FakeRecordListener(client=listener, service_type="_test._tcp.local.")
        listener._InstanceListener__listener = fake_rl

        record_name = "AnotherService.local."
        port = 54321
        addresses_bytes = [b'\x0a\x00\x00\x05'] # 10.0.0.5
        expected_ips = ["10.0.0.5"]
        txt_record: Dict[bytes, bytes | None] = {} 

        fake_rl.simulate_service_added(record_name, port, addresses_bytes, txt_record)
        await asyncio.sleep(0)

        mock_run_on_event_loop.assert_called_once()
        mock_instance_listener_client._on_service_added.assert_called_once() # type: ignore
        received_info = mock_instance_listener_client._on_service_added.call_args[0][0] # type: ignore
        assert isinstance(received_info, ServiceInfo)
        assert received_info.name == record_name 
        assert received_info.port == port
        assert sorted(received_info.addresses) == sorted(expected_ips)
        assert received_info.mdns_name == record_name

    @pytest.mark.asyncio
    async def test_service_added_partial_ip_conversion_failure(
        self, mocker: MockerFixture, mock_instance_listener_client: InstanceListener.Client
    ):
        mock_run_on_event_loop = mocker.patch('tsercom.threading.aio.aio_utils.run_on_event_loop', side_effect=mock_run_on_event_loop_force_execute)
        mock_inet_ntoa = mocker.patch("socket.inet_ntoa", side_effect=realistic_inet_ntoa_mock)
        
        listener = InstanceListener[ServiceInfo](mock_instance_listener_client, None) # type: ignore
        fake_rl = FakeRecordListener(client=listener, service_type="_test._tcp.local.")
        listener._InstanceListener__listener = fake_rl

        record_name = "PartialFailService.local."
        port = 8080
        addresses_bytes = [b'\x0a\x00\x00\x01', b"fail_marker", b'\x0a\x00\x00\x02']
        expected_ips = ["10.0.0.1", "10.0.0.2"]
        txt_record = {b"name": b"Partial"}

        fake_rl.simulate_service_added(record_name, port, addresses_bytes, txt_record)
        await asyncio.sleep(0)

        mock_run_on_event_loop.assert_called_once()
        mock_instance_listener_client._on_service_added.assert_called_once() # type: ignore
        received_info = mock_instance_listener_client._on_service_added.call_args[0][0] # type: ignore
        assert isinstance(received_info, ServiceInfo)
        assert received_info.name == "Partial"
        assert sorted(received_info.addresses) == sorted(expected_ips)

    @pytest.mark.asyncio
    async def test_service_added_all_ip_conversion_failure(
        self, mocker: MockerFixture, mock_instance_listener_client: InstanceListener.Client
    ):
        mock_run_on_event_loop = mocker.patch('tsercom.threading.aio.aio_utils.run_on_event_loop', side_effect=mock_run_on_event_loop_force_execute)
        mocker.patch("socket.inet_ntoa", side_effect=socket.error("Simulated conversion failure for all"))

        listener = InstanceListener[ServiceInfo](mock_instance_listener_client, None) # type: ignore
        fake_rl = FakeRecordListener(client=listener, service_type="_test._tcp.local.")
        listener._InstanceListener__listener = fake_rl
        
        record_name = "TotalFailService.local."
        port = 8081
        addresses_bytes = [b"fail1", b"fail2"] # These will cause realistic_inet_ntoa_mock to raise ValueError
        txt_record = {b"name": b"Fail"}

        fake_rl.simulate_service_added(record_name, port, addresses_bytes, txt_record)
        await asyncio.sleep(0)

        mock_run_on_event_loop.assert_not_called()
        mock_instance_listener_client._on_service_added.assert_not_called() # type: ignore

    @pytest.mark.asyncio
    async def test_service_added_no_ip_addresses(
        self, mocker: MockerFixture, mock_instance_listener_client: InstanceListener.Client
    ):
        mock_run_on_event_loop = mocker.patch('tsercom.threading.aio.aio_utils.run_on_event_loop', side_effect=mock_run_on_event_loop_force_execute)
        mock_socket_inet_ntoa = mocker.patch("socket.inet_ntoa") 

        listener = InstanceListener[ServiceInfo](mock_instance_listener_client, None) # type: ignore
        fake_rl = FakeRecordListener(client=listener, service_type="_test._tcp.local.")
        listener._InstanceListener__listener = fake_rl
        
        record_name = "NoIPService.local."
        port = 8082
        addresses_bytes: List[bytes] = [] 
        txt_record = {b"name": b"NoIP"}

        fake_rl.simulate_service_added(record_name, port, addresses_bytes, txt_record)
        await asyncio.sleep(0)

        mock_run_on_event_loop.assert_not_called()
        mock_instance_listener_client._on_service_added.assert_not_called() # type: ignore
        mock_socket_inet_ntoa.assert_not_called()

    @pytest.mark.asyncio
    async def test_service_added_uses_convert_service_info_output(
        self, mocker: MockerFixture, mock_instance_listener_client: InstanceListener.Client
    ):
        mock_run_on_event_loop = mocker.patch('tsercom.threading.aio.aio_utils.run_on_event_loop', side_effect=mock_run_on_event_loop_force_execute)
        mock_inet_ntoa = mocker.patch("socket.inet_ntoa", side_effect=realistic_inet_ntoa_mock)

        listener = InstanceListener[ServiceInfo](mock_instance_listener_client, None) # type: ignore
        fake_rl = FakeRecordListener(client=listener, service_type="_test._tcp.local.")
        listener._InstanceListener__listener = fake_rl
        
        custom_service_info = ServiceInfo("CustomName", 9999, ["custom_ip"], "CustomMdnsName")
        # Patch the _convert_service_info method on the listener instance
        mocker.patch.object(listener, "_convert_service_info", return_value=custom_service_info)

        record_name = "ConvertTestService.local."
        port = 7000
        addresses_bytes = [b'\x01\x02\x03\x04'] 
        expected_converted_ip = "1.2.3.4" # From realistic_inet_ntoa_mock
        txt_record: Dict[bytes, bytes | None] = {}

        fake_rl.simulate_service_added(record_name, port, addresses_bytes, txt_record)
        await asyncio.sleep(0)

        mock_run_on_event_loop.assert_called_once()
        mock_instance_listener_client._on_service_added.assert_called_once() # type: ignore
        
        received_info_from_client = mock_instance_listener_client._on_service_added.call_args[0][0] # type: ignore
        assert received_info_from_client is custom_service_info 

        listener._convert_service_info.assert_called_once() # type: ignore
        # Check arguments passed to the (now mocked) _convert_service_info
        call_args_to_convert = listener._convert_service_info.call_args[0] # type: ignore
        base_info_passed_to_convert = call_args_to_convert[0]
        assert isinstance(base_info_passed_to_convert, ServiceInfo)
        assert base_info_passed_to_convert.name == record_name 
        assert base_info_passed_to_convert.port == port
        assert base_info_passed_to_convert.addresses == [expected_converted_ip]
        assert base_info_passed_to_convert.mdns_name == record_name
        assert call_args_to_convert[1] == txt_record

    def test_convert_service_info_default_behavior(
        self, mock_instance_listener_client: InstanceListener.Client
    ):
        # For this test, the RecordListener isn't strictly needed, can pass None or a dummy.
        # However, InstanceListener constructor expects a RecordListenerProtocol.
        # We can use a simple mock for RecordListenerProtocol if FakeRecordListener is too complex here.
        dummy_rl_protocol_mock = mocker.Mock(spec=RecordListenerProtocol)
        listener = InstanceListener[ServiceInfo](mock_instance_listener_client, dummy_rl_protocol_mock)
        
        input_service_info = ServiceInfo("Test", 123, ["1.2.3.4"], "test.local")
        txt_record: Dict[bytes, bytes | None] = {b"key": b"value"}

        output_service_info = listener._convert_service_info(input_service_info, txt_record)

        assert output_service_info is input_service_info
        assert output_service_info.name == "Test"
        assert output_service_info.port == 123
        assert output_service_info.addresses == ["1.2.3.4"]
        assert output_service_info.mdns_name == "test.local"

    # This fixture was used by old tests, not needed by new ones using FakeRecordListener directly or specific mocks.
    # @pytest.fixture
    # def mock_record_listener(mocker: MockerFixture) -> ConcreteRecordListener: 
    #     return mocker.Mock(spec=ConcreteRecordListener)

```
