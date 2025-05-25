import asyncio
import pytest
from unittest.mock import patch, AsyncMock, MagicMock, call
import functools  # For functools.partial
import socket  # For socket.inet_ntoa and AF_INET type

from tsercom.discovery.mdns.instance_listener import InstanceListener
from tsercom.discovery.service_info import ServiceInfo

# Import the module where RecordListener is defined to patch it
import tsercom.discovery.mdns.record_listener as record_listener_module

# Import the module where run_on_event_loop is defined to patch it
import tsercom.threading.aio.aio_utils as aio_utils_module


# Mock for run_on_event_loop
async def mock_run_on_event_loop_for_instance_listener(
    func_partial, event_loop=None, *args, **kwargs
):
    """
    Mock for run_on_event_loop. Assumes func_partial returns a coroutine.
    This mock will be patched into tsercom.threading.aio.aio_utils.run_on_event_loop.
    """
    print(
        f"MOCK_RUN_ON_EVENT_LOOP_FOR_INSTANCE_LISTENER CALLED with func_partial: {func_partial}"
    )
    # func_partial is expected to be functools.partial(self._InstanceListener__on_service_added_impl, service_info_instance)
    # Calling it should return a coroutine if __on_service_added_impl is async.
    coro = func_partial()

    if not asyncio.iscoroutine(coro):
        print(
            f"  WARNING: Mock run_on_event_loop received non-coroutine: {type(coro)}"
        )
    else:
        print(f"  Awaiting coroutine from partial: {coro}")
        await coro  # Execute the coroutine
        print(f"  Coroutine awaited: {coro}")

    # Original run_on_event_loop returns a Future.
    f = asyncio.Future()
    try:
        current_loop = asyncio.get_running_loop()
        if not current_loop.is_closed():
            asyncio.ensure_future(f, loop=current_loop)
    except RuntimeError:  # pragma: no cover
        pass
    f.set_result(None)
    return f


@pytest.mark.asyncio
class TestInstanceListener:

    @pytest.fixture
    def mock_client(self):
        client = MagicMock(name="MockInstanceListenerClient")
        # _on_service_added on the client is expected to be a regular method,
        # but the call is wrapped by run_on_event_loop in the SUT,
        # which implies the actual implementation might be async or called from an async context.
        # For testing the InstanceListener's call to it via the mocked run_on_event_loop,
        # a simple MagicMock is sufficient to capture calls.
        # If __on_service_added_impl itself awaits client._on_service_added, then this needs to be AsyncMock.
        # Looking at InstanceListener, __on_service_added_impl is async and awaits client._on_service_added.
        client._on_service_added = AsyncMock(
            name="client._on_service_added_async_mock"
        )
        return client

    def test_init_creates_record_listener(self, mock_client):
        print("\n--- Test: test_init_creates_record_listener ---")
        service_type = "test_service._tcp.local."

        with patch.object(
            record_listener_module, "RecordListener", autospec=True
        ) as MockRecordListenerCtor:
            print(
                f"  RecordListener patched. Mock object: {MockRecordListenerCtor}"
            )

            instance_listener = InstanceListener(mock_client, service_type)
            print("  InstanceListener instantiated.")

            # Assert RecordListener was initialized with self (the InstanceListener instance) and service_type
            MockRecordListenerCtor.assert_called_once_with(
                instance_listener, service_type
            )
            print(
                "  Assertion: RecordListener constructor called correctly - PASSED"
            )
            assert (
                instance_listener._record_listener
                is MockRecordListenerCtor.return_value
            )
            print(
                "  Assertion: instance_listener._record_listener is set - PASSED"
            )
        print("--- Test: test_init_creates_record_listener finished ---")

    @patch.object(socket, "inet_ntoa", autospec=True)  # Patch socket.inet_ntoa
    async def test_on_service_added_successful_population_with_txt_name(
        self, mock_inet_ntoa, mock_client
    ):
        print(
            "\n--- Test: test_on_service_added_successful_population_with_txt_name ---"
        )
        service_type = "test_service_txt._tcp.local."
        mdns_record_name = "MyDevice._test_service_txt._tcp.local."
        port = 8080
        addr_bytes1 = b"\x01\x02\x03\x04"  # 1.2.3.4
        addr_bytes2 = b"\x05\x06\x07\x08"  # 5.6.7.8
        txt_record = {b"name": b"Friendly Device Name"}

        # Configure mock_inet_ntoa
        def inet_ntoa_side_effect(addr_bytes_param):
            # Simple conversion for test, assuming 4 bytes to dot-decimal
            if len(addr_bytes_param) == 4:
                return ".".join(map(str, addr_bytes_param))
            return f"unmocked_format_for_{addr_bytes_param!r}"  # Fallback for unexpected format

        mock_inet_ntoa.side_effect = inet_ntoa_side_effect
        print(f"  socket.inet_ntoa patched with side effect: {mock_inet_ntoa}")

        # Patch run_on_event_loop at its source module
        with patch.object(
            aio_utils_module,
            "run_on_event_loop",
            new=mock_run_on_event_loop_for_instance_listener,
        ) as mock_run_on_event_loop_patch_obj:
            print(
                f"  run_on_event_loop patched. Mock object: {mock_run_on_event_loop_patch_obj}"
            )

            # We need to bypass RecordListener creation for this direct test of _on_service_added
            with patch.object(
                record_listener_module, "RecordListener"
            ):  # Keep RecordListener from starting
                instance_listener = InstanceListener(mock_client, service_type)
                print(
                    "  InstanceListener instantiated for _on_service_added test."
                )

                # Call the method to be tested (which is a callback for RecordListener)
                # This method is synchronous, but it calls __populate_service_info (sync)
                # and then schedules __on_service_added_impl (async) via run_on_event_loop.
                instance_listener._on_service_added(
                    mdns_record_name,
                    port,
                    [addr_bytes1, addr_bytes2],
                    txt_record,
                )
                print("  instance_listener._on_service_added called.")

                # Assert run_on_event_loop (our mock) was called
                mock_run_on_event_loop_patch_obj.assert_called_once()
                print(
                    "  Assertion: mock_run_on_event_loop_patch_obj.assert_called_once() - PASSED"
                )

                # Assert client._on_service_added was called (via the mocked run_on_event_loop and __on_service_added_impl)
                mock_client._on_service_added.assert_called_once()
                print(
                    "  Assertion: mock_client._on_service_added.assert_called_once() - PASSED"
                )

                # Inspect the ServiceInfo object passed to client._on_service_added
                call_args = mock_client._on_service_added.call_args
                assert (
                    call_args is not None
                ), "Client's _on_service_added was not called with arguments"
                service_info_arg: ServiceInfo = call_args[0][0]

                assert isinstance(service_info_arg, ServiceInfo)
                assert service_info_arg.name == "Friendly Device Name"
                assert service_info_arg.port == port
                assert sorted(service_info_arg.addresses) == sorted(
                    ["1.2.3.4", "5.6.7.8"]
                )
                assert service_info_arg.mdns_name == mdns_record_name
                print("  Assertion: ServiceInfo content verified - PASSED")
        print(
            "--- Test: test_on_service_added_successful_population_with_txt_name finished ---"
        )

    @patch.object(socket, "inet_ntoa", autospec=True)
    async def test_on_service_added_successful_population_no_txt_name(
        self, mock_inet_ntoa, mock_client
    ):
        print(
            "\n--- Test: test_on_service_added_successful_population_no_txt_name ---"
        )
        service_type = "test_service_no_txt._tcp.local."
        mdns_record_name = "RawDeviceName._test_service_no_txt._tcp.local."
        port = 8081
        addr_bytes = [b"\x0a\x00\x00\x01"]  # 10.0.0.1
        txt_record_empty = {}
        txt_record_other_keys = {b"other_key": b"other_value"}

        mock_inet_ntoa.side_effect = lambda addr_bytes_param: ".".join(
            map(str, addr_bytes_param)
        )

        with patch.object(
            aio_utils_module,
            "run_on_event_loop",
            new=mock_run_on_event_loop_for_instance_listener,
        ) as mock_run_patch_obj:
            with patch.object(record_listener_module, "RecordListener"):
                instance_listener = InstanceListener(mock_client, service_type)

                # Scenario 1: Empty TXT record
                instance_listener._on_service_added(
                    mdns_record_name, port, addr_bytes, txt_record_empty
                )
                mock_run_patch_obj.assert_called_once()
                mock_client._on_service_added.assert_called_once()
                service_info_arg1: ServiceInfo = (
                    mock_client._on_service_added.call_args[0][0]
                )
                assert (
                    service_info_arg1.name == mdns_record_name
                )  # Should default to mdns_record_name
                print(
                    f"  Assertion (empty TXT): ServiceInfo.name defaulted to mdns_name ('{service_info_arg1.name}') - PASSED"
                )

                # Reset for Scenario 2
                mock_run_patch_obj.reset_mock()
                mock_client._on_service_added.reset_mock()

                # Scenario 2: TXT record with other keys but not 'name'
                instance_listener._on_service_added(
                    mdns_record_name, port, addr_bytes, txt_record_other_keys
                )
                mock_run_patch_obj.assert_called_once()
                mock_client._on_service_added.assert_called_once()
                service_info_arg2: ServiceInfo = (
                    mock_client._on_service_added.call_args[0][0]
                )
                assert (
                    service_info_arg2.name == mdns_record_name
                )  # Should also default
                print(
                    f"  Assertion (other keys TXT): ServiceInfo.name defaulted to mdns_name ('{service_info_arg2.name}') - PASSED"
                )
        print(
            "--- Test: test_on_service_added_successful_population_no_txt_name finished ---"
        )

    @patch.object(
        socket, "inet_ntoa", side_effect=socket.error("inet_ntoa failed")
    )  # inet_ntoa raises error
    async def test_on_service_added_populate_service_info_returns_none_bad_address(
        self, mock_inet_ntoa_raising_error, mock_client
    ):
        print(
            "\n--- Test: test_on_service_added_populate_service_info_returns_none_bad_address ---"
        )
        service_type = "test_bad_addr._tcp.local."
        with patch.object(
            aio_utils_module,
            "run_on_event_loop",
            new=mock_run_on_event_loop_for_instance_listener,
        ) as mock_run_patch_obj:
            with patch.object(record_listener_module, "RecordListener"):
                instance_listener = InstanceListener(mock_client, service_type)

                # Call _on_service_added. __populate_service_info should return None due to inet_ntoa error.
                instance_listener._on_service_added(
                    "rec_bad_addr", 123, [b"badip"], {}
                )

                # Assert client._on_service_added was NOT called
                mock_client._on_service_added.assert_not_called()
                print(
                    "  Assertion: client._on_service_added.assert_not_called() - PASSED"
                )
                # Assert run_on_event_loop was NOT called
                mock_run_patch_obj.assert_not_called()
                print(
                    "  Assertion: mock_run_on_event_loop_patch_obj.assert_not_called() - PASSED"
                )
        print(
            "--- Test: test_on_service_added_populate_service_info_returns_none_bad_address finished ---"
        )

    async def test_on_service_added_empty_address_bytes_list(
        self, mock_client
    ):
        print("\n--- Test: test_on_service_added_empty_address_bytes_list ---")
        service_type = "test_empty_addr_list._tcp.local."
        with patch.object(
            aio_utils_module,
            "run_on_event_loop",
            new=mock_run_on_event_loop_for_instance_listener,
        ) as mock_run_patch_obj:
            with patch.object(record_listener_module, "RecordListener"):
                instance_listener = InstanceListener(mock_client, service_type)

                # Call _on_service_added with an empty list for addresses_bytes
                instance_listener._on_service_added(
                    "rec_empty_addr", 123, [], {}
                )

                mock_client._on_service_added.assert_not_called()
                print(
                    "  Assertion: client._on_service_added.assert_not_called() - PASSED"
                )
                mock_run_patch_obj.assert_not_called()
                print(
                    "  Assertion: mock_run_on_event_loop_patch_obj.assert_not_called() - PASSED"
                )
        print(
            "--- Test: test_on_service_added_empty_address_bytes_list finished ---"
        )

    def test_convert_service_info_is_identity(
        self, mock_client
    ):  # This is a synchronous method
        print("\n--- Test: test_convert_service_info_is_identity ---")
        instance_listener = InstanceListener(mock_client, "any_service")
        mock_service_info = MagicMock(spec=ServiceInfo)

        # Call the method (it's protected, but we test its behavior)
        returned_info = instance_listener._convert_service_info(mock_service_info)  # type: ignore

        assert (
            returned_info is mock_service_info
        ), "Expected _convert_service_info to return the same object"
        print("  Assertion: _convert_service_info returns identity - PASSED")
        print("--- Test: test_convert_service_info_is_identity finished ---")
