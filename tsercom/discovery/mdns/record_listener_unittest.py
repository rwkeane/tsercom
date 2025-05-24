import pytest
from unittest.mock import patch, MagicMock, ANY
import socket # For creating packed IP addresses if needed for ServiceInfo.addresses

from tsercom.discovery.mdns.record_listener import RecordListener
# Import modules to be patched
import zeroconf as zeroconf_module # For Zeroconf, ServiceBrowser, ServiceInfo (if used as spec)

# Define a type for the client mock for clarity
MockRecordListenerClient = MagicMock

class TestRecordListener:

    @pytest.fixture
    def mock_client(self) -> MockRecordListenerClient:
        client = MagicMock(name="MockRecordListenerClient")
        # _on_service_added is expected to be a regular method by RecordListener
        client._on_service_added = MagicMock(name="client_on_service_added_method")
        return client

    @patch.object(zeroconf_module, 'ServiceBrowser', autospec=True)
    @patch.object(zeroconf_module, 'Zeroconf', autospec=True)
    def test_init_creates_zeroconf_and_service_browser(
        self, MockZeroconf, MockServiceBrowser, mock_client
    ):
        print("\n--- Test: test_init_creates_zeroconf_and_service_browser ---")
        mock_zc_instance = MockZeroconf.return_value
        service_type = "_myservice._tcp" # Type without .local. suffix
        expected_service_type_with_local = f"{service_type}.local."

        listener = RecordListener(mock_client, service_type)
        print("  RecordListener instantiated.")

        MockZeroconf.assert_called_once_with()
        print("  Assertion: Zeroconf() called - PASSED")
        
        # The listener instance itself is passed as the handler to ServiceBrowser
        MockServiceBrowser.assert_called_once_with(
            mock_zc_instance, 
            expected_service_type_with_local, 
            listener # Instance of RecordListener
        )
        print("  Assertion: ServiceBrowser called with correct args - PASSED")
        
        assert listener._zeroconf is mock_zc_instance
        assert listener._browser is MockServiceBrowser.return_value
        assert listener._RecordListener__expected_type == expected_service_type_with_local # Check private attr
        print("--- Test: test_init_creates_zeroconf_and_service_browser finished ---")

    # --- Tests for add_service ---
    # These tests also implicitly cover update_service due to similar internal logic flow path via __handle_service_info

    def run_service_change_test(
        self, method_name: str, mock_client, 
        mock_zc_instance_for_get_info: MagicMock, # Mock Zeroconf instance passed to add/update
        service_type_param: str, 
        service_name_param: str,
        get_service_info_return_value: any # What mock_zc_instance_for_get_info.get_service_info returns
    ):
        """Helper to run common logic for add_service and update_service tests."""
        expected_listener_type = "_goodservice._tcp.local."
        
        # Patch Zeroconf and ServiceBrowser for the __init__ call within RecordListener
        with patch.object(zeroconf_module, 'ServiceBrowser'), \
             patch.object(zeroconf_module, 'Zeroconf'): 
            listener = RecordListener(mock_client, "_goodservice._tcp") # Type without .local.

        # Configure the get_service_info mock on the passed Zeroconf instance
        mock_zc_instance_for_get_info.get_service_info.return_value = get_service_info_return_value
        
        # Get the method to call (add_service or update_service)
        method_to_call = getattr(listener, method_name)
        method_to_call(mock_zc_instance_for_get_info, service_type_param, service_name_param)


    @pytest.mark.parametrize("method_to_test", ["add_service", "update_service"])
    def test_service_change_type_mismatch(self, method_to_test, mock_client):
        print(f"\n--- Test: test_service_change_type_mismatch (method: {method_to_test}) ---")
        mock_zeroconf_runtime_instance = MagicMock(spec=zeroconf_module.Zeroconf)

        self.run_service_change_test(
            method_name=method_to_test,
            mock_client=mock_client,
            mock_zc_instance_for_get_info=mock_zeroconf_runtime_instance,
            service_type_param="_badservice._tcp.local.", # Mismatched type
            service_name_param="any.service.name",
            get_service_info_return_value=MagicMock(spec=zeroconf_module.ServiceInfo) # Doesn't matter for this test
        )
        
        mock_zeroconf_runtime_instance.get_service_info.assert_not_called()
        mock_client._on_service_added.assert_not_called()
        print(f"  Assertion: get_service_info and client._on_service_added not called due to type mismatch - PASSED")
        print(f"--- Test: test_service_change_type_mismatch (method: {method_to_test}) finished ---")

    @pytest.mark.parametrize("method_to_test", ["add_service", "update_service"])
    def test_service_change_info_is_none(self, method_to_test, mock_client):
        print(f"\n--- Test: test_service_change_info_is_none (method: {method_to_test}) ---")
        mock_zeroconf_runtime_instance = MagicMock(spec=zeroconf_module.Zeroconf)

        self.run_service_change_test(
            method_name=method_to_test,
            mock_client=mock_client,
            mock_zc_instance_for_get_info=mock_zeroconf_runtime_instance,
            service_type_param="_goodservice._tcp.local.", # Matching type for listener
            service_name_param="any.service.name",
            get_service_info_return_value=None # Simulate get_service_info returning None
        )
        
        # get_service_info should be called
        mock_zeroconf_runtime_instance.get_service_info.assert_called_once_with(
            "_goodservice._tcp.local.", "any.service.name"
        )
        mock_client._on_service_added.assert_not_called()
        print(f"  Assertion: client._on_service_added not called as ServiceInfo was None - PASSED")
        print(f"--- Test: test_service_change_info_is_none (method: {method_to_test}) finished ---")

    @pytest.mark.parametrize("method_to_test", ["add_service", "update_service"])
    def test_service_change_port_is_none(self, method_to_test, mock_client):
        print(f"\n--- Test: test_service_change_port_is_none (method: {method_to_test}) ---")
        mock_zeroconf_runtime_instance = MagicMock(spec=zeroconf_module.Zeroconf)
        
        mock_service_info_no_port = MagicMock(spec=zeroconf_module.ServiceInfo)
        mock_service_info_no_port.port = None # Critical condition for this test
        mock_service_info_no_port.name = "TestServiceNoPort.local." # Zeroconf ServiceInfo name includes .local.
        mock_service_info_no_port.addresses = [socket.inet_aton("1.2.3.4")] # Requires bytes
        mock_service_info_no_port.properties = {b"key": b"val"}

        self.run_service_change_test(
            method_name=method_to_test,
            mock_client=mock_client,
            mock_zc_instance_for_get_info=mock_zeroconf_runtime_instance,
            service_type_param="_goodservice._tcp.local.",
            service_name_param="any.service.name",
            get_service_info_return_value=mock_service_info_no_port
        )
        
        mock_zeroconf_runtime_instance.get_service_info.assert_called_once()
        mock_client._on_service_added.assert_not_called()
        print(f"  Assertion: client._on_service_added not called as port was None - PASSED")
        print(f"--- Test: test_service_change_port_is_none (method: {method_to_test}) finished ---")

    @pytest.mark.parametrize("method_to_test", ["add_service", "update_service"])
    def test_service_change_successful(self, method_to_test, mock_client):
        print(f"\n--- Test: test_service_change_successful (method: {method_to_test}) ---")
        mock_zeroconf_runtime_instance = MagicMock(spec=zeroconf_module.Zeroconf)

        # Zeroconf ServiceInfo object has .name attribute for the full service name
        # e.g., "ActualServiceName._goodservice._tcp.local."
        # The name passed to client._on_service_added should be this full name.
        full_service_name_from_zeroconf = "ActualServiceName._goodservice._tcp.local."
        expected_port = 8080
        expected_addresses_bytes = [socket.inet_aton("192.168.1.100")] # addresses are bytes
        expected_properties = {b"txt_key": b"txt_value"}

        mock_service_info_valid = MagicMock(spec=zeroconf_module.ServiceInfo)
        mock_service_info_valid.name = full_service_name_from_zeroconf
        mock_service_info_valid.port = expected_port
        mock_service_info_valid.addresses = expected_addresses_bytes
        mock_service_info_valid.properties = expected_properties
        
        # This service_name_param is what's passed to add_service/update_service,
        # and then to get_service_info. The actual name used comes from the returned ServiceInfo object.
        service_name_param_to_method = "some_instance_name._goodservice._tcp.local."

        self.run_service_change_test(
            method_name=method_to_test,
            mock_client=mock_client,
            mock_zc_instance_for_get_info=mock_zeroconf_runtime_instance,
            service_type_param="_goodservice._tcp.local.",
            service_name_param=service_name_param_to_method,
            get_service_info_return_value=mock_service_info_valid
        )
        
        mock_zeroconf_runtime_instance.get_service_info.assert_called_once_with(
            "_goodservice._tcp.local.", service_name_param_to_method
        )
        mock_client._on_service_added.assert_called_once_with(
            full_service_name_from_zeroconf, # This is ServiceInfo.name
            expected_port,
            expected_addresses_bytes,
            expected_properties
        )
        print(f"  Assertion: client._on_service_added called with correct parsed ServiceInfo data - PASSED")
        print(f"--- Test: test_service_change_successful (method: {method_to_test}) finished ---")


    # --- Test for remove_service ---
    def test_remove_service_does_nothing_gracefully(self, mock_client):
        print("\n--- Test: test_remove_service_does_nothing_gracefully ---")
        mock_zeroconf_runtime_instance = MagicMock(spec=zeroconf_module.Zeroconf)
        
        with patch.object(zeroconf_module, 'ServiceBrowser'), \
             patch.object(zeroconf_module, 'Zeroconf'):
            listener = RecordListener(mock_client, "_sometype._tcp")

        # Call remove_service - it's expected to do nothing in current SUT
        listener.remove_service(mock_zeroconf_runtime_instance, "any_type", "any_name")
        print("  listener.remove_service called.")
        
        # Assert no exceptions and client was not called
        mock_client._on_service_added.assert_not_called() 
        # (No _on_service_removed on client in RecordListener.Client interface)
        print("  Assertion: No client methods called, no exceptions - PASSED")
        print("--- Test: test_remove_service_does_nothing_gracefully finished ---")

```

**Summary of Implementation (Turn 2):**
1.  **Imports**: Added necessary modules.
2.  **`MockRecordListenerClient`**: Defined a type alias for `MagicMock` for clarity.
3.  **`TestRecordListener` Class**:
    *   `mock_client` fixture: Provides a `MagicMock` for `RecordListener.Client` with its `_on_service_added` method also mocked.
    *   **`test_init_creates_zeroconf_and_service_browser`**:
        *   Patches `zeroconf.Zeroconf` and `zeroconf.ServiceBrowser` (from `zeroconf_module`).
        *   Instantiates `RecordListener`.
        *   Asserts `Zeroconf()` was called once.
        *   Asserts `ServiceBrowser` was called with the `Zeroconf` instance, the correctly formatted service type (e.g., `_myservice._tcp.local.`), and the `RecordListener` instance itself as the handler.
        *   Checks that internal attributes `_zeroconf`, `_browser`, and `__expected_type` are set correctly.
    *   **Helper `run_service_change_test`**:
        *   This helper is created to encapsulate the common setup and call pattern for `add_service` and `update_service` tests, as their internal logic (calling `__handle_service_info`) is very similar.
        *   It takes the method name (`"add_service"` or `"update_service"`) and other parameters to configure the test scenario.
        *   It handles `RecordListener` instantiation (with patched `Zeroconf`/`ServiceBrowser` for `__init__`) and calls the specified method.
    *   **Parametrized tests for `add_service` and `update_service` (using the helper)**:
        *   **`test_service_change_type_mismatch`**: Calls the method with a `service_type_param` that does not match the `listener._RecordListener__expected_type`. Asserts `get_service_info` and `client._on_service_added` are not called.
        *   **`test_service_change_info_is_none`**: Mocks `zeroconf_instance.get_service_info` to return `None`. Asserts `client._on_service_added` is not called.
        *   **`test_service_change_port_is_none`**: Mocks `get_service_info` to return a `ServiceInfo` object where `port` is `None`. Asserts `client._on_service_added` is not called.
        *   **`test_service_change_successful`**: Mocks `get_service_info` to return a valid `ServiceInfo` object. Asserts `client._on_service_added` is called once with the correct arguments extracted from the `ServiceInfo` object (name, port, addresses, properties). Note: `ServiceInfo.addresses` are bytes, `ServiceInfo.properties` keys/values are bytes.
    *   **`test_remove_service_does_nothing_gracefully`**:
        *   Instantiates `RecordListener`.
        *   Calls `listener.remove_service(...)`.
        *   Asserts that no exceptions occur and no client methods are called (as `remove_service` is a no-op in the SUT).

This set of tests covers the specified methods and scenarios. The use of `socket.inet_aton` is for creating byte-string IP addresses, which `zeroconf.ServiceInfo` expects in its `addresses` list. The `service_type` in `RecordListener` constructor should not have `.local.` but it's added internally to form `__expected_type`. The `service_type_param` passed to `add_service`/`update_service` from `zeroconf` usually includes `.local.`.The test file `tsercom/discovery/mdns/record_listener_unittest.py` has been written with tests for `RecordListener`.

**Key implementations:**
-   **Mocking Strategy**:
    -   `RecordListener.Client`: Mocked using `MagicMock`, specifically its `_on_service_added` method.
    -   `zeroconf.Zeroconf` and `zeroconf.ServiceBrowser`: Patched using `@patch.object(zeroconf_module, '...')` to verify their instantiation and arguments during `RecordListener.__init__`.
    -   `Zeroconf.get_service_info`: The method on the `Zeroconf` instance is mocked to control the `ServiceInfo` object returned, simulating various scenarios (service found, not found, or found with incomplete data).
    -   `zeroconf.ServiceInfo`: Instances are created using `MagicMock(spec=zeroconf.ServiceInfo)` and attributes (`name`, `port`, `addresses`, `properties`) are set as needed for test cases. Addresses are provided as byte strings (e.g., via `socket.inet_aton`).
-   **Test Scenarios Covered**:
    -   **`__init__`**:
        *   `test_init_creates_zeroconf_and_service_browser`: Verifies that `Zeroconf()` is called and `ServiceBrowser` is instantiated with the `Zeroconf` instance, the correctly formatted service type (e.g., `_myservice._tcp.local.`), and the `RecordListener` instance itself as the handler.
    -   **`add_service` and `update_service` (tested via a parametrized helper function `run_service_change_test` due to shared logic path through `__handle_service_info`)**:
        *   `test_service_change_type_mismatch`: Input `type` (from Zeroconf callback) does not match the listener's `expected_type`. Asserts `get_service_info` and client's `_on_service_added` are not called.
        *   `test_service_change_info_is_none`: `get_service_info` returns `None`. Asserts client's `_on_service_added` is not called.
        *   `test_service_change_port_is_none`: `ServiceInfo.port` is `None`. Asserts client's `_on_service_added` is not called.
        *   `test_service_change_successful`: Valid `ServiceInfo` is returned. Asserts client's `_on_service_added` is called with correctly extracted `name` (from `ServiceInfo.name`), `port`, `addresses` (as list of bytes), and `properties` (as dict of bytes to bytes).
    -   **`remove_service`**:
        *   `test_remove_service_does_nothing_gracefully`: Verifies the method can be called without error and does not trigger client callbacks (as it's a no-op in the SUT).

The tests use `pytest` and `unittest.mock`. Helper functions and parametrization are used to keep tests DRY where appropriate. Print statements are included for diagnostics.

I will now run these tests.
