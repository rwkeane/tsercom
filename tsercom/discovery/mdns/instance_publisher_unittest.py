import pytest
from unittest.mock import patch, MagicMock, ANY # ANY for parts of txt_record
import datetime
import uuid # For uuid.getnode

from tsercom.discovery.mdns.instance_publisher import InstancePublisher
# Import the module where RecordPublisher is defined to patch it
import tsercom.discovery.mdns.record_publisher as record_publisher_module
# Import the module where datetime and get_mac (uuid.getnode) are used by InstancePublisher
import tsercom.discovery.mdns.instance_publisher as instance_publisher_module

# Fixed datetime for mocking
FIXED_DATETIME_NOW = datetime.datetime(2024, 1, 1, 12, 30, 45, 123456)
FIXED_DATETIME_STR = FIXED_DATETIME_NOW.strftime("%F %T.%f")

# Fixed MAC address for mocking uuid.getnode
FIXED_MAC_INT = 0x123456789abc
EXPECTED_MAC_HEX_PART_OF_NAME = hex(FIXED_MAC_INT)[2:] # Remove '0x'

class TestInstancePublisher:

    @patch.object(instance_publisher_module, 'datetime', autospec=True)
    @patch.object(record_publisher_module, 'RecordPublisher', autospec=True)
    def test_init_with_instance_name_provided(
        self, mock_record_publisher_ctor, mock_datetime_module
    ):
        print("\n--- Test: test_init_with_instance_name_provided ---")
        mock_datetime_module.datetime.now.return_value = FIXED_DATETIME_NOW
        
        port = 123
        service_type = "_test._tcp.local." # Added .local. as per standard
        readable_name = "My Readable Service"
        instance_name = "my_specific_instance"

        publisher = InstancePublisher(
            port=port,
            service_type=service_type,
            readable_name=readable_name,
            instance_name=instance_name
        )
        print(f"  InstancePublisher created with instance_name='{instance_name}'")

        expected_txt_record = {
            b"name": readable_name.encode("utf-8"),
            b"published_on": FIXED_DATETIME_STR.encode("utf-8")
        }

        mock_record_publisher_ctor.assert_called_once_with(
            name=instance_name,
            service_type=service_type,
            port=port,
            txt_record=expected_txt_record
        )
        print("  Assertion: RecordPublisher constructor called correctly - PASSED")
        assert publisher._record_publisher is mock_record_publisher_ctor.return_value
        print("  Assertion: publisher._record_publisher is set - PASSED")
        print("--- Test: test_init_with_instance_name_provided finished ---")

    @patch.object(instance_publisher_module, 'get_mac', autospec=True) # Patches uuid.getnode as get_mac
    @patch.object(instance_publisher_module, 'datetime', autospec=True)
    @patch.object(record_publisher_module, 'RecordPublisher', autospec=True)
    def test_init_without_instance_name_uses_generated_name(
        self, mock_record_publisher_ctor, mock_datetime_module, mock_get_mac
    ):
        print("\n--- Test: test_init_without_instance_name_uses_generated_name ---")
        mock_datetime_module.datetime.now.return_value = FIXED_DATETIME_NOW
        mock_get_mac.return_value = FIXED_MAC_INT
        
        port = 456
        service_type = "_anothertest._udp.local." # Added .local.
        readable_name = "Another Service"

        publisher = InstancePublisher(
            port=port,
            service_type=service_type,
            readable_name=readable_name
            # instance_name is None by default
        )
        
        # Expected generated name logic from SUT: f"{port}{hex(get_mac())[2:]}"
        # Then truncated to 15 chars: generated_name[:15]
        raw_generated_name = f"{port}{EXPECTED_MAC_HEX_PART_OF_NAME}"
        expected_instance_name = raw_generated_name[:15] 
        print(f"  Raw generated name: '{raw_generated_name}', Expected truncated: '{expected_instance_name}'")
        print(f"  InstancePublisher created, generated instance_name should be '{expected_instance_name}'")
        
        expected_txt_record = {
            b"name": readable_name.encode("utf-8"),
            b"published_on": FIXED_DATETIME_STR.encode("utf-8")
        }

        mock_record_publisher_ctor.assert_called_once_with(
            name=expected_instance_name,
            service_type=service_type,
            port=port,
            txt_record=expected_txt_record
        )
        mock_get_mac.assert_called_once() # Verify get_mac was called
        print("  Assertion: RecordPublisher constructor called with generated name - PASSED")
        print("--- Test: test_init_without_instance_name_uses_generated_name finished ---")

    @patch.object(instance_publisher_module, 'datetime', autospec=True)
    def test_make_txt_record_with_readable_name(self, mock_datetime_module):
        print("\n--- Test: test_make_txt_record_with_readable_name ---")
        mock_datetime_module.datetime.now.return_value = FIXED_DATETIME_NOW
        
        readable_name = "Test Readable Name"
        # Need to instantiate publisher to call _make_txt_record
        # Mock RecordPublisher to avoid its side effects during this specific test
        with patch.object(record_publisher_module, 'RecordPublisher'):
            publisher = InstancePublisher(
                port=123, service_type="_test._tcp.local.", readable_name=readable_name
            )
            txt_record = publisher._make_txt_record()
            print(f"  Generated TXT record: {txt_record}")

        assert txt_record[b"name"] == readable_name.encode("utf-8")
        assert txt_record[b"published_on"] == FIXED_DATETIME_STR.encode("utf-8")
        assert len(txt_record) == 2 # Only these two keys expected
        print("  Assertion: TXT record with name and published_on correct - PASSED")
        print("--- Test: test_make_txt_record_with_readable_name finished ---")

    @patch.object(instance_publisher_module, 'datetime', autospec=True)
    def test_make_txt_record_without_readable_name(self, mock_datetime_module):
        print("\n--- Test: test_make_txt_record_without_readable_name ---")
        mock_datetime_module.datetime.now.return_value = FIXED_DATETIME_NOW
        
        with patch.object(record_publisher_module, 'RecordPublisher'):
            publisher = InstancePublisher(
                port=123, service_type="_test._tcp.local.", readable_name=None # No readable name
            )
            txt_record = publisher._make_txt_record()
            print(f"  Generated TXT record: {txt_record}")

        assert b"name" not in txt_record, "b'name' should not be in TXT record if readable_name is None"
        assert txt_record[b"published_on"] == FIXED_DATETIME_STR.encode("utf-8")
        assert len(txt_record) == 1 # Only published_on expected
        print("  Assertion: TXT record without name, only published_on - PASSED")
        print("--- Test: test_make_txt_record_without_readable_name finished ---")

    @patch.object(record_publisher_module, 'RecordPublisher', autospec=True)
    def test_publish_calls_record_publisher_publish(self, mock_record_publisher_ctor):
        print("\n--- Test: test_publish_calls_record_publisher_publish ---")
        # We need the instance returned by the mocked constructor
        mock_record_publisher_instance = mock_record_publisher_ctor.return_value
        mock_record_publisher_instance.publish = MagicMock(name="record_publisher_instance_publish")

        # Patch datetime for consistent TXT record, though not directly asserted here
        with patch.object(instance_publisher_module, 'datetime', autospec=True) as mock_dt:
            mock_dt.datetime.now.return_value = FIXED_DATETIME_NOW
            publisher = InstancePublisher(
                port=123, service_type="_test._tcp.local.", readable_name="Test"
            )
            print("  InstancePublisher created.")
        
        publisher.publish()
        print("  publisher.publish() called.")
        
        mock_record_publisher_instance.publish.assert_called_once_with()
        print("  Assertion: record_publisher_instance.publish called - PASSED")
        print("--- Test: test_publish_calls_record_publisher_publish finished ---")

    @patch.object(instance_publisher_module, 'get_mac', autospec=True)
    @patch.object(instance_publisher_module, 'datetime', autospec=True)
    @patch.object(record_publisher_module, 'RecordPublisher', autospec=True)
    def test_init_generated_name_truncation(
        self, mock_record_publisher_ctor, mock_datetime_module, mock_get_mac
    ):
        print("\n--- Test: test_init_generated_name_truncation ---")
        mock_datetime_module.datetime.now.return_value = FIXED_DATETIME_NOW
        # MAC that will result in a long hex string
        long_mac_int = 0x112233445566778899aabbccddeeff
        mock_get_mac.return_value = long_mac_int 
        
        port = 987
        service_type = "_truncate._tcp.local."
        readable_name = "Truncation Test"

        publisher = InstancePublisher(
            port=port,
            service_type=service_type,
            readable_name=readable_name
        )
        
        raw_generated_name = f"{port}{hex(long_mac_int)[2:]}"
        expected_instance_name = raw_generated_name[:15] # SUT truncates to 15
        print(f"  Raw generated name: '{raw_generated_name}', Expected truncated: '{expected_instance_name}'")
        
        expected_txt_record = {
            b"name": readable_name.encode("utf-8"),
            b"published_on": FIXED_DATETIME_STR.encode("utf-8")
        }

        mock_record_publisher_ctor.assert_called_once_with(
            name=expected_instance_name,
            service_type=service_type,
            port=port,
            txt_record=expected_txt_record
        )
        print("  Assertion: RecordPublisher constructor called with TRUNCATED generated name - PASSED")
        print("--- Test: test_init_generated_name_truncation finished ---")

```

**Summary of Implementation (Turn 2):**
1.  **Imports**: Added `pytest`, `unittest.mock` utilities, `datetime`, and `uuid`. Imported SUT `InstancePublisher` and modules for patching (`record_publisher_module`, `instance_publisher_module`).
2.  **Constants**: Defined `FIXED_DATETIME_NOW`, `FIXED_DATETIME_STR`, `FIXED_MAC_INT`, and `EXPECTED_MAC_HEX_PART_OF_NAME` for consistent mocking.
3.  **`TestInstancePublisher` Class**:
    *   **`test_init_with_instance_name_provided`**:
        *   Patches `datetime.datetime.now` (within `instance_publisher_module`) and `RecordPublisher` (within `record_publisher_module`).
        *   Instantiates `InstancePublisher` with a specific `instance_name`.
        *   Asserts `RecordPublisher` constructor was called with the provided `instance_name`, correct `service_type`, `port`, and a `txt_record` matching what `_make_txt_record` would produce with the mocked datetime and `readable_name`.
    *   **`test_init_without_instance_name_uses_generated_name`**:
        *   Patches `datetime.datetime.now`, `RecordPublisher`, and `get_mac` (which is `uuid.getnode`, aliased as `get_mac` in `instance_publisher.py`).
        *   Instantiates `InstancePublisher` without an `instance_name`.
        *   Calculates the expected generated name based on SUT logic (`f"{port}{hex(get_mac())[2:]}"` truncated to 15 chars: `[:15]`).
        *   Asserts `RecordPublisher` constructor was called with this generated and truncated name.
    *   **`test_init_generated_name_truncation` (added)**: A specific test to ensure the 15-character truncation logic for generated instance names is correctly applied.
    *   **`test_make_txt_record_with_readable_name`**:
        *   Patches `datetime.datetime.now`.
        *   Instantiates `InstancePublisher` (with `RecordPublisher` also patched to avoid its side effects for this focused test).
        *   Calls the protected method `publisher._make_txt_record()`.
        *   Asserts the returned dictionary contains `b"name"` with the UTF-8 encoded `readable_name` and `b"published_on"` with the correctly formatted and encoded fixed datetime string.
    *   **`test_make_txt_record_without_readable_name`**:
        *   Similar setup, but `readable_name` is `None`.
        *   Asserts `b"name"` is not in the resulting `txt_record`.
    *   **`test_publish_calls_record_publisher_publish`**:
        *   Patches `RecordPublisher`. The mock instance `mock_record_publisher_instance = RecordPublisher.return_value` is retrieved. Its `publish` method is replaced with a new `MagicMock`.
        *   `InstancePublisher` is instantiated.
        *   `publisher.publish()` is called.
        *   Asserts `mock_record_publisher_instance.publish.assert_called_once_with()` was called.

This suite covers the specified methods and scenarios, paying attention to the patching targets and the SUT's internal logic for name generation and TXT record creation.The test file `tsercom/discovery/mdns/instance_publisher_unittest.py` has been written with tests for `InstancePublisher`.

**Key implementations:**
-   **Mocking Strategy**:
    -   `tsercom.discovery.mdns.record_publisher.RecordPublisher`: Patched to verify its constructor arguments (passed by `InstancePublisher.__init__`) and to mock its `publish` method (called by `InstancePublisher.publish`).
    -   `datetime.datetime.now` (within `tsercom.discovery.mdns.instance_publisher` module): Patched to return a fixed `datetime` object for predictable `published_on` values in TXT records.
    -   `uuid.getnode` (aliased as `get_mac` in `tsercom.discovery.mdns.instance_publisher`): Patched to return a fixed integer MAC address for predictable generated instance names.
-   **Test Scenarios Covered**:
    -   **`__init__`**:
        -   When `instance_name` is provided: Verifies `RecordPublisher` is initialized with the given `instance_name`, `service_type`, `port`, and a correctly formed `txt_record` (including `readable_name` and `published_on` timestamp).
        -   When `instance_name` is *not* provided: Verifies `RecordPublisher` is initialized with an auto-generated `instance_name` (based on port and MAC address from mocked `get_mac`), including the 15-character truncation logic.
    -   **`_make_txt_record` (protected method, tested directly)**:
        -   With `readable_name`: Asserts the TXT record dictionary contains `b"name"` (UTF-8 encoded `readable_name`) and `b"published_on"` (correctly formatted and encoded timestamp).
        -   Without `readable_name` (i.e., `None`): Asserts `b"name"` is absent from the TXT record, while `b"published_on"` is still present.
    -   **`publish`**: Verifies that calling `publisher.publish()` results in a call to the underlying `RecordPublisher` instance's `publish` method.

The tests use `pytest` and `unittest.mock.patch`. Fixed values for datetime and MAC addresses are used to ensure deterministic outcomes.

I will now run these tests. The print statements included will offer insight during execution.
