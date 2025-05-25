import pytest
from unittest.mock import (
    patch,
    MagicMock,
    ANY,
)  # ANY might not be needed if properties are exact
import zeroconf  # For IPVersion and potentially for spec if ServiceInfo is directly instantiated

from tsercom.discovery.mdns.record_publisher import RecordPublisher

# Modules to patch (where the names are looked up by RecordPublisher)
import tsercom.util.ip as tsercom_ip_module  # For get_all_addresses
import tsercom.discovery.mdns.record_publisher as record_publisher_sut_module  # For Zeroconf, ServiceInfo


class TestRecordPublisher:

    def test_init_sets_attributes_correctly(self):
        print("\n--- Test: test_init_sets_attributes_correctly ---")
        name = "MyInstance"
        type_ = "_myproto._tcp"  # Input type without .local.
        port = 1234
        properties = {b"key1": b"val1", b"key2": b"val2"}

        # Patch Zeroconf and ServiceInfo within __init__ scope if they are used there directly,
        # but for attribute checking, no patches are needed for __init__ itself unless it calls them.
        # RecordPublisher.__init__ doesn't call external Zeroconf methods, only sets attributes.

        publisher = RecordPublisher(name, type_, port, properties)
        print(
            f"  RecordPublisher instantiated with name='{name}', type_='{type_}'"
        )

        expected_ptr = f"{type_}.local."
        expected_srv = f"{name}.{type_}.local."

        assert (
            publisher._RecordPublisher__name == name
        ), "Name not set correctly"
        assert (
            publisher._RecordPublisher__ptr == expected_ptr
        ), "PTR record not set correctly"
        assert (
            publisher._RecordPublisher__srv == expected_srv
        ), "SRV record not set correctly"
        assert (
            publisher._RecordPublisher__port == port
        ), "Port not set correctly"
        assert (
            publisher._RecordPublisher__txt == properties
        ), "TXT properties not set correctly"
        print("  Assertions for attributes passed.")
        print("--- Test: test_init_sets_attributes_correctly finished ---")

    def test_init_properties_defaults_to_empty_dict(self):
        print("\n--- Test: test_init_properties_defaults_to_empty_dict ---")
        name = "TestInstanceDefaultProps"
        type_ = "_testservice._udp"  # Input type without .local.
        port = 8080

        publisher = RecordPublisher(
            name, type_, port, properties=None
        )  # Pass None for properties
        print(
            f"  RecordPublisher instantiated with name='{name}', properties=None"
        )

        assert (
            publisher._RecordPublisher__txt == {}
        ), "TXT properties should default to empty dict if None"
        print("  Assertion for default TXT properties passed.")
        print(
            "--- Test: test_init_properties_defaults_to_empty_dict finished ---"
        )

    @patch.object(record_publisher_sut_module, "Zeroconf", autospec=True)
    @patch.object(record_publisher_sut_module, "ServiceInfo", autospec=True)
    @patch.object(tsercom_ip_module, "get_all_addresses", autospec=True)
    def test_publish_calls_dependencies_correctly(
        self, mock_get_all_addresses, MockServiceInfo, MockZeroconf
    ):
        print("\n--- Test: test_publish_calls_dependencies_correctly ---")

        # Setup mock return values
        mock_packed_ips = [
            b"\x01\x02\x03\x04",
            b"\x0a\x0b\x0c\x0d",
        ]  # e.g., 1.2.3.4, 10.11.12.13
        mock_get_all_addresses.return_value = mock_packed_ips
        print(
            f"  mock_get_all_addresses configured to return: {mock_packed_ips}"
        )

        mock_zc_instance = MockZeroconf.return_value
        mock_zc_instance.register_service = MagicMock(
            name="zc_instance_register_service"
        )
        print(f"  MockZeroconf instance configured: {mock_zc_instance}")

        mock_service_info_instance = MockServiceInfo.return_value
        print(
            f"  MockServiceInfo instance configured: {mock_service_info_instance}"
        )

        # Instantiate RecordPublisher
        name = "TestPublishInstance"
        type_ = "_publishservice._tcp"
        port = 9876
        properties = {b"data_prop": b"data_val"}

        publisher = RecordPublisher(name, type_, port, properties)
        print(
            f"  RecordPublisher for publish test instantiated: name='{name}'"
        )

        # Call the publish method
        publisher.publish()
        print("  publisher.publish() called.")

        # Assertions
        print("  Checking assertions...")
        mock_get_all_addresses.assert_called_once_with()
        print("  Assertion: get_all_addresses called - PASSED")

        # Verify ServiceInfo constructor call
        # publisher._RecordPublisher__ptr is f"{type_}.local."
        # publisher._RecordPublisher__srv is f"{name}.{type_}.local."
        expected_ptr_for_si = f"{type_}.local."
        expected_srv_for_si = f"{name}.{type_}.local."

        MockServiceInfo.assert_called_once_with(
            expected_ptr_for_si,
            expected_srv_for_si,
            addresses=mock_packed_ips,
            port=port,
            properties=properties,
            # server=f'{socket.gethostname()}.local.', # This is a default in ServiceInfo, not passed by RecordPublisher
            # host_ttl=120, other_ttl=4500, priority=0, weight=0 # Also defaults
        )
        # To be more precise, if ServiceInfo has defaults we don't want to overspecify,
        # we can check individual args if the above fails due to unmocked defaults.
        # For now, assume this full signature check is intended.
        # If this fails due to `server` kwarg, we can use mock_calls or check args individually.
        print("  Assertion: ServiceInfo constructor called correctly - PASSED")

        MockZeroconf.assert_called_once_with(
            ip_version=zeroconf.IPVersion.V4Only
        )
        print(
            "  Assertion: Zeroconf initialized with IPVersion.V4Only - PASSED"
        )

        mock_zc_instance.register_service.assert_called_once_with(
            mock_service_info_instance
        )
        print(
            "  Assertion: zeroconf_instance.register_service called with ServiceInfo instance - PASSED"
        )

        # Check that the internal _zeroconf is stored
        assert (
            publisher._record_publisher is mock_zc_instance
        ), "Internal _zeroconf not stored"
        print("  Assertion: Internal _zeroconf correctly stored - PASSED")
        print(
            "--- Test: test_publish_calls_dependencies_correctly finished ---"
        )
