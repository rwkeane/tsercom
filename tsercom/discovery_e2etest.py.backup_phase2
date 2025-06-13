import asyncio
import gc  # Moved import gc to top level
import ipaddress
import uuid

import pytest
import pytest_asyncio

from tsercom.discovery.mdns.instance_listener import InstanceListener
from tsercom.discovery.mdns.instance_publisher import InstancePublisher
from tsercom.discovery.service_info import ServiceInfo

# pytest_asyncio is not directly imported but used via pytest.mark.asyncio
from tsercom.threading.aio.global_event_loop import (
    clear_tsercom_event_loop,
    set_tsercom_event_loop,
)


class DiscoveryTestClient(InstanceListener.Client):
    __test__ = False

    def __init__(
        self, discovery_event: asyncio.Event, discovered_services: list
    ):
        super().__init__()
        self._discovery_event = discovery_event
        self._discovered_services = discovered_services

    async def _on_service_added(self, connection_info: ServiceInfo):
        self._discovered_services.append(connection_info)
        self._discovery_event.set()

    async def _on_service_removed(self, service_name: str):
        # Basic implementation for existing tests, can be expanded if needed
        print(
            f"Service removed (not actively handled in this client): {service_name}"
        )
        pass


@pytest_asyncio.fixture(scope="function", autouse=True)
async def manage_tsercom_global_event_loop():
    """Ensures tsercom's global event loop is set for asyncio tests."""
    try:
        loop = asyncio.get_running_loop()
        set_tsercom_event_loop(loop)
        yield
    finally:
        clear_tsercom_event_loop()


@pytest.mark.asyncio
async def test_successful_registration_and_discovery():
    service_type_suffix = uuid.uuid4().hex[:8]
    service_type = f"_e2e-{service_type_suffix}._tcp.local."
    service_port = 50001
    readable_name = f"TestService_{service_type_suffix}"
    instance_name = f"TestInstance_{service_type_suffix}"

    discovery_event = asyncio.Event()
    discovered_services = []
    client = DiscoveryTestClient(discovery_event, discovered_services)

    # Listener needs to be started before publisher to catch the announcement
    # Assign to specific variable names to manage lifecycle explicitly in finally block
    listener_obj = InstanceListener(client=client, service_type=service_type)
    publisher_obj = None  # Initialize to None for the finally block

    try:
        publisher_obj = InstancePublisher(
            port=service_port,
            service_type=service_type,
            readable_name=readable_name,
            instance_name=instance_name,
        )
        # Start listening - happens in InstanceListener constructor
        # Publish the service
        await publisher_obj.publish()

        await asyncio.wait_for(discovery_event.wait(), timeout=10.0)
    except asyncio.TimeoutError:
        pytest.fail(
            f"Service {readable_name} on port {service_port} of type {service_type} not discovered within timeout."
        )
    finally:
        # Explicitly delete to manage lifecycle and satisfy linters,
        # relying on __del__ in publisher for unpublishing.
        if publisher_obj:
            await publisher_obj.close()  # Explicitly close
        if (
            listener_obj
        ):  # listener_obj is always created if this block is reached
            del listener_obj
        gc.collect()  # Encourage faster cleanup

    assert discovery_event.is_set(), "Discovery event was not set"
    assert (
        len(discovered_services) == 1
    ), "Incorrect number of services discovered"

    service_info = discovered_services[0]
    assert (
        service_info.name == readable_name
    ), "Discovered service name does not match"
    assert (
        service_info.port == service_port
    ), "Discovered service port does not match"
    # mDNS instance name can have ".local." or similar suffixes, so we check startswith
    assert service_info.mdns_name.startswith(
        instance_name
    ), f"Discovered mDNS name '{service_info.mdns_name}' does not start with '{instance_name}'"

    assert service_info.addresses, "Discovered service addresses list is empty"
    for addr_str in service_info.addresses:
        try:
            ipaddress.ip_address(
                addr_str
            )  # Validate if it's a valid IP address string
        except ValueError:
            pytest.fail(f"Invalid IP address string found: {addr_str}")

    # Small sleep to allow background tasks to settle, if necessary.
    await asyncio.sleep(0.1)


class SelectiveDiscoveryClient(InstanceListener.Client):
    __test__ = False

    def __init__(self):
        super().__init__()
        self.service_added_event = asyncio.Event()
        self.service_removed_event = asyncio.Event()
        self.discovered_services: list[ServiceInfo] = []
        self.removed_service_names: list[str] = []
        self._lock = asyncio.Lock()

    async def _on_service_added(self, connection_info: ServiceInfo):
        async with self._lock:
            # Avoid duplicates if service already in list by mdns_name
            if not any(
                s.mdns_name == connection_info.mdns_name
                for s in self.discovered_services
            ):
                self.discovered_services.append(connection_info)
            # Set event even if it's a duplicate, signifies an add event occurred
            self.service_added_event.set()

    async def _on_service_removed(self, service_name: str):
        async with self._lock:
            self.removed_service_names.append(service_name)
            # Remove from discovered_services if present
            self.discovered_services = [
                s
                for s in self.discovered_services
                if s.mdns_name != service_name
            ]
            self.service_removed_event.set()

    def clear_events(self):
        self.service_added_event.clear()
        self.service_removed_event.clear()


@pytest.mark.asyncio
async def test_concurrent_publishing_with_selective_unpublish():
    service_type_suffix = uuid.uuid4().hex[:8]
    service_type = f"_conc-{service_type_suffix}._tcp.local."

    publisher1_obj = None
    publisher2_obj = None
    listener_obj = None

    try:
        # Create Client
        client = SelectiveDiscoveryClient()
        listener_obj = InstanceListener(
            client=client, service_type=service_type
        )

        # Create two InstancePublisher instances
        publisher1_port = 50010
        publisher1_readable_name = f"ConcService1_{service_type_suffix}"
        publisher1_instance_name = f"ConcInstance1_{service_type_suffix}"
        publisher1_obj = InstancePublisher(
            port=publisher1_port,
            service_type=service_type,
            readable_name=publisher1_readable_name,
            instance_name=publisher1_instance_name,
        )

        publisher2_port = 50011
        publisher2_readable_name = f"ConcService2_{service_type_suffix}"
        publisher2_instance_name = f"ConcInstance2_{service_type_suffix}"
        publisher2_obj = InstancePublisher(
            port=publisher2_port,
            service_type=service_type,
            readable_name=publisher2_readable_name,
            instance_name=publisher2_instance_name,
        )

        # Publish concurrently
        client.clear_events()
        await asyncio.gather(
            publisher1_obj.publish(),
            publisher2_obj.publish(),
        )

        # Verify both services are discovered
        # Wait for at least two 'added' events.
        # This is a bit simplistic; mDNS might send multiple "added" events.
        # A more robust check would be to wait until len(client.discovered_services) == 2
        # with a timeout.
        timeout = 15.0
        start_time = asyncio.get_event_loop().time()
        while len(client.discovered_services) < 2:
            if asyncio.get_event_loop().time() - start_time > timeout:
                pytest.fail(
                    f"Timeout waiting for both services to be discovered. Discovered: {len(client.discovered_services)}."
                )
            try:
                await asyncio.wait_for(
                    client.service_added_event.wait(), timeout=0.5
                )  # Short wait for event
            except asyncio.TimeoutError:
                pass  # Event didn't fire, continue loop and check time
            client.service_added_event.clear()  # Clear event for next potential service

        async with client._lock:  # Protect access to discovered_services
            assert (
                len(client.discovered_services) == 2
            ), f"Both services were not discovered. Found {len(client.discovered_services)}"
            # Ensure mdns_names are unique, as readable_name might not be if not set carefully for test
            discovered_mdns_names = {
                s.mdns_name for s in client.discovered_services
            }
            # Check that the instance names are prefixes of the discovered mDNS names
            assert any(
                name.startswith(publisher1_instance_name)
                for name in discovered_mdns_names
            ), f"Publisher 1 (instance: {publisher1_instance_name}) not discovered in {discovered_mdns_names}"
            assert any(
                name.startswith(publisher2_instance_name)
                for name in discovered_mdns_names
            ), f"Publisher 2 (instance: {publisher2_instance_name}) not discovered in {discovered_mdns_names}"
            # Also check readable names for completeness
            discovered_readable_names = {
                s.name for s in client.discovered_services
            }
            assert (
                publisher1_readable_name in discovered_readable_names
            ), "Publisher 1 not discovered by readable name"
            assert (
                publisher2_readable_name in discovered_readable_names
            ), "Publisher 2 not discovered by readable name"

        # Unpublish the first service
        client.clear_events()
        await publisher1_obj.close()  # This should trigger unpublish

        # Verify first service is removed, second is still present
        try:
            await asyncio.wait_for(
                client.service_removed_event.wait(), timeout=10.0
            )
        except asyncio.TimeoutError:
            pytest.fail(
                f"Timeout waiting for service {publisher1_instance_name} to be removed."
            )

        async with client._lock:  # Protect access to lists
            assert any(
                name.startswith(publisher1_instance_name)
                for name in client.removed_service_names
            ), f"Service {publisher1_instance_name} was not reported as removed in {client.removed_service_names}."

            # Check that publisher1 is no longer in discovered_services
            assert not any(
                s.mdns_name.startswith(publisher1_instance_name)
                for s in client.discovered_services
            ), f"Service {publisher1_instance_name} still in discovered list after removal."

            # Check that publisher2 is still in discovered_services
            assert any(
                s.mdns_name.startswith(publisher2_instance_name)
                for s in client.discovered_services
            ), f"Service {publisher2_instance_name} not found in discovered list after p1 removal."
            assert (
                len(client.discovered_services) == 1
            ), "Incorrect number of services remaining after unpublishing one."
            assert (
                client.discovered_services[0].name == publisher2_readable_name
            ), "The remaining service is not publisher 2."

    finally:
        if publisher1_obj:
            await publisher1_obj.close()
        if publisher2_obj:
            await publisher2_obj.close()
        if listener_obj:
            del listener_obj  # Rely on __del__ for cleanup
        gc.collect()

    await asyncio.sleep(0.1)  # Final small sleep


class UpdateTestClient(InstanceListener.Client):
    __test__ = False

    def __init__(
        self, events: list[asyncio.Event], services_list: list[ServiceInfo]
    ):
        super().__init__()
        self.events = events
        self.services_list = services_list
        self.call_count = 0

    async def _on_service_added(self, connection_info: ServiceInfo) -> None:
        self.services_list.append(connection_info)
        if self.call_count < len(self.events):
            self.events[self.call_count].set()
        self.call_count += 1
        # Note: mDNS can sometimes send multiple "added" events for the same service initially.
        # This test relies on the new publication with the same instance name causing a new
        # "added" event after the old one might have been removed or TTL expired.
        # Robust handling might require `_on_service_removed` and more complex logic
        # if the underlying zeroconf library guarantees remove-then-add for updates.
        # For this test, we assume two distinct "added" events for V1 and V2.

    async def _on_service_removed(self, service_name: str):
        # Basic implementation for existing tests, can be expanded if needed
        print(
            f"Service removed (not actively handled in this client): {service_name}"
        )
        pass


@pytest.mark.asyncio
async def test_instance_update_reflects_changes():
    service_type_suffix = uuid.uuid4().hex[:8]
    service_type = f"_upd-{service_type_suffix}._tcp.local."
    instance_name = f"UpdateInstance_{service_type_suffix}"  # Critical: This stays the same

    service_port1 = 50002
    readable_name1 = f"UpdateTestService_V1_{service_type_suffix}"

    discovery_event1 = asyncio.Event()
    discovery_event2 = asyncio.Event()  # For the updated service
    discovered_services = []

    client = UpdateTestClient(
        [discovery_event1, discovery_event2], discovered_services
    )
    listener_obj = InstanceListener(client=client, service_type=service_type)

    publisher1_obj = None  # Initialize for finally block
    publisher2_obj = None  # Initialize for finally block

    try:
        publisher1_obj = InstancePublisher(
            port=service_port1,
            service_type=service_type,
            readable_name=readable_name1,
            instance_name=instance_name,
        )
        await publisher1_obj.publish()
        await asyncio.wait_for(discovery_event1.wait(), timeout=10.0)

        assert discovery_event1.is_set(), "Initial discovery event was not set"
        assert (
            len(discovered_services) >= 1
        ), "No services discovered after initial publication"

        initial_service_info = next(
            (s for s in discovered_services if s.name == readable_name1), None
        )
        assert (
            initial_service_info is not None
        ), "Initial service not found in discovered list"
        assert initial_service_info.port == service_port1
        assert initial_service_info.mdns_name.startswith(instance_name)

        # Publish the second (updated) version
        service_port2 = 50003
        readable_name2 = f"UpdateTestService_V2_{service_type_suffix}"

        # Explicitly close the first publisher to unregister its service
        # so the second publisher can register the same instance name.
        if publisher1_obj:
            await publisher1_obj.close()
            # Allow a brief moment for the unregistration to propagate.
            await asyncio.sleep(0.2)

        publisher2_obj = InstancePublisher(
            port=service_port2,
            service_type=service_type,
            readable_name=readable_name2,
            instance_name=instance_name,  # SAME instance_name
        )
        await publisher2_obj.publish()
        await asyncio.wait_for(discovery_event2.wait(), timeout=10.0)

    except asyncio.TimeoutError as e:
        if not discovery_event1.is_set():
            pytest.fail(
                f"Initial service {readable_name1} not discovered within timeout. Error: {e}"
            )
        elif not discovery_event2.is_set():
            pytest.fail(
                f"Updated service {readable_name2} not discovered within timeout. Error: {e}"
            )
        else:
            pytest.fail(f"A timeout error occurred: {e}")
    finally:
        if publisher1_obj:
            await publisher1_obj.close()  # Explicitly close
        if publisher2_obj:
            await publisher2_obj.close()  # Explicitly close
        if listener_obj:  # listener_obj is always created
            del listener_obj
        gc.collect()  # Encourage faster cleanup

    assert (
        discovery_event2.is_set()
    ), "Second discovery event (for update) was not set"
    # We expect at least two events now: one for the initial, one for the update.
    # mDNS might send multiple add events, so we check that the count increased.
    # And more importantly, that the new service details are present.
    assert (
        client.call_count >= 2
    ), "Expected at least two service added calls (initial and update)"

    # The latest discovered service should be the updated one.
    # Iterating backwards to find the most recent one matching the updated name.
    updated_service_info = None
    for service in reversed(discovered_services):
        if service.name == readable_name2:
            updated_service_info = service
            break

    assert (
        updated_service_info is not None
    ), "Updated service not found in discovered list"
    assert updated_service_info.port == service_port2
    assert updated_service_info.mdns_name.startswith(instance_name)

    # Verify that the old service is either gone or the new one is preferred/seen later.
    # This part is tricky without explicit on_service_removed or more knowledge of underlying behavior.
    # For now, we've confirmed the new one is present.

    await asyncio.sleep(0.1)


@pytest.mark.asyncio
async def test_instance_unpublishing():
    service_type_suffix = uuid.uuid4().hex[:8]
    service_type = f"_unpub-{service_type_suffix}._tcp.local."
    service_port = 50004
    readable_name = f"UnpublishTestService_{service_type_suffix}"
    instance_name = f"UnpublishInstance_{service_type_suffix}"

    # Phase 1: Publish and discover the service
    discovery_event1 = asyncio.Event()
    discovered_services1 = []
    # Use the original DiscoveryTestClient for simplicity in this phase
    client1 = DiscoveryTestClient(discovery_event1, discovered_services1)
    listener1 = InstanceListener(client=client1, service_type=service_type)

    publisher = InstancePublisher(
        port=service_port,
        service_type=service_type,
        readable_name=readable_name,
        instance_name=instance_name,
    )

    try:
        await publisher.publish()
        await asyncio.wait_for(discovery_event1.wait(), timeout=10.0)
    except asyncio.TimeoutError:
        pytest.fail(
            f"Service {readable_name} (instance {instance_name}) not discovered during initial publishing phase."
        )
    finally:
        # No explicit stop for listener1 as per instructions, rely on GC
        # No explicit unpublish for publisher yet.
        # Publisher is closed in the new finally block below.
        pass

    assert discovery_event1.is_set(), "Initial discovery event was not set."
    assert (
        len(discovered_services1) == 1
    ), "Incorrect number of services discovered initially."
    assert (
        discovered_services1[0].name == readable_name
    ), "Incorrect service name discovered initially."

    # Clean up listener1 by removing reference. Actual cleanup depends on GC and InstanceListener's __del__ if any.
    del listener1
    # If InstanceListener held onto client1, that also needs to be considered.
    # For this test, we assume client1 is simple and listener1 going out of scope is enough for its mDNS parts.

    # Phase 2: "Unpublish" the service
    # Explicitly close the publisher to unregister the service.
    if publisher:  # Ensure publisher was created
        await publisher.close()
    # import gc # gc is now imported at top level
    gc.collect()  # GC still useful for other objects

    # Allow time for mDNS "goodbye" packets to propagate and network state to settle.
    # This duration is heuristic. Zeroconf's default TTL for records is often 60 or 120 seconds.
    # A client might cache for that long. However, "goodbye" packets (if sent) should update clients sooner.
    # If RecordPublisher.unpublish_all() correctly calls unregister_service, goodbye packets should be sent.
    await asyncio.sleep(5.0)  # Increased sleep for network propagation

    # Phase 3: Attempt to discover the service with a new listener, expecting no discovery
    discovery_event2 = asyncio.Event()
    discovered_services2 = []
    client2 = DiscoveryTestClient(discovery_event2, discovered_services2)
    # Create a new listener to ensure it's not using cached data from the old one (if any such cache existed)
    listener2 = InstanceListener(client=client2, service_type=service_type)

    try:
        # Wait for a shorter period, as we expect a timeout (no service found).
        await asyncio.wait_for(discovery_event2.wait(), timeout=5.0)
        # If the event is set, it means the service was discovered, which is a failure for this test phase.
        # Explicitly check discovered_services2 as well, as event might be set but list empty (less likely)
        if discovered_services2:
            pytest.fail(
                f"Service {readable_name} (instance {instance_name}) was unexpectedly discovered after being unpublished. Discovered: {discovered_services2[0]}"
            )
        # Even if event is set but list is empty, it's unexpected if event implies discovery.
        # However, DiscoveryTestClient only sets event if a service is added.
    except asyncio.TimeoutError:
        # This is the expected outcome: the service was not found, so wait_for timed out.
        pass
    finally:
        # No explicit stop for listener2, rely on GC
        del listener2  # Clean up listener2
        gc.collect()

    assert (
        not discovery_event2.is_set()
    ), f"Discovery event was unexpectedly set for unpublished service. Call count: {client2._call_count}"
    assert (
        len(discovered_services2) == 0
    ), f"Service list not empty. Found {len(discovered_services2)} services after unpublishing. Details: {discovered_services2}"

    await asyncio.sleep(0.1)


class MultiDiscoveryTestClient(InstanceListener.Client):
    __test__ = False

    def __init__(
        self,
        services_list: list[ServiceInfo],
        expected_discoveries: int,
        all_discovered_event: asyncio.Event,
    ):
        super().__init__()
        self.services_list = services_list
        self.expected_discoveries = expected_discoveries
        self.all_discovered_event = all_discovered_event
        self.lock = asyncio.Lock()  # To protect access to shared list & event

    async def _on_service_added(self, connection_info: ServiceInfo) -> None:
        async with self.lock:
            # Avoid duplicates if mDNS sends multiple add notifications for the same service instance
            if not any(
                s.mdns_name == connection_info.mdns_name
                for s in self.services_list
            ):
                self.services_list.append(connection_info)
                if len(self.services_list) >= self.expected_discoveries:
                    if not self.all_discovered_event.is_set():
                        self.all_discovered_event.set()

    async def _on_service_removed(self, service_name: str):
        # Basic implementation for existing tests, can be expanded if needed
        print(
            f"Service removed (not actively handled in this client): {service_name}"
        )
        # Potentially remove from self.services_list if necessary for test logic
        # For now, keeping it simple as these tests focus on additions.
        pass


@pytest.mark.asyncio
async def test_multiple_publishers_one_listener():
    service_type_suffix = uuid.uuid4().hex[:8]
    service_type = f"_mpub-{service_type_suffix}._tcp.local."

    all_discovered_event = asyncio.Event()
    discovered_services = []
    # Expecting 2 services to be discovered
    client = MultiDiscoveryTestClient(
        discovered_services,
        expected_discoveries=2,
        all_discovered_event=all_discovered_event,
    )
    listener = InstanceListener(client=client, service_type=service_type)

    publishers = []
    expected_service_details = []

    # Publisher 1
    p1_port = 50005
    p1_readable_name = f"MultiPubService1_{service_type_suffix}"
    p1_instance_name = f"MultiPubInstance1_{service_type_suffix}"
    publisher1 = InstancePublisher(
        port=p1_port,
        service_type=service_type,
        readable_name=p1_readable_name,
        instance_name=p1_instance_name,
    )
    publishers.append(publisher1)
    expected_service_details.append(
        {
            "name": p1_readable_name,
            "port": p1_port,
            "instance_prefix": p1_instance_name,
        }
    )

    # Publisher 2
    p2_port = 50006
    p2_readable_name = f"MultiPubService2_{service_type_suffix}"
    p2_instance_name = (
        f"MultiPubInstance2_{service_type_suffix}"  # Different instance name
    )
    publisher2 = InstancePublisher(
        port=p2_port,
        service_type=service_type,
        readable_name=p2_readable_name,
        instance_name=p2_instance_name,
    )
    publishers.append(publisher2)
    expected_service_details.append(
        {
            "name": p2_readable_name,
            "port": p2_port,
            "instance_prefix": p2_instance_name,
        }
    )

    try:
        for p in publishers:
            await p.publish()

        await asyncio.wait_for(all_discovered_event.wait(), timeout=15.0)
    except asyncio.TimeoutError:
        pytest.fail(
            f"Not all services ({len(expected_service_details)}) discovered within timeout. Found {len(discovered_services)}."
        )
    finally:
        for p in publishers:  # Close all publishers
            if p:
                await p.close()
        del listener
        # import gc # gc is now imported at top level
        gc.collect()

    assert (
        all_discovered_event.is_set()
    ), "Event for all discoveries was not set."
    assert (
        len(discovered_services) == 2
    ), f"Expected 2 services, discovered {len(discovered_services)}"

    names_found = {s.name for s in discovered_services}
    ports_found = {s.port for s in discovered_services}
    mdns_names_found = {s.mdns_name for s in discovered_services}

    for expected in expected_service_details:
        assert (
            expected["name"] in names_found
        ), f"Expected name {expected['name']} not found."
        assert (
            expected["port"] in ports_found
        ), f"Expected port {expected['port']} not found."
        assert any(
            mname.startswith(expected["instance_prefix"])
            for mname in mdns_names_found
        ), f"No mDNS name starting with {expected['instance_prefix']} found."

    await asyncio.sleep(0.1)


@pytest.mark.asyncio
async def test_one_publisher_multiple_listeners():
    service_type_suffix = uuid.uuid4().hex[:8]
    service_type = f"_mlis-{service_type_suffix}._tcp.local."
    service_port = 50007
    readable_name = f"MultiListenService_{service_type_suffix}"
    instance_name = f"MultiListenInstance_{service_type_suffix}"

    publisher = InstancePublisher(
        port=service_port,
        service_type=service_type,
        readable_name=readable_name,
        instance_name=instance_name,
    )

    listeners_data = []
    tasks = []

    try:
        await publisher.publish()
        # Give publisher a moment to ensure it's up before listeners start
        # This is important as mDNS registration can take a moment.
        await asyncio.sleep(1.0)

        # Listener 1
        listener1_event = asyncio.Event()
        listener1_services = []
        client1 = MultiDiscoveryTestClient(
            listener1_services,
            expected_discoveries=1,
            all_discovered_event=listener1_event,
        )
        listener1 = InstanceListener(client=client1, service_type=service_type)
        listeners_data.append(
            {
                "event": listener1_event,
                "services": listener1_services,
                "listener_obj": listener1,
            }
        )
        tasks.append(asyncio.wait_for(listener1_event.wait(), timeout=10.0))

        await asyncio.sleep(3.0)

        # Listener 2
        listener2_event = asyncio.Event()
        listener2_services = []
        client2 = MultiDiscoveryTestClient(
            listener2_services,
            expected_discoveries=1,
            all_discovered_event=listener2_event,
        )
        listener2 = InstanceListener(client=client2, service_type=service_type)
        listeners_data.append(
            {
                "event": listener2_event,
                "services": listener2_services,
                "listener_obj": listener2,
            }
        )
        tasks.append(asyncio.wait_for(listener2_event.wait(), timeout=10.0))

        await asyncio.gather(*tasks)

    except asyncio.TimeoutError:
        for i, data in enumerate(listeners_data):
            if not data["event"].is_set():
                pytest.fail(
                    f"Listener {i+1} did not discover the service within timeout."
                )
        # If gather fails due to timeout, this part might not be reached directly,
        # but individual timeouts in tasks will raise TimeoutError.
    finally:
        if publisher:
            await publisher.close()
        for data in listeners_data:
            del data["listener_obj"]  # Remove reference to listener
        # import gc # gc is now imported at top level
        gc.collect()

    for i, data in enumerate(listeners_data):
        assert data[
            "event"
        ].is_set(), f"Listener {i+1}'s discovery event was not set."
        assert (
            len(data["services"]) == 1
        ), f"Listener {i+1} discovered {len(data['services'])} services, expected 1."
        service_info = data["services"][0]
        assert (
            service_info.name == readable_name
        ), f"Listener {i+1} discovered name {service_info.name}, expected {readable_name}."
        assert (
            service_info.port == service_port
        ), f"Listener {i+1} discovered port {service_info.port}, expected {service_port}."
        assert service_info.mdns_name.startswith(
            instance_name
        ), f"Listener {i+1} discovered mdns_name {service_info.mdns_name}, expected to start with {instance_name}."

    await asyncio.sleep(0.1)


@pytest.mark.asyncio
async def test_publisher_starts_after_listener():
    service_type_suffix = uuid.uuid4().hex[:8]
    service_type = f"_res-{service_type_suffix}._tcp.local."
    service_port = 50008
    readable_name = f"ResilienceTestService_{service_type_suffix}"
    instance_name = f"ResilienceInstance_{service_type_suffix}"

    discovery_event = asyncio.Event()
    discovered_services = []

    # Ensure DiscoveryTestClient is available (it was defined in the first test)
    # If not, it should be redefined here or imported.
    # For this exercise, we assume it's available in the same file.
    client = DiscoveryTestClient(discovery_event, discovered_services)

    listener = None
    publisher = None

    try:
        # Start the listener first
        listener = InstanceListener(client=client, service_type=service_type)
        # Give the listener a moment to fully initialize and start listening.
        # This is crucial for mDNS, as the ServiceBrowser needs to be active.
        await asyncio.sleep(1.0)

        # Then, start the publisher
        publisher = InstancePublisher(
            port=service_port,
            service_type=service_type,
            readable_name=readable_name,
            instance_name=instance_name,
        )
        await publisher.publish()

        # Wait for discovery
        await asyncio.wait_for(discovery_event.wait(), timeout=10.0)

    except asyncio.TimeoutError:
        pytest.fail(
            f"Service {readable_name} not discovered when publisher started after listener."
        )
    finally:
        if publisher:
            await publisher.close()
        del listener
        # import gc # gc is now imported at top level
        gc.collect()

    assert discovery_event.is_set(), "Discovery event was not set."
    assert (
        len(discovered_services) == 1
    ), f"Expected 1 service, discovered {len(discovered_services)}."

    service_info = discovered_services[0]
    assert (
        service_info.name == readable_name
    ), f"Discovered name {service_info.name} != {readable_name}"
    assert (
        service_info.port == service_port
    ), f"Discovered port {service_info.port} != {service_port}"
    assert service_info.mdns_name.startswith(
        instance_name
    ), f"Discovered mDNS name {service_info.mdns_name} does not start with {instance_name}"
    assert service_info.addresses, "Discovered service addresses list is empty"
    # Basic validation of addresses (already done in first test, good to have here too)
    # import ipaddress # ipaddress is now imported at top level
    for addr_str in service_info.addresses:
        try:
            ipaddress.ip_address(addr_str)
        except ValueError:
            pytest.fail(f"Invalid IP address string found: {addr_str}")

    await asyncio.sleep(0.1)
