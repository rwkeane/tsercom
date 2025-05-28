from abc import ABC, abstractmethod


class MdnsPublisher(ABC):
    @abstractmethod
    def publish(self) -> None:
        """Publishes the service to mDNS.

        Constructs a `zeroconf.ServiceInfo` object with the configured details
        and registers it with a `Zeroconf` instance. The service is published
        using IPVersion.V4Only.
        """
        pass
