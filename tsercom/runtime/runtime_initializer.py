from typing import Generic, TypeVar

from tsercom.data.exposed_data import ExposedData
from tsercom.data.remote_data_aggregator import RemoteDataAggregator


TDataType = TypeVar("TDataType", bound=ExposedData)


class RuntimeInitializer(Generic[TDataType]):
    """
    A base class for server and client runtime initializer instances. Mainly
    used to simplify sharing of code between client and server.
    """

    def client(self) -> RemoteDataAggregator[TDataType].Client | None:
        """
        Returns the client that should be informed when new data is provided to
        the RemoteDataAggregator instance created for the runtime created from
        this initializer, or None if no such instance should be used.
        """
        return None

    def timeout(self) -> int | None:
        """
        Returns the timeout (in seconds) that should be used for data received
        by the runtime created from this initializer, or None if data should not
        time out.
        """
        return 60
