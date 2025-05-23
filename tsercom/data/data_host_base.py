from typing import Generic, Optional, TypeVar

from tsercom.data.data_host import DataHost
from tsercom.data.data_timeout_tracker import DataTimeoutTracker
from tsercom.data.exposed_data import ExposedData
from tsercom.data.remote_data_aggregator import RemoteDataAggregator
from tsercom.data.remote_data_aggregator_impl import RemoteDataAggregatorImpl
from tsercom.data.remote_data_reader import RemoteDataReader
from tsercom.threading.thread_watcher import ThreadWatcher

TDataType = TypeVar("TDataType", bound=ExposedData)


from typing import Any # Added Any for *args, **kwargs

class DataHostBase(
    Generic[TDataType], DataHost[TDataType], RemoteDataReader[TDataType]
):
    """A base implementation of DataHost and RemoteDataReader.

    This class provides a concrete implementation for data aggregation and
    handling data readiness, intended to be reused by various ServerHost and
    ClientHost implementations. It sets up a `RemoteDataAggregatorImpl`
    and optionally a `DataTimeoutTracker`.
    """

    def __init__(
        self,
        watcher: ThreadWatcher,
        aggregation_client: Optional[RemoteDataAggregator[TDataType].Client] = None,
        timeout_seconds: int = 60,
        *args: Any,  # Pass through additional arguments to superclass
        **kwargs: Any, # Pass through additional keyword arguments to superclass
    ) -> None:
        """Initializes the DataHostBase.

        Args:
            watcher: A ThreadWatcher to monitor threads created for the
                     data aggregator.
            aggregation_client: An optional client for the RemoteDataAggregator
                                to notify when new data is available.
            timeout_seconds: The duration in seconds after which data is
                             considered timed out. If <= 0, timeout tracking
                             is disabled.
            *args: Variable length argument list passed to the superclass constructor.
            **kwargs: Arbitrary keyword arguments passed to the superclass constructor.
        """
        # Create a single-threaded executor for the aggregator.
        # This ensures sequential processing of data aggregation tasks.
        thread_pool = watcher.create_tracked_thread_pool_executor(
            max_workers=1
        )

        # Initialize and start a DataTimeoutTracker if a positive timeout is specified.
        # This tracker monitors data for staleness.
        tracker: Optional[DataTimeoutTracker] = None
        if timeout_seconds > 0:
            tracker = DataTimeoutTracker(timeout_seconds)
            tracker.start()

        # Instantiate the core data aggregator.
        self.__aggregator: RemoteDataAggregatorImpl[TDataType] = RemoteDataAggregatorImpl[TDataType](
            thread_pool, aggregation_client, tracker
        )

        # Call the constructor of the superclass (likely DataHost or RemoteDataReader,
        # or ultimately object if they don't have __init__ taking *args, **kwargs).
        super().__init__(*args, **kwargs)

    def _on_data_ready(self, new_data: TDataType) -> None:
        """Handles new data by passing it to the internal data aggregator.

        This method is called when new data is available, typically from a
        source that this DataHostBase is reading from (as a RemoteDataReader).

        Args:
            new_data: The new data item that has become available.
        """
        self.__aggregator._on_data_ready(new_data)

    def _remote_data_aggregator(self) -> RemoteDataAggregator[TDataType]:
        """Provides the internal `RemoteDataAggregator` instance.

        This method fulfills the `DataHost` abstract contract.

        Returns:
            The `RemoteDataAggregator[TDataType]` instance used by this host.
        """
        return self.__aggregator
