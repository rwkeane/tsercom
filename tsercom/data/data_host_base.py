"""DataHostBase: reusable base for DataHost & RemoteDataReader."""

from typing import Any, Generic, Optional, TypeVar

from tsercom.data.data_host import DataHost
from tsercom.data.data_timeout_tracker import DataTimeoutTracker
from tsercom.data.exposed_data import ExposedData
from tsercom.data.remote_data_aggregator import RemoteDataAggregator
from tsercom.data.remote_data_aggregator_impl import RemoteDataAggregatorImpl
from tsercom.data.remote_data_reader import RemoteDataReader
from tsercom.threading.thread_watcher import ThreadWatcher

DataTypeT = TypeVar("DataTypeT", bound=ExposedData)


# pylint: disable=R0903 # Base class providing concrete interface implementations
class DataHostBase(
    Generic[DataTypeT], DataHost[DataTypeT], RemoteDataReader[DataTypeT]
):
    """A base implementation of DataHost and RemoteDataReader.

    This class provides a concrete implementation for data aggregation and
    handling data readiness, intended to be reused by various ServerHost and
    ClientHost implementations. It sets up a `RemoteDataAggregatorImpl`
    and optionally a `DataTimeoutTracker`.
    """

    # pylint: disable=W1113 # False positive with complex defaults and *args
    def __init__(
        self,
        watcher: ThreadWatcher,
        aggregation_client: Optional[RemoteDataAggregator.Client] = None,
        timeout_seconds: int = 60,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initializes the DataHostBase.

        Args:
            watcher: ThreadWatcher to monitor threads for the data aggregator.
            aggregation_client: Optional client for RemoteDataAggregator
                                to notify of new data.
            timeout_seconds: Duration in seconds for data timeout. If <= 0,
                             timeout tracking disabled.
            *args: Variable length arguments for superclass.
            **kwargs: Keyword arguments for superclass.
        """
        # This ensures sequential processing of data aggregation tasks.
        thread_pool = watcher.create_tracked_thread_pool_executor(
            max_workers=1
        )

        tracker: Optional[DataTimeoutTracker] = None
        if timeout_seconds > 0:
            tracker = DataTimeoutTracker(timeout_seconds)
            tracker.start()

        # Assign to a local variable first, then to self.__aggregator to avoid redefinition error.
        aggregator_instance: RemoteDataAggregatorImpl[DataTypeT]
        if tracker is not None:
            aggregator_instance = RemoteDataAggregatorImpl[DataTypeT](
                thread_pool, aggregation_client, tracker=tracker
            )
        else:
            aggregator_instance = RemoteDataAggregatorImpl[DataTypeT](
                thread_pool, aggregation_client
            )
        self.__aggregator = aggregator_instance

        super().__init__(*args, **kwargs)

    def _on_data_ready(self, new_data: DataTypeT) -> None:
        """Handles new data by passing it to the internal data aggregator.

        This method is called when new data is available, typically from a
        source that this DataHostBase is reading from (as a RemoteDataReader).

        Args:
            new_data: The new data item that has become available.
        """
        # pylint: disable=W0212 # Calling client's data ready method
        self.__aggregator._on_data_ready(new_data)

    def _remote_data_aggregator(self) -> RemoteDataAggregator[DataTypeT]:
        """Provides the internal `RemoteDataAggregator` instance.

        This method fulfills the `DataHost` abstract contract.

        Returns:
            The `RemoteDataAggregator[DataTypeT]` instance used by this host.
        """
        return self.__aggregator
