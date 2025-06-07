# pylint: disable=C0301
"""Provides RemoteDataAggregatorImpl, a concrete implementation of the RemoteDataAggregator interface.

This class manages RemoteDataOrganizer instances for each data source (identified by
CallerIdentifier) and handles data timeout tracking. It acts as a central point
for collecting and accessing data from multiple remote endpoints.
"""

import datetime
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Generic, List, Optional, TypeVar, overload

from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.data.data_timeout_tracker import DataTimeoutTracker
from tsercom.data.exposed_data import ExposedData
from tsercom.data.remote_data_aggregator import RemoteDataAggregator
from tsercom.data.remote_data_organizer import RemoteDataOrganizer
from tsercom.data.remote_data_reader import RemoteDataReader

DataTypeT = TypeVar("DataTypeT", bound=ExposedData)


class RemoteDataAggregatorImpl(
    # pylint: disable=C0301
    Generic[DataTypeT],
    RemoteDataAggregator[DataTypeT],
    RemoteDataOrganizer.Client,
    RemoteDataReader[DataTypeT],
):
    """Concrete implementation of `RemoteDataAggregator`.

    This class manages a collection of `RemoteDataOrganizer` instances, one for
    each unique `CallerIdentifier` from which data is received. It handles
    thread-safe access to these organizers and orchestrates data timeout
    tracking if configured. It also acts as a client to its own organizers
    to propagate data availability signals and as a `RemoteDataReader` to
    initiate the data processing pipeline.
    """

    @overload
    def __init__(
        self,
        thread_pool: ThreadPoolExecutor,
        client: Optional[RemoteDataAggregator.Client] = None,
    ):
        """Initializes with thread pool and optional client.

        Args:
            thread_pool: Executor for asynchronous tasks.
            client: Optional client for data event callbacks.
        """
        ...

    @overload
    def __init__(
        self,
        thread_pool: ThreadPoolExecutor,
        client: Optional[RemoteDataAggregator.Client] = None,
        *,
        tracker: DataTimeoutTracker,
    ):
        """Initializes with thread pool, client, and custom data timeout tracker.

        Args:
            thread_pool: Executor for asynchronous tasks.
            client: Optional client for data event callbacks.
            tracker: Custom `DataTimeoutTracker` instance.
        """
        ...

    @overload
    def __init__(
        self,
        thread_pool: ThreadPoolExecutor,
        client: Optional[RemoteDataAggregator.Client] = None,
        *,
        timeout: int,
    ):
        """Initializes with thread pool, client, and data timeout value.

        A `DataTimeoutTracker` will be created internally using this timeout.

        Args:
            thread_pool: Executor for asynchronous tasks.
            client: Optional client for data event callbacks.
            timeout: Timeout duration in seconds for data tracking.
        """
        ...

    def __init__(
        self,
        thread_pool: ThreadPoolExecutor,
        client: Optional[RemoteDataAggregator.Client] = None,
        *,
        tracker: Optional[DataTimeoutTracker] = None,
        timeout: Optional[int] = None,
    ) -> None:
        """Initializes the RemoteDataAggregatorImpl.

        This constructor is overloaded. It can be called with:
        - `thread_pool` and optional `client`.
        - `thread_pool`, optional `client`, and `tracker` (for timeout tracking).
        - `thread_pool`, optional `client`, and `timeout` (to create a default tracker).

        Args:
            thread_pool: A `ThreadPoolExecutor` for running asynchronous tasks,
                         primarily for `RemoteDataOrganizer` instances.
            client: An optional client implementing `RemoteDataAggregator.Client`
                    to receive callbacks for data events.
            tracker: An optional `DataTimeoutTracker` instance. If provided, it's
                     used to track data timeouts. `timeout` should not be provided.
            timeout: An optional integer specifying the timeout duration in seconds.
                     If provided, a `DataTimeoutTracker` is created and started.
                     `tracker` should not be provided.

        Raises:
            AssertionError: If both `tracker` and `timeout` are provided.
        """
        assert not (
            timeout is not None and tracker is not None
        ), "Cannot specify both 'timeout' and 'tracker' simultaneously."

        if tracker is None and timeout is not None and timeout > 0:
            tracker = DataTimeoutTracker(timeout)
            tracker.start()

        self.__thread_pool = thread_pool
        self.__client = client
        self.__tracker = tracker

        self.__organizers: Dict[
            CallerIdentifier, RemoteDataOrganizer[DataTypeT]
        ] = {}
        self.__lock: threading.Lock = threading.Lock()

    def stop(
        self, identifier: Optional[CallerIdentifier] = None
    ) -> None:  # Renamed id to identifier
        """Stops data processing for one or all callers.

        If an `identifier` is provided, stops the `RemoteDataOrganizer` for that specific
        caller. Otherwise, stops all organizers managed by this aggregator.

        Args:
            identifier: Optional `CallerIdentifier` of the caller to stop.

        Raises:
            KeyError: If `identifier` is provided but not found among active organizers.
        """
        with self.__lock:
            if identifier is not None:
                organizer = self.__organizers.get(identifier)
                if organizer is None:
                    raise KeyError(
                        f"Caller ID '{identifier}' not found in active organizers during stop."
                    )
                organizer.stop()
                return

            for organizer in self.__organizers.values():
                organizer.stop()

    # pylint: disable=arguments-differ # Signature matches base, Pylint false positive with @overload
    @overload
    def has_new_data(self) -> Dict[CallerIdentifier, bool]:
        """Checks for new data for all callers.

        Returns:
            Dict[CallerIdentifier, bool]: True if new data for a caller.
        """
        ...

    @overload
    def has_new_data(self, identifier: CallerIdentifier) -> bool:
        """Checks if new data is available for a specific caller.

        Args:
            identifier: The `CallerIdentifier` to check.

        Returns:
            bool: True if new data is available, False otherwise.
        """
        ...

    def has_new_data(
        self, identifier: Optional[CallerIdentifier] = None
    ) -> Dict[CallerIdentifier, bool] | bool:
        """Checks for new data for one or all callers.

        Args:
            identifier: Optional `CallerIdentifier`. If provided, checks for this specific
                caller. Otherwise, checks for all callers.

        Returns:
            If `identifier` is provided, returns a boolean indicating if new data is available for that caller (returns False if the `identifier` is not found).
            If `identifier` is None, returns a dictionary mapping each `CallerIdentifier`
            to a boolean.
        """
        with self.__lock:
            if identifier is not None:
                organizer = self.__organizers.get(identifier)
                if organizer is None:
                    return False
                return organizer.has_new_data()

            results = {}
            for key, org_item in self.__organizers.items():
                results[key] = org_item.has_new_data()
            return results

    # pylint: disable=arguments-differ # Signature matches base, Pylint false positive with @overload
    @overload
    def get_new_data(self) -> Dict[CallerIdentifier, List[DataTypeT]]:
        """Retrieves all new data items for all callers.

        Returns:
            Dict[CallerIdentifier, List[DataTypeT]]: New data from each caller.
        """
        ...

    @overload
    def get_new_data(self, identifier: CallerIdentifier) -> List[DataTypeT]:
        """Retrieves all new data items for a specific caller.

        Args:
            identifier: The `CallerIdentifier` for which to retrieve new data.

        Returns:
            List[DataTypeT]: New data items from the specified caller.
        """
        ...

    def get_new_data(
        self, identifier: Optional[CallerIdentifier] = None
    ) -> Dict[CallerIdentifier, List[DataTypeT]] | List[DataTypeT]:
        """Retrieves new data for one or all callers.

        Args:
            identifier: Optional `CallerIdentifier`. If provided, retrieves data for this
                specific caller. Otherwise, retrieves data for all callers.

        Returns:
            If `identifier` is provided, returns a list of new data items for that caller.
            If `identifier` is None, returns a dictionary mapping each `CallerIdentifier`
            to a list of its new data items.

        Raises:
            KeyError: If `identifier` is provided but not found.
        """
        with self.__lock:
            if identifier is not None:
                organizer = self.__organizers.get(identifier)
                if organizer is None:
                    raise KeyError(
                        f"Caller ID '{identifier}' not found for get_new_data."
                    )
                return organizer.get_new_data()

            results = {}
            for key, organizer in self.__organizers.items():
                results[key] = organizer.get_new_data()
            return results

    @overload
    def get_most_recent_data(
        self,
    ) -> Dict[CallerIdentifier, Optional[DataTypeT]]:
        """Retrieves the most recent data item for all callers.

        Returns `None` for a caller if no data or if timed out.

        Returns:
            Dict[CallerIdentifier, Optional[DataTypeT]]: Most recent data or None.
        """
        ...

    @overload
    def get_most_recent_data(
        self, identifier: CallerIdentifier
    ) -> Optional[DataTypeT]:
        """Retrieves the most recent data item for a specific caller.

        Returns `None` if no data for this caller or if timed out.

        Args:
            identifier: The `CallerIdentifier` for which to retrieve data.

        Returns:
            Optional[DataTypeT]: The most recent data item or `None`.
        """
        ...

    # pylint: disable=arguments-differ # Signature matches base, Pylint false positive with @overload
    def get_most_recent_data(
        self, identifier: Optional[CallerIdentifier] = None
    ) -> Dict[CallerIdentifier, DataTypeT | None] | DataTypeT | None:
        """Retrieves the most recent data for one or all callers.

        Args:
            identifier: Optional `CallerIdentifier`. If provided, retrieves data for this
                specific caller. Otherwise, retrieves data for all callers.

        Returns:
            If `identifier` is provided, returns the most recent data item (or None) for
            that caller.
            If `identifier` is None, returns a dictionary mapping each `CallerIdentifier`
            to its most recent data item (or None).

        Raises:
            KeyError: If `identifier` is provided but not found.
        """
        with self.__lock:
            if identifier is not None:
                organizer = self.__organizers.get(identifier)
                if organizer is None:
                    raise KeyError(
                        f"Caller ID '{identifier}' not found for get_most_recent_data."
                    )
                return organizer.get_most_recent_data()

            results = {}
            for key, organizer in self.__organizers.items():
                results[key] = organizer.get_most_recent_data()
            return results

    @overload
    def get_data_for_timestamp(
        self, timestamp: datetime.datetime
    ) -> Dict[CallerIdentifier, Optional[DataTypeT]]:
        """Retrieves data before or at `timestamp` for all callers.

        Returns `None` for a caller if no suitable data exists.

        Args:
            timestamp: The `datetime` to compare data timestamps against.

        Returns:
            Dict[CallerIdentifier, Optional[DataTypeT]]: Relevant data or None.
        """
        ...

    @overload
    def get_data_for_timestamp(
        self, timestamp: datetime.datetime, identifier: CallerIdentifier
    ) -> Optional[DataTypeT]:
        """Retrieves data before or at `timestamp` for a specific caller.

        Returns `None` if no suitable data exists for this caller.

        Args:
            timestamp: The `datetime` to compare data timestamps against.
            identifier: The `CallerIdentifier` for which to retrieve data.

        Returns:
            Optional[DataTypeT]: Relevant data item or `None`.
        """
        ...

    # pylint: disable=arguments-differ # Signature matches base, Pylint false positive with @overload
    def get_data_for_timestamp(
        self,
        timestamp: datetime.datetime,
        identifier: Optional[CallerIdentifier] = None,
    ) -> Dict[CallerIdentifier, DataTypeT | None] | DataTypeT | None:
        """Retrieves data for a specific timestamp for one or all callers.

        Args:
            timestamp: The `datetime` to compare data against.
            identifier: Optional `CallerIdentifier`. If provided, retrieves data for this
                specific caller. Otherwise, retrieves data for all callers.

        Returns:
            If `identifier` is provided, returns the data item (or None) for that caller
            at the given timestamp.
            If `identifier` is None, returns a dictionary mapping each `CallerIdentifier`
            to its data item (or None) at the given timestamp.

        Raises:
            KeyError: If `identifier` is provided but not found.
        """
        with self.__lock:
            if identifier is not None:
                organizer = self.__organizers.get(identifier)
                if organizer is None:
                    raise KeyError(
                        f"Caller ID '{identifier}' not found for get_data_for_timestamp."
                    )
                return organizer.get_data_for_timestamp(timestamp)

            results = {}
            for key, organizer_item in self.__organizers.items():
                results[key] = organizer_item.get_data_for_timestamp(
                    timestamp=timestamp
                )
            return results

    def _on_data_available(
        self,
        data_organizer: RemoteDataOrganizer[DataTypeT],  # type: ignore[override]
    ) -> None:
        """Callback from a `RemoteDataOrganizer` when it has new data.

        This method is part of the `RemoteDataOrganizer.Client` interface.
        It notifies the aggregator's client, if one is registered.

        Args:
            data_organizer: The `RemoteDataOrganizer` instance that has new data.
        """
        if self.__client is not None:
            # pylint: disable=W0212 # Calling listener method
            self.__client._on_data_available(self, data_organizer.caller_id)

    def _on_data_ready(self, new_data: DataTypeT) -> None:
        """Handles incoming raw data, routing it to the appropriate `RemoteDataOrganizer`.

        This method is part of the `RemoteDataReader` interface. If an organizer
        for the data's `caller_id` doesn't exist, one is created and started.
        The data is then passed to the organizer. If a new organizer is created,
        the aggregator's client is notified.

        Args:
            new_data: The incoming data item, which must be a subclass of `ExposedData`.

        Raises:
            TypeError: If `new_data` is not a subclass of `ExposedData`.
        """

        if not isinstance(new_data, ExposedData):
            raise TypeError(
                f"Expected new_data to be an instance of ExposedData, but got {type(new_data).__name__}."
            )

        data_organizer: RemoteDataOrganizer[DataTypeT]
        is_new_organizer = False

        with self.__lock:
            if new_data.caller_id not in self.__organizers:
                data_organizer = RemoteDataOrganizer(
                    self.__thread_pool,
                    new_data.caller_id,
                    self,
                )

                if self.__tracker is not None:
                    self.__tracker.register(data_organizer)

                data_organizer.start()
                self.__organizers[new_data.caller_id] = data_organizer
                is_new_organizer = True

            else:
                data_organizer = self.__organizers[new_data.caller_id]
        # pylint: disable=W0212 # Calling data host method
        data_organizer._on_data_ready(new_data)

        if is_new_organizer and self.__client is not None:
            # pylint: disable=W0212 # Calling listener method
            self.__client._on_new_endpoint_began_transmitting(
                self, data_organizer.caller_id
            )
