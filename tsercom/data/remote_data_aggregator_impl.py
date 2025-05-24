from concurrent.futures import ThreadPoolExecutor
import datetime
import threading
from typing import Dict, Generic, List, Optional, TypeVar, overload

from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.data.data_timeout_tracker import DataTimeoutTracker
from tsercom.data.exposed_data import ExposedData
from tsercom.data.remote_data_aggregator import RemoteDataAggregator
from tsercom.data.remote_data_organizer import RemoteDataOrganizer
from tsercom.data.remote_data_reader import RemoteDataReader

# Generic type for the data being aggregated, bound by ExposedData.
TDataType = TypeVar("TDataType", bound=ExposedData)


class RemoteDataAggregatorImpl(
    Generic[TDataType],
    RemoteDataAggregator[TDataType],
    RemoteDataOrganizer.Client, # Implements callback client for organizers
    RemoteDataReader[TDataType],  # Can receive data directly
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
        client: Optional[RemoteDataAggregator.Client[TDataType]] = None,
    ): ...

    @overload
    def __init__(
        self,
        thread_pool: ThreadPoolExecutor,
        client: Optional[RemoteDataAggregator.Client[TDataType]] = None,
        *,
        tracker: DataTimeoutTracker,
    ): ...

    @overload
    def __init__(
        self,
        thread_pool: ThreadPoolExecutor,
        client: Optional[RemoteDataAggregator.Client[TDataType]] = None,
        *,
        timeout: int,
    ): ...

    def __init__(
        self,
        thread_pool: ThreadPoolExecutor,
        client: Optional[RemoteDataAggregator.Client[TDataType]] = None,
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
        # Ensure that tracker and timeout are not simultaneously specified.
        assert not (timeout is not None and tracker is not None), \
            "Cannot specify both 'timeout' and 'tracker' simultaneously."

        # If a timeout duration is given but no tracker, create and start a default tracker.
        if tracker is None and timeout is not None and timeout > 0:
            tracker = DataTimeoutTracker(timeout)
            tracker.start()

        self.__thread_pool: ThreadPoolExecutor = thread_pool
        self.__client: Optional[RemoteDataAggregator.Client[TDataType]] = client
        self.__tracker: Optional[DataTimeoutTracker] = tracker
        
        # Dictionary to hold RemoteDataOrganizer instances, keyed by CallerIdentifier.
        # Access to this dictionary is protected by a lock.
        self.__organizers: Dict[
            CallerIdentifier, RemoteDataOrganizer[TDataType]
        ] = {}
        self.__lock: threading.Lock = threading.Lock()


    def stop(self, id: Optional[CallerIdentifier] = None) -> None:
        """Stops data processing for one or all callers.

        If an `id` is provided, stops the `RemoteDataOrganizer` for that specific
        caller. Otherwise, stops all organizers managed by this aggregator.

        Args:
            id: Optional `CallerIdentifier` of the caller to stop.

        Raises:
            KeyError: If `id` is provided but not found among active organizers.
        """
        with self.__lock:
            if id is not None:
                organizer = self.__organizers.get(id)
                if organizer is None:
                    raise KeyError(f"Caller ID '{id}' not found in active organizers during stop.")
                organizer.stop()
                # Optionally remove from __organizers dict if it won't be restarted.
                # self.__organizers.pop(id, None)
                return

            # Stop all organizers if no specific ID is given.
            for organizer in self.__organizers.values():
                organizer.stop()
            # Optionally clear the dictionary if all are stopped and won't be restarted.
            # self.__organizers.clear()

    def has_new_data(
        self, id: Optional[CallerIdentifier] = None
    ) -> Dict[CallerIdentifier, bool] | bool:
        """Checks for new data for one or all callers.

        Args:
            id: Optional `CallerIdentifier`. If provided, checks for this specific
                caller. Otherwise, checks for all callers.

        Returns:
            If `id` is provided, returns a boolean indicating if new data is
            available for that caller.
            If `id` is None, returns a dictionary mapping each `CallerIdentifier`
            to a boolean.

        Raises:
            KeyError: If `id` is provided but not found.
        """
        with self.__lock:
            if id is not None:
                organizer = self.__organizers.get(id)
                if organizer is None:
                    raise KeyError(f"Caller ID '{id}' not found for has_new_data.")
                return organizer.has_new_data()

            results = {}
            for key, organizer in self.__organizers.items():
                results[key] = organizer.has_new_data()
            return results

    def get_new_data(
        self, id: Optional[CallerIdentifier] = None
    ) -> Dict[CallerIdentifier, List[TDataType]] | List[TDataType]:
        """Retrieves new data for one or all callers.

        Args:
            id: Optional `CallerIdentifier`. If provided, retrieves data for this
                specific caller. Otherwise, retrieves data for all callers.

        Returns:
            If `id` is provided, returns a list of new data items for that caller.
            If `id` is None, returns a dictionary mapping each `CallerIdentifier`
            to a list of its new data items.

        Raises:
            KeyError: If `id` is provided but not found.
        """
        with self.__lock:
            if id is not None:
                organizer = self.__organizers.get(id)
                if organizer is None:
                    raise KeyError(f"Caller ID '{id}' not found for get_new_data.")
                return organizer.get_new_data()

            results = {}
            for key, organizer in self.__organizers.items():
                results[key] = organizer.get_new_data()
            return results

    def get_most_recent_data(
        self, id: Optional[CallerIdentifier] = None
    ) -> Dict[CallerIdentifier, TDataType | None] | TDataType | None:
        """Retrieves the most recent data for one or all callers.

        Args:
            id: Optional `CallerIdentifier`. If provided, retrieves data for this
                specific caller. Otherwise, retrieves data for all callers.

        Returns:
            If `id` is provided, returns the most recent data item (or None) for
            that caller.
            If `id` is None, returns a dictionary mapping each `CallerIdentifier`
            to its most recent data item (or None).

        Raises:
            KeyError: If `id` is provided but not found.
        """
        with self.__lock:
            if id is not None:
                organizer = self.__organizers.get(id)
                if organizer is None:
                    raise KeyError(f"Caller ID '{id}' not found for get_most_recent_data.")
                return organizer.get_most_recent_data()

            results = {}
            for key, organizer in self.__organizers.items():
                results[key] = organizer.get_most_recent_data()
            return results

    def get_data_for_timestamp(
        self,
        timestamp: datetime.datetime, # Corrected order: timestamp first
        id: Optional[CallerIdentifier] = None, # Corrected order: id second and optional
    ) -> Dict[CallerIdentifier, TDataType | None] | TDataType | None:
        """Retrieves data for a specific timestamp for one or all callers.
        
        Args:
            timestamp: The `datetime` to compare data against.
            id: Optional `CallerIdentifier`. If provided, retrieves data for this
                specific caller. Otherwise, retrieves data for all callers.

        Returns:
            If `id` is provided, returns the data item (or None) for that caller
            at the given timestamp.
            If `id` is None, returns a dictionary mapping each `CallerIdentifier`
            to its data item (or None) at the given timestamp.

        Raises:
            KeyError: If `id` is provided but not found.
        """
        with self.__lock:
            if id is not None:
                organizer = self.__organizers.get(id)
                if organizer is None:
                    raise KeyError(f"Caller ID '{id}' not found for get_data_for_timestamp.")
                return organizer.get_data_for_timestamp(timestamp)

            results = {}
            for key, organizer in self.__organizers.items():
                # Pass only timestamp to organizer's get_data_for_timestamp
                results[key] = organizer.get_data_for_timestamp(timestamp)
            return results

    def _on_data_available(
        self, data_organizer: RemoteDataOrganizer[TDataType] # Removed quotes
    ) -> None:
        """Callback from a `RemoteDataOrganizer` when it has new data.

        This method is part of the `RemoteDataOrganizer.Client` interface.
        It notifies the aggregator's client, if one is registered.

        Args:
            data_organizer: The `RemoteDataOrganizer` instance that has new data.
        """
        if self.__client is not None:
            # Notify the aggregator's client that data is available for the specific caller_id.
            self.__client._on_data_available(self, data_organizer.caller_id)

    def _on_data_ready(self, new_data: TDataType) -> None:
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
        if not isinstance(new_data, ExposedData): # Changed from issubclass to isinstance for instance check
            raise TypeError(f"Expected new_data to be an instance of ExposedData, but got {type(new_data).__name__}.")

        data_organizer: RemoteDataOrganizer[TDataType]
        is_new_organizer = False # Flag to track if a new organizer was created

        with self.__lock:
            # Check if an organizer already exists for this caller_id.
            if new_data.caller_id not in self.__organizers:
                # Create a new RemoteDataOrganizer for this caller_id.
                data_organizer = RemoteDataOrganizer(
                    self.__thread_pool, new_data.caller_id, self # Pass self as client to organizer
                )
                # If a timeout tracker is configured, register the new organizer.
                if self.__tracker is not None:
                    self.__tracker.register(data_organizer)
                data_organizer.start() # Start the new organizer.
                self.__organizers[new_data.caller_id] = data_organizer
                is_new_organizer = True # Mark that a new organizer was created.
            else:
                # Use the existing organizer.
                data_organizer = self.__organizers[new_data.caller_id]
        
        # Pass the new data to the responsible organizer.
        data_organizer._on_data_ready(new_data)

        # If a new organizer was created and a client is registered, notify the client.
        if is_new_organizer and self.__client is not None:
            self.__client._on_new_endpoint_began_transmitting(
                self, data_organizer.caller_id
            )
