"""Defines the RemoteDataAggregator abstract base class, an interface for aggregating and accessing data from remote sources."""

from abc import ABC, abstractmethod
import datetime
from typing import Dict, Generic, List, Optional, TypeVar, overload

from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.data.exposed_data import ExposedData

# Generic type for the data being aggregated, bound by ExposedData.
TDataType = TypeVar("TDataType", bound=ExposedData)


class RemoteDataAggregator(ABC, Generic[TDataType]):
    """Abstract interface for aggregating and accessing remote data.

    This class defines a contract for services that provide a unified view of
    data received from multiple remote sources or callers. It handles the
    organization of data by `CallerIdentifier` and provides various methods
    to query this data, such as checking for new data, retrieving new or
    most recent data, and accessing data based on timestamps.

    It also defines a `Client` interface for receiving callbacks about data
    availability and new endpoint discovery.
    """

    class Client(ABC, Generic[TDataType]):
        """Interface for clients wishing to receive callbacks from a RemoteDataAggregator.

        Implementers of this interface can register with a `RemoteDataAggregator`
        to be notified about changes in data state or endpoint connectivity.
        """

        @abstractmethod
        def _on_data_available(
            self,
            aggregator: "RemoteDataAggregator[TDataType]",
            caller_id: CallerIdentifier,
        ) -> None:
            """Callback invoked when new data becomes available for a specific caller.

            Args:
                aggregator: The `RemoteDataAggregator` instance that detected
                            the new data.
                caller_id: The `CallerIdentifier` for which new data is available.
            """
            pass

        @abstractmethod
        def _on_new_endpoint_began_transmitting(
            self,
            aggregator: "RemoteDataAggregator[TDataType]",
            caller_id: CallerIdentifier,
        ) -> None:
            """Callback invoked when a new endpoint associated with a caller_id starts transmitting data.

            Args:
                aggregator: The `RemoteDataAggregator` instance reporting the event.
                caller_id: The `CallerIdentifier` of the newly discovered endpoint.
            """
            pass

    @overload
    def stop(self) -> None:
        """Stops all data organizers and data processing for all callers."""
        ...

    @overload
    def stop(self, id: CallerIdentifier) -> None:
        """Stops the data organizer and data processing for a specific caller.

        Args:
            id: The `CallerIdentifier` for which to stop data processing.
        """
        ...

    @abstractmethod
    def stop(self, id: Optional[CallerIdentifier] = None) -> None:
        """Stops data processing.

        If `id` is provided, stops processing for that specific caller.
        Otherwise, stops all data processing managed by this aggregator.
        Implementations should handle cleanup of resources associated with
        the stopped caller(s).
        """
        pass

    @overload
    def has_new_data(self) -> Dict[CallerIdentifier, bool]:
        """Checks for new data for all callers.

        Returns:
            A dictionary mapping each `CallerIdentifier` to a boolean indicating
            if new data is available for that caller.
        """
        ...

    @overload
    def has_new_data(self, id: CallerIdentifier) -> bool:
        """Checks if new data is available for a specific caller.

        Args:
            id: The `CallerIdentifier` to check for new data.

        Returns:
            True if new data is available for the specified caller, False otherwise.
        """
        ...

    @abstractmethod
    def has_new_data(
        self, id: Optional[CallerIdentifier] = None
    ) -> Dict[CallerIdentifier, bool] | bool:
        """Checks for new data.

        If `id` is provided, checks for the specific caller. Otherwise, checks
        for all callers. Refer to overloaded signatures for specific return types.
        """
        pass

    def any_new_data(self) -> bool:
        """Checks if there is any new data available from any caller.

        This is a convenience method that iterates over the results of
        `has_new_data()` for all callers.

        Returns:
            True if at least one caller has new data, False otherwise.
        """
        all_data_status = self.has_new_data()
        # If the result is a dictionary (meaning no specific ID was passed to has_new_data),
        # check if any value in the dictionary is True.
        if isinstance(all_data_status, dict):
            return any(all_data_status.values())
        # If all_data_status is a bool (meaning a specific ID was passed),
        # this path should ideally not be taken by this method's logic,
        # as any_new_data() implies checking across all.
        # However, if has_new_data() was called with an ID before this,
        # and its result (bool) was somehow used to call this, we just return it.
        # This case suggests a potential misuse or misunderstanding of the API.
        # For robustness, we handle it, but typical usage implies all_data_status will be a dict.
        return bool(all_data_status)

    @overload
    def get_new_data(self) -> Dict[CallerIdentifier, List[TDataType]]:
        """Retrieves all new data items for all callers.

        Returns:
            A dictionary mapping each `CallerIdentifier` to a list of new
            data items (`TDataType`) received from that caller.
        """
        ...

    @overload
    def get_new_data(self, id: CallerIdentifier) -> List[TDataType]:
        """Retrieves all new data items for a specific caller.

        Args:
            id: The `CallerIdentifier` for which to retrieve new data.

        Returns:
            A list of new data items (`TDataType`) received from the specified caller.
        """
        ...

    @abstractmethod
    def get_new_data(
        self, id: Optional[CallerIdentifier] = None
    ) -> Dict[CallerIdentifier, List[TDataType]] | List[TDataType]:
        """Retrieves new data.

        If `id` is provided, retrieves data for the specific caller.
        Otherwise, retrieves data for all callers. Refer to overloaded
        signatures for specific return types.
        """
        pass

    @overload
    def get_most_recent_data(self) -> Dict[CallerIdentifier, TDataType | None]:
        """Retrieves the most recently received data item for all callers.

        Returns `None` for a caller if no data has been received or if it has timed out.

        Returns:
            A dictionary mapping each `CallerIdentifier` to its most recent
            data item (`TDataType`) or `None`.
        """
        ...

    @overload
    def get_most_recent_data(self, id: CallerIdentifier) -> TDataType | None:
        """Retrieves the most recently received data item for a specific caller.

        Returns `None` if no data has been received for this caller or if it has timed out.

        Args:
            id: The `CallerIdentifier` for which to retrieve the most recent data.

        Returns:
            The most recent data item (`TDataType`) or `None`.
        """
        ...

    @abstractmethod
    def get_most_recent_data(
        self, id: Optional[CallerIdentifier] = None
    ) -> Dict[CallerIdentifier, TDataType | None] | TDataType | None:
        """Retrieves the most recent data.

        If `id` is provided, retrieves data for the specific caller.
        Otherwise, retrieves data for all callers. Refer to overloaded
        signatures for specific return types. Returns `None` if data is
        unavailable or timed out.
        """
        pass

    @overload
    def get_data_for_timestamp(
        self, timestamp: datetime.datetime
    ) -> Dict[CallerIdentifier, TDataType | None]:
        """Retrieves the most recent data item received before or at a specific timestamp for all callers.

        Returns `None` for a caller if no suitable data exists or if it has timed out.

        Args:
            timestamp: The `datetime` object to compare against data timestamps.

        Returns:
            A dictionary mapping each `CallerIdentifier` to the relevant data
            item (`TDataType`) or `None`.
        """
        ...

    @overload
    def get_data_for_timestamp(
        self, timestamp: datetime.datetime, id: CallerIdentifier
    ) -> TDataType | None:
        """Retrieves the most recent data item received before or at a specific timestamp for a specific caller.

        Returns `None` if no suitable data exists for this caller or if it has timed out.

        Args:
            timestamp: The `datetime` object to compare against data timestamps.
            id: The `CallerIdentifier` for which to retrieve the data.

        Returns:
            The relevant data item (`TDataType`) or `None`.
        """
        ...

    @abstractmethod
    def get_data_for_timestamp(
        self,
        timestamp: datetime.datetime,
        id: CallerIdentifier | None = None,
    ) -> Dict[CallerIdentifier, TDataType | None] | TDataType | None:
        """Retrieves data for a specific timestamp.

        If `id` is provided, retrieves data for the specific caller.
        Otherwise, retrieves data for all callers. Refer to overloaded
        signatures for specific return types. Returns `None` if data is
        unavailable or timed out.
        """
        pass
