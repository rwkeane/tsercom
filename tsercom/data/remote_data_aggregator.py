"""Defines the RemoteDataAggregator abstract base class.

An interface for aggregating and accessing data from remote sources.
"""

import datetime
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, overload

from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.data.exposed_data import ExposedData

DataTypeT = TypeVar("DataTypeT", bound=ExposedData)


class RemoteDataAggregator(ABC, Generic[DataTypeT]):
    """Abstract interface for aggregating and accessing remote data.

    This class defines a contract for services that provide a unified view of
    data received from multiple remote sources or callers. It handles the
    organization of data by `CallerIdentifier` and provides various methods
    to query this data, such as checking for new data, retrieving new or
    most recent data, and accessing data based on timestamps.

    It also defines a `Client` interface for receiving callbacks about data
    availability and new endpoint discovery.
    """

    class Client(ABC):
        """Interface for clients wishing to receive callbacks from a RemoteDataAggregator.

        Implementers of this interface can register with a `RemoteDataAggregator`
        to be notified about changes in data state or endpoint connectivity.
        """

        @abstractmethod
        def _on_data_available(
            self,
            aggregator: "RemoteDataAggregator[DataTypeT]",
            caller_id: CallerIdentifier,
        ) -> None:
            """Handle callback invoked when new data becomes available for a specific caller.

            Args:
                aggregator: The `RemoteDataAggregator` instance that detected
                            the new data.
                caller_id: The `CallerIdentifier` for which new data is available.

            """
            raise NotImplementedError()

        @abstractmethod
        def _on_new_endpoint_began_transmitting(
            self,
            aggregator: "RemoteDataAggregator[DataTypeT]",
            caller_id: CallerIdentifier,
        ) -> None:
            """Handle callback invoked when a new endpoint associated with a caller_id.

            Starts transmitting data.

            Args:
                aggregator: The `RemoteDataAggregator` instance reporting the event.
                caller_id: The `CallerIdentifier` of the newly discovered endpoint.

            """
            raise NotImplementedError()

        @abstractmethod
        def _on_new_endpoint_stopped_transmitting(
            self,
            aggregator: "RemoteDataAggregator[DataTypeT]",
            caller_id: CallerIdentifier,
        ) -> None:
            """Handle callback invoked when an endpoint associated with a caller_id.

            Stops transmitting data.

            Args:
                aggregator: The `RemoteDataAggregator` instance reporting the event.
                caller_id: The `CallerIdentifier` of the endpoint that stopped.

            """
            raise NotImplementedError()

    @overload
    def stop(self) -> None:
        """Stop data processing for all callers."""
        ...

    @overload
    def stop(self, identifier: CallerIdentifier) -> None:  # Renamed id to identifier
        """Stop data processing for a specific caller.

        Args:
            identifier: The caller for which to stop data processing.
        """
        ...

    @abstractmethod
    def stop(
        self, identifier: CallerIdentifier | None = None
    ) -> None:  # Renamed id to identifier
        """Stop data processing.

        If `identifier` is provided, stops processing for that specific caller.
        Otherwise, stops all data processing managed by this aggregator.
        Implementations should handle cleanup of resources associated with
        the stopped caller(s).
        """
        raise NotImplementedError()

    @overload
    def has_new_data(self) -> dict[CallerIdentifier, bool]:
        """Check for new data for all callers.

        Returns:
            A dictionary mapping each `CallerIdentifier` to a boolean indicating
            if new data is available.
        """
        ...

    @overload
    def has_new_data(
        self, identifier: CallerIdentifier
    ) -> bool:  # Renamed id to identifier
        """Check for new data for a specific caller.

        Args:
            identifier: The caller to check for new data.

        Returns:
            True if new data is available for the specified caller, False otherwise.
        """
        ...

    @abstractmethod
    def has_new_data(
        self,
        identifier: CallerIdentifier | None = None,  # Renamed id to identifier
    ) -> dict[CallerIdentifier, bool] | bool:
        """Check for new data.

        If `identifier` is provided, checks for the specific caller. Otherwise, checks
        for all callers. Refer to overloaded signatures for specific return types.
        """
        raise NotImplementedError()

    def any_new_data(self) -> bool:
        """Check if there is any new data available from any caller.

        This is a convenience method that iterates over the results of
        `has_new_data()` for all callers.

        Returns:
            True if at least one caller has new data, False otherwise.

        """
        all_data_status = self.has_new_data()
        # If the result is a dictionary (meaning no specific ID was passed to
        # has_new_data), check if any value in the dictionary is True.
        if isinstance(all_data_status, dict):
            return any(all_data_status.values())
        # If all_data_status is a bool (meaning a specific ID was passed),
        # this path should ideally not be taken by this method's logic,
        # as any_new_data() implies checking across all.
        # However, if has_new_data() was called with an ID before this,
        # and its result (bool) was somehow used to call this, we just return it.
        # This case suggests a potential misuse or misunderstanding of the API.
        # For robustness, we handle it, but typical usage implies all_data_status
        # will be a dict.
        return bool(all_data_status)

    @overload
    def get_new_data(self) -> dict[CallerIdentifier, list[DataTypeT]]:
        """Retrieve new data for all callers.

        Returns:
            A dictionary mapping `CallerIdentifier` to a list of new data items.
        """
        ...

    @overload
    def get_new_data(
        self, identifier: CallerIdentifier
    ) -> list[DataTypeT]:  # Renamed id to identifier
        """Retrieve new data for a specific caller.

        Args:
            identifier: The caller for which to retrieve new data.

        Returns:
            A list of new data items for the specified caller.
        """
        ...

    @abstractmethod
    def get_new_data(
        self,
        identifier: CallerIdentifier | None = None,  # Renamed id to identifier
    ) -> dict[CallerIdentifier, list[DataTypeT]] | list[DataTypeT]:
        """Retrieve new data.

        If `identifier` is provided, retrieves data for the specific caller.
        Otherwise, retrieves data for all callers. Refer to overloaded
        signatures for specific return types.
        """
        raise NotImplementedError()

    @overload
    def get_most_recent_data(self) -> dict[CallerIdentifier, DataTypeT | None]:
        """Retrieve the most recent data for all callers.

        Returns:
            A dictionary mapping `CallerIdentifier` to its most recent data item
            (or None if unavailable).
        """
        ...

    @overload
    def get_most_recent_data(
        self, identifier: CallerIdentifier
    ) -> DataTypeT | None:  # Renamed id to identifier
        """Retrieve the most recent data for a specific caller.

        Args:
            identifier: The caller for which to retrieve the most recent data.

        Returns:
            The most recent data item for the specified caller, or None if
            unavailable.
        """
        ...

    @abstractmethod
    def get_most_recent_data(
        self,
        identifier: CallerIdentifier | None = None,  # Renamed id to identifier
    ) -> dict[CallerIdentifier, DataTypeT | None] | DataTypeT | None:
        """Retrieve the most recent data.

        If `identifier` is provided, retrieves data for the specific caller.
        Otherwise, retrieves data for all callers. Refer to overloaded
        signatures for specific return types. Returns `None` if data is
        unavailable or timed out.
        """
        raise NotImplementedError()

    @overload
    def get_data_for_timestamp(
        self, timestamp: datetime.datetime
    ) -> dict[CallerIdentifier, DataTypeT | None]:
        """Retrieve data for a specific timestamp for all callers.

        Args:
            timestamp: The timestamp to retrieve data for.

        Returns:
            A dictionary mapping `CallerIdentifier` to its data item (or None)
            at the specified timestamp.
        """
        ...

    @overload
    def get_data_for_timestamp(
        self,
        timestamp: datetime.datetime,
        identifier: CallerIdentifier,  # Renamed id to identifier
    ) -> DataTypeT | None:
        """Retrieve data for a specific timestamp for a specific caller.

        Args:
            timestamp: The timestamp to retrieve data for.
            identifier: The caller for which to retrieve data.

        Returns:
            The data item for the specified caller at the timestamp, or None.
        """
        ...

    @abstractmethod
    def get_data_for_timestamp(
        self,
        timestamp: datetime.datetime,
        identifier: CallerIdentifier | None = None,  # Renamed id to identifier
    ) -> dict[CallerIdentifier, DataTypeT | None] | DataTypeT | None:
        """Retrieve data for a specific timestamp.

        If `identifier` is provided, retrieves data for the specific caller.
        Otherwise, retrieves data for all callers. Refer to overloaded
        signatures for specific return types. Returns `None` if data is
        unavailable or timed out.
        """
        raise NotImplementedError()

    @overload
    def get_interpolated_at(
        self, timestamp: datetime.datetime
    ) -> dict[CallerIdentifier, DataTypeT]:
        """Interpolate data at a specific time for all callers.

        Args:
            timestamp: The time for which to estimate data.

        Returns:
            A dictionary mapping `CallerIdentifier` to its interpolated data.
            Callers for whom interpolation fails are omitted.
        """
        ...

    @overload
    def get_interpolated_at(
        self, timestamp: datetime.datetime, identifier: CallerIdentifier
    ) -> DataTypeT | None:
        """Interpolate data at a specific time for a specific caller.

        Args:
            timestamp: The time for which to estimate data.
            identifier: The caller for which to perform interpolation.

        Returns:
            Interpolated data for the caller, or None if interpolation fails.
        """
        ...

    @abstractmethod
    def get_interpolated_at(
        self, timestamp: datetime.datetime, identifier: CallerIdentifier | None = None
    ) -> DataTypeT | None | dict[CallerIdentifier, DataTypeT]:
        """Perform linear interpolation to estimate data at a specific time.

        This method estimates the data value at the given `timestamp` by
        linearly interpolating between the two closest data points available
        in the internal buffer.

        - If `identifier` is specified, interpolation is performed for that
          single caller. Returns `DataTypeT` or `None`.
        - If `identifier` is `None`, interpolation is performed for all known
          callers. Returns a `Dict[CallerIdentifier, DataTypeT]`, omitting
          callers for whom interpolation was not successful.

        If the `timestamp` matches an existing data point exactly for a given
        `identifier`, that data point is returned (or included in the dictionary).

        If the `timestamp` is outside the range of available data, or if
        insufficient data points exist for interpolation for an `identifier`,
        `None` is returned (if `identifier` was specified), or the caller is
        omitted from the dictionary (if `identifier` was `None`).

        Args:
            timestamp: The specific time (as a `datetime.datetime` object)
                for which to estimate the data.
            identifier: Optional. The `CallerIdentifier` for which to perform
                interpolation. If `None`, interpolation is attempted for all callers.

        Returns:
            If `identifier` is provided: An instance of `DataTypeT` representing
            the interpolated data, or `None`.
            If `identifier` is `None`: A dictionary mapping each `CallerIdentifier`
            to its successfully interpolated data (`DataTypeT`). Callers for whom
            interpolation failed are omitted.

        """
        raise NotImplementedError()
