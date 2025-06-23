"""Defines RemoteDataAggregator ABC for aggregating data from remote sources."""

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
        """Interface for clients wishing to receive callbacks from the aggregator.

        Implementers of this interface can register with a `RemoteDataAggregator`
        to be notified about changes in data state or endpoint connectivity.
        """

        @abstractmethod
        def _on_data_available(
            self,
            aggregator: "RemoteDataAggregator[DataTypeT]",
            caller_id: CallerIdentifier,
        ) -> None:
            """Invoke callback when new data becomes available for a specific caller.

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
            """Invoke callback when a new endpoint starts transmitting data.

            Args:
                aggregator: The `RemoteDataAggregator` instance reporting the event.
                caller_id: The `CallerIdentifier` of the newly discovered endpoint.

            """
            raise NotImplementedError()

    @overload
    def stop(self) -> None:
        """Stop all data organizers and data processing for all callers."""
        raise NotImplementedError()  # Replaced ...

    @overload
    def stop(self, identifier: CallerIdentifier) -> None:  # Renamed id to identifier
        """Stop the data organizer and data processing for a specific caller.

        Args:
            identifier: The `CallerIdentifier` for which to stop data processing.

        """
        raise NotImplementedError()  # Replaced ...

    @abstractmethod
    def stop(
        self, identifier: CallerIdentifier | None = None
    ) -> None:  # Renamed id to identifier
        """Stop data processing. See overloads for details."""
        # Main impl docstring is minimal per prompt (overloads documented).
        raise NotImplementedError()

    @overload
    def has_new_data(self) -> dict[CallerIdentifier, bool]:
        """Check for new data for all callers.

        Returns:
            A dictionary mapping each `CallerIdentifier` to a boolean indicating
            if new data is available for that caller.

        """
        raise NotImplementedError()  # Replaced ...

    @overload
    def has_new_data(
        self, identifier: CallerIdentifier
    ) -> bool:  # Renamed id to identifier
        """Check if new data is available for a specific caller.

        Args:
            identifier: The `CallerIdentifier` to check for new data.

        Returns:
            True if new data is available for the specified caller, False otherwise.

        """
        raise NotImplementedError()  # Replaced ...

    @abstractmethod
    def has_new_data(
        self,
        identifier: CallerIdentifier | None = None,  # Renamed id to identifier
    ) -> dict[CallerIdentifier, bool] | bool:
        """Check for new data. See overloads for details."""
        # Main impl docstring is minimal per prompt (overloads documented).
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
        """Retrieve all new data items for all callers.

        Returns:
            A dictionary mapping each `CallerIdentifier` to a list of new
            data items (`DataTypeT`) received from that caller.

        """
        raise NotImplementedError()  # Replaced ...

    @overload
    def get_new_data(
        self, identifier: CallerIdentifier
    ) -> list[DataTypeT]:  # Renamed id to identifier
        """Retrieve all new data items for a specific caller.

        Args:
            identifier: The `CallerIdentifier` for which to retrieve new data.

        Returns:
            A list of new data items (`DataTypeT`) received from the specified caller.

        """
        raise NotImplementedError()  # Replaced ...

    @abstractmethod
    def get_new_data(
        self,
        identifier: CallerIdentifier | None = None,  # Renamed id to identifier
    ) -> dict[CallerIdentifier, list[DataTypeT]] | list[DataTypeT]:
        """Retrieve new data. See overloads for details."""
        # Main impl docstring is minimal per prompt (overloads documented).
        raise NotImplementedError()

    @overload
    def get_most_recent_data(self) -> dict[CallerIdentifier, DataTypeT | None]:
        """Retrieve the most recently received data item for all callers.

        Returns `None` for a caller if no data has been received or if it has timed out.

        Returns:
            A dictionary mapping each `CallerIdentifier` to its most recent
            data item (`DataTypeT`) or `None`.

        """
        raise NotImplementedError()  # Replaced ...

    @overload
    def get_most_recent_data(
        self, identifier: CallerIdentifier
    ) -> DataTypeT | None:  # Renamed id to identifier
        """Retrieve the most recently received data item for a specific caller.

        Returns `None` if no data has been received for this caller or if it has
        timed out.

        Args:
            identifier: The `CallerIdentifier` for which to retrieve the most
                recent data.

        Returns:
            The most recent data item (`DataTypeT`) or `None`.

        """
        raise NotImplementedError()  # Replaced ...

    @abstractmethod
    def get_most_recent_data(
        self,
        identifier: CallerIdentifier | None = None,  # Renamed id to identifier
    ) -> dict[CallerIdentifier, DataTypeT | None] | DataTypeT | None:
        """Retrieve the most recent data. See overloads for details."""
        # Main impl docstring is minimal per prompt (overloads documented).
        raise NotImplementedError()

    @overload
    def get_data_for_timestamp(
        self, timestamp: datetime.datetime
    ) -> dict[CallerIdentifier, DataTypeT | None]:
        """Get data at or before a timestamp for all callers.

        Returns `None` for a caller if no suitable data exists or if it has
        timed out.

        Args:
            timestamp: The `datetime` object to compare against data timestamps.

        Returns:
            A dictionary mapping each `CallerIdentifier` to the relevant data
            item (`DataTypeT`) or `None`.

        """
        raise NotImplementedError()  # Replaced ...

    @overload
    def get_data_for_timestamp(
        self,
        timestamp: datetime.datetime,
        identifier: CallerIdentifier,  # Renamed id to identifier
    ) -> DataTypeT | None:
        """Get data at or before a timestamp for a specific caller.

        Returns `None` if no suitable data exists for this caller or if it has
        timed out.

        Args:
            timestamp: The `datetime` object to compare against data timestamps.
            identifier: The `CallerIdentifier` for which to retrieve the data.

        Returns:
            The relevant data item (`DataTypeT`) or `None`.

        """
        raise NotImplementedError()  # Replaced ...

    @abstractmethod
    def get_data_for_timestamp(
        self,
        timestamp: datetime.datetime,
        identifier: CallerIdentifier | None = None,  # Renamed id to identifier
    ) -> dict[CallerIdentifier, DataTypeT | None] | DataTypeT | None:
        """Retrieve data for a specific timestamp. See overloads for details."""
        # Main impl docstring is minimal per prompt (overloads documented).
        raise NotImplementedError()

    @overload
    def get_interpolated_at(
        self, timestamp: datetime.datetime
    ) -> dict[CallerIdentifier, DataTypeT]:
        """Perform interpolation for a single timestamp across all callers.

        Returns a dictionary mapping each `CallerIdentifier` to its successfully
        interpolated data (`DataTypeT`). Callers for which interpolation is
        not possible (e.g., out of bounds, insufficient data) are omitted
        from the dictionary.

        Args:
            timestamp: The specific time for which to estimate data.

        Returns:
            A dictionary where keys are `CallerIdentifier` and values are
            the successfully interpolated data instances (`DataTypeT`).

        """
        ...

    @overload
    def get_interpolated_at(
        self, timestamp: datetime.datetime, identifier: CallerIdentifier
    ) -> DataTypeT | None:
        """Perform interpolation for a specific caller at a single timestamp.

        Estimates data for the given `identifier` at the specified `timestamp`
        using linear interpolation.

        Args:
            timestamp: The specific time for which to estimate data.
            identifier: The `CallerIdentifier` for which to interpolate.

        Returns:
            The interpolated data instance (`DataTypeT`) for the specified
            caller, or `None` if interpolation is not possible (e.g.,
            out of bounds, insufficient data).

        """
        ...

    @abstractmethod
    def get_interpolated_at(
        self, timestamp: datetime.datetime, identifier: CallerIdentifier | None = None
    ) -> DataTypeT | None | dict[CallerIdentifier, DataTypeT]:
        """Perform linear interpolation. See overloads for details."""
        # Main impl docstring is minimal per prompt (overloads documented).
        raise NotImplementedError()
