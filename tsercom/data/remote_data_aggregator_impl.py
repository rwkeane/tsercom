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

TDataType = TypeVar("TDataType", bound=ExposedData)


class RemoteDataAggregatorImpl(
    Generic[TDataType],
    RemoteDataAggregator[TDataType],
    RemoteDataOrganizer.Client,
    RemoteDataReader[TDataType],
):
    """
    Main implementation of the RemoteDataAggregator. This instance is separate
    from the interface to limit what is shown to a user of the class.
    """

    @overload
    def __init__(
        self,
        thread_pool: ThreadPoolExecutor,
        client: Optional[RemoteDataAggregator.Client] = None,
    ):
        ...

    @overload
    def __init__(
        self,
        thread_pool: ThreadPoolExecutor,
        client: Optional[RemoteDataAggregator.Client] = None,
        *,
        tracker: DataTimeoutTracker,
    ):
        ...
        
    @overload
    def __init__(
        self,
        thread_pool: ThreadPoolExecutor,
        client: Optional[RemoteDataAggregator.Client] = None,
        *,
        timeout: int,
    ):
        ...
        
    def __init__(
        self,
        thread_pool: ThreadPoolExecutor,
        client: Optional[RemoteDataAggregator.Client] = None,
        *,
        tracker: Optional[DataTimeoutTracker] = None,
        timeout: Optional[int] = None,
    ):
        assert not timeout or not tracker
        if tracker is None and timeout is not None:
            tracker = DataTimeoutTracker(timeout)
            tracker.start()

        self.__thread_pool = thread_pool
        self.__client = client
        self.__tracker = tracker

        # TODO: Can this be a CallerIdMap?
        self.__organizers: Dict[
            CallerIdentifier, RemoteDataOrganizer[TDataType]
        ] = {}
        self.__lock = threading.Lock()

    def stop(self, id: Optional[CallerIdentifier] = None) -> None:
        with self.__lock:
            if id is not None:
                assert id in self.__organizers
                self.__organizers[id].stop()
                return

            for key, val in self.__organizers.items():
                val.stop()

    def has_new_data(  # type: ignore
        self, id: Optional[CallerIdentifier] = None
    ) -> Dict[CallerIdentifier, bool] | bool:
        with self.__lock:
            if id is not None:
                assert id in self.__organizers
                return self.__organizers[id].has_new_data()

            results = {}
            for key, val in self.__organizers.items():
                results[key] = val.has_new_data()
            return results

    def get_new_data(  # type: ignore
        self, id: Optional[CallerIdentifier] = None
    ) -> Dict[CallerIdentifier, List[TDataType]] | List[TDataType]:
        with self.__lock:
            if id is not None:
                assert id in self.__organizers
                return self.__organizers[id].get_new_data()

            results = {}
            for key, val in self.__organizers.items():
                results[key] = val.get_new_data()
            return results

    def get_most_recent_data(  # type: ignore
        self, id: Optional[CallerIdentifier] = None
    ) -> Dict[CallerIdentifier, TDataType | None] | TDataType | None:
        with self.__lock:
            if id is not None:
                assert id in self.__organizers
                return self.__organizers[id].get_most_recent_data()

            results = {}
            for key, val in self.__organizers.items():
                results[key] = val.get_most_recent_data()
            return results

    def get_data_for_timestamp(  # type: ignore
        self,
        id: CallerIdentifier,
        timestamp: datetime.datetime,
    ) -> Dict[CallerIdentifier, TDataType | None] | TDataType | None:
        with self.__lock:
            if id is not None:
                assert id in self.__organizers
                return self.__organizers[id].get_data_for_timestamp(timestamp)

            results = {}
            for key, val in self.__organizers.items():
                results[key] = val.get_data_for_timestamp(id, timestamp)
            return results

    def _on_data_available(  # type: ignore
        self, data_organizer: "RemoteDataOrganizer[TDataType]"
    ) -> None:
        if self.__client is not None:
            self.__client._on_data_available(self, data_organizer.caller_id)

    def _on_data_ready(self, new_data: TDataType) -> None:
        assert issubclass(type(new_data), ExposedData), type(new_data)

        # Find or create the RemoteDataOrganizer.
        found = False
        data_organizer: RemoteDataOrganizer = None  # type: ignore
        with self.__lock:
            found = new_data.caller_id in self.__organizers
            if not found:
                data_organizer = RemoteDataOrganizer(
                    self.__thread_pool, new_data.caller_id, self
                )
                if self.__tracker is not None:
                    self.__tracker.register(data_organizer)
                data_organizer.start()
                self.__organizers[new_data.caller_id] = data_organizer
            else:
                data_organizer = self.__organizers[new_data.caller_id]

        # Call into it.
        assert data_organizer is not None
        data_organizer._on_data_ready(new_data)

        # Inform the client if needed
        if not found and self.__client is not None:
            self.__client._on_new_endpoint_began_transmitting(
                self, data_organizer.caller_id
            )
