"""Thread-safe map for CallerIdentifier to generic object association."""

import threading
from typing import Callable, Dict, Generic, TypeVar

from tsercom.caller_id.caller_identifier import CallerIdentifier

MapValueT = TypeVar("MapValueT")


class CallerIdMap(Generic[MapValueT]):
    """A thread-safe map associating CallerIdentifiers with generic objects.

    Keys are CallerIdentifier objects, values are instances of MapValueT.
    Access is synchronized using a threading.Lock.
    """

    def __init__(self) -> None:
        """Initializes a new CallerIdMap instance."""
        self.__lock = threading.Lock()
        self.__map: Dict[CallerIdentifier, MapValueT] = {}

    def find_instance(
        self, caller_id: CallerIdentifier, factory: Callable[[], MapValueT]
    ) -> MapValueT:
        """Finds an instance associated with a caller_id, creating it if necessary.

        If an instance for the given `caller_id` already exists, it's returned.
        Otherwise, the `factory` function is called to create a new instance,
        which is stored and returned. This operation is thread-safe.
        """
        with self.__lock:
            if caller_id not in self.__map:
                self.__map[caller_id] = factory()
            return self.__map[caller_id]

    def for_all_items(self, function: Callable[[MapValueT], None]) -> None:
        """Executes a function for all items currently in the map.

        This method iterates over a snapshot of the map's values. Modifications
        to the map during iteration won't affect processed items. Thread-safe.
        """
        # Create a snapshot of values to iterate over.
        # This avoids issues with map modification during iteration.
        items = []
        with self.__lock:
            items = list(self.__map.values())

        for val in items:
            function(val)

    def count(self) -> int:
        """Returns the number of items in the map.

        This operation is thread-safe.
        """
        with self.__lock:
            return len(self.__map)

    def __len__(self) -> int:
        """Returns item count, making `len()` usable. Thread-safe."""
        return self.count()
