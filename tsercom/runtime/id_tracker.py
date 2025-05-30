"""Provides IdTracker for managing bidirectional mappings between CallerIdentifiers and network addresses."""

import threading
from typing import Dict, Optional, overload, Iterator

from tsercom.caller_id.caller_identifier import CallerIdentifier


class IdTracker:
    """
    Helper object for creating and managing a thread-safe bidirectional
    dictionary, to map between the local CallerID and the address / port.
    """

    def __init__(self) -> None:
        """Initializes the IdTracker with internal lock and dictionaries."""
        self.__lock = threading.Lock()
        self.__address_to_id: Dict[tuple[str, int], CallerIdentifier] = {}
        self.__id_to_address: Dict[CallerIdentifier, tuple[str, int]] = {}

    @overload
    def try_get(self, id: CallerIdentifier) -> Optional[tuple[str, int]]:
        pass

    @overload
    def try_get(self, address: str, port: int) -> Optional[CallerIdentifier]:
        pass

    def try_get(
        self,
        id: Optional[CallerIdentifier] = None,
        address: Optional[str] = None,
        port: Optional[int] = None,
    ) -> Optional[tuple[str, int] | CallerIdentifier]:
        """Attempts to retrieve a value from the tracker.

        Can be called either with a `CallerIdentifier` (to get address/port)
        or with an address and port (to get `CallerIdentifier`).

        Args:
            id: The `CallerIdentifier` to look up.
            address: The IP address or hostname to look up.
            port: The port number to look up (required if address is provided).

        Returns:
            A tuple (address, port) if looking up by ID and found.
            A `CallerIdentifier` if looking up by address/port and found.
            `None` if the lookup key is not found.

        Raises:
            ValueError: If incorrect arguments are provided (e.g., both id and address,
                        or address without port).
        """
        if (id is None) == (address is None):
            raise ValueError(
                f"Exactly one of 'id' or 'address' must be provided to try_get. Got id={id}, address={address}."
            )
        if (address is None) != (port is None):
            raise ValueError(
                f"If 'address' is provided, 'port' must also be provided, and vice-versa. Got address={address}, port={port}."
            )

        with self.__lock:
            if id is not None:
                return self.__id_to_address.get(id)
            elif (
                address is not None and port is not None
            ):  # Ensure port is not None for type safety
                return self.__address_to_id.get((address, port))
        return None  # Should not be reached given the initial checks, but as a fallback

    @overload
    def get(self, id: CallerIdentifier) -> tuple[str, int]:
        pass

    @overload
    def get(self, address: str, port: int) -> CallerIdentifier:
        pass

    def get(
        self,
        id: Optional[CallerIdentifier] = None,
        address: Optional[str] = None,
        port: Optional[int] = None,
    ) -> tuple[str, int] | CallerIdentifier:
        """Retrieves a value from the tracker, raising KeyError if not found.

        Can be called either with a `CallerIdentifier` (to get address/port)
        or with an address and port (to get `CallerIdentifier`).

        Args:
            id: The `CallerIdentifier` to look up.
            address: The IP address or hostname to look up.
            port: The port number to look up (required if address is provided).

        Returns:
            A tuple (address, port) if looking up by ID.
            A `CallerIdentifier` if looking up by address/port.

        Raises:
            ValueError: If incorrect arguments are provided.
            KeyError: If the lookup key is not found.
        """
        result = self.try_get(
            id=id, address=address, port=port
        )  # Use named args for clarity
        if result is None:
            raise KeyError(
                f"Key not found for query: id={id}, address={address}, port={port}"
            )

        return result  # Type of result is correctly inferred by mypy here based on overload

    def add(self, id: CallerIdentifier, address: str, port: int) -> None:
        """Adds a new bidirectional mapping to the tracker.

        Args:
            id: The `CallerIdentifier`.
            address: The IP address or hostname.
            port: The port number.

        Raises:
            KeyError: If the ID or the address/port combination already exists.
        """
        # --- New Prints ---
        print(
            f"IdTracker.add: Called with caller_id={id}, endpoint=({address}, {port})",
            flush=True,
        )
        print(
            f"IdTracker.add: Current self.__id_to_address before add: {self.__id_to_address}",
            flush=True,
        )
        print(
            f"IdTracker.add: Current self.__address_to_id before add: {self.__address_to_id}",
            flush=True,
        )
        # --- End New Prints ---
        with self.__lock:
            if id in self.__id_to_address:
                print(
                    f"IdTracker.add: Raising KeyError because caller_id {id} is already in self.__id_to_address.",
                    flush=True,
                )
                raise KeyError(f"ID {id} already exists in tracker.")
            if (address, port) in self.__address_to_id:
                print(
                    f"IdTracker.add: Raising KeyError because key ({address}, {port}) is already in self.__address_to_id.",
                    flush=True,
                )
                raise KeyError(
                    f"Address ({address}:{port}) already exists in tracker."
                )

            self.__address_to_id[(address, port)] = id
            self.__id_to_address[id] = (address, port)
            print(
                f"IdTracker.add: Successfully added caller_id={id}, endpoint=({address}, {port})",
                flush=True,
            )

    def has_id(self, id: CallerIdentifier) -> bool:
        """Checks if the given `CallerIdentifier` exists in the tracker.

        Args:
            id: The `CallerIdentifier` to check.

        Returns:
            True if the ID exists, False otherwise.
        """
        with self.__lock:
            return id in self.__id_to_address

    def has_address(self, address: str, port: int) -> bool:
        """Checks if the given address and port combination exists in the tracker.

        Args:
            address: The IP address or hostname.
            port: The port number.

        Returns:
            True if the address/port combination exists, False otherwise.
        """
        with self.__lock:
            return (address, port) in self.__address_to_id

    def __len__(self) -> int:
        """Returns the number of mappings in the tracker."""
        with self.__lock:
            # Assertion ensures internal consistency.
            assert len(self.__id_to_address) == len(self.__address_to_id)
            return len(self.__id_to_address)

    def __iter__(self) -> Iterator[CallerIdentifier]:
        """Returns an iterator over the `CallerIdentifier`s in the tracker.

        Note: This iterates over a copy of the keys if modification during
        iteration is a concern, or directly if thread-safety is ensured by the lock.
        Current implementation iterates directly over the dictionary's iterator,
        which is safe due to the lock in typical use cases but might behave
        unexpectedly if the lock is released and reacquired by another thread
        modifying the dictionary *during* the iteration of a single __next__ call.
        However, for simple iteration loops, the lock protects the whole loop.
        """
        with self.__lock:
            # But for most common loops, this direct iteration is fine under the lock.
            return iter(self.__id_to_address)

    def remove(self, id: CallerIdentifier) -> bool:
        """
        Removes the given CallerIdentifier and its associated address/port
        from the tracker.
        Returns True if the id was found and removed, False otherwise.
        """
        with self.__lock:
            if id not in self.__id_to_address:
                return False

            address_port_tuple = self.__id_to_address[id]
            del self.__id_to_address[id]
            if address_port_tuple in self.__address_to_id:
                del self.__address_to_id[address_port_tuple]
            # It's possible that the address_port_tuple is not in __address_to_id
            # if there was some inconsistency, but we prioritize removing the id.
            # Consider logging a warning here if such a state is unexpected.
            return True
