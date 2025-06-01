"""Provides IdTracker for managing bidirectional mappings between CallerIdentifiers and network addresses."""

import threading
from typing import Dict, Optional, overload, Iterator, Any, cast

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
        self, *args: Any, **kwargs: Any
    ) -> Optional[tuple[str, int] | CallerIdentifier]:
        """Attempts to retrieve a value from the tracker.

        Can be called either with a `CallerIdentifier` (to get address/port)
        or with an address and port (to get `CallerIdentifier`).

        Args (interpreted based on overloads):
            id: The `CallerIdentifier` to look up (via keyword or single positional arg).
            address: The IP address or hostname to look up (via keyword or first of two positional args).
            port: The port number to look up (via keyword or second of two positional args).

        Returns:
            A tuple (address, port) if looking up by ID and found.
            A `CallerIdentifier` if looking up by address/port and found.
            `None` if the lookup key is not found.

        Raises:
            ValueError: If incorrect arguments are provided based on the overloads.
        """
        _id: Optional[CallerIdentifier] = None
        _address: Optional[str] = None
        _port: Optional[int] = None

        if args:
            if len(args) == 1 and isinstance(args[0], CallerIdentifier):
                _id = args[0]
            elif (
                len(args) == 2
                and isinstance(args[0], str)
                and isinstance(args[1], int)
            ):
                _address = args[0]
                _port = args[1]
            elif len(args) > 0:  # Incorrect positional arguments
                raise ValueError(
                    f"Invalid positional arguments. Provide a CallerIdentifier OR (address, port). Got: {args}"
                )

        # Process kwargs, potentially overriding positional if they were also passed (though that's unusual)
        # or providing them if no positional args were given.
        if "id" in kwargs:
            if (
                _id is not None or _address is not None or _port is not None
            ):  # id kwarg with other args
                raise ValueError(
                    "Cannot mix 'id' keyword argument with positional arguments or other keyword arguments for address/port."
                )
            kw_id = kwargs.pop("id")
            if not isinstance(kw_id, CallerIdentifier):
                raise ValueError(
                    f"'id' keyword argument must be a CallerIdentifier. Got {type(kw_id)}"
                )
            _id = kw_id

        if "address" in kwargs or "port" in kwargs:
            if _id is not None:  # address/port kwargs with id arg
                raise ValueError(
                    "Cannot mix 'address'/'port' keyword arguments with 'id' argument."
                )
            if "address" in kwargs:
                kw_address = kwargs.pop("address")
                if not isinstance(kw_address, str):
                    raise ValueError(
                        f"'address' keyword argument must be a string. Got {type(kw_address)}"
                    )
                _address = kw_address
            if "port" in kwargs:
                kw_port = kwargs.pop("port")
                if not isinstance(kw_port, int):
                    raise ValueError(
                        f"'port' keyword argument must be an int. Got {type(kw_port)}"
                    )
                _port = kw_port

        if kwargs:  # Any remaining kwargs are unexpected
            raise ValueError(f"Unexpected keyword arguments: {kwargs.keys()}")

        # Validation based on processed arguments
        if (_id is None) == (
            _address is None and _port is None
        ):  # XOR logic: one group must be present
            raise ValueError(
                "Exactly one of (CallerIdentifier) or (address and port) must be provided."
            )
        if (_address is None) != (
            _port is None
        ):  # If one of address/port is provided, the other must be too
            raise ValueError(
                "If 'address' is provided, 'port' must also be provided, and vice-versa."
            )

        with self.__lock:
            if _id is not None:
                return self.__id_to_address.get(_id)
            # By this point, if _id is None, then _address and _port must be not None due to validation
            elif _address is not None and _port is not None:
                return self.__address_to_id.get((_address, _port))
        return None  # Should be unreachable due to validation ensuring one path is taken

    @overload
    def get(self, id: CallerIdentifier) -> tuple[str, int]:
        pass

    @overload
    def get(self, address: str, port: int) -> CallerIdentifier:
        pass

    def get(
        self, *args: Any, **kwargs: Any
    ) -> tuple[str, int] | CallerIdentifier:
        """Retrieves a value from the tracker, raising KeyError if not found.

        Can be called either with a `CallerIdentifier` (to get address/port)
        or with an address and port (to get `CallerIdentifier`).

        Args (interpreted based on overloads):
            id: The `CallerIdentifier` to look up.
            address: The IP address or hostname to look up.
            port: The port number to look up.

        Returns:
            A tuple (address, port) if looking up by ID.
            A `CallerIdentifier` if looking up by address/port.

        Raises:
            ValueError: If incorrect arguments are provided.
            KeyError: If the lookup key is not found.
        """
        # try_get already has the complex argument parsing and validation logic.
        # We call it and then check the result.
        # The *args and **kwargs are passed directly to try_get.
        resolved_result = self.try_get(*args, **kwargs)

        if resolved_result is None:
            # Determine the original query for a more informative error message.
            # This is a bit of a simplification; full reconstruction of original args might be needed for perfect message.
            query_repr = ""
            if args:
                query_repr = f"args={args}"
            if kwargs:
                query_repr += f"{', ' if args else ''}kwargs={kwargs}"

            raise KeyError(f"Key not found for query: {query_repr}")

        # Type of resolved_result is Optional[tuple[str, int] | CallerIdentifier].
        # Since we've checked for None, it's now tuple[str, int] | CallerIdentifier.

        # Explicit assertion to help mypy with the type after the None check.
        # assert resolved_result is not None # Keep this for runtime, but cast is for mypy
        return cast(tuple[str, int] | CallerIdentifier, resolved_result)

    def add(self, id: CallerIdentifier, address: str, port: int) -> None:
        """Adds a new bidirectional mapping to the tracker.

        Args:
            id: The `CallerIdentifier`.
            address: The IP address or hostname.
            port: The port number.

        Raises:
            KeyError: If the ID or the address/port combination already exists.
        """
        with self.__lock:
            if id in self.__id_to_address:
                # ID already exists. Remove old mapping before adding the new one.
                old_address_port = self.__id_to_address.pop(id)
                if (
                    old_address_port in self.__address_to_id
                    and self.__address_to_id[old_address_port] == id
                ):
                    self.__address_to_id.pop(old_address_port)

            # Now, check if the new address/port is already mapped to a *different* ID.
            # If (address, port) is the same as old_address_port, this check is fine.
            # If (address, port) is new, this check is also fine.
            # If (address, port) is currently mapped to another ID, we should still raise an error.
            if (
                address,
                port,
            ) in self.__address_to_id and self.__address_to_id[
                (address, port)
            ] != id:
                raise KeyError(
                    f"New address ({address}:{port}) already mapped to a different ID ({self.__address_to_id[(address,port)]}). Cannot reassign to {id}."
                )

            self.__address_to_id[(address, port)] = id
            self.__id_to_address[id] = (address, port)

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
