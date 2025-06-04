"""Manages bidirectional mappings: CallerIdentifier <=> network address."""

import threading
from typing import Dict, Optional, overload, Iterator, Any, cast

from tsercom.caller_id.caller_identifier import CallerIdentifier


class IdTracker:
    """
    Thread-safe bidirectional dictionary mapping CallerID to address/port.
    """

    def __init__(self) -> None:
        """Initializes IdTracker with internal lock and dictionaries."""
        self.__lock = threading.Lock()
        self.__address_to_id: Dict[tuple[str, int], CallerIdentifier] = {}
        self.__id_to_address: Dict[CallerIdentifier, tuple[str, int]] = {}

    # pylint: disable=too-many-branches # Handles multiple lookup methods
    @overload
    def try_get(
        self, caller_id_obj: CallerIdentifier
    ) -> Optional[tuple[str, int]]:
        pass

    @overload
    def try_get(self, address: str, port: int) -> Optional[CallerIdentifier]:
        pass

    def try_get(
        self, *args: Any, **kwargs: Any
    ) -> Optional[tuple[str, int] | CallerIdentifier]:
        """Attempts to retrieve a value from the tracker.

        Can be called with a `CallerIdentifier` (gets address/port)
        or with address and port (gets `CallerIdentifier`).

        Args:
            caller_id_obj: `CallerIdentifier` to look up.
            address: IP address or hostname to look up.
            port: Port number to look up.

        Returns:
            A tuple (address, port) if looking up by CallerIdentifier & found.
            A `CallerIdentifier` if looking up by address/port & found.
            `None` if the lookup key is not found.

        Raises:
            ValueError: If incorrect arguments are provided.
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
            elif len(args) > 0:
                # pylint: disable=consider-using-f-string
                raise ValueError(
                    "Invalid positional args. Use CallerID or (addr, port). "
                    "Got: %s" % (args,)
                )

        if "id" in kwargs:  # Callers still use 'id' as kwarg for now.
            if _id is not None or _address is not None or _port is not None:
                raise ValueError(
                    "Cannot mix 'id' kwarg with positional or other "
                    "address/port kwargs."
                )
            kw_id = kwargs.pop("id")
            if not isinstance(kw_id, CallerIdentifier):
                # pylint: disable=consider-using-f-string
                raise ValueError(
                    "'id' kwarg must be CallerIdentifier. Got %s" % type(kw_id)
                )
            _id = kw_id

        if "address" in kwargs or "port" in kwargs:
            if _id is not None:
                raise ValueError(
                    "Cannot mix 'address'/'port' kwargs with 'id' argument."
                )
            if "address" in kwargs:
                kw_address = kwargs.pop("address")
                if not isinstance(kw_address, str):
                    # pylint: disable=consider-using-f-string
                    raise ValueError(
                        "'address' kwarg must be str. Got %s" % type(kw_address)
                    )
                _address = kw_address
            if "port" in kwargs:
                kw_port = kwargs.pop("port")
                if not isinstance(kw_port, int):
                    # pylint: disable=consider-using-f-string
                    raise ValueError(
                        "'port' kwarg must be int. Got %s" % type(kw_port)
                    )
                _port = kw_port

        if kwargs:  # Any remaining kwargs are unexpected
            # pylint: disable=consider-using-f-string
            raise ValueError("Unexpected kwargs: %s" % list(kwargs.keys()))

        if (_id is None) == (_address is None and _port is None):
            raise ValueError(
                "Provide (CallerID) or (address and port), not both/neither."
            )
        if (_address is None) != (_port is None):
            raise ValueError(
                "If 'address' is given, 'port' must also be, and vice-versa."
            )

        with self.__lock:
            if _id is not None:
                return self.__id_to_address.get(_id)
            if _address is not None and _port is not None:
                return self.__address_to_id.get((_address, _port))
        return None  # Should be unreachable

    @overload
    def get(self, caller_id_obj: CallerIdentifier) -> tuple[str, int]:
        pass

    @overload
    def get(self, address: str, port: int) -> CallerIdentifier:
        pass

    def get(
        self, *args: Any, **kwargs: Any
    ) -> tuple[str, int] | CallerIdentifier:
        """Retrieves a value, raising KeyError if not found.

        Args:
            caller_id_obj: `CallerIdentifier` to look up.
            address: IP address or hostname.
            port: Port number.

        Returns:
            (address, port) by ID, or `CallerIdentifier` by address/port.

        Raises:
            ValueError: If incorrect arguments provided.
            KeyError: If lookup key not found.
        """
        resolved_result = self.try_get(*args, **kwargs)

        if resolved_result is None:
            query_repr = ""
            if args:
                # pylint: disable=consider-using-f-string
                query_repr = "args=%s" % (args,)
            if kwargs:
                sep = ", " if args else ""
                # pylint: disable=consider-using-f-string
                query_repr += "%skwargs=%s" % (sep, kwargs)
            # pylint: disable=consider-using-f-string
            raise KeyError("Key not found for query: %s" % query_repr)
        return cast(tuple[str, int] | CallerIdentifier, resolved_result)

    def add(
        self, caller_id_obj: CallerIdentifier, address: str, port: int
    ) -> None:
        """Adds a new bidirectional mapping.

        Args:
            caller_id_obj: The `CallerIdentifier`.
            address: IP address or hostname.
            port: Port number.

        Raises:
            KeyError: If ID or address/port combination already exists mapped
                      to a different counterpart.
        """
        with self.__lock:
            if caller_id_obj in self.__id_to_address:
                old_address_port = self.__id_to_address.pop(caller_id_obj)
                if (
                    old_address_port in self.__address_to_id
                    and self.__address_to_id[old_address_port] == caller_id_obj
                ):
                    self.__address_to_id.pop(old_address_port)

            if (
                address,
                port,
            ) in self.__address_to_id and self.__address_to_id[
                (address, port)
            ] != caller_id_obj:
                # pylint: disable=consider-using-f-string
                raise KeyError(
                    "New address (%s:%s) already mapped to a different ID "
                    "(%s). Cannot reassign to %s."
                    % (
                        address,
                        port,
                        self.__address_to_id[(address, port)],
                        caller_id_obj,
                    )
                )

            self.__address_to_id[(address, port)] = caller_id_obj
            self.__id_to_address[caller_id_obj] = (address, port)

    def has_id(self, caller_id_obj: CallerIdentifier) -> bool:
        """Checks if `CallerIdentifier` exists."""
        with self.__lock:
            return caller_id_obj in self.__id_to_address

    def has_address(self, address: str, port: int) -> bool:
        """Checks if address/port combination exists."""
        with self.__lock:
            return (address, port) in self.__address_to_id

    def __len__(self) -> int:
        """Returns number of mappings."""
        with self.__lock:
            assert len(self.__id_to_address) == len(self.__address_to_id)
            return len(self.__id_to_address)

    def __iter__(self) -> Iterator[CallerIdentifier]:
        """Returns iterator over `CallerIdentifier`s.

        Note: Iterates over dictionary's iterator directly. Safe for simple
        loops due to lock, but concurrent modification during a single
        __next__ call might be an issue if lock is managed externally.
        """
        with self.__lock:
            return iter(self.__id_to_address)

    def remove(self, caller_id_obj: CallerIdentifier) -> bool:
        """Removes `CallerIdentifier` and its mapping.

        Returns:
            True if found and removed, False otherwise.
        """
        with self.__lock:
            if caller_id_obj not in self.__id_to_address:
                return False
            address_port_tuple = self.__id_to_address.pop(caller_id_obj)
            if address_port_tuple in self.__address_to_id:
                # Ensure we only delete if mapped to the ID we are removing
                if self.__address_to_id[address_port_tuple] == caller_id_obj:
                    del self.__address_to_id[address_port_tuple]
            return True
