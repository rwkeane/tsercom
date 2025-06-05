"""Threading utils for tsercom: custom threads, watchers."""

# Comment out the direct import of AsyncPoller to enable lazy loading.
# from tsercom.threading.async_poller import AsyncPoller
from tsercom.threading.atomic import Atomic
from tsercom.threading.thread_watcher import ThreadWatcher
from tsercom.threading.thread_safe_queue import ThreadSafeQueue

__all__ = [
    "AsyncPoller",  # pylint: disable=undefined-all-variable # Used via __getattr__
    "Atomic",
    "ThreadWatcher",
    "ThreadSafeQueue",
]


# Implement __getattr__ for lazy loading of AsyncPoller.
def __getattr__(name: str) -> type:
    if name == "AsyncPoller":
        # pylint: disable=import-outside-toplevel
        from tsercom.threading.async_poller import (
            AsyncPoller as _AsyncPoller,
        )

        return _AsyncPoller
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
