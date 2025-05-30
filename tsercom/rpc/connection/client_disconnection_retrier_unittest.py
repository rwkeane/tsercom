import asyncio
import pytest
import pytest_asyncio
import logging
from unittest.mock import (
    AsyncMock,
    MagicMock,
)
# SUT
from tsercom.rpc.connection.client_disconnection_retrier import (
    ClientDisconnectionRetrier,
)
from tsercom.util.stopable import Stopable
from tsercom.threading.thread_watcher import ThreadWatcher

# Module to patch for aio_utils
import tsercom.rpc.connection.client_disconnection_retrier as retrier_module_to_patch

# The user-provided script logic starts here, adapted slightly
file_path = "tsercom/rpc/connection/client_disconnection_retrier_unittest.py" # Path is fixed for overwrite

# Read original lines (conceptually, as we are overwriting, this is for the script's logic)
# In a real overwrite, we'd effectively be generating this from scratch or a transformation.
# For this tool, the entire content below will BE the new file.
# So, the script needs to be self-contained in producing the final output lines.

original_lines_content = """\
import asyncio
import pytest
import pytest_asyncio  # Added for async fixtures
import logging  # Added
from unittest.mock import (
    AsyncMock,
    MagicMock,
)  # pytest-mock uses these via mocker
from functools import partial

# SUT
from tsercom.rpc.connection.client_disconnection_retrier import (
    ClientDisconnectionRetrier,
)
from tsercom.util.stopable import Stopable
from tsercom.threading.thread_watcher import ThreadWatcher

# Module to patch for aio_utils
import tsercom.rpc.connection.client_disconnection_retrier as retrier_module_to_patch


# Helper Test Classes
class MockConnectable(Stopable):
    def __init__(self, name="mock_instance"):
        self.name = name
        self.stopped = False
        self.stop_call_count = 0

    async def stop(self) -> None:
        self.stopped = True
        self.stop_call_count += 1
        print(f"{self.name} stopped")

    def __repr__(self):
        return f"<MockConnectable {self.name}>"


class _TestableRetrier(ClientDisconnectionRetrier[MockConnectable]):  # Renamed
    __test__ = False  # Add this line

    def set_on_connect(self, mock_connect_func: AsyncMock):
        self.connect_func = mock_connect_func

    async def _connect(self) -> MockConnectable:
        print(
            f"_TestableRetrier._connect called, about to call self.connect_func ({self.connect_func})"
        )
        instance = await self.connect_func()
        print(
            f"_TestableRetrier._connect: self.connect_func returned {instance}"
        )
        return instance


# Pytest Fixtures
@pytest.fixture
def mock_watcher(mocker):
    return mocker.MagicMock(spec=ThreadWatcher)


@pytest.fixture
def mock_safe_disconnection_handler(mocker):
    # Making it async as the SUT might await it
    return mocker.AsyncMock(name="safe_disconnection_handler")


@pytest.fixture
def mock_delay_func(mocker):
    return mocker.AsyncMock(name="delay_func")


@pytest.fixture
def mock_is_grpc_error_func(mocker):
    return mocker.MagicMock(name="is_grpc_error_func")


@pytest.fixture
def mock_is_server_unavailable_error_func(mocker):
    return mocker.MagicMock(name="is_server_unavailable_error_func")


@pytest.fixture
def mock_connect_func(mocker):
    return mocker.AsyncMock(name="connect_func")


@pytest_asyncio.fixture  # Changed to pytest_asyncio.fixture
async def current_event_loop():  # Made async
    return asyncio.get_running_loop()


@pytest_asyncio.fixture  # Changed to pytest_asyncio.fixture
async def mock_aio_utils(
    mocker, monkeypatch, current_event_loop
):  # Made async
    # current_event_loop is now the resolved loop instance, no need to await it here.
    mock_get_loop = mocker.MagicMock(return_value=current_event_loop)
    mock_is_on_loop = mocker.MagicMock(
        return_value=True
    )  # Default to True for most tests

    # This mock needs to handle partials correctly, similar to discoverable_grpc_endpoint_connector
    def simplified_run_on_loop_side_effect(
        func_to_run, loop_arg, *args, **kwargs
    ):
        # Use current_event_loop (which is loop_instance from the outer scope) as default
        resolved_loop = loop_arg if loop_arg else current_event_loop
        print(
            f"MOCKED run_on_event_loop CALLED for {getattr(func_to_run, 'func', func_to_run).__name__} on loop {resolved_loop}"
        )
        if isinstance(func_to_run, partial):
            # If it's a partial, it might be an async method.
            # The SUT uses it for self.stop and self._on_disconnect
            # These are async. We need to schedule them.
            asyncio.ensure_future(func_to_run(), loop=resolved_loop)
        else:  # Direct call for simple functions/coroutines
            asyncio.ensure_future(
                func_to_run(*args, **kwargs),
                loop=resolved_loop,
            )

        f = asyncio.Future()
        f.set_result(None)  # SUT doesn't await this future.
        return f

    mock_run_on_loop = mocker.MagicMock(
        side_effect=simplified_run_on_loop_side_effect
    )

    monkeypatch.setattr(
        retrier_module_to_patch, "get_running_loop_or_none", mock_get_loop
    )
    monkeypatch.setattr(
        retrier_module_to_patch, "is_running_on_event_loop", mock_is_on_loop
    )
    monkeypatch.setattr(
        retrier_module_to_patch, "run_on_event_loop", mock_run_on_loop
    )

    return {
        "get_running_loop_or_none": mock_get_loop,
        "is_running_on_event_loop": mock_is_on_loop,
        "run_on_event_loop": mock_run_on_loop,
    }


@pytest.mark.asyncio
class TestClientDisconnectionRetrier:

    async def test_initial_connection_success(
        self,
        mock_watcher,
        mock_connect_func,
        current_event_loop,  # Keep as is, will be awaited
        mock_aio_utils,  # Now receives resolved fixture
    ):
        loop = current_event_loop  # No await
        _mock_aio_utils = mock_aio_utils  # No await

        mock_instance = MockConnectable("instance1")
        mock_connect_func.return_value = mock_instance

        retrier = _TestableRetrier(  # Renamed
            watcher=mock_watcher, event_loop=loop
        )
        retrier.set_on_connect(mock_connect_func)

        assert await retrier.start() is True
        mock_connect_func.assert_called_once()
        assert retrier._ClientDisconnectionRetrier__instance is mock_instance
        mock_watcher.on_exception_seen.assert_not_called()

    async def test_initial_connection_server_unavailable(
        self,
        mock_watcher,
        mock_connect_func,
        mock_is_server_unavailable_error_func,
        current_event_loop,  # Resolved
        mock_aio_utils,  # Resolved
    ):
        loop = current_event_loop  # No await
        _mock_aio_utils = mock_aio_utils  # No await

        test_error = ConnectionRefusedError("Server unavailable")
        mock_connect_func.side_effect = test_error
        mock_is_server_unavailable_error_func.return_value = True

        retrier = _TestableRetrier(  # Renamed
            watcher=mock_watcher,
            is_server_unavailable_error_func=mock_is_server_unavailable_error_func,
            event_loop=loop,
        )
        retrier.set_on_connect(mock_connect_func)

        assert await retrier.start() is False
        mock_connect_func.assert_called_once()
        mock_is_server_unavailable_error_func.assert_called_once_with(
            test_error
        )
        assert retrier._ClientDisconnectionRetrier__instance is None
        mock_watcher.on_exception_seen.assert_not_called()  # Should not be called for server unavailable on start

    async def test_initial_connection_other_error(
        self,
        mock_watcher,
        mock_connect_func,
        mock_is_server_unavailable_error_func,
        current_event_loop,  # Resolved
        mock_aio_utils,  # Resolved
    ):
        loop = current_event_loop  # No await
        _mock_aio_utils = mock_aio_utils  # No await

        test_error = ValueError("Some other connection error")
        mock_connect_func.side_effect = test_error
        mock_is_server_unavailable_error_func.return_value = (
            False  # Not server unavailable
        )

        retrier = _TestableRetrier(  # Renamed
            watcher=mock_watcher,
            is_server_unavailable_error_func=mock_is_server_unavailable_error_func,
            event_loop=loop,
        )
        retrier.set_on_connect(mock_connect_func)

        with pytest.raises(ValueError, match="Some other connection error"):
            await retrier.start()

        mock_connect_func.assert_called_once()
        mock_is_server_unavailable_error_func.assert_called_once_with(
            test_error
        )
        assert retrier._ClientDisconnectionRetrier__instance is None
        # The error is raised by start(), not reported to watcher via _on_disconnect path
        mock_watcher.on_exception_seen.assert_not_called()

    async def test_stop_method_basic(
        self,
        mock_watcher,
        mock_connect_func,
        current_event_loop,  # Resolved
        mock_aio_utils,  # Resolved
    ):
        loop = current_event_loop  # No await
        _mock_aio_utils = mock_aio_utils  # No await

        mock_instance = MockConnectable("instance_to_stop")
        mock_connect_func.return_value = mock_instance

        retrier = _TestableRetrier(  # Renamed
            watcher=mock_watcher, event_loop=loop
        )
        retrier.set_on_connect(mock_connect_func)
        await retrier.start()  # Connect
        assert retrier._ClientDisconnectionRetrier__instance is mock_instance

        await retrier.stop()

        assert mock_instance.stopped is True
        assert mock_instance.stop_call_count == 1
        assert retrier._ClientDisconnectionRetrier__instance is None

    async def test_disconnect_and_reconnect_success(
        self,
        mock_watcher,
        mock_connect_func,
        mock_delay_func,
        mock_is_server_unavailable_error_func,
        current_event_loop,  # Resolved
        mock_aio_utils,  # Resolved
    ):
        loop = current_event_loop  # No await
        _mock_aio_utils = mock_aio_utils  # No await

        # Initial successful connection
        instance1 = MockConnectable("instance1")
        mock_connect_func.return_value = instance1

        retrier = _TestableRetrier(  # Renamed
            watcher=mock_watcher,
            delay_before_retry_func=mock_delay_func,
            is_server_unavailable_error_func=mock_is_server_unavailable_error_func,
            event_loop=loop,
            max_retries=3,
        )
        retrier.set_on_connect(mock_connect_func)
        await retrier.start()
        assert retrier._ClientDisconnectionRetrier__instance is instance1
        mock_connect_func.reset_mock()  # Reset for a new sequence of calls

        # Simulate disconnect with a server unavailable error
        disconnect_error = ConnectionAbortedError("Server connection lost")
        mock_is_server_unavailable_error_func.return_value = True

        # Prepare for reconnect success
        instance2 = MockConnectable("instance2")
        # Mock _connect to fail once then succeed
        mock_connect_func.side_effect = [
            instance2
        ]  # Succeeds on first retry call

        # Trigger disconnect
        await retrier._on_disconnect(disconnect_error)
        await asyncio.sleep(
            0
        )  # Allow scheduled tasks like _on_disconnect's body to run

        # Verifications
        mock_is_server_unavailable_error_func.assert_called_with(
            disconnect_error
        )
        assert instance1.stopped is True, "Original instance should be stopped"

        mock_delay_func.assert_called_once()  # Called before the successful retry

        # _connect should be called once for the successful retry
        mock_connect_func.assert_called_once()

        assert (
            retrier._ClientDisconnectionRetrier__instance is instance2
        ), "Should have reconnected to new instance"
        mock_watcher.on_exception_seen.assert_not_called()

    async def test_disconnect_reaches_max_retries(
        self,
        mock_watcher,
        mock_connect_func,
        mock_delay_func,
        mock_is_server_unavailable_error_func,
        current_event_loop,  # Resolved
        mock_aio_utils,  # Resolved
        caplog,
    ):
        loop = current_event_loop  # No await
        _mock_aio_utils = mock_aio_utils  # No await

        caplog.set_level(logging.INFO)
        # Initial successful connection
        instance1 = MockConnectable("instance_max_retry")
        mock_connect_func.return_value = instance1
        max_retries = 2  # Keep it small for the test

        retrier = _TestableRetrier(  # Renamed
            watcher=mock_watcher,
            delay_before_retry_func=mock_delay_func,
            is_server_unavailable_error_func=mock_is_server_unavailable_error_func,
            event_loop=loop,
            max_retries=max_retries,
        )
        retrier.set_on_connect(mock_connect_func)
        await retrier.start()
        mock_connect_func.reset_mock()

        # Simulate disconnect
        disconnect_error = ConnectionRefusedError(
            "Server unavailable persistently"
        )
        mock_is_server_unavailable_error_func.return_value = True

        # Mock _connect to always fail for retries
        retry_connect_error = ConnectionRefusedError("Still unavailable")
        mock_connect_func.side_effect = [retry_connect_error] * max_retries

        # Trigger disconnect
        await retrier._on_disconnect(disconnect_error)
        await asyncio.sleep(0)  # Allow _on_disconnect body to run
        await asyncio.sleep(0)  # Allow potential retry loop tasks to run

        # Verifications
        assert instance1.stopped is True
        assert mock_delay_func.call_count == max_retries
        assert mock_connect_func.call_count == max_retries

        assert (
            retrier._ClientDisconnectionRetrier__instance is None
        ), "Instance should be None after max retries"
        assert (
            f"Max retries ({max_retries}) reached. Stopping reconnection attempts."
            in caplog.text
        )
        mock_watcher.on_exception_seen.assert_not_called()  # Server unavailable errors are not reported to watcher by default
        _mock_aio_utils[  # Use awaited version
            "run_on_event_loop"
        ].assert_not_called()  # Should be on the same loop

    async def test_stop_method_when_not_on_event_loop(
        self,
        mock_watcher,
        mock_connect_func,
        current_event_loop,  # Resolved
        mock_aio_utils,  # Resolved
    ):
        loop = current_event_loop  # No await
        _mock_aio_utils = mock_aio_utils  # No await

        mock_instance = MockConnectable("instance_to_stop_off_loop")
        mock_connect_func.return_value = mock_instance

        # Simulate being on a different event loop for the first call to stop(),
        # then on the event loop for the rescheduled call.
        is_on_loop_mock = _mock_aio_utils["is_running_on_event_loop"]
        is_on_loop_mock.side_effect = [False, True]

        retrier = _TestableRetrier(  # Renamed
            watcher=mock_watcher,
            event_loop=loop,  # This is the SUT's main loop
        )
        retrier.set_on_connect(mock_connect_func)
        await retrier.start()

        await retrier.stop()  # This should trigger run_on_event_loop

        # Check that run_on_event_loop was called to reschedule stop
        _mock_aio_utils["run_on_event_loop"].assert_called_once()
        call_args = _mock_aio_utils["run_on_event_loop"].call_args[0]
        # func_to_run is self.stop. loop is loop.
        assert call_args[0] == retrier.stop
        assert call_args[1] == loop

        # Because the mock 'run_on_event_loop' schedules 'retrier.stop' on the current_event_loop
        # and we `await asyncio.sleep(0)` after this call in the test usually,
        # the stop() method on the instance should eventually be called.
        # The mock simplified_run_on_loop_side_effect uses ensure_future.
        # We need to allow this task to run.
        await asyncio.sleep(0)

        assert mock_instance.stopped is True
        assert mock_instance.stop_call_count == 1
        assert retrier._ClientDisconnectionRetrier__instance is None

""" # End of original_lines_content

lines = original_lines_content.splitlines(keepends=True)
output_lines = []
# Flags to help with context-specific comment removal or docstring addition
in_mock_connectable = False
in_testable_retrier = False
in_mock_aio_utils_fixture = False
in_test_class = False

# 1. Add Module Docstring
if not lines[0].strip().startswith('"""') and not lines[0].strip().startswith("'''"):
    output_lines.append('"""Unit tests for the ClientDisconnectionRetrier class."""\n')

for i, line in enumerate(lines):
    # 1. Remove Unused Import (functools.partial)
    if "from functools import partial" in line: # This was in original imports
        continue
    if "from functools import partial" in line and "SUT" in lines[i-2 if i>1 else 0]: # Check context for safety
        # This condition is too specific, the general one above should catch it.
        continue

    # 2. Remove print() statements
    # General approach: replace print(xxx) with empty string or just the xxx if it's essential
    new_line = line
    if "print(f\"{self.name} stopped\")" in line: # In MockConnectable.stop
        new_line = line.replace("print(f\"{self.name} stopped\")", "")
    if "print(f\"_TestableRetrier._connect called, about to call self.connect_func ({self.connect_func})\")" in line: # In _TestableRetrier._connect
        new_line = new_line.replace("print(f\"_TestableRetrier._connect called, about to call self.connect_func ({self.connect_func})\")", "")
    if "print(f\"_TestableRetrier._connect: self.connect_func returned {instance}\")" in line: # In _TestableRetrier._connect
        new_line = new_line.replace("print(f\"_TestableRetrier._connect: self.connect_func returned {instance}\")", "")
    if "print(f\"MOCKED run_on_event_loop CALLED for {getattr(func_to_run, 'func', func_to_run).__name__} on loop {resolved_loop}\")" in line: # In mock_aio_utils
        new_line = new_line.replace("print(f\"MOCKED run_on_event_loop CALLED for {getattr(func_to_run, 'func', func_to_run).__name__} on loop {resolved_loop}\")", "")


    # 3. Clean up Dev Comments
    new_line = new_line.replace("  # Added for async fixtures", "")
    new_line = new_line.replace("  # Added", "")
    new_line = new_line.replace("  # Renamed", "")
    new_line = new_line.replace("  # No await", "")
    new_line = new_line.replace("  # Resolved", "")
    new_line = new_line.replace("  # Keep as is, will be awaited", "")
    new_line = new_line.replace("  # Now receives resolved fixture", "")
    new_line = new_line.replace("  # pytest-mock uses these via mocker", "")
    new_line = new_line.replace("  # Changed to pytest_asyncio.fixture", "")
    new_line = new_line.replace("  # Made async", "")
    new_line = new_line.replace("  # Add this line", "")


    # Remove trailing whitespace that might be left if only comment was on line
    if new_line.strip() == "#":
        new_line = "\n" # Keep newline if it was a full line comment
    elif new_line.strip().endswith("#"):
        new_line = new_line.rstrip("#").rstrip() + "\n"

    # Add docstrings
    if "class MockConnectable(Stopable):" in new_line:
        output_lines.append(new_line)
        if not lines[i+1].strip().startswith('"""'):
            output_lines.append("    \"\"\"A mock Stopable class for testing.\"\"\"\n")
        continue
    if "class _TestableRetrier(ClientDisconnectionRetrier[MockConnectable]):" in new_line:
        output_lines.append(new_line)
        if not lines[i+1].strip().startswith('"""'):
             output_lines.append("    \"\"\"A subclass of ClientDisconnectionRetrier for testing _connect behavior.\"\"\"\n")
        continue

    if "async def mock_aio_utils(" in new_line:
        # Check if previous line was the fixture decorator
        if "@pytest_asyncio.fixture" in lines[i-1]:
            output_lines.append(new_line) # Add the def line
            if not lines[i+1].strip().startswith('"""'): # Check for existing docstring on next line
                indent = new_line.split("async")[0]
                output_lines.append(indent + "    \"\"\"Mocks tsercom.threading.aio.aio_utils for testing event loop interactions.\"\"\"\n")
            continue

    if "class TestClientDisconnectionRetrier:" in new_line:
        # Check if previous line was the marker
        if "@pytest.mark.asyncio" in lines[i-1]:
            output_lines.append(new_line) # Add class def line
            if not lines[i+1].strip().startswith('"""'): # Check for existing docstring
                indent = new_line.split("class")[0]
                output_lines.append(indent + "    \"\"\"Tests for the ClientDisconnectionRetrier functionality.\"\"\"\n")
            continue

    # Avoid multiple blank lines if comments/prints were removed and new_line became empty
    if new_line.strip() == "" and len(output_lines) > 0 and output_lines[-1].strip() == "":
        continue

    # Ensure we don't add empty lines if new_line is solely a newline character from comment removal
    if new_line == "\n" and line.strip().startswith("#") and len(output_lines) > 0 and output_lines[-1].strip() == "":
         pass # Skip adding this newline if it makes multiple blanks
    elif new_line.strip() != "" or (new_line == "\n" and (len(output_lines) == 0 or output_lines[-1].strip() != "")):
        output_lines.append(new_line)


# Final pass to remove consecutive blank lines that might have been created
final_output_lines = []
if output_lines: # Ensure output_lines is not empty
    # Add module docstring if it was generated and not added due to initial lines being comments
    if output_lines[0].strip().startswith('"""Unit tests for the ClientDisconnectionRetrier class."""') and \
       not original_lines_content.strip().startswith('"""'):
        final_output_lines.append(output_lines.pop(0))

    for k, line_content in enumerate(output_lines):
        is_empty_line = line_content.strip() == ""
        is_prev_empty = k > 0 and final_output_lines[-1].strip() == ""

        if is_empty_line and is_prev_empty:
            continue
        final_output_lines.append(line_content)

# The original script's writing part:
# with open(file_path, "w") as f:
#     f.writelines(final_output_lines)
# This will be implicitly handled by overwrite_file_with_block using final_output_lines

# Ensure all imports are at the top, after the module docstring
processed_imports = []
processed_body = []
module_docstring_present = False
if final_output_lines and final_output_lines[0].strip().startswith('"""'):
    module_docstring_present = True
    processed_imports.append(final_output_lines.pop(0))


for line_content in final_output_lines:
    if line_content.strip().startswith("import ") or line_content.strip().startswith("from "):
        if "SUT" in line_content or "Module to patch" in line_content: # Heuristic for comments separating imports
             processed_body.append(line_content)
        elif "# SUT" in line_content or "# Module to patch" in line_content: # Comment lines themselves
            processed_body.append(line_content)
        else:
             processed_imports.append(line_content)
    else:
        processed_body.append(line_content)

# Remove duplicate blank lines from imports and body separately
temp_imports = []
if processed_imports:
    temp_imports.append(processed_imports[0]) # Add first line (module docstring or first import)
    for i in range(1, len(processed_imports)):
        if processed_imports[i].strip() == "" and temp_imports[-1].strip() == "":
            continue
        temp_imports.append(processed_imports[i])
processed_imports = temp_imports

temp_body = []
if processed_body:
    # Remove leading blank lines from body if imports are present
    first_real_line_idx = 0
    if processed_imports:
        while first_real_line_idx < len(processed_body) and processed_body[first_real_line_idx].strip() == "":
            first_real_line_idx += 1

    if first_real_line_idx < len(processed_body): # If body is not all blanks
        temp_body.append(processed_body[first_real_line_idx]) # Add first non-blank line
        for i in range(first_real_line_idx + 1, len(processed_body)):
            if processed_body[i].strip() == "" and temp_body[-1].strip() == "":
                continue
            temp_body.append(processed_body[i])
processed_body = temp_body

# Combine imports and body
final_output_lines = processed_imports + ["\n"] + processed_body # Add a blank line between imports and body if both exist

# This is what the tool will write:
for l in final_output_lines:
    print(l, end="")

# The print statements in the original subtask script like "Finished code polishing..." are not part of file content.
