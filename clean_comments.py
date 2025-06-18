import tokenize
import io

# Patterns for comments to remove (case-insensitive)
PATTERNS_TO_REMOVE = [
    r"# For storing.*type",
    r"# Import .* directly",
    r"# Renamed for pylint.*",
    r"# Added cast",
    r"# Ensure .* is imported",
    r"# Changed type hint to use.*",
    r"# Initialize variable.*",
    r"# Loop through items",
    r"# Create a new.*",
    r"# Check if .*",
    r"# This will be .*",
    r"# Get the .*",
    r"# Store the .*",
    r"# Make shared dict say.*",
    r"# Configure the .*",
    r"# Type hint for .*",
    r"# Fallback if .*",
    r"# Use actual torch availability",
    r"# Mock reflects reality",
    r"# This test explicitly requires torch",
    r"# This will be converted to a real tensor",
    r"# No real tensors",
    r"# No barrier",
    r"# Long timeout for get",
    r"# Let source start polling",
    r"# Give some time for source to potentially pick up",
    r"# Using mp_sync.Barrier for type hint clarity",
    r"# Increased timeout",
    r"# Placeholder class for when torch is not available",
    r"# Keep as object for now, cast later",
    r"# Fallback for TypeAlias if torch is not installed",
    r"# This was already defined above.*",
    r"# direct import",
    r"# Accessing private members for testing purposes.",
    r"# We mocked is_torch_available to False, so torch_manager should not be called.",
    r"# Should still be called only once",
    r"# One of them should have been called by __get_manager()",
    r"# Access the underlying dict store provided by the mock_manager fixture",
    r"# Use standard manager for predictability here",
    r"# Spy on the manager's shutdown method",
    r"# Ensure it's a mock even if torch isn't installed",
    r"# This is the mock DictProxy",
    r"# This dict store will be used by the mock_dict_proxy",
    r"# Create a mock DictProxy that uses the _actual_dict_store",
    r"# Add __getitem__ if it's used by the application",
    r"# Ensure the item itself is a tensor for isinstance",
    r"# Even if torch is available, if item is not tensor, torch factory should not be called.",
    r"# Should not be called even if installed, due to is_torch_available mock",
    r"# First call initializes",
    r"# Subsequent calls on the same sink instance should not re-initialize if already done.",
    r"# If the sink itself has __real_sink_internal, it won't proceed.",
    r"# This test checks the scenario where the shared dict is already initialized",
    r"# Simulate shared dict already being initialized",
    r"# For this test, we don't strictly need it set if we're just checking factory calls.",
    r"# The application code's __initialize_real_sink has an assertion:",
    r"# This assertion will fail if INITIALIZED_KEY is true",
    r"# The default factory mock from sink_fixture_setup should not have its call_count increased by this.",
    r"# Initialize the sink first",
    r"# Sets __closed_flag = True",
    r"# Ensure it initializes",
    r"# This test now more accurately mocks the scenario where",
    r"# This mock simulates the path where the shared dict was initialized by another process,",
    r"# The original code would assert",
    r"# We are testing the consequence if that assertion was not there or passed.",
    r"# Ensure internal sink is None before calling put_blocking",
    r"# The patch is still in effect for __initialize_real_sink",
    r"# Use the same dict store approach as mock_manager_setup for consistency",
    r"# Patch time.sleep to avoid actual sleeping during tests",
    r"# Manually set the internal source",
    r"# Setup shared dict to initialize on first get call",
    r"# time.sleep should not have been called if it initialized on the first check",
    r"# Simulate _ensure_real_source_initialized raising queue.Empty (timeout)",
    r"# Simulate _ensure_real_source_initialized completing successfully (no exception)",
    r"# Force this state",
    r"# _ensure_real_source_initialized would be called, then if __real_source_internal is still None, returns None",
    r"# Using self attributes to store factories and the patch object",
    r"# so they can be accessed by helper methods like _create_factory and test methods.",
    r"# Patch is_torch_available for the duration of the test class or per method if needed",
    r"# If the patch was started, it should be stopped, though mocker might handle this.",
    r"# For explicit control:",
    r"# Use a unique queue for each run to avoid interference if tests run in parallel",
    r"# Prepare items, creating real tensors if specified and torch is available",
    r"# If not using real tensors, or torch not available, use items as is",
    r"# Give source process a moment to start polling",
    r"# Store the actual item put (e.g. real tensor)",
    r"# Small delay between puts",
    r"# Try to clean up hanging process",
    r"# Add tensor items only if torch is actually installed",
    r"# Add placeholder strings if torch is not installed",
    r"# Mock is_torch_available to True",
    r"# Critical for this test",
    r"# Simulate torch not available",
    r"# Example from a previous run",
    r"# This is a placeholder for the actual file editing an IDE would perform.",
    r"# The actual work will be to transform the unittest code into pytest style.",
    r"# The following is a conceptual representation of the transformation.",
    r"# A full transformation is too complex for a single shell script here,",
    r"# For now, since I need to provide a concrete file, I will start by",
    r"# transforming the imports and basic structure. The detailed test logic",
    r"# will be adjusted in subsequent steps or within this subtask if possible.",
    r"# This subtask performs a significant transformation of the unittest file:",
    r"# This is a substantial first pass. The next step would be to run `pytest`",
    r"# The subtask will now execute and replace the file.",
    r"# It is expected that running pytest on this file will now highlight further adjustments needed.",
    r"# For example, if any mock setups are slightly off or if some multiprocess interactions",
    r"# behave differently than the original unittest setup anticipated.",
    r"# (Not strictly necessary for the sed commands, but good for context if doing complex parsing)",
    r"# These factories are within delegating_multiprocess_queue_factory.py in this version of the codebase,",
    r"# based on the initial file read. If they were separate, paths would differ.",
    r"# The subtask report implies they are in separate files:",
    r"# The original file listing confirms they are separate. I need to edit those.",
    r"# Add manager import if not present",
    r"# If SyncManager was already there but not Optional",
    r"# Ensure super().__init__() is not duplicated if it was already there and we just added the manager line",
    r"# Avoid double-adding self.__manager if script reruns",
    r"# This requires more complex parsing. Let's assume a simple structure for now or do it manually.",
    r"# For DefaultMultiprocessQueueFactory, it uses multiprocessing.Queue()",
    r"# Replace:",
    r"# With:",
    r"# This is hard with sed. Let's use awk or a Python script for this part.",
    r"# Check for self.max_queue_size vs self._max_queue_size in original file for accuracy",
    r"# The original default_multiprocess_queue_factory.py uses self.max_queue_size (public).",
    r"# Similar changes for TorchMultiprocessQueueFactory",
    r"# For Torch, it uses torch_mp.Queue()",
    r"# Ensure torch_mp is available",
    r"# Check torch_mp too",
    r"# Fallback to torch_mp.Queue if no manager but torch_mp is there",
    r"# Should not happen if torch_available guard is effective",
    r"# The original torch_multiprocess_queue_factory.py uses self.max_queue_size (public).",
    r"# File: tsercom/threading/multiprocess/delegating_multiprocess_queue_factory.py",
    r"# __init__ already has shared_manager_dict, shared_lock. It does NOT take manager_instance.",
    r"# The manager is retrieved by the factory. The sink needs to call the Default/Torch factories *with* the manager.",
    r"# This means the manager instance needs to be available to __initialize_real_sink.",
    r"# The factory (`DelegatingMultiprocessQueueFactory`) creates the manager.",
    r"# It should pass this manager to the Sink.",
    r"# Modify DelegatingMultiprocessQueueSink.__init__ to accept manager",
    r"# Check if DictProxy is still bytes,bytes or just DictProxy after previous tool run.",
    r"# Previous run changed it to DictProxy only.",
    r"# Added manager_instance",
    r"# Store the manager",
    r"# Replace the whole __init__ block for DelegatingMultiprocessQueueSink to be safe.",
    r"# First, find the class definition",
    r"# Ends before next class",
    r"# Replace __init__ within this class block",
    r"# Modify __initialize_real_sink to pass self.__manager",
    r"# queue_factory = TorchMultiprocessQueueFactory() -> queue_factory = TorchMultiprocessQueueFactory(manager=self.__manager)",
    r"# Replace the old class content part with the new one",
    r"# Needs to pass the manager it creates/retrieves to DelegatingMultiprocessQueueSink constructor.",
    r"# Add manager_instance=manager to the call.",
    r"# Pass the manager instance",
    r"# Add SyncManager to imports if not already there for the sink's new type hint",
    r"# Check if SyncManager is used",
    r"# If DictProxy is there, add SyncManager next to it",
    r"# Else # Add it fresh",
    r"# Could try to find a good spot, or just add at top of other typing imports",
    r"# For now, assume the previous subtask's change to plain `DictProxy` was correct for its Python version/linter.",
    r"# One final check on the import for SyncManager in default and torch factories",
    r"# Ensure `from typing import Optional` is also present.",
    r"# The Python scripts for Default and Torch factories attempt to add this.",
    r"# --- Helper to get current module imports for context ---",
    r"# (Not strictly needed for replacement if class names are fully qualified or already imported)",
    r"# For this fix, we assume MultiprocessQueueSource, MultiprocessQueueSink,",
    r"# INITIALIZED_KEY, REAL_QUEUE_SOURCE_REF_KEY,",
    r"# torch, torch_mp, is_torch_available, DefaultMultiprocessQueueFactory, TorchMultiprocessQueueFactory are all appropriately",
    r"# available in the scope where DelegatingMultiprocessQueueSink is defined.",
    r"# --- Define the new __initialize_real_sink method text ---",
    r"# Using correct name mangling for private attributes like _DelegatingMultiprocessQueueSink__shared_lock",
    r"# Forward ref if needed",
    r"# Ensure torch, torch_mp, and is_torch_available are used as defined in the module scope",
    r"# Assuming 'torch' is the imported torch module, 'torch_mp' is torch.multiprocessing (or None)",
    r"# and 'is_torch_available()' is the function.",
    r"# Need to resolve how torch, torch_mp are named in the module.",
    r"# From original file:",
    r"#    import torch",
    r"#    import torch.multiprocessing as torch_mp_local_var # Renamed to avoid conflict",
    r"#    _torch_available = True",
    r"#    _torch_available = False",
    r"#    torch_mp_local_var = None # And torch would be undefined",
    r"# So, need to use _torch_available and torch_mp_local_var, or ensure global torch/torch_mp exist.",
    r"# The file uses global `torch` and `torch_mp` (which is `torch.multiprocessing` or `None`)",
    r"# and `is_torch_available()`.",
    r"# Mark as initialized *after* REAL_QUEUE_SOURCE_REF_KEY is set.",
    r"# This instance initializes its own real sink.",
    r"# Successfully initialized by this instance.",
    r"# If we reach here, INITIALIZED_KEY was True in the shared_dict.",
    r"# This means another process/sink has already set up the shared queue.",
    r"# This current sink instance needs to get a reference to it.",
    r"# Crucially, self.__real_sink_internal is still None for *this* instance.",
    r"# This check is key",
    r"# This accesses a private member of another class, which is not ideal",
    r"# but necessary given the current structure to get the actual mp.Queue.",
    r"# Type\[QueueItemType\] is not available here, so just use Any or QueueItemType",
    r"# This sink instance remains uninitialized if queue cannot be accessed. Put will fail.",
    r"# Shared state is inconsistent. This sink remains uninitialized. Put will fail.",
    r"# If __real_sink_internal is still None here, put_X methods will raise RuntimeError.",
    r"# --- Replace the old method in the class body ---",
    r"# Try to find the end of the class or the start of the next one.",
    r"# This assumes DelegatingMultiprocessQueueSource is the next class or it's near EOF.",
    r"# Non-capturing group for end",
    r"# Pattern needs to match the whole method signature and body",
    r"# Ensuring the new_method_text has correct leading indentation for methods \(4 spaces\)",
    r"# Replace the old class text in the original content with the updated class text",
    r"# Need to ensure all necessary imports are present for the new logic",
    r"# Specifically: MultiprocessQueueSource, MultiprocessQueueSink, INITIALIZED_KEY, REAL_QUEUE_SOURCE_REF_KEY",
    r"# These should already be imported in the module.",
    r"# Target: DelegatingMultiprocessQueueSink.__initialize_real_sink method",
    r"# The part of the method to modify is after the initial check for self.__real_sink_internal,",
    r"# and specifically in the block where self.__shared_lock is acquired.",
    r"# This part sets self.__real_sink_internal for the first process",
    r"# <<<< WE NEED TO ADD LOGIC HERE >>>>",
    r"# If INITIALIZED_KEY is True \(meaning another process did the above\),",
    r"# this current instance of the Sink still has self.__real_sink_internal = None.",
    r"# It needs to hook into the already existing queue.",
    r"# The original code had an `assert self.__real_sink_internal is not None` here,",
    r"# which would fail for subsequent processes. This assertion might have been removed by prior automated edits.",
    r"# If INITIALIZED_KEY is True and self.__real_sink_internal is still None:",
    r"# Access the underlying queue object from the source instance.",
    r"# This requires accessing its private '__queue' member.",
    r"# The actual name after mangling is _MultiprocessQueueSource__queue.",
    r"# Now, create a new MultiprocessQueueSink for this process using that queue.",
    r"# The generic MultiprocessQueueSink directly takes the queue object.",
    r"# This would be an error state, inconsistent shared dictionary.",
    r"# Consider raising an error or logging. For now, if it's not a valid source,",
    r"# __real_sink_internal will remain None and subsequent put calls will fail.",
    r"# Let's try to find the end of the 'if not self.__shared_dict.get\(INITIALIZED_KEY, False\):' block",
    # r"# which is marked by its 'return' statement.", # Problematic line removed
    r"# Regex to find the relevant part of __initialize_real_sink",
    r"# Looking for the end of the primary initialization block and before the final assertion \(if it exists\)",
    r"# It's safer to reconstruct the method carefully.",
    r"# Find the start of the method:",
    r"# Find the class content for DelegatingMultiprocessQueueSink",
    r"# Find where the next class starts or EOF, assuming Source is next or it's the last one here.",
    r"# Find the __initialize_real_sink method within this body",
    r"# Locate the 'return' statement inside the 'if not self.__shared_dict.get\(INITIALIZED_KEY, False\):'",
    r"# This 'return' signifies the end of the "first initializer" path.",
    r"# The code to add goes *after* this 'if' block but still within the 'with self.__shared_lock:'.",
    r"# The logic should be:",
    r"# Ensure MultiprocessQueueSink import is available for the new logic below",
    r"# This might happen if _MultiprocessQueueSource__queue is not found or source is malformed.",
    r"# Let __real_sink_internal remain None, put will fail.",
    r"# Log or handle error: INITIALIZED_KEY is true, but no valid source ref.",
    r"# We need to inject this block *inside* the `with self.__shared_lock:`",
    r"# but *after* the `if not self.__shared_dict.get\(INITIALIZED_KEY, False\): ... return` block.",
    r"# A slightly simplified regex for the insertion point might be to find the end of that 'if' block.",
    r"# The 'return' is a good marker.",
    r"# Let's try to replace the entire __initialize_real_sink method with a corrected version.",
    r"# This is often safer than complex regex surgery on parts of it.",
    r"# Using correct name mangling for private attributes like _DelegatingMultiprocessQueueSink__shared_lock",
    r"# The f-string above uses _DelegatingMultiprocessQueueSink__... which is correct.",
    r"# The new_initialize_real_sink_method f-string needs to be part of the replacement.",
    r"# It's a complete method definition starting with "def __initialize_real_sink..."",
    r"# We need to replace the entire original method found by method_match.",
    r"# Replace the old sink class body part in the main content",
    r"# After attempting to initialize, if __real_sink_internal is still None,",
    r"# put_blocking/put_nowait will raise a RuntimeError. This is intended.",
    r"# The original assertion `assert self.__real_sink_internal is not None` might have been here.",
    r"# If it was, it should now pass if the above logic correctly sets __real_sink_internal.",
    r"# If the logic above fails \(e.g. AttributeError\), then this assertion would correctly fail too.",
    r"# For robustness, let's ensure this assertion is present or that put_X methods handle None robustly.",
    r"# The put_X methods *do* check for None and raise, so the assertion here is belt-and-suspenders.",
    r"# If the assertion was removed by a prior tool, that's fine.",
    r"# If we want to ensure it's there:",
    r"#    "__real_sink_internal remained None after initialization attempt in DelegatingMultiprocessQueueSink"",
]


# More generic patterns for "what" comments
# These are applied after the specific ones.
GENERIC_PATTERNS_TO_REMOVE = [
    r"^\s*# Re-check inside lock",
    r"^\s*# Check if already initialized for this specific instance",
    r"^\s*# This is the first process .* to initialize",
    r"^\s*# Access the underlying queue from the source instance",
    r"^\s*# Create a MultiprocessQueueSink wrapper for this instance",
    r"^\s*# Log or handle: Could not get underlying queue",
    r"^\s*# Log or handle: INITIALIZED_KEY is true, but REAL_QUEUE_SOURCE_REF_KEY is not a valid source",
]

PATTERNS_TO_REMOVE.extend(GENERIC_PATTERNS_TO_REMOVE)

import re
import sys # Ensure sys is imported for sys.argv

def clean_file_comments(filepath):
    print(f"Cleaning comments in: {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return

    new_lines = []
    for line in lines:
        stripped_line = line.strip()
        is_junk_comment = False
        if stripped_line.startswith('#'):
            # Check against all patterns
            for pattern_str in PATTERNS_TO_REMOVE:
                # Ensure pattern is treated as raw string for regex
                try:
                    if re.match(pattern_str, stripped_line, re.IGNORECASE):
                        is_junk_comment = True
                        # print(f"Removing comment: {line.strip()} (Matches: {pattern_str})")
                        break
                except re.error as e_re:
                    # This can happen if a pattern string is not a valid regex
                    # print(f"Regex error with pattern '{pattern_str}': {e_re}")
                    pass # Skip this pattern if it's bad

            # Heuristic: if a line is *only* a comment and very short, it might be trivial
            # This is risky, so disabled for now.
            # if not is_junk_comment and len(stripped_line) < 25 and stripped_line.count(' ') < 3:
            #    is_junk_comment = True
            #    print(f"Removing potentially trivial short comment: {line.strip()}")

        if not is_junk_comment:
            new_lines.append(line)
        #else:
            # To see what's being removed:
            # print(f"Removed line: {line.strip()}")


    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        print(f"Finished cleaning comments in: {filepath}")
    except Exception as e:
        print(f"Error writing file {filepath}: {e}")

if __name__ == "__main__":
    files_to_process = sys.argv[1:]
    if not files_to_process:
        print("No files provided to clean_comments.py for comment removal.")
    for f_path in files_to_process:
        clean_file_comments(f_path)

print("Comment cleaning script finished.")
