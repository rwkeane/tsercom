import re

test_file_path = "tsercom/data/remote_data_organizer_unittest.py"
output_lines = []

# Pattern for lines accessing the internal __data attribute directly on an 'organizer' instance
# e.g., organizer._RemoteDataOrganizer__data.add(...)
# e.g., assert len(organizer._RemoteDataOrganizer__data) == ...
# e.g., assert organizer._RemoteDataOrganizer__data[0] == ...
# Also matches organizer_no_client and organizer_custom_type
internal_data_access_pattern = re.compile(r"((?:organizer|organizer_no_client|organizer_custom_type)\._RemoteDataOrganizer__data)")
internal_is_running_access_pattern = re.compile(r"((?:organizer|organizer_no_client|organizer_custom_type)\._RemoteDataOrganizer__is_running)")
internal_on_data_ready_impl_access_pattern = re.compile(r"((?:organizer|organizer_no_client|organizer_custom_type)\._RemoteDataOrganizer__on_data_ready_impl)")
internal_last_access_pattern = re.compile(r"((?:organizer|organizer_no_client|organizer_custom_type)\._RemoteDataOrganizer__last_access)")
internal_client_pattern = re.compile(r"((?:organizer|organizer_no_client|organizer_custom_type)\._RemoteDataOrganizer__client)")
internal_thread_pool_pattern = re.compile(r"((?:organizer|organizer_no_client|organizer_custom_type)\._RemoteDataOrganizer__thread_pool)")
internal_timeout_old_data_pattern = re.compile(r"((?:organizer|organizer_no_client|organizer_custom_type)\._RemoteDataOrganizer__timeout_old_data)")


# Lines that might need a general ignore due to complex fixture interactions or untyped mocks
# This is very heuristic.
problematic_lines_heuristics = [
    "mock_is_running_tracker.start.side_effect = RuntimeError", # Mock assignment
    "mock_is_running_tracker.stop.side_effect = RuntimeError",  # Mock assignment
    "mock_pool.submit = mocker.MagicMock(side_effect=immediate_submit)", # Mock assignment
    "submitted_callable.func", # Accessing .func on a 'partial' object
    "organizer_custom_type._RemoteDataOrganizer__is_running = mocker.MagicMock" # Direct mock assignment
]


with open(test_file_path, "r") as f:
    lines = f.readlines()

for line_content in lines:
    stripped_line = line_content.strip()
    original_line = line_content
    added_ignore = False

    # Check for internal attribute access
    patterns_for_attr_defined = [
        internal_data_access_pattern,
        internal_is_running_access_pattern,
        internal_on_data_ready_impl_access_pattern,
        internal_last_access_pattern,
        internal_client_pattern,
        internal_thread_pool_pattern,
        internal_timeout_old_data_pattern,
    ]

    for pattern in patterns_for_attr_defined:
        if pattern.search(line_content):
            if not line_content.rstrip().endswith("# type: ignore[attr-defined]"):
                line_content = line_content.rstrip() + "  # type: ignore[attr-defined]\n"
                added_ignore = True
                break

    if added_ignore:
        output_lines.append(line_content)
        continue

    # Heuristic ignores for known problematic patterns (often with mocks)
    for heuristic in problematic_lines_heuristics:
        if heuristic in stripped_line:
            # Check if a general ignore or specific one is already there
            if not re.search(r"# type: ignore(\[.+\])?$", line_content.rstrip()):
                 line_content = original_line.rstrip() + "  # type: ignore[misc]\n"
            break # Apply only one heuristic ignore if multiple match

    output_lines.append(line_content)

with open(test_file_path, "w") as f:
    f.writelines(output_lines)

print(f"Attempted to add '# type: ignore' comments to {test_file_path}.")
