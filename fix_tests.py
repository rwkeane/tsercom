import re
import ast
from typing import List, Any, Tuple

test_file_path = "tsercom/data/remote_data_organizer_unittest.py"

with open(test_file_path, "r") as f:
    lines = f.readlines()

output_lines = []
# Regex to find direct data manipulations on common organizer instance names
data_manipulation_pattern_appendleft = re.compile(r"((?:organizer|organizer_no_client|organizer_custom_type)\._RemoteDataOrganizer__data)\.appendleft\((.*?)\)")
data_manipulation_pattern_append = re.compile(r"((?:organizer|organizer_no_client|organizer_custom_type)\._RemoteDataOrganizer__data)\.append\((.*?)\)")

# Regex for the specific fixture error in the signature
fixture_factory_pattern_sig = re.compile(r"(def\s+test_get_interpolated_at_unsupported_datatype_for_interpolation\s*\((?:[^,\)]+,)*\s*)mock_is_running_tracker_factory(\s*[,)])")


for i, line_content in enumerate(lines):
    modified_line = line_content

    # 1. Replace .appendleft() and .append() with .add() for relevant organizer instances
    modified_line = data_manipulation_pattern_appendleft.sub(r".add()", modified_line)
    modified_line = data_manipulation_pattern_append.sub(r".add()", modified_line)

    # 2. Fix fixture 'mock_is_running_tracker_factory' in the specific test signature
    # The body of this test already uses mocker.MagicMock directly for __is_running.
    if "def test_get_interpolated_at_unsupported_datatype_for_interpolation" in modified_line:
        modified_line = fixture_factory_pattern_sig.sub(r"mocker\2", modified_line)


    output_lines.append(modified_line)

with open(test_file_path, "w") as f:
    f.writelines(output_lines)

print(f"Attempted to fix test failures in {test_file_path}.")
print("Primarily replaced .appendleft/.append with .add() and fixed mock_is_running_tracker_factory.")
print("Review of list order assertions and specific test logic for interpolation failures might still be needed if tests fail.")
