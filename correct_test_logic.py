import re
import ast
from typing import List, Any, Tuple

# Helper to parse variable names from a list string like "[var1, var2, var3]"
# Returns a list of variable name strings.
def parse_vars_from_list_str(list_str: str) -> List[str]:
    try:
        cleaned_str = list_str.strip()
        if not cleaned_str.startswith("[") or not cleaned_str.endswith("]"):
            return [] # Not a list string
        cleaned_str = cleaned_str[1:-1] # Remove brackets
        if not cleaned_str: return [] # Empty list
        return [var.strip() for var in cleaned_str.split(',')]
    except Exception:
        return []

test_file_path = "tsercom/data/remote_data_organizer_unittest.py"

with open(test_file_path, "r") as f:
    lines = f.readlines()

output_lines = []
# Regex to find direct data manipulations on common organizer instance names
data_manipulation_pattern_appendleft = re.compile(r"((?:organizer|organizer_no_client|organizer_custom_type)\._RemoteDataOrganizer__data)\.appendleft\((.*?)\)")
data_manipulation_pattern_append = re.compile(r"((?:organizer|organizer_no_client|organizer_custom_type)\._RemoteDataOrganizer__data)\.append\((.*?)\)")

# Regex for the specific fixture error in the signature
# This was to change mock_is_running_tracker_factory to mocker in the test signature.
# This change was already applied successfully in the previous subtask by directly editing the test file content.
# So, this regex might not find anything, which is fine.
fixture_factory_pattern_sig = re.compile(r"(def\s+test_get_interpolated_at_unsupported_datatype_for_interpolation\s*\((?:[^,\)]+,)*\s*)mock_is_running_tracker_factory(\s*[,)])")

# Regex for list assertions to ensure they reflect SortedList's oldest-first order.
# Example: assert list(organizer._RemoteDataOrganizer__data) == [data_t1, data_t2, data_t3, data_t4]
# This is very complex to automate safely without full context.
# The previous subtask's diffs already corrected many of these.
# This script will focus on the .add() replacements and the specific fixture name.

for i, line_content in enumerate(lines):
    modified_line = line_content

    # 1. Replace .appendleft() and .append() with .add() for relevant organizer instances
    # These should have been fixed by the syntax error fix script, this is a safeguard.
    modified_line = data_manipulation_pattern_appendleft.sub(r"\1.add(\2)", modified_line)
    modified_line = data_manipulation_pattern_append.sub(r"\1.add(\2)", modified_line)

    # 2. Fix fixture 'mock_is_running_tracker_factory' in the specific test signature
    # This was handled in the previous subtask by directly editing the generated test code.
    # If the name 'mock_is_running_tracker_factory' is still in the signature, this will change it.
    if "def test_get_interpolated_at_unsupported_datatype_for_interpolation" in modified_line:
        modified_line = fixture_factory_pattern_sig.sub(r"\1mocker\2", modified_line)

    output_lines.append(modified_line)

with open(test_file_path, "w") as f:
    f.writelines(output_lines)

print(f"Applied functional fixes (primarily .add() and fixture name) to {test_file_path}.")
