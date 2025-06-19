import re

file_path = "tsercom/data/remote_data_organizer_unittest.py"
lines = []
sorted_list_imported = False
isinstance_updated = False

with open(file_path, "r") as f:
    lines = f.readlines()

output_lines = []

# Ensure SortedList is imported
# Try to add it along with other `collections` imports or standard library imports for neatness
# For simplicity, add it after `from collections import deque` if present, or after `import datetime`
# More robust: check if already imported.

# First pass: check if SortedList is already imported
for line_content in lines: # Renamed variable to avoid conflict
    if "from sortedcontainers import SortedList" in line_content:
        sorted_list_imported = True
        break

current_line_index = 0
while current_line_index < len(lines):
    line = lines[current_line_index]

    # Add SortedList import if not already present
    if not sorted_list_imported:
        # Add after 'from collections import deque' or 'import datetime' or 'import pytest'
        if ("from collections import deque" in line or \
            "import datetime" in line or \
            "import pytest" in line):
            # Check if next line is already the import to avoid double import if script is re-run
            if not (current_line_index + 1 < len(lines) and "from sortedcontainers import SortedList" in lines[current_line_index+1]):
                output_lines.append(line) # current import line
                output_lines.append("from sortedcontainers import SortedList\n")
                sorted_list_imported = True
                current_line_index += 1
                continue
            else: # It is already imported in the next line
                sorted_list_imported = True


    # Update isinstance check in test_initialization
    if "def test_initialization(" in line:
        output_lines.append(line) # Add the test definition line

        processed_lines_in_block = 0
        for i in range(current_line_index + 1, len(lines)):
            test_line = lines[i]
            if "isinstance(organizer_instance._RemoteDataOrganizer__data, deque)" in test_line:
                updated_line = test_line.replace(
                    "isinstance(organizer_instance._RemoteDataOrganizer__data, deque)",
                    "isinstance(organizer_instance._RemoteDataOrganizer__data, SortedList)"
                )
                output_lines.append(updated_line)
                isinstance_updated = True
            else:
                output_lines.append(test_line)

            processed_lines_in_block += 1
            # Heuristic to break if we are likely outside the test method block
            # (e.g., another 'def test_' starts or indentation changes significantly)
            # This is simplified; a proper AST parser would be more robust for complex structures.
            if not test_line.startswith(" ") and not test_line.strip() == "" and "def test_" in test_line: # next test
                 break
            if not test_line.startswith("    ") and not test_line.strip() == "" and not test_line.startswith("def"): # dedented line not a new test
                 break

        current_line_index += processed_lines_in_block
        continue

    output_lines.append(line)
    current_line_index += 1

# If SortedList import was still not added (e.g. file was empty or had no other imports, or no suitable anchor)
if not sorted_list_imported and lines: # ensure lines is not empty
    # Prepend to the very beginning if all else failed and not already done by initial check
    # This check is important if the file had no recognized anchor points
    already_prepended = False
    if output_lines and "from sortedcontainers import SortedList" in output_lines[0]:
        already_prepended = True

    if not already_prepended:
        # Check if the first line of original content was an import, to decide where to put it
        first_line_is_import = lines[0].startswith("import ") or lines[0].startswith("from ")
        insertion_point = 0
        # Try to put it after any initial comments or shebangs
        for idx, ol_line in enumerate(output_lines):
            if not ol_line.startswith("#") and not ol_line.startswith("!"):
                insertion_point = idx
                break

        # If all lines are comments, it will insert at the end of comments.
        # If no comments, at the beginning.
        if not any(ol.startswith("import ") or ol.startswith("from ") for ol in output_lines): # if no other imports yet in output_lines
             output_lines.insert(insertion_point, "from sortedcontainers import SortedList\n")
        # If there ARE other imports but we just didn't find a specific anchor, add it near the first actual import.
        # This part is tricky; the earlier logic should ideally handle it. This is a fallback.
        # The current logic for adding import is tied to specific lines, this fallback might be too simple.
        # For now, rely on the primary import logic. If it fails, manual check is better.
        # print("Debug: Fallback import logic considered.")


with open(file_path, "w") as f:
    f.writelines(output_lines)

if isinstance_updated and sorted_list_imported:
    print(f"Successfully updated isinstance in test_initialization and ensured SortedList import in {file_path}")
else:
    if not isinstance_updated:
        print(f"Warning: Could not find or update the isinstance check in test_initialization in {file_path}")
    if not sorted_list_imported:
        print(f"Warning: Failed to add SortedList import to {file_path}. This might happen if no anchor imports (datetime, pytest, collections.deque) were found or if it was already imported in an unexpected place.")
