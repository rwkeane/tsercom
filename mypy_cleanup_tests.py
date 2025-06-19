import re
import ast

test_file_path = "tsercom/data/remote_data_organizer_unittest.py"

# Read the file content first
with open(test_file_path, "r") as f:
    source_code = f.read()

# Use AST to find function definitions and their line numbers accurately
tree = ast.parse(source_code)
lines = source_code.splitlines()

# Store line numbers of function definitions to modify
# Also store lines that need attr-defined ignore
lines_to_add_return_none = {} # lineno -> original_def_line
lines_needing_attr_defined_ignore = set() # lineno (0-indexed)

# Patterns for specific ignores based on Mypy error messages (if known)
internal_attribute_pattern = re.compile(r"_RemoteDataOrganizer__([a-zA-Z0-9_]+)")
# Pattern for specific mock assignments that cause [misc] or [assignment]
problematic_mock_assignments = [
    "mock_is_running_tracker.start.side_effect = RuntimeError",
    "mock_is_running_tracker.stop.side_effect = RuntimeError",
    "mock_pool.submit = mocker.MagicMock(side_effect=immediate_submit)",
    "organizer_custom_type._RemoteDataOrganizer__is_running = mocker.MagicMock"
]

for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef):
        if node.name.startswith("test_"):
            if node.returns is None:
                def_line_str = lines[node.lineno - 1]
                if not def_line_str.strip().endswith("-> None:"):
                    lines_to_add_return_none[node.lineno - 1] = def_line_str
        # For immediate_submit inside a fixture
        elif node.name == "immediate_submit" and node.returns is None:
            def_line_str = lines[node.lineno -1]
            if not def_line_str.strip().endswith("-> Any:"): # It returns Any
                 lines_to_add_return_none[node.lineno -1] = def_line_str # Will be modified to -> Any

    elif isinstance(node, ast.Attribute):
        # Check if the attribute name matches the internal pattern
        if internal_attribute_pattern.search(node.attr):
            # Check if the value being accessed is a Name node (e.g., 'organizer.attr')
            # or a Subscript node (e.g., 'organizer.data[0].attr')
            current_node = node.value
            is_on_organizer_instance = False
            while isinstance(current_node, ast.Attribute) or isinstance(current_node, ast.Subscript):
                if isinstance(current_node, ast.Attribute):
                    current_node = current_node.value
                elif isinstance(current_node, ast.Subscript):
                    current_node = current_node.value # Check the object being subscripted

            if isinstance(current_node, ast.Name) and \
               current_node.id in ["organizer", "organizer_no_client", "organizer_custom_type", "self"]:
                lines_needing_attr_defined_ignore.add(node.lineno - 1)


# Reconstruct the file content with modifications
output_lines = []
for i, line_str in enumerate(lines):
    modified_line = line_str.rstrip()
    original_line_for_debug = modified_line # For printing if needed

    if i in lines_to_add_return_none:
        original_def_line = lines_to_add_return_none[i]
        # Add "-> None:" or "-> Any:" before the colon
        return_type = "-> None:" if not "immediate_submit" in original_def_line else "-> Any:"
        if ":" in modified_line and not "->" in modified_line:
            # Ensure it's really the def line we want to change
            if original_def_line.strip().startswith("def "):
                 modified_line = modified_line.replace(":", f" {return_type}", 1)

    if i in lines_needing_attr_defined_ignore:
        if "# type: ignore[attr-defined]" not in modified_line:
            modified_line += "  # type: ignore[attr-defined]"

    for assignment_heuristic in problematic_mock_assignments:
        if assignment_heuristic in modified_line:
            if "# type: ignore" not in modified_line:
                # More specific ignores are better if possible, e.g. [assignment] or [misc]
                # For side_effect, [misc] is common. For direct mock assignment, [assignment]
                if ".side_effect =" in modified_line:
                    modified_line += "  # type: ignore[misc]"
                elif "= mocker.MagicMock" in modified_line:
                     modified_line += "  # type: ignore[assignment]"
                else: # Fallback
                    modified_line += "  # type: ignore[misc]"
            break

    output_lines.append(modified_line + "\n")

with open(test_file_path, "w") as f:
    f.writelines(output_lines)

print(f"Attempted to add Mypy compliance type hints and ignores to {test_file_path}.")
