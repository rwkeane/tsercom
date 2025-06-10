import sys
import toml # Ensure toml is available

def main_add_ruff_exclude():
    pyproject_file = "pyproject.toml"
    new_exclude_line_content = "**/*_component_test.py"
    # target_line_content = '"**/*_unittest.py", # Clearer exclusion' # This was the old target

    try:
        with open(pyproject_file, "r") as f:
            # data = toml.load(f) # Read with toml first to ensure validity
            # Re-reading raw lines for insertion is not ideal if toml sorts keys.
            # It's better to modify the toml object then dump it.
            pass # Let's use the toml object modification approach
    except FileNotFoundError:
        print(f"Error: {pyproject_file} not found.", file=sys.stderr)
        return 1
    except Exception as e: # Catch other read errors too
        print(f"Error reading {pyproject_file} initially: {e}", file=sys.stderr)
        return 1

    try:
        data = toml.load(pyproject_file) # Load the TOML data
    except Exception as e:
        print(f"Error parsing TOML file {pyproject_file}: {e}", file=sys.stderr)
        return 1

    # Navigate to the exclude list or create if necessary
    try:
        tool_ruff_lint = data.setdefault('tool', {}).setdefault('ruff', {}).setdefault('lint', {})
        exclude_list = tool_ruff_lint.setdefault('exclude', [])
    except Exception as e: # Broad exception if setdefault chain fails for some reason
        print(f"Error accessing/creating [tool.ruff.lint.exclude] in TOML data: {e}", file=sys.stderr)
        return 1


    if not isinstance(exclude_list, list):
        print(f"Error: 'tool.ruff.lint.exclude' is not a list in {pyproject_file}. Current value: {exclude_list}", file=sys.stderr)
        # Attempt to fix if it's a common incorrect type, e.g. string
        if isinstance(exclude_list, str):
            print("Attempting to convert string exclude to list.", file=sys.stderr)
            data['tool']['ruff']['lint']['exclude'] = [exclude_list]
            exclude_list = data['tool']['ruff']['lint']['exclude']
        else:
            return 1 # Unhandled type

    # Add new exclusion if not present
    if new_exclude_line_content not in exclude_list:
        exclude_list.append(new_exclude_line_content)
        print(f"Added Ruff exclusion: {new_exclude_line_content}")
    else:
        print(f"Ruff exclusion already exists: {new_exclude_line_content}")

    # Write the updated data back to pyproject.toml
    try:
        with open(pyproject_file, "w") as f:
            toml.dump(data, f)
        print("Minimal Ruff exclusion added to pyproject.toml using TOML library.")
    except Exception as e:
        print(f"Error writing {pyproject_file} using TOML library: {e}", file=sys.stderr)
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main_add_ruff_exclude())
