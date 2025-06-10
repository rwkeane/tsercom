import toml
import sys # For stderr

def main():
    try:
        with open("pyproject.toml", "r") as f:
            data = toml.load(f)
    except toml.TomlDecodeError as e:
        print(f"Error decoding TOML: {e}", file=sys.stderr)
        return 1 # Indicate failure
    except FileNotFoundError:
        print("pyproject.toml not found", file=sys.stderr)
        return 1 # Indicate failure

    if 'tool' not in data:
        data['tool'] = {}
    if 'ruff' not in data['tool']:
        data['tool']['ruff'] = {}
    if 'lint' not in data['tool']['ruff']:
        data['tool']['ruff']['lint'] = {}
    if 'exclude' not in data['tool']['ruff']['lint']:
        data['tool']['ruff']['lint']['exclude'] = []

    # Ensure it's a list
    if not isinstance(data['tool']['ruff']['lint']['exclude'], list):
        print("Warning: tool.ruff.lint.exclude is not a list. Converting.", file=sys.stderr)
        current_exclude = data['tool']['ruff']['lint']['exclude']
        if isinstance(current_exclude, str):
            data['tool']['ruff']['lint']['exclude'] = [current_exclude]
        else:
            print(f"Warning: tool.ruff.lint.exclude was an unexpected type: {type(current_exclude)}. Re-initializing as empty list.", file=sys.stderr)
            data['tool']['ruff']['lint']['exclude'] = []


    new_pattern = "**/*_component_test.py"

    if new_pattern not in data['tool']['ruff']['lint']['exclude']:
        data['tool']['ruff']['lint']['exclude'].append(new_pattern)
        print(f"Added '{new_pattern}' to tool.ruff.lint.exclude")
    else:
        print(f"'{new_pattern}' already in tool.ruff.lint.exclude")

    try:
        with open("pyproject.toml", "w") as f:
            toml.dump(data, f)
        print("pyproject.toml updated successfully.")
    except Exception as e:
        print(f"Error writing updated pyproject.toml: {e}", file=sys.stderr)
        return 1 # Indicate failure

    return 0 # Indicate success

if __name__ == "__main__":
    sys.exit(main()) # Propagate return code for bash script
