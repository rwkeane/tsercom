import toml

pyproject_path = "pyproject.toml"

try:
    # Read the pyproject.toml file
    with open(pyproject_path, "r", encoding="utf-8") as f:
        data = toml.load(f)

    # Check if the section and key exist
    if (
        "tool" in data
        and "pylint" in data["tool"]
        and "MESSAGES CONTROL" in data["tool"]["pylint"]
        and "disable" in data["tool"]["pylint"]["MESSAGES CONTROL"]
    ):

        disable_list = data["tool"]["pylint"]["MESSAGES CONTROL"]["disable"]

        # Items to remove
        items_to_remove = ["W0221", "W0238"]

        # Filter out items to remove
        new_disable_list = [
            item for item in disable_list if item not in items_to_remove
        ]

        if len(new_disable_list) < len(disable_list):
            data["tool"]["pylint"]["MESSAGES CONTROL"][
                "disable"
            ] = new_disable_list

            # Write the modified content back to pyproject.toml
            with open(pyproject_path, "w", encoding="utf-8") as f:
                toml.dump(data, f)
            print(
                f"Successfully removed items {items_to_remove} (if they existed) from Pylint disable list in {pyproject_path}"
            )
            # Print the new list for verification
            print(f"New disable list: {new_disable_list}")
        else:
            print(
                f"Items {items_to_remove} not found in Pylint disable list. No changes made."
            )
            print(f"Current disable list: {disable_list}")

    else:
        print(
            "Error: Pylint disable list not found in the expected structure in pyproject.toml"
        )
        if "tool" not in data:
            print("Missing 'tool' section.")
        elif "pylint" not in data.get("tool", {}):
            print("Missing 'pylint' in 'tool' section.")
        elif "MESSAGES CONTROL" not in data.get("tool", {}).get("pylint", {}):
            print("Missing 'MESSAGES CONTROL' in 'tool.pylint' section.")
        elif "disable" not in data.get("tool", {}).get("pylint", {}).get(
            "MESSAGES CONTROL", {}
        ):
            print(
                "Missing 'disable' in 'tool.pylint.MESSAGES CONTROL' section."
            )


except FileNotFoundError:
    print(f"Error: {pyproject_path} not found.")
except Exception as e:
    print(f"An error occurred: {e}")
