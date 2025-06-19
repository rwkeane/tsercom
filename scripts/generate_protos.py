import ast
import re
import subprocess
import os
import sys
from pathlib import Path
from typing import Dict, Iterable

import grpc

kGeneratedDir = "generated"


def generate_proto_file(
    package_dir: Path, proto_file_path: str, import_paths: Iterable[str]
) -> None:
    """Generates Python files for a single .proto file.

    Args:
        package_dir: The root directory of the package.
        proto_file_path: The path to the .proto file, relative to package_dir.
        import_paths: An iterable of import paths relative to package_dir.
    """
    # Construct the absolute path to the .proto file.
    absolute_proto_path = Path.joinpath(package_dir, proto_file_path)
    output_dir = make_versioned_output_dir(absolute_proto_path.parent)

    # Construct the protoc command.
    command = [
        sys.executable,  # Path to the Python interpreter.
        "-m",  # Run library module as a script.
        "grpc_tools.protoc",  # The gRPC tools protoc module.
        *[
            f"-I{package_dir / path}" for path in import_paths
        ],  # Include paths for proto imports.
        f"--python_out={output_dir}",  # Output directory for generated Python code.
        f"--mypy_out={output_dir}",  # Output directory for generated mypy stubs.
        f"--grpc_python_out={output_dir}",  # Output directory for gRPC stubs.
        str(absolute_proto_path),  # The .proto file to compile.
    ]
    print(f"Running command: {' '.join(command)}")

    # Ensure plugins are in PATH
    env = os.environ.copy()
    local_bin_path = os.path.expanduser("~/.local/bin")
    if local_bin_path not in env["PATH"]:
        env["PATH"] = f"{local_bin_path}:{env['PATH']}"

    result = subprocess.run(command, capture_output=True, text=True, env=env)

    if result.returncode != 0:
        print(f"Error compiling {proto_file_path}:")
        print(result.stderr)
        raise RuntimeError(
            f"protoc compilation failed for {proto_file_path}:\n{result.stderr}"
        )
    else:
        print(f"Successfully compiled {proto_file_path}")

    name = absolute_proto_path.name.split(".")[0]

    generate_init(package_dir, proto_file_path, output_dir.parent)

    # --- Get generated file paths (relative to output_dir) ---
    pb2_file = Path.joinpath(output_dir, f"{name}_pb2.py")
    pyi_file = Path.joinpath(output_dir, f"{name}_pb2.pyi")

    # String substitution for _pb2.py and _pb2.pyi files
    modify_generated_file(pb2_file)
    modify_generated_file(pyi_file)
    # Also modify the _pb2_grpc.py file
    pb2_grpc_file = Path.joinpath(output_dir, f"{name}_pb2_grpc.py")
    modify_generated_file(pb2_grpc_file)


def make_versioned_output_dir(base_dir: Path) -> Path:
    """Creates a versioned output directory for generated protobuf files.

    The version is derived from the currently installed grpcio-tools package.
    This ensures that generated code is compatible with the gRPC runtime.

    Args:
        base_dir: The base directory where the 'generated/vX_Y' structure
            will be created.

    Returns:
        The path to the versioned output directory (e.g., base_dir/generated/v1_62).

    Raises:
        RuntimeError: If the grpcio-tools version cannot be determined.
    """
    try:
        version = grpc.__version__
        major_minor_version = ".".join(version.split(".")[:2])
        version_string = f"v{major_minor_version.replace('.', '_')}"
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        AttributeError,
    ) as e:
        raise RuntimeError(
            f"Could not determine grpcio-tools version. Is it installed? Error: {e}"
        ) from e

    print(f"Generating protos for grpcio-tools version: {version_string}")

    generated_dir = Path.joinpath(base_dir, kGeneratedDir)
    os.makedirs(generated_dir, exist_ok=True)
    versioned_output_base = Path.joinpath(generated_dir, f"{version_string}")
    os.makedirs(versioned_output_base, exist_ok=True)

    return versioned_output_base


def modify_generated_file(file_path: Path) -> None:
    """Performs string replacement in the generated file."""
    updates: Dict[str, str] = {
        "import caller_id_pb2\n": "import tsercom.caller_id.proto as caller_id_pb2\n",
        "import time_pb2\n": "import tsercom.timesync.common.proto as time_pb2\n",
        "import common_pb2\n": "import tsercom.rpc.proto as common_pb2\n",
        "import caller_id_pb2 ": "import tsercom.caller_id.proto ",
        "import time_pb2 ": "import tsercom.timesync.common.proto ",
        "import common_pb2 ": "import tsercom.rpc.proto ",
        # Ensure e2e_test_service_pb2 is imported relatively from e2e_test_service_pb2_grpc.py
    }
    # Additional specific replacements can be added here
    # tensor_ops_pb2.py is no longer generated, so its specific fix is removed.

    if not file_path.exists():
        print(f"Warning: Generated file not found, cannot modify: {file_path}")
        return
    try:
        with open(file_path, "r+") as f:
            content = f.read()
            for original, replacement in updates.items():
                content = content.replace(original, replacement)
            # Special handling for e2e_test_service_pb2_grpc.py's import of e2e_test_service_pb2
            if file_path.name == "e2e_test_service_pb2_grpc.py":
                try:
                    path_parts = list(
                        file_path.parent.parts
                    )  # e.g. ('app', 'tsercom', 'test', 'proto', 'generated', 'v1_62')
                    tsercom_index = -1
                    # Find the 'tsercom' directory in the path parts
                    for i_part_idx, part_name in enumerate(path_parts):
                        if part_name == "tsercom":
                            tsercom_index = i_part_idx
                            break

                    if tsercom_index != -1:
                        # Construct the module path from 'tsercom' up to the directory containing the file
                        absolute_module_prefix = ".".join(
                            path_parts[tsercom_index:]
                        )

                        original_import_line = "import e2e_test_service_pb2 as e2e__test__service__pb2"
                        correct_module_name = "e2e_test_service_pb2"
                        correct_alias = "e2e__test__service__pb2"

                        new_import_line = f"from {absolute_module_prefix} import {correct_module_name} as {correct_alias}"

                        if original_import_line in content:
                            content = content.replace(
                                original_import_line, new_import_line
                            )
                            print(
                                f"Applied absolute import fix for {file_path.name}: replaced '{original_import_line}' with '{new_import_line}'"
                            )
                        else:
                            print(
                                f"Warning: Original import line '{original_import_line}' not found as expected in {file_path}"
                            )
                    else:
                        print(
                            f"Warning: 'tsercom' directory not found in path for {file_path}, cannot construct absolute import for e2e_test_service sibling."
                        )
                except (
                    ValueError
                ):  # Handles potential .index() error if "tsercom" is not found
                    print(
                        f"Warning: 'tsercom' not found in path for {file_path} (ValueError), cannot construct absolute import for e2e_test_service sibling."
                    )
                except Exception as e_abs:
                    print(
                        f"Warning: Could not form absolute import for e2e_test_service sibling in {file_path}: {e_abs}"
                    )

            # After all replacements, including the special one above
            # The original script intends to insert this python code block *before* f.seek(0)
            # but *after* the existing loop for general updates.
            # So, the target for insertion is just before `f.seek(0)`

            f.seek(0)
            f.write(content)
            f.truncate()
    except Exception as e:
        print(f"Error modifying file {file_path}: {e}")
        raise


def _update_pyproject_version_range(versioned_dirs: list[str]) -> None:
    """Updates pyproject.toml with the detected gRPC version range."""
    if not versioned_dirs:
        print(
            "Warning: No versioned directories found. Skipping pyproject.toml update."
        )
        return

    versions = []
    for v_str in versioned_dirs:
        match = re.match(r"v(\d+)_(\d+)", v_str)
        if match:
            major, minor = int(match.group(1)), int(match.group(2))
            versions.append((major, minor))
        else:
            match_single = re.match(r"v(\d+)", v_str)
            if match_single:
                major = int(match_single.group(1))
                versions.append((major, 0))
            else:
                print(f"Warning: Could not parse version string: {v_str}")

    if not versions:
        print(
            "Warning: No valid versions parsed. Skipping pyproject.toml update."
        )
        return

    min_version_tuple = min(versions)
    max_version_tuple = max(versions)
    min_version_str = f"{min_version_tuple[0]}.{min_version_tuple[1]}.0"
    next_minor_version_str = (
        f"{max_version_tuple[0]}.{max_version_tuple[1] + 1}.0"
    )

    try:
        script_dir = Path(__file__).parent.resolve()
        pyproject_path = script_dir.parent / "pyproject.toml"
        if not pyproject_path.exists():
            pyproject_path = Path("pyproject.toml")

        with open(pyproject_path, "r") as f:
            pyproject_content = f.read()

        patterns_to_update = [
            r'(grpcio\s*=\s*")[^"]*(")',
            r'(grpcio-tools\s*=\s*")[^"]*(")',
            r'(grpcio-status\s*=\s*")[^"]*(")',
            r'(grpcio-health-checking\s*=\s*")[^"]*(")',
        ]
        replacement_string = (
            f"\1>={min_version_str}, <{next_minor_version_str}\2"
        )
        for pattern in patterns_to_update:
            pyproject_content = re.sub(
                pattern, replacement_string, pyproject_content
            )

        with open(pyproject_path, "w") as f:
            f.write(pyproject_content)
        print(
            f"Successfully updated {pyproject_path} with gRPC versions: >={min_version_str}, <{next_minor_version_str}"
        )

    except FileNotFoundError:
        print(
            f"Error: pyproject.toml not found at expected locations: {script_dir.parent / 'pyproject.toml'} or {Path('pyproject.toml')}"
        )
    except Exception as e:
        print(f"Error updating pyproject.toml: {e}")


def get_public_symbols_from_file(filepath: Path) -> list[str]:
    """
    Gets a list of public class and function names defined in a Python file.
    """
    if not filepath.exists():
        return []
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            tree = ast.parse(file.read(), filename=str(filepath))
    except SyntaxError:
        print(f"Warning: Syntax error in {filepath}. Returning empty list.")
        return []
    except FileNotFoundError:
        print(
            f"Warning: File not found for get_public_symbols_from_file: {filepath}"
        )
        return []

    symbols = []
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            if not node.name.startswith("_"):
                symbols.append(node.name)
    return symbols


def generate_init(
    package_dir: Path, proto_path: str, generated_path: Path
) -> None:
    """Generates an __init__.py file for the generated protobuf package."""

    def _parse_ver(v_str: str) -> tuple[int, int]:
        match = re.match(r"v(\d+)_(\d+)", v_str)
        if match:
            return int(match.group(1)), int(match.group(2))
        match_single = re.match(r"v(\d+)", v_str)
        if match_single:
            return int(match_single.group(1)), 0
        return (0, 0)

    init_file_content = """
import grpc
import subprocess
from typing import TYPE_CHECKING

if not TYPE_CHECKING:
    try:
        version = grpc.__version__
        major_minor_version = ".".join(version.split(".")[:2])
    except (AttributeError, subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Warning: Failed to get grpc.__version__ ({e}), defaulting to a common version for proto loading.")
        major_minor_version = "1.62" # Fallback version

    version_string = f"v{major_minor_version.replace('.', '_')}"

    if False:
        pass
"""
    name = Path(proto_path).name.split(".")[0]
    versioned_dirs_data = []
    base_package = (
        generated_path.relative_to(package_dir).as_posix().replace("/", ".")
    )

    for item in generated_path.iterdir():
        if item.is_dir() and item.name.startswith("v"):
            pb2_pyi_path = item.joinpath(f"{name}_pb2.pyi")
            message_symbols = get_public_symbols_from_file(pb2_pyi_path)
            pb2_grpc_py_path = item.joinpath(f"{name}_pb2_grpc.py")
            grpc_symbols = get_public_symbols_from_file(pb2_grpc_py_path)
            if message_symbols or grpc_symbols:
                versioned_dirs_data.append(
                    (item.name, message_symbols, grpc_symbols)
                )

    if not versioned_dirs_data:
        print(
            f"Warning: No versioned symbols found for {name} in {generated_path}. __init__.py may be incomplete."
        )

    _update_pyproject_version_range(
        [vd[0] for vd in versioned_dirs_data if vd[0]]
    )

    if versioned_dirs_data:
        runtime_sorted_dirs = sorted(
            versioned_dirs_data, key=lambda x: _parse_ver(x[0]), reverse=True
        )
        for version_dir_name, msg_symbols, grpc_syms in runtime_sorted_dirs:
            current_ver = version_dir_name[1:]
            init_file_content += (
                f'\n    elif version_string == "v{current_ver}":'
            )
            if msg_symbols:
                init_file_content += f"\n        from tsercom.{base_package}.{version_dir_name}.{name}_pb2 import {', '.join(msg_symbols)}"
            if grpc_syms:
                init_file_content += f"\n        from tsercom.{base_package}.{version_dir_name}.{name}_pb2_grpc import {', '.join(grpc_syms)}"

        # Pre-calculate available_versions string for the error message
        # _parse_ver is available in this scope (generate_init)
        # versioned_dirs_data is also available here.
        available_versions_list = sorted(
            [vd[0] for vd in versioned_dirs_data], key=_parse_ver
        )
        available_versions_str = str(available_versions_list)

        # name_in_error_message refers to the 'name' variable from the outer scope of generate_init
        # version and version_string are defined inside the __init__.py, so they can be used directly.
        init_file_content += f"""
    else:
        # The 'name' variable for the error message is '{name}'
        # The 'available_versions' for the error message is {available_versions_str}
        raise ImportError(
            f"Error: No code for version {{version}}, name '{name}', available_versions {available_versions_str}, version_string {{version_string}}."
        )
"""
    else:  # No versioned_dirs_data at all
        init_file_content += f"""
    else:
        raise ImportError(
            f"Error: No versioned protobuf code found at all for proto '{name}'. Check generation script and paths."
        )
"""  # Closes the runtime 'else' block for ImportError

    # Now, handle the TYPE_CHECKING block
    init_file_content += "\nelse: # When TYPE_CHECKING\n"  # Add the static "else: # When TYPE_CHECKING" line

    if not versioned_dirs_data:
        init_file_content += "    pass # No versioned data for TYPE_CHECKING\n"
    else:
        # This logic is now Python executed by generate_init, appending strings
        versioned_dirs_data.sort(
            key=lambda x: _parse_ver(x[0])
        )  # Ensure it's sorted for "latest"
        latest_ver_name, latest_msg_syms, latest_grpc_syms = (
            versioned_dirs_data[-1]
        )

        type_checking_imports = []
        if latest_msg_syms:
            for sym in latest_msg_syms:
                type_checking_imports.append(
                    f"    from tsercom.{base_package}.{latest_ver_name}.{name}_pb2 import {sym} as {sym}"
                )
        if latest_grpc_syms:
            for sym in latest_grpc_syms:
                type_checking_imports.append(
                    f"    from tsercom.{base_package}.{latest_ver_name}.{name}_pb2_grpc import {sym} as {sym}"
                )

        if type_checking_imports:  # Only add if there are any imports to add
            init_file_content += "\n".join(type_checking_imports) + "\n"
        elif (
            not latest_msg_syms and not latest_grpc_syms
        ):  # If both are empty, still add a pass
            init_file_content += (
                "    pass # No symbols for TYPE_CHECKING for latest version\n"
            )

    init_py_path = generated_path.parent.joinpath("__init__.py")
    try:
        with open(init_py_path, "w") as f:
            f.write(init_file_content)
        print(f"Generated __init__.py at {init_py_path}")
    except IOError as e:
        print(f"Error writing __init__.py at {init_py_path}: {e}")
        raise


def generate_protos(project_root: Path) -> None:
    """Generates Python files from .proto files for the entire project."""
    assert isinstance(project_root, Path)
    package_dir = Path.joinpath(project_root, "tsercom")

    generate_proto_file(
        package_dir,
        "timesync/common/proto/time.proto",
        ["timesync/common/proto"],
    )
    generate_proto_file(
        package_dir, "caller_id/proto/caller_id.proto", ["caller_id/proto"]
    )
    generate_proto_file(
        package_dir,
        "rpc/proto/common.proto",
        ["caller_id/proto", "rpc/proto", "timesync/common/proto"],
    )
    generate_proto_file(
        package_dir,
        "tensor/proto/tensor.proto",  # This now contains all tensor-related messages
        ["tensor/proto", "timesync/common/proto"],
    )
    generate_proto_file(
        package_dir, "test/proto/e2e_test_service.proto", ["test/proto"]
    )


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    generate_protos(project_root)
