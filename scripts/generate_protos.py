import ast
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

    # String substitution.
    modify_generated_file(pb2_file)
    modify_generated_file(pyi_file)


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
    # --- Get current grpcio-tools version (MAJOR.MINOR only) ---
    try:
        version = grpc.__version__
        major_minor_version = ".".join(
            version.split(".")[:2]
        )  # Extract major.minor
        version_string = (
            f"v{major_minor_version.replace('.', '_')}"  # e.g., v1_62
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise RuntimeError(
            f"Could not determine grpcio-tools version. Is it installed? Error: {e}"
        ) from e

    print(f"Generating protos for grpcio-tools version: {version_string}")

    # --- Create Versioned Output Directory ---
    generated_dir = Path.joinpath(base_dir, kGeneratedDir)
    os.makedirs(generated_dir, exist_ok=True)
    versioned_output_base = Path.joinpath(generated_dir, f"{version_string}")
    os.makedirs(versioned_output_base, exist_ok=True)

    return versioned_output_base


def modify_generated_file(file_path: Path) -> None:
    """Performs string replacement in the generated file.

    This function modifies the import statements in the generated protobuf files
    to use fully qualified paths. This is a workaround for how protoc generates
    imports.

    Args:
        file_path: The path to the generated file to modify.

    Raises:
        FileNotFoundError: If the specified file_path does not exist.
        Exception: For other file I/O errors.
    """
    # TODO: Make this less hacky.
    # Dictionary of import statements to find and replace.
    # This is necessary because protoc generates relative imports, but we need
    # fully qualified imports for our project structure.
    updates: Dict[str, str] = {
        "import caller_id_pb2\n": "import tsercom.caller_id.proto as caller_id_pb2\n",
        "import time_pb2\n": "import tsercom.timesync.common.proto as time_pb2\n",
        "import common_pb2\n": "import tsercom.rpc.proto as common_pb2\n",
        "import caller_id_pb2 ": "import tsercom.caller_id.proto ",  # For typing imports
        "import time_pb2 ": "import tsercom.timesync.common.proto ",  # For typing imports
        "import common_pb2 ": "import tsercom.rpc.proto ",  # For typing imports
    }

    try:
        with open(file_path, "r+") as f:
            content = f.read()
            for original, replacement in updates.items():
                content = content.replace(original, replacement)
            f.seek(0)
            f.write(content)
            f.truncate()
    except FileNotFoundError:
        print(f"Warning: Generated file not found: {file_path}")
    except Exception as e:
        print(f"Error modifying file {file_path}: {e}")
        raise


def generate_init(
    package_dir: Path, proto_path: str, generated_path: Path
) -> None:
    """Generates an __init__.py file for the generated protobuf package.

    This __init__.py file dynamically imports the correct version of the
    protobuf modules based on the installed grpcio-tools version. It also
    provides type hints for static analysis when TYPE_CHECKING is true.

    Args:
        package_dir: The root directory of the tsercom package.
        proto_path: The path to the original .proto file (used to derive the module name).
        generated_path: The path to the 'generated' directory where versioned
            protobuf files are located (e.g., tsercom/rpc/proto/generated).
    """
    # Initial content for the __init__.py file.
    # This part handles runtime imports.
    init_file_content = """
import grpc
import subprocess
from typing import TYPE_CHECKING

if not TYPE_CHECKING:
    try:
        version = grpc.__version__
        major_minor_version = ".".join(version.split(".")[:2])
    except (AttributeError, subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Warning: Failed to get grpc.__version__ ({e}), defaulting to 1.71 for proto loading.")
        major_minor_version = "1.71" # Default to a version used in TYPE_CHECKING

    version_string = f"v{major_minor_version.replace('.', '_')}"

    if False:
        pass
"""
    name = Path(proto_path).name.split(".")[0]
    versioned_dirs = []
    base_package = (
        generated_path.relative_to(package_dir)
        .__str__()
        .replace("/", ".")
        .replace("\\", ".")
    )
    # Iterate through versioned directories (e.g., v1_62, v1_63)
    # to find all generated _pb2.pyi files and extract class names.
    for item in generated_path.iterdir():
        if item.is_dir() and item.name.startswith("v"):
            file_path = item.joinpath(f"{name}_pb2.pyi")
            classes = get_classes_from_file(file_path)
            versioned_dirs.append((item.name, classes))

    # Add runtime import statements for each version found.
    for versioned_dir_name, classes in versioned_dirs:
        current_version = versioned_dir_name[1:]  # Remove the 'v' prefix
        init_file_content += f"""
    elif version_string == "v{current_version}":
        from tsercom.{base_package}.{versioned_dir_name}.{name}_pb2 import {", ".join(classes)}
"""
    # Add a fallback else clause for runtime if no matching version is found.
    init_file_content += """
    else:
        raise ImportError(
            f"No pre-generated protobuf code found for grpcio version: {version}.\\n"
            f"Please generate the code for your grpcio version by running 'python scripts/build.py'."
        )

# This part handles type hinting for static analysis (e.g., mypy).
# It imports symbols from the latest available version.
else: # When TYPE_CHECKING
"""
    # Get the latest version for type checking.
    # Assumes versioned_dirs is sorted or the last one is the newest.
    # TODO: Consider explicitly sorting by version if not guaranteed.
    versioned_dir_name, classes = versioned_dirs[-1]
    current_version = versioned_dir_name[1:]
    for clazz in classes:
        init_file_content += f"""
    from tsercom.{base_package}.{versioned_dir_name}.{name}_pb2 import {clazz} as {clazz}"""

    # Write the generated __init__.py content.
    # The __init__.py is placed one level up from the 'generated' directory,
    # e.g., tsercom/rpc/proto/__init__.py
    f = open(generated_path.parent.joinpath("__init__.py"), "w")
    f.write(init_file_content)
    f.close()


def get_classes_from_file(filepath: Path) -> list[str]:
    """
    Gets a list of class names defined in a Python file (.py or .pyi).

    Args:
        filepath: The path to the Python source file.

    Returns:
        A list of strings (class names). Returns an empty list if the
        file exists but has syntax errors or contains no class definitions.

    Raises:
        FileNotFoundError: If the specified filepath does not exist.
    """

    try:
        with open(filepath, "r", encoding="utf-8") as file:
            tree = ast.parse(file.read())
    except SyntaxError:
        print(f"Warning: Syntax error in {filepath}. Returning empty list.")
        return []  # Or raise, depending on your needs

    class_names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            class_names.append(node.name)
    return class_names


def generate_protos(project_root: Path) -> None:
    """Generates Python files from .proto files for the entire project.

    This function orchestrates the protobuf generation process for all
    defined .proto files in the project.

    Args:
        project_root: The root directory of the project.
    """
    assert isinstance(project_root, Path)
    package_dir = Path.joinpath(project_root, "tsercom")

    # --- Define your proto files and their import paths ---
    generate_proto_file(
        package_dir,
        "timesync/common/proto/time.proto",
        ["timesync/common/proto"],
    )
    generate_proto_file(
        package_dir,
        "caller_id/proto/caller_id.proto",
        ["caller_id/proto"],
    )
    generate_proto_file(
        package_dir,
        "rpc/proto/common.proto",
        ["caller_id/proto", "rpc/proto", "timesync/common/proto"],
    )


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    generate_protos(project_root)
