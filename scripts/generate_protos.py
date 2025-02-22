import ast
from importlib.metadata import version as lib_version
import subprocess
import os
import sys
import re
from pathlib import Path
from typing import Dict, Iterable, List

import grpc

kGeneratedDir = "generated"


def generate_proto_file(package_dir : Path,
                        proto_file_path: str,
                        import_paths: Iterable[str]):
    """Generates Python files for a single .proto file."""

    absolute_proto_path = Path.joinpath(package_dir, proto_file_path)
    output_dir = make_versioned_output_dir(absolute_proto_path.parent)

    command = [
        sys.executable,  # python executable path
        "-m",
        "grpc_tools.protoc",
        *[f"-I{package_dir / path}" for path in import_paths],
        f"--python_out={output_dir}",
        f"--mypy_out={output_dir}",
        str(absolute_proto_path),
    ]
    print(f"Running command: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error compiling {proto_file_path}:")
        print(result.stderr)
        raise RuntimeError(
                f"protoc compilation failed for {proto_file_path}:\n{result.stderr}")
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

def make_versioned_output_dir(base_dir : Path):
    # --- Get current grpcio-tools version (MAJOR.MINOR only) ---
    try:
        version = grpc.__version__
        major_minor_version = ".".join(version.split(".")[:2])  # Extract major.minor
        version_string = f"v{major_minor_version.replace('.', '_')}" # e.g., v1_62
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

def modify_generated_file(file_path: Path):
    """Performs string replacement in the generated file."""

    # TODO: Make this less hacky.
    updates: Dict[str, str] = {
        "import caller_id_pb2\n": "import tsercom.caller_id.proto as caller_id_pb2\n",
        "import time_pb2\n": "import tsercom.timesync.common.proto as time_pb2\n",
        "import common_pb2\n": "import tsercom.rpc.proto as common_pb2\n",
        "import caller_id_pb2 ": "import tsercom.caller_id.proto ",
        "import time_pb2 ": "import tsercom.timesync.common.proto ",
        "import common_pb2 ": "import tsercom.rpc.proto ",
    }

    try:
        with open(file_path, 'r+') as f:
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

def generate_init(package_dir, proto_path : str, generated_path : Path):
    init_file_content = f'''
import grpc
import subprocess

try:
    version = grpc.__version__
    major_minor_version = ".".join(version.split(".")[:2])  # Extract major.minor
    version_string = f"v{{major_minor_version.replace('.', '_')}}" # e.g., v1_62
except (subprocess.CalledProcessError, FileNotFoundError) as e:
    raise RuntimeError(
        f"Could not determine grpcio-tools version. Is it installed? Error: {{e}}"
    ) from e

if False:
    pass
'''
    name = Path(proto_path).name.split(".")[0]
    versioned_dirs = []
    base_package = generated_path.relative_to(package_dir).__str__().replace("/", ".").replace("\\",".")
    for item in generated_path.iterdir():
        if item.is_dir() and item.name.startswith("v"):
            file_path = item.joinpath(f"{name}_pb2.pyi")
            classes = get_classes_from_file(file_path)
            versioned_dirs.append((item.name, classes))
    for versioned_dir_name, classes in versioned_dirs:
        current_version = versioned_dir_name[1:]
        init_file_content += f'''
elif version_string == "v{current_version}":
    from tsercom.{base_package}.{versioned_dir_name}.{name}_pb2 import {", ".join(classes)}
'''
    init_file_content += f'''
else:
    raise ImportError(
        f"No pre-generated protobuf code found for grpcio version: {{version}}.\\n"
        f"Please generate the code for your grpcio version by running 'python scripts/build.py'."
    )
'''
    
    init_file = Path.joinpath(Path.joinpath(package_dir, proto_path).parent, "__init__.py")
    with open(init_file, 'w') as f:  # Open in write mode ('w')
        f.write(init_file_content)

def get_classes_from_file(filepath):
    """
    Gets a list of class names defined in a Python file (.py or .pyi).

    Args:
        module_name: The name of the module (without the .py or .pyi extension).

    Returns:
        A list of strings (class names), or raises FileNotFoundError if
        neither .py nor .pyi files exist.  Returns an empty list if the
        file exists but has syntax errors.
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

def generate_protos(project_root : Path):
    """Generates Python files from .proto files."""

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
        ["caller_id/proto",
         "rpc/proto",
         "timesync/common/proto"],
    )

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    generate_protos(project_root)