import os
import grpc
import importlib.util
import sys
from pathlib import Path

def load_versioned_protos(current_file_path):
    try:
        version = grpc.__version__
        major_minor_version = ".".join(version.split(".")[:2])  # Extract major.minor
        version_string = f"v{major_minor_version.replace('.', '_')}" # e.g., v1_62
        generated_dir = Path.joinpath(
                Path(current_file_path).resolve().parent, "generated", version_string)

        if not generated_dir.exists():
            # ---  Construct a helpful error message ---
            available_versions = [
                p.name for p in (Path(current_file_path).resolve().parent / "generated").iterdir()
                if p.is_dir() and p.name.startswith("v")
            ]
            version_string = ", ".join(available_versions) or "None"
            raise ImportError(
                f"No pre-generated protobuf code found for grpcio version: {version}.\n"
                f"Available versions: {version_string}.\n"
                f"Please generate the code for your grpcio version by running 'python scripts/build.py'."
            )
            # --- End helpful error message ---

        # --- Dynamically Import ALL .py Files ---
        for root, _, files in os.walk(generated_dir):
            for file in files:
                if file.endswith(".py") and file != "__init__.py":
                    module_path = Path(root) / file
                    relative_module_path = module_path.relative_to(generated_dir.parent)  # Path relative to tsercom/
                    module_name = ".".join(relative_module_path.with_suffix("").parts) # Build module name
                    # print(f"Attempting to load: {module_name}") # Debug print

                    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
                    if spec is None:  # Check if spec is None
                        print(f"Warning: Could not create spec for {module_name} at {module_path}")
                        continue
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    try:
                        spec.loader.exec_module(module)
                    except Exception as e:
                        print(f"Error loading module {module_name}: {e}")
                        raise
                    # Make the module's contents available in the tsercom namespace:
                    globals().update(vars(module))
        # --- End Dynamic Import ---
    except ImportError as e:
        raise ImportError(
            "Could not import gRPC stubs.  Ensure you have a compatible version of "
            "grpcio installed and that the Python files have been generated."
        ) from e