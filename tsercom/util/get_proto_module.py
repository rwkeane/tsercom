import grpc
import importlib.util
from pathlib import Path

def get_proto_module(module_name: str, path : str):
    """
    Dynamically imports a protobuf module based on the gRPC version.

    Args:
        module_name: The base name of the module (e.g., "time_pb2").

    Returns:
        The imported module object, or None if not found.  Raises ImportError
        if no compatible version is found.
    """
    version = grpc.__version__
    major_minor = ".".join(version.split(".")[:2])
    version_str = f"v{major_minor.replace('.', '_')}"
    generated_dir = Path(path).resolve().parent / "generated" / version_str

    if not generated_dir.exists():
        available_versions = [
            p.name for p in (Path(path).resolve().parent / "generated").iterdir()
            if p.is_dir() and p.name.startswith("v")
        ]
        version_string = ", ".join(available_versions) or "None"
        raise ImportError(
            f"No pre-generated protobuf code found for grpcio version: {version}.\n"
            f"Available versions: {version_string}.\n"
            f"Please generate the code for your grpcio version."
        )

    module_path = generated_dir / f"{module_name}.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Could not find module {module_name} at {module_path}")

    spec = importlib.util.spec_from_file_location(f"tsercom.generated.{version_str}.{module_name}", str(module_path))
    if spec is None:
        raise ImportError(f"Could not create spec for {module_name} at {module_path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise ImportError(f"Error loading module {module_name}: {e}") from e

    return module