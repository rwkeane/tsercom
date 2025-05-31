import grpc

try:
    grpc.enable_fork_support()
    print("gRPC fork support enabled via conftest.py.")
except Exception as e:
    # Catching broad exception because the conditions under which this can fail
    # (e.g., already initialized, platform support) can vary.
    # Printing is for visibility during test runs.
    print(f"INFO: Failed to enable gRPC fork support or already enabled (from conftest.py): {e}")

# Other global test configurations or fixtures can be added below if needed.
