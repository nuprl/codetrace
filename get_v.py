import importlib.metadata

try:
    version = importlib.metadata.version("nnsight")
    print(f"Transformers version: {version}")
except importlib.metadata.PackageNotFoundError:
    print("Transformers library is not installed in the current environment.")