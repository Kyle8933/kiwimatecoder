from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("kiwimatecoder")
except PackageNotFoundError:  # pragma: no cover - dev/untagged checkouts
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
