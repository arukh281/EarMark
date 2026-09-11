"""Weight-blob export and golden tensors for the C++ engine.

* :mod:`earmark.export.blob`: the versioned ``.emwb`` weight blob, its JSON manifest,
  the Python reader and the random-weight export
  (``python -m earmark.export.blob random --config M``).
* :mod:`earmark.export.golden`: per-layer golden files for the engine's Catch2 tests
  (``python -m earmark.export.golden``).
* :mod:`earmark.export.reference`: the float64 reference implementations behind them.

Only the blob API is imported eagerly. ``golden`` pulls in torch and the model DSP, so
import it explicitly.
"""

from earmark.export.blob import (
    Blob,
    BlobFormatError,
    ContractMismatchError,
    ExportResult,
    TensorEntry,
    export_model,
    export_random,
    load_into,
    pack_blob,
    read_blob,
    read_manifest,
    unpack_blob,
    verify_manifest,
    write_blob,
)

__all__ = [
    "Blob",
    "BlobFormatError",
    "ContractMismatchError",
    "ExportResult",
    "TensorEntry",
    "export_model",
    "export_random",
    "load_into",
    "pack_blob",
    "read_blob",
    "read_manifest",
    "unpack_blob",
    "verify_manifest",
    "write_blob",
]
