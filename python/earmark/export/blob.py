"""Versioned weight blob (``.emwb``) for the C++ engine, plus its JSON manifest.

The blob is one flat little-endian file: a 64-byte header, a table of 128-byte tensor
entries, then the tensor data, each tensor starting on a 64-byte boundary. Weights are
float32. int32 is also allowed, for golden-file metadata such as block sizes. The
engine parses the file without allocating (``engine/src/weights.cpp``);
:func:`unpack_blob` is the Python reader and mirrors every check the engine makes.

Header, 64 bytes::

    offset  size  field
    0       8     magic b"EMWBLOB\\0"
    8       4     u32 format_version (FORMAT_VERSION = 1)
    12      4     u32 tensor_count
    16      16    contract_hash: 16 ASCII hex digits (EARMARK_CONTRACT_HASH)
    32      8     u64 table_offset (always 64)
    40      8     u64 data_offset (64-aligned, at or after the end of the table)
    48      8     u64 file_bytes (total file size)
    56      4     u32 table_crc32: CRC-32 of header bytes [0, 56) followed by the table
    60      4     u32 data_crc32: CRC-32 of bytes [data_offset, file_bytes)

Tensor entry, 128 bytes::

    0    72  name: UTF-8, NUL-padded, at most 71 bytes, so it always contains a NUL
    72   4   u32 dtype (1 = float32, 2 = int32)
    76   4   u32 ndim (0..6)
    80   24  u32 shape[6]; unused trailing dimensions are 0
    104  8   u64 offset of the data relative to data_offset, a multiple of 64
    112  8   u64 numel: product of the shape, 1 for a 0-d tensor
    120  4   u32 crc32 of the tensor's bytes
    124  4   u32 reserved (0)

Arrays are stored in C (row-major) order, and every CRC is CRC-32/ISO-HDLC, the one
:func:`zlib.crc32` computes. Model blobs name each tensor by its PyTorch
``state_dict`` key, so ``body.gru.weight_ih_l0`` keeps nn.GRU's ``[3H, in]`` layout
with gates r, z, n. Two derived constants are added under ``const.``:
``const.erb_norm_init`` and ``const.spec_norm_init``.

The JSON manifest sits next to the blob. Its first keys are flat (``contract_hash``,
``blob_bytes``, ``blob_data_crc32``, ...) so that the engine's minimal scanner can
cross-check them against the blob. It also records the model config, the streaming
state layout and every convention that is not part of the signal contract (see
:func:`conventions`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import subprocess
import sys
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from earmark import constants as C

if TYPE_CHECKING:
    from earmark.model.earmark_net import EarmarkNet

MAGIC: Final[bytes] = b"EMWBLOB\x00"
FORMAT_VERSION: Final[int] = 1
HEADER_BYTES: Final[int] = 64
ENTRY_BYTES: Final[int] = 128
NAME_BYTES: Final[int] = 72
MAX_NDIM: Final[int] = 6
ALIGNMENT: Final[int] = 64
MAX_TENSORS: Final[int] = 65535
DTYPE_FLOAT32: Final[int] = 1
DTYPE_INT32: Final[int] = 2
BLOB_SUFFIX: Final[str] = ".emwb"
MANIFEST_SUFFIX: Final[str] = ".json"
MANIFEST_FORMAT: Final[str] = "earmark-weights-blob"

#: Derived constants appended to every model blob (they are not in the state_dict).
ERB_NORM_INIT_TENSOR: Final[str] = "const.erb_norm_init"
SPEC_NORM_INIT_TENSOR: Final[str] = "const.spec_norm_init"

_HEADER = struct.Struct("<8sII16sQQQII")
_HEADER_CRC_SPAN: Final[int] = 56
_ENTRY = struct.Struct("<72sII6IQQII")
assert _HEADER.size == HEADER_BYTES, _HEADER.size
assert _ENTRY.size == ENTRY_BYTES, _ENTRY.size

_NUMPY_DTYPES: Final[dict[int, np.dtype]] = {
    DTYPE_FLOAT32: np.dtype("<f4"),
    DTYPE_INT32: np.dtype("<i4"),
}
_DTYPE_NAMES: Final[dict[int, str]] = {DTYPE_FLOAT32: "float32", DTYPE_INT32: "int32"}


class BlobFormatError(ValueError):
    """The blob or manifest is malformed, corrupt or inconsistent."""


class ContractMismatchError(BlobFormatError):
    """The blob was exported under a different signal contract (``CONTRACT_HASH``)."""


@dataclass(frozen=True)
class TensorEntry:
    """One row of the tensor table."""

    name: str
    dtype: int
    shape: tuple[int, ...]
    offset: int
    numel: int
    crc32: int

    @property
    def nbytes(self) -> int:
        """Bytes of tensor data (both dtypes are 4 bytes wide)."""
        return self.numel * 4

    def as_json(self) -> dict[str, Any]:
        """JSON-friendly description for the manifest."""
        return {
            "name": self.name,
            "dtype": _DTYPE_NAMES[self.dtype],
            "shape": list(self.shape),
            "offset": self.offset,
            "numel": self.numel,
            "crc32": self.crc32,
        }


@dataclass(frozen=True)
class Blob:
    """A parsed blob: header fields, the tensor table and the arrays (copies)."""

    version: int
    contract_hash: str
    file_bytes: int
    data_offset: int
    table_crc32: int
    data_crc32: int
    entries: tuple[TensorEntry, ...]
    tensors: dict[str, np.ndarray]

    def __getitem__(self, name: str) -> np.ndarray:
        return self.tensors[name]

    def __contains__(self, name: object) -> bool:
        return name in self.tensors

    def names(self) -> list[str]:
        """Tensor names in file order."""
        return [entry.name for entry in self.entries]

    def entry(self, name: str) -> TensorEntry:
        for entry in self.entries:
            if entry.name == name:
                return entry
        raise KeyError(name)


# ------------------------------------------------------------------------------ writing


def align_up(value: int, alignment: int = ALIGNMENT) -> int:
    """Smallest multiple of ``alignment`` that is at least ``value``."""
    return (value + alignment - 1) // alignment * alignment


def as_blob_array(value: Any) -> np.ndarray:
    """Convert a tensor or array to a C-contiguous little-endian float32 or int32 array.

    Floating inputs, including torch bf16/fp16, become float32 and must be finite.
    Integer and bool inputs become int32 and must fit in its range.
    """
    if hasattr(value, "detach"):  # torch.Tensor, without importing torch here
        tensor = value.detach().cpu()
        if tensor.is_floating_point():
            tensor = tensor.float()
        value = tensor.numpy()
    array = np.asarray(value)
    # np.array(..., order="C") rather than np.ascontiguousarray, which turns 0-d
    # tensors into shape (1,).
    if array.dtype == np.bool_ or np.issubdtype(array.dtype, np.integer):
        if array.size and (int(array.min()) < -(2**31) or int(array.max()) > 2**31 - 1):
            raise BlobFormatError("integer tensor does not fit in int32")
        return np.array(array, dtype=_NUMPY_DTYPES[DTYPE_INT32], order="C")
    if np.issubdtype(array.dtype, np.floating):
        out = np.array(array, dtype=_NUMPY_DTYPES[DTYPE_FLOAT32], order="C")
        if not np.all(np.isfinite(out)):
            raise BlobFormatError("float tensor contains NaN or infinity")
        return out
    raise BlobFormatError(f"unsupported dtype {array.dtype}")


def _encode_name(name: str) -> bytes:
    if not isinstance(name, str) or not name:
        raise BlobFormatError(f"tensor name must be a non-empty string, got {name!r}")
    encoded = name.encode("utf-8")
    if b"\x00" in encoded:
        raise BlobFormatError(f"tensor name {name!r} contains NUL")
    if len(encoded) >= NAME_BYTES:
        raise BlobFormatError(f"tensor name {name!r} is longer than {NAME_BYTES - 1} bytes")
    return encoded


def _check_hash(contract_hash: str) -> bytes:
    encoded = contract_hash.encode("ascii")
    if len(encoded) != 16 or any(ch not in b"0123456789abcdef" for ch in encoded):
        raise BlobFormatError(f"contract hash must be 16 lowercase hex digits, got {contract_hash!r}")
    return encoded


def pack_blob(tensors: Mapping[str, Any], contract_hash: str = C.CONTRACT_HASH) -> bytes:
    """Serialise ``{name: array}`` (in iteration order) into blob bytes."""
    hash_bytes = _check_hash(contract_hash)
    items = [(_encode_name(name), as_blob_array(value)) for name, value in tensors.items()]
    if len(items) > MAX_TENSORS:
        raise BlobFormatError(f"too many tensors ({len(items)} > {MAX_TENSORS})")
    table_end = HEADER_BYTES + ENTRY_BYTES * len(items)
    data_offset = align_up(table_end)

    rows: list[bytes] = []
    chunks: list[tuple[int, bytes]] = []
    cursor = 0
    for encoded, array in items:
        if array.ndim > MAX_NDIM:
            raise BlobFormatError(f"{encoded.decode()}: {array.ndim} dimensions (max {MAX_NDIM})")
        raw = array.tobytes(order="C")
        cursor = align_up(cursor)
        dtype = DTYPE_INT32 if array.dtype == _NUMPY_DTYPES[DTYPE_INT32] else DTYPE_FLOAT32
        shape = list(array.shape) + [0] * (MAX_NDIM - array.ndim)
        rows.append(
            _ENTRY.pack(encoded, dtype, array.ndim, *shape, cursor, array.size, zlib.crc32(raw), 0)
        )
        chunks.append((cursor, raw))
        cursor += len(raw)

    data = bytearray(align_up(cursor))
    for offset, raw in chunks:
        data[offset : offset + len(raw)] = raw
    table = b"".join(rows)
    file_bytes = data_offset + len(data)
    head = _HEADER.pack(
        MAGIC, FORMAT_VERSION, len(items), hash_bytes, HEADER_BYTES, data_offset, file_bytes, 0, 0
    )[:_HEADER_CRC_SPAN]
    table_crc = zlib.crc32(table, zlib.crc32(head))
    header = head + struct.pack("<II", table_crc, zlib.crc32(data))
    return header + table + bytes(data_offset - table_end) + bytes(data)


def write_blob(path: str | Path, tensors: Mapping[str, Any], contract_hash: str = C.CONTRACT_HASH) -> bytes:
    """Pack ``tensors`` and write them to ``path``; returns the bytes written."""
    data = pack_blob(tensors, contract_hash)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return data


# ------------------------------------------------------------------------------ reading


def unpack_blob(data: bytes | bytearray | memoryview, *, verify_crc: bool = True) -> Blob:
    """Parse and validate blob bytes (the checks match ``earmark::Blob::parse``)."""
    buf = bytes(data)
    if len(buf) < HEADER_BYTES:
        raise BlobFormatError(f"blob is {len(buf)} bytes, shorter than the {HEADER_BYTES}-byte header")
    (magic, version, count, hash_raw, table_offset, data_offset, file_bytes, table_crc, data_crc) = (
        _HEADER.unpack_from(buf, 0)
    )
    if magic != MAGIC:
        raise BlobFormatError(f"bad magic {magic!r}")
    if version != FORMAT_VERSION:
        raise BlobFormatError(f"unsupported format version {version} (expected {FORMAT_VERSION})")
    if count > MAX_TENSORS:
        raise BlobFormatError(f"tensor count {count} exceeds {MAX_TENSORS}")
    try:
        contract_hash = hash_raw.decode("ascii")
        _check_hash(contract_hash)
    except (UnicodeDecodeError, BlobFormatError) as exc:
        raise BlobFormatError(f"bad contract hash field {hash_raw!r}") from exc
    if table_offset != HEADER_BYTES:
        raise BlobFormatError(f"table offset {table_offset} != {HEADER_BYTES}")
    table_end = table_offset + count * ENTRY_BYTES
    if data_offset % ALIGNMENT or data_offset < table_end:
        raise BlobFormatError(f"bad data offset {data_offset}")
    if file_bytes != len(buf):
        raise BlobFormatError(f"header says {file_bytes} bytes but the blob has {len(buf)}")
    if data_offset > file_bytes:
        raise BlobFormatError("data offset lies past the end of the blob")
    if verify_crc:
        table_crc_now = zlib.crc32(buf[table_offset:table_end], zlib.crc32(buf[:_HEADER_CRC_SPAN]))
        if table_crc_now != table_crc:
            raise BlobFormatError("header/table CRC mismatch")
        if zlib.crc32(buf[data_offset:file_bytes]) != data_crc:
            raise BlobFormatError("data CRC mismatch")

    entries: list[TensorEntry] = []
    tensors: dict[str, np.ndarray] = {}
    for index in range(count):
        row = _ENTRY.unpack_from(buf, table_offset + index * ENTRY_BYTES)
        raw_name, dtype, ndim = row[0], row[1], row[2]
        dims, offset, numel, crc = row[3:9], row[9], row[10], row[11]
        if b"\x00" not in raw_name:
            raise BlobFormatError(f"entry {index}: name is not NUL-terminated")
        try:
            name = raw_name.split(b"\x00", 1)[0].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise BlobFormatError(f"entry {index}: name is not UTF-8") from exc
        if not name:
            raise BlobFormatError(f"entry {index}: empty name")
        if name in tensors:
            raise BlobFormatError(f"duplicate tensor name {name!r}")
        if dtype not in _NUMPY_DTYPES:
            raise BlobFormatError(f"{name}: unknown dtype code {dtype}")
        if ndim > MAX_NDIM:
            raise BlobFormatError(f"{name}: ndim {ndim} > {MAX_NDIM}")
        if any(dims[ndim:]):
            raise BlobFormatError(f"{name}: non-zero shape beyond ndim")
        shape = tuple(int(d) for d in dims[:ndim])
        if math.prod(shape) != numel:
            raise BlobFormatError(f"{name}: numel {numel} != prod{shape}")
        if offset % ALIGNMENT:
            raise BlobFormatError(f"{name}: offset {offset} is not {ALIGNMENT}-aligned")
        start = data_offset + offset
        end = start + numel * 4
        if end > file_bytes:
            raise BlobFormatError(f"{name}: data runs past the end of the blob")
        payload = buf[start:end]
        if verify_crc and zlib.crc32(payload) != crc:
            raise BlobFormatError(f"{name}: tensor CRC mismatch")
        array = np.frombuffer(payload, dtype=_NUMPY_DTYPES[dtype], count=numel).reshape(shape).copy()
        entries.append(TensorEntry(name, dtype, shape, offset, numel, crc))
        tensors[name] = array
    return Blob(
        version=version,
        contract_hash=contract_hash,
        file_bytes=file_bytes,
        data_offset=data_offset,
        table_crc32=table_crc,
        data_crc32=data_crc,
        entries=tuple(entries),
        tensors=tensors,
    )


def read_blob(path: str | Path, *, verify_crc: bool = True) -> Blob:
    """Read and validate a blob file."""
    return unpack_blob(Path(path).read_bytes(), verify_crc=verify_crc)


# ----------------------------------------------------------------------------- manifest


def blob_fields(blob_bytes: bytes, blob_file: str) -> dict[str, Any]:
    """The flat top-level manifest keys that the engine cross-checks against the blob."""
    blob = unpack_blob(blob_bytes)
    return {
        "format": MANIFEST_FORMAT,
        "format_version": FORMAT_VERSION,
        "contract_version": C.CONTRACT_VERSION,
        "contract_hash": blob.contract_hash,
        "blob_file": blob_file,
        "blob_bytes": len(blob_bytes),
        "blob_sha256": hashlib.sha256(blob_bytes).hexdigest(),
        "blob_table_crc32": blob.table_crc32,
        "blob_data_crc32": blob.data_crc32,
    }


def verify_manifest(
    manifest: Mapping[str, Any], blob_bytes: bytes, *, expect_contract: str | None = C.CONTRACT_HASH
) -> Blob:
    """Check that ``manifest`` describes ``blob_bytes``; returns the parsed blob.

    Raises :class:`ContractMismatchError` if the blob or manifest carries a different
    contract hash from ``expect_contract`` (pass ``None`` to skip that check), and
    :class:`BlobFormatError` for any other disagreement.
    """
    blob = unpack_blob(blob_bytes)
    if manifest.get("format") != MANIFEST_FORMAT:
        raise BlobFormatError(f"manifest format {manifest.get('format')!r} != {MANIFEST_FORMAT!r}")
    if manifest.get("format_version") != FORMAT_VERSION:
        raise BlobFormatError(f"manifest format_version {manifest.get('format_version')!r}")
    if manifest.get("contract_hash") != blob.contract_hash:
        raise ContractMismatchError(
            f"manifest contract {manifest.get('contract_hash')!r} != blob contract {blob.contract_hash!r}"
        )
    if expect_contract is not None and blob.contract_hash != expect_contract:
        raise ContractMismatchError(
            f"blob contract {blob.contract_hash!r} != current contract {expect_contract!r}; re-export"
        )
    expected = blob_fields(blob_bytes, str(manifest.get("blob_file", "")))
    for key in ("blob_bytes", "blob_sha256", "blob_table_crc32", "blob_data_crc32"):
        if manifest.get(key) != expected[key]:
            raise BlobFormatError(f"manifest {key}={manifest.get(key)!r} but the blob has {expected[key]!r}")
    listed = manifest.get("tensors")
    if listed is not None:
        described = [entry.as_json() for entry in blob.entries]
        if list(listed) != described:
            raise BlobFormatError("manifest tensor table does not match the blob")
    return blob


def conventions(net: EarmarkNet) -> dict[str, Any]:
    """Model conventions that are not part of the signal contract but that the engine must copy."""
    from earmark.model import dsp

    cfg = net.config
    return {
        "framing": (
            "Frame t is the previous hop (state in_buf, initially zeros) followed by the current "
            "hop: 320 samples. Output lags input by output_delay_samples, and model frame t is "
            "contract frame t - model_frame_offset."
        ),
        "analysis": "X = rfft(frame * w, n=320), w[n] = sin(pi n / 320); forward DFT unscaled",
        "synthesis": "y = irfft(Y, n=320) * w (inverse scaled 1/320); out = y[:160] + ola; ola = y[160:]",
        "erb_features": (
            "db = 10*log10(mean |X|^2 over the band + power_eps); m = a*m + (1-a)*db (update "
            "first, a = NORM_ALPHA); feature = (db - m) / erb_feature_scale_db"
        ),
        "unit_norm": "s = a*s + (1-a)*|X| on bins 0..DF_BINS-1 (update first); feature = X / sqrt(s)",
        "power_eps": dsp.POWER_EPS,
        "erb_feature_scale_db": dsp.ERB_FEATURE_SCALE_DB,
        "norm_alpha": C.NORM_ALPHA,
        "encoder": (
            "Two branches. erb_enc: input [1 ch, 32 bins]; df_enc: input [2 ch (re, im), 64 bins]. "
            "Conv2d weights [out, in/groups, kt, kf]. first: kernel (2, 3), kt index 0 multiplies "
            "frame t-1 and index 1 frame t (state enc_*_prev), frequency padding 1, ungrouped. "
            f"down.*: kernel (1, 3), frequency stride 2, padding 1, groups {cfg.enc_groups}. ReLU "
            "after every conv. ERB branch 32->16->8 bins, low band 64->32->16->8. Each branch is "
            "flattened channel-major (c * bins + f), then ERB features precede low-band features."
        ),
        "grouped_linear": (
            "weight [G, out/G, in/G]; input g*in/G + i feeds group g and output g*out/G + o comes "
            "from it; y = W x + bias"
        ),
        "enc_proj": "GroupedLinear followed by ReLU",
        "film": (
            "e = L2-normalise(embedding, or conditioner.null_embedding when none); "
            "c = tanh(conditioner.proj(e)); (dg, b) = conditioner.pre(c) split in half "
            "(conditioner.post for post-FiLM); x * (1 + dg) + b. Computed once per embedding."
        ),
        "gru": (
            "PyTorch nn.GRU. weight_ih_l{k} [3H, in], weight_hh_l{k} [3H, H], rows stacked as gates "
            "r, z, n. r = sigmoid(W_ir x + b_ir + W_hr h + b_hr); z likewise; "
            "n = tanh(W_in x + b_in + r * (W_hn h + b_hn)); h' = (1 - z) * n + z * h"
        ),
        "heads": (
            "vad_head reads the body output before post-FiLM (vad = sigmoid(logit)); gain_head "
            "(sigmoid, 32 ERB gains) and df_head read the post-FiLM features"
        ),
        "gains": "ERB gains expand rectangularly over ERB_WIDTHS bins and multiply X before the deep filter",
        "deep_filter": (
            "df_head output (DF_BINS*DF_ORDER*2 values) is laid out [bin, tap, re/im]; taps = "
            "tanh(raw) plus 1 on the real part of tap 0; Y[t,k] = sum_i C_i[t,k] X_gained[t-i,k] "
            "for k < DF_BINS (history oldest first in state df_hist); bins >= DF_BINS keep X_gained"
        ),
        "embedding_dim": C.EMBEDDING_DIM,
    }


def model_tensors(net: EarmarkNet) -> dict[str, np.ndarray]:
    """Every persistent state_dict tensor (float32), then the ``const.*`` derived tensors."""
    import torch

    from earmark.model import dsp

    tensors = {name: as_blob_array(value) for name, value in net.state_dict().items()}
    tensors[ERB_NORM_INIT_TENSOR] = as_blob_array(dsp.erb_norm_init(dtype=torch.float64))
    tensors[SPEC_NORM_INIT_TENSOR] = as_blob_array(dsp.unit_norm_init(dtype=torch.float64))
    return tensors


def build_manifest(
    net: EarmarkNet,
    blob_bytes: bytes,
    *,
    blob_file: str,
    seed: int | None = None,
    random_weights: bool = False,
    git_sha: str | None = None,
) -> dict[str, Any]:
    """Manifest for a model blob. Deterministic: no timestamps."""
    import torch

    from earmark.model.earmark_net import MODEL_FRAME_OFFSET, OUTPUT_DELAY_SAMPLES
    from earmark.model.stream import STATE_FIELDS, state_size_bytes

    blob = unpack_blob(blob_bytes)
    return {
        **blob_fields(blob_bytes, blob_file),
        "created_by": "earmark.export.blob",
        "random_weights": random_weights,
        "seed": seed,
        "git_sha": git_sha,
        "torch_version": torch.__version__,
        "model": {
            "name": net.config.name,
            "config": asdict(net.config),
            "params": sum(p.numel() for p in net.parameters()),
            "state_fields": list(STATE_FIELDS),
            "state_layout": {name: list(shape) for name, shape in net.state_layout().items()},
            "state_bytes_per_stream": state_size_bytes(net),
            "output_delay_samples": OUTPUT_DELAY_SAMPLES,
            "model_frame_offset": MODEL_FRAME_OFFSET,
        },
        "norm_init": {
            "erb_norm_db": [float(v) for v in blob[ERB_NORM_INIT_TENSOR]],
            "spec_norm": [float(v) for v in blob[SPEC_NORM_INIT_TENSOR]],
        },
        "conventions": conventions(net),
        "tensors": [entry.as_json() for entry in blob.entries],
    }


@dataclass(frozen=True)
class ExportResult:
    """Paths and summary of one export."""

    blob_path: Path
    manifest_path: Path
    blob_bytes: int
    blob_sha256: str
    tensor_count: int
    params: int


def export_model(
    net: EarmarkNet,
    out_dir: str | Path,
    stem: str,
    *,
    seed: int | None = None,
    random_weights: bool = False,
    git_sha: str | None = None,
) -> ExportResult:
    """Write ``<stem>.emwb`` and ``<stem>.json`` for ``net`` into ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    blob_path = out / f"{stem}{BLOB_SUFFIX}"
    manifest_path = out / f"{stem}{MANIFEST_SUFFIX}"
    data = write_blob(blob_path, model_tensors(net))
    manifest = build_manifest(
        net, data, blob_file=blob_path.name, seed=seed, random_weights=random_weights, git_sha=git_sha
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return ExportResult(
        blob_path=blob_path,
        manifest_path=manifest_path,
        blob_bytes=len(data),
        blob_sha256=manifest["blob_sha256"],
        tensor_count=len(manifest["tensors"]),
        params=manifest["model"]["params"],
    )


def random_stem(config: str, seed: int) -> str:
    """Default file stem of a random-weight export, e.g. ``earmark-m-random-seed0``."""
    return f"earmark-{config.strip().lower().replace('_', '-')}-random-seed{seed}"


def export_random(
    config: str = "M",
    seed: int = 0,
    out_dir: str | Path = "results/export",
    *,
    stem: str | None = None,
    git_sha: str | None = None,
) -> ExportResult:
    """Random-weight export (fixed seed) that unblocks engine and web work before training."""
    import torch

    from earmark.model import build

    torch.manual_seed(seed)
    net = build(config).eval()
    return export_model(
        net, out_dir, stem or random_stem(config, seed), seed=seed, random_weights=True, git_sha=git_sha
    )


def load_into(net: EarmarkNet, blob: Blob) -> None:
    """Load a model blob's state_dict tensors into ``net`` (strict: every key must be present)."""
    import torch

    expected = net.state_dict()
    missing = [name for name in expected if name not in blob]
    if missing:
        raise BlobFormatError(f"blob is missing tensors {missing[:5]}{'...' if len(missing) > 5 else ''}")
    state = {}
    for name, current in expected.items():
        array = blob[name]
        if tuple(array.shape) != tuple(current.shape):
            raise BlobFormatError(f"{name}: blob shape {array.shape} != model shape {tuple(current.shape)}")
        state[name] = torch.from_numpy(array.copy()).to(dtype=current.dtype)
    net.load_state_dict(state, strict=True)


def read_manifest(path: str | Path) -> dict[str, Any]:
    """Load a manifest JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------------- CLI


def _git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _describe(blob: Blob) -> str:
    lines = [
        f"format v{blob.version}, contract {blob.contract_hash}, {blob.file_bytes} bytes, "
        f"{len(blob.entries)} tensors"
    ]
    for entry in blob.entries:
        lines.append(f"  {entry.name:<48} {_DTYPE_NAMES[entry.dtype]:<8} {list(entry.shape)}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m earmark.export.blob", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    rnd = sub.add_parser("random", help="export a random-weight model (fixed seed)")
    rnd.add_argument("--config", default="M", help="S-GRU, S-SSM, M or M-256")
    rnd.add_argument("--seed", type=int, default=0)
    rnd.add_argument("--out", default="results/export", help="output directory")
    rnd.add_argument("--stem", default=None, help="file stem (default earmark-<config>-random-seed<N>)")
    ins = sub.add_parser("inspect", help="validate a blob (and optionally its manifest) and list tensors")
    ins.add_argument("blob")
    ins.add_argument("--manifest", default=None)
    args = parser.parse_args(argv)

    if args.command == "random":
        result = export_random(args.config, args.seed, args.out, stem=args.stem, git_sha=_git_sha())
        print(f"wrote {result.blob_path} ({result.blob_bytes} bytes, {result.tensor_count} tensors, "
              f"{result.params} params) and {result.manifest_path}")
        return 0
    data = Path(args.blob).read_bytes()
    try:
        blob = unpack_blob(data)
        if args.manifest:
            verify_manifest(read_manifest(args.manifest), data)
    except BlobFormatError as exc:
        print(f"invalid: {exc}", file=sys.stderr)
        return 1
    print(_describe(blob))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ALIGNMENT",
    "BLOB_SUFFIX",
    "DTYPE_FLOAT32",
    "DTYPE_INT32",
    "ENTRY_BYTES",
    "ERB_NORM_INIT_TENSOR",
    "FORMAT_VERSION",
    "HEADER_BYTES",
    "MAGIC",
    "MANIFEST_FORMAT",
    "MAX_NDIM",
    "NAME_BYTES",
    "SPEC_NORM_INIT_TENSOR",
    "Blob",
    "BlobFormatError",
    "ContractMismatchError",
    "ExportResult",
    "TensorEntry",
    "align_up",
    "as_blob_array",
    "blob_fields",
    "build_manifest",
    "conventions",
    "export_model",
    "export_random",
    "load_into",
    "model_tensors",
    "pack_blob",
    "random_stem",
    "read_blob",
    "read_manifest",
    "unpack_blob",
    "verify_manifest",
    "write_blob",
]
