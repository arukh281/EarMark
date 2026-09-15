"""Tests for the ``.emwb`` weight blob, its manifest and the random-weight export."""

from __future__ import annotations

import hashlib
import json
import struct
import zlib
from pathlib import Path

import numpy as np
import pytest
import torch

from earmark import constants as C
from earmark.export import blob as B
from earmark.model import build
from earmark.model import dsp


def sample_tensors() -> dict[str, object]:
    return {
        "a.f32": np.arange(24, dtype=np.float32).reshape(2, 3, 4) * 0.25,
        "b.i32": np.array([[1, -2], [3, 2**31 - 1]], dtype=np.int64),
        "scalar": np.float32(2.5),
        "empty": np.zeros(0, dtype=np.float32),
        "rank6": np.arange(8, dtype=np.float64).reshape(1, 2, 1, 2, 1, 2),
        "t.bf16": torch.full((3,), 1.5, dtype=torch.bfloat16),
        "t.bool": np.array([True, False]),
    }


def test_round_trip_keeps_names_dtypes_shapes_and_values() -> None:
    tensors = sample_tensors()
    blob = B.unpack_blob(B.pack_blob(tensors))
    assert blob.names() == list(tensors)
    assert blob.contract_hash == C.CONTRACT_HASH
    assert blob.version == B.FORMAT_VERSION
    np.testing.assert_array_equal(blob["a.f32"], tensors["a.f32"])
    assert blob["b.i32"].dtype == np.int32
    np.testing.assert_array_equal(blob["b.i32"], [[1, -2], [3, 2**31 - 1]])
    assert blob["scalar"].shape == () and blob["scalar"] == np.float32(2.5)
    assert blob["empty"].shape == (0,)
    assert blob["rank6"].dtype == np.float32 and blob["rank6"].shape == (1, 2, 1, 2, 1, 2)
    np.testing.assert_array_equal(blob["t.bf16"], np.full(3, 1.5, dtype=np.float32))
    np.testing.assert_array_equal(blob["t.bool"], [1, 0])
    assert "a.f32" in blob and "missing" not in blob
    assert blob.entry("scalar").numel == 1


def test_layout_is_aligned_and_the_header_is_consistent() -> None:
    data = B.pack_blob(sample_tensors())
    magic, version, count, hash_raw, table, data_offset, file_bytes, table_crc, data_crc = struct.unpack_from(
        "<8sII16sQQQII", data
    )
    assert magic == B.MAGIC and version == 1 and count == len(sample_tensors())
    assert hash_raw.decode() == C.CONTRACT_HASH
    assert table == B.HEADER_BYTES
    assert data_offset % B.ALIGNMENT == 0 and data_offset >= table + count * B.ENTRY_BYTES
    assert file_bytes == len(data) and len(data) % B.ALIGNMENT == 0
    assert table_crc == zlib.crc32(data[64 : 64 + count * 128], zlib.crc32(data[:56]))
    assert data_crc == zlib.crc32(data[data_offset:])
    for entry in B.unpack_blob(data).entries:
        assert entry.offset % B.ALIGNMENT == 0


def test_pack_is_deterministic_and_order_preserving() -> None:
    tensors = sample_tensors()
    assert B.pack_blob(tensors) == B.pack_blob(tensors)
    reordered = dict(reversed(list(tensors.items())))
    assert B.unpack_blob(B.pack_blob(reordered)).names() == list(reordered)


def _flip(data: bytes, offset: int, mask: int = 0x01) -> bytes:
    raw = bytearray(data)
    raw[offset] ^= mask
    return bytes(raw)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d[:40], "shorter than"),
        (lambda d: b"X" + d[1:], "bad magic"),
        (lambda d: d[:8] + struct.pack("<I", 2) + d[12:], "format version"),
        (lambda d: d[:16] + b"Z" + d[17:], "contract hash"),
        (lambda d: d + bytes(64), "header says"),
        (lambda d: _flip(d, len(d) - 1), "data CRC"),
        (lambda d: _flip(d, 64 + 80), "table CRC"),
    ],
)
def test_corrupt_blobs_are_rejected(mutate, message: str) -> None:
    data = B.pack_blob(sample_tensors())
    with pytest.raises(B.BlobFormatError, match=message):
        B.unpack_blob(mutate(data))


def test_crc_checks_can_be_skipped_but_structure_is_still_checked() -> None:
    data = B.pack_blob(sample_tensors())
    B.unpack_blob(_flip(data, len(data) - 1), verify_crc=False)
    with pytest.raises(B.BlobFormatError, match="numel"):
        B.unpack_blob(_flip(data, 64 + 80), verify_crc=False)


@pytest.mark.parametrize(
    ("tensors", "message"),
    [
        ({"x": np.array([1.0, np.nan])}, "NaN"),
        ({"y" * 72: np.zeros(1)}, "longer than"),
        ({"a\x00b": np.zeros(1)}, "NUL"),
        ({"": np.zeros(1)}, "non-empty"),
        ({"big": np.array([2**40])}, "int32"),
        ({"deep": np.zeros((1,) * 7)}, "dimensions"),
        ({"text": np.array(["a"])}, "unsupported dtype"),
    ],
)
def test_writer_rejects_bad_tensors(tensors: dict[str, object], message: str) -> None:
    with pytest.raises(B.BlobFormatError, match=message):
        B.pack_blob(tensors)


def test_writer_rejects_a_bad_contract_hash() -> None:
    with pytest.raises(B.BlobFormatError, match="16 lowercase hex"):
        B.pack_blob({"x": np.zeros(1)}, contract_hash="ABC")


def _manifest_for(data: bytes) -> dict[str, object]:
    blob = B.unpack_blob(data)
    return {**B.blob_fields(data, "x.emwb"), "tensors": [e.as_json() for e in blob.entries]}


def test_manifest_verification() -> None:
    data = B.pack_blob(sample_tensors())
    manifest = _manifest_for(data)
    assert B.verify_manifest(manifest, data).names() == list(sample_tensors())
    assert manifest["blob_sha256"] == hashlib.sha256(data).hexdigest()

    with pytest.raises(B.BlobFormatError, match="blob_bytes"):
        B.verify_manifest({**manifest, "blob_bytes": 1}, data)
    with pytest.raises(B.BlobFormatError, match="tensor table"):
        B.verify_manifest({**manifest, "tensors": manifest["tensors"][:-1]}, data)
    with pytest.raises(B.BlobFormatError, match="format"):
        B.verify_manifest({**manifest, "format": "other"}, data)
    with pytest.raises(B.ContractMismatchError):
        B.verify_manifest({**manifest, "contract_hash": "0" * 16}, data)

    stale = B.pack_blob(sample_tensors(), contract_hash="0123456789abcdef")
    with pytest.raises(B.ContractMismatchError, match="re-export"):
        B.verify_manifest(_manifest_for(stale), stale)
    B.verify_manifest(_manifest_for(stale), stale, expect_contract=None)


def test_random_export_round_trips_into_a_fresh_model(tmp_path: Path) -> None:
    result = B.export_random("S-GRU", seed=3, out_dir=tmp_path)
    assert result.blob_path.name == "earmark-s-gru-random-seed3.emwb"
    data = result.blob_path.read_bytes()
    manifest = B.read_manifest(result.manifest_path)
    blob = B.verify_manifest(manifest, data)

    torch.manual_seed(3)
    reference = build("S-GRU")
    expected = reference.state_dict()
    assert blob.names()[: len(expected)] == list(expected)
    assert blob.names()[len(expected) :] == [B.ERB_NORM_INIT_TENSOR, B.SPEC_NORM_INIT_TENSOR]
    assert blob["conditioner.null_embedding"].shape == (C.EMBEDDING_DIM,)
    np.testing.assert_array_equal(blob[B.ERB_NORM_INIT_TENSOR], dsp.erb_norm_init().numpy())
    np.testing.assert_array_equal(blob[B.SPEC_NORM_INIT_TENSOR], dsp.unit_norm_init().numpy())

    torch.manual_seed(99)
    fresh = build("S-GRU")
    B.load_into(fresh, blob)
    for name, value in fresh.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)

    assert manifest["random_weights"] is True and manifest["seed"] == 3
    assert manifest["contract_hash"] == C.CONTRACT_HASH
    assert manifest["model"]["name"] == "S-GRU"
    assert manifest["model"]["params"] == result.params == sum(p.numel() for p in reference.parameters())
    assert manifest["model"]["state_bytes_per_stream"] > 0
    assert manifest["model"]["state_fields"][:4] == ["in_buf", "ola_buf", "erb_norm", "spec_norm"]
    for key in ("gru", "film", "deep_filter", "erb_features", "encoder", "power_eps", "erb_feature_scale_db"):
        assert key in manifest["conventions"]
    assert manifest["conventions"]["power_eps"] == dsp.POWER_EPS
    json.dumps(manifest)  # fully JSON-serialisable


def test_random_export_is_deterministic_per_seed(tmp_path: Path) -> None:
    first = B.export_random("S-GRU", seed=5, out_dir=tmp_path / "a")
    second = B.export_random("S-GRU", seed=5, out_dir=tmp_path / "b")
    other = B.export_random("S-GRU", seed=6, out_dir=tmp_path / "c")
    assert first.blob_sha256 == second.blob_sha256
    assert first.manifest_path.read_text() == second.manifest_path.read_text()
    assert first.blob_sha256 != other.blob_sha256


def test_load_into_rejects_missing_or_misshapen_tensors() -> None:
    net = build("S-GRU")
    tensors = {name: value for name, value in net.state_dict().items()}
    missing = dict(tensors)
    missing.pop("vad_head.bias")
    with pytest.raises(B.BlobFormatError, match="missing"):
        B.load_into(net, B.unpack_blob(B.pack_blob(missing)))
    misshapen = dict(tensors)
    misshapen["vad_head.bias"] = np.zeros(7, dtype=np.float32)
    with pytest.raises(B.BlobFormatError, match="shape"):
        B.load_into(net, B.unpack_blob(B.pack_blob(misshapen)))


def test_cli_random_and_inspect(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert B.main(["random", "--config", "S-GRU", "--seed", "1", "--out", str(tmp_path)]) == 0
    blob_path = tmp_path / "earmark-s-gru-random-seed1.emwb"
    manifest_path = tmp_path / "earmark-s-gru-random-seed1.json"
    assert B.main(["inspect", str(blob_path), "--manifest", str(manifest_path)]) == 0
    assert "body.gru.weight_ih_l0" in capsys.readouterr().out
    corrupt = tmp_path / "corrupt.emwb"
    corrupt.write_bytes(_flip(blob_path.read_bytes(), blob_path.stat().st_size - 1))
    assert B.main(["inspect", str(corrupt)]) == 1
    assert "invalid" in capsys.readouterr().err
