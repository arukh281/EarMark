"""The asset manifest and the fetch script that reads it (no network access)."""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from earmark.eval import assets as A

SCRIPT = A.REPO_ROOT / "scripts" / "fetch_eval_data.sh"


def test_manifest_rows_are_pinned() -> None:
    rows = A.load_assets()
    assert {"clean_testset_wav.zip", "noisy_testset_wav.zip", "model_trained_on_vctk.tar"} <= set(rows)
    for asset in rows.values():
        assert re.fullmatch(r"[0-9a-f]{64}", asset.sha256)
        assert asset.bytes > 0
        assert asset.url.startswith("https://")
        assert not Path(asset.dest).is_absolute() and ".." not in Path(asset.dest).parts
    assert A.get_asset("clean_testset_wav.zip").extract == "data/vbd"
    assert A.get_asset("model_trained_on_vctk.tar").extract is None
    with pytest.raises(A.AssetError):
        A.get_asset("nope.zip")


def test_manifest_parser_rejects_bad_rows() -> None:
    with pytest.raises(A.AssetError, match="7 tab-separated"):
        A._parse_manifest("a\tb\tc\n")
    with pytest.raises(A.AssetError, match="sha256"):
        A._parse_manifest("n\tg\t1\tNOTAHASH\td\t-\thttps://x\n")


def test_verify_asset_checks_size_and_hash(tmp_path: Path) -> None:
    bogus = tmp_path / "model_trained_on_vctk.tar"
    bogus.write_bytes(b"not a checkpoint")
    with pytest.raises(A.AssetError, match="bytes"):
        A.verify_asset("model_trained_on_vctk.tar", bogus)
    with pytest.raises(A.AssetError, match="not found"):
        A.verify_asset("model_trained_on_vctk.tar", tmp_path / "missing.tar")
    assert A.sha256_file(bogus, chunk_bytes=3) == hashlib.sha256(b"not a checkpoint").hexdigest()


def test_cache_root_follows_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("EARMARK_CACHE", str(tmp_path))
    assert A.cache_root() == tmp_path
    assert A.get_asset("logfiles.zip").path() == tmp_path / "data/vbd/zips/logfiles.zip"


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_fetch_script_syntax_list_and_usage() -> None:
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    listed = subprocess.run([str(SCRIPT), "--list"], check=True, capture_output=True, text=True).stdout
    for name, asset in A.load_assets().items():
        assert name in listed and asset.sha256 in listed
    bad = subprocess.run([str(SCRIPT), "no-such-group"], capture_output=True, text=True)
    assert bad.returncode == 2 and "unknown group" in bad.stderr


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_fetch_script_verify_only_reports_missing_files(tmp_path: Path) -> None:
    res = subprocess.run(
        [str(SCRIPT), "--verify-only", "gtcrn"],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "EARMARK_CACHE": str(tmp_path)},
    )
    assert res.returncode == 1
    assert "MISSING" in res.stderr and list(tmp_path.iterdir()) == []
