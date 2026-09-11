"""Evaluation asset manifest: download URLs, sizes and SHA-256 pins.

``assets.tsv`` (next to this module) is the single source of truth. ``scripts/fetch_eval_data.sh``
downloads and verifies the files it lists; this module lets Python code find the same files and
re-check their pins before trusting them (for example before loading a checkpoint).
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from functools import cache
from pathlib import Path

__all__ = [
    "MANIFEST_PATH",
    "REPO_ROOT",
    "Asset",
    "AssetError",
    "cache_root",
    "get_asset",
    "load_assets",
    "sha256_file",
    "verify_asset",
]

MANIFEST_PATH = Path(__file__).with_name("assets.tsv")
REPO_ROOT = Path(__file__).resolve().parents[3]

_FETCH_HINT = "run scripts/fetch_eval_data.sh to download and verify it"


class AssetError(RuntimeError):
    """An asset is missing, unknown or fails its size/SHA-256 pin."""


@dataclass(frozen=True)
class Asset:
    """One row of ``assets.tsv``."""

    name: str
    group: str
    bytes: int
    sha256: str
    dest: str
    extract: str | None
    url: str

    def path(self, root: Path | None = None) -> Path:
        """Location of the downloaded file under the cache root."""
        return (root or cache_root()) / self.dest

    def extract_dir(self, root: Path | None = None) -> Path | None:
        """Directory the archive is unpacked into, or ``None`` for files kept as is."""
        return None if self.extract is None else (root or cache_root()) / self.extract


def cache_root() -> Path:
    """Cache root: ``$EARMARK_CACHE`` if set, else ``<repo>/.cache`` (same rule as the script)."""
    env = os.environ.get("EARMARK_CACHE")
    return Path(env) if env else REPO_ROOT / ".cache"


def _parse_manifest(text: str) -> dict[str, Asset]:
    assets: dict[str, Asset] = {}
    for lineno, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip() or raw.startswith("#"):
            continue
        fields = raw.split("\t")
        if len(fields) != 7:
            raise AssetError(f"{MANIFEST_PATH.name}:{lineno}: expected 7 tab-separated fields, got {len(fields)}")
        name, group, size, sha, dest, extract, url = fields
        if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            raise AssetError(f"{MANIFEST_PATH.name}:{lineno}: {name} has no valid sha256 pin")
        if name in assets:
            raise AssetError(f"{MANIFEST_PATH.name}:{lineno}: duplicate asset {name}")
        assets[name] = Asset(
            name=name,
            group=group,
            bytes=int(size),
            sha256=sha,
            dest=dest,
            extract=None if extract == "-" else extract,
            url=url,
        )
    return assets


@cache
def load_assets() -> dict[str, Asset]:
    """All assets keyed by name, parsed once from ``assets.tsv``."""
    return _parse_manifest(MANIFEST_PATH.read_text(encoding="utf-8"))


def get_asset(name: str) -> Asset:
    """Look up one asset by its file name, e.g. ``"model_trained_on_vctk.tar"``."""
    try:
        return load_assets()[name]
    except KeyError:
        raise AssetError(f"unknown asset {name!r}; known: {sorted(load_assets())}") from None


def sha256_file(path: Path, chunk_bytes: int = 1 << 20) -> str:
    """Hex SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def verify_asset(name: str, path: Path | None = None) -> Path:
    """Return the asset's path after checking it exists and matches its size and SHA-256 pin."""
    asset = get_asset(name)
    target = path if path is not None else asset.path()
    if not target.is_file():
        raise AssetError(f"{name} not found at {target}; {_FETCH_HINT}")
    size = target.stat().st_size
    if size != asset.bytes:
        raise AssetError(f"{name} at {target} has {size} bytes, pinned {asset.bytes}; {_FETCH_HINT}")
    digest = sha256_file(target)
    if digest != asset.sha256:
        raise AssetError(f"{name} at {target} has sha256 {digest}, pinned {asset.sha256}; {_FETCH_HINT}")
    return target
