"""Durable endpoint response cache (National Enrichment Engine).

Key:  IGME_RECORD_ID + ENDPOINT + REQUEST_VERSION
Store: request metadata, raw response, response hash, retrieval timestamp,
       endpoint version.

Guarantees:
- Survives process restarts (file-per-record on disk).
- Never overwrites an existing raw response silently: if the same key later
  resolves to different content, the original is preserved and the new
  response is stored under a versioned sibling file flagged as a change.
- Every write is atomic (tmp file in same directory → fsync → rename).
"""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Same canonical-JSON hashing convention as aquascan.enrichment.api.sha256_json.
from aquascan.enrichment.api import sha256_json
from aquascan.national.io_utils import atomic_write_text


def _hash_response(data: Any) -> str:
    return sha256_json(data)


class DurableCache:
    """Durable per-record/per-endpoint response cache."""

    def __init__(self, cache_dir: Path, request_version: str = "v1") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.request_version = request_version
        self._lock = threading.RLock()

    # ── paths ────────────────────────────────────────────────────────

    def _record_dir(self, igme_id: str) -> Path:
        # igme_id may contain '/'-unsafe chars only in tests; sanitize for safety.
        safe = igme_id.replace("/", "_").replace("\\", "_")
        return self.cache_dir / safe

    def _main_path(self, igme_id: str, endpoint: str) -> Path:
        return self._record_dir(igme_id) / f"{endpoint}@{self.request_version}.json"

    def _conflict_path(self, igme_id: str, endpoint: str, new_hash: str) -> Path:
        fname = f"{endpoint}@{self.request_version}__changed_{new_hash[:12]}.json"
        return self._record_dir(igme_id) / fname

    def _entry(self, igme_id: str, endpoint: str, data: Any, response_hash: str) -> dict[str, Any]:
        return {
            "key": {
                "igme_record_id": igme_id,
                "endpoint": endpoint,
                "request_version": self.request_version,
            },
            "request_metadata": {
                "api": "igme_bdaguas",
                "endpoint_version": "v1",
                "request_version": self.request_version,
            },
            "raw_response": data,
            "response_hash": response_hash,
            "retrieval_timestamp": datetime.now(UTC).isoformat(),
            "endpoint_version": "v1",
        }

    # ── read / write ─────────────────────────────────────────────────

    def get(self, igme_id: str, endpoint: str) -> tuple[list[dict[str, Any]] | None, str]:
        """Return (raw_response, response_hash) or (None, '') on a miss."""
        path = self._main_path(igme_id, endpoint)
        if not path.exists():
            return None, ""
        try:
            with open(path) as f:
                entry = json.load(f)
        except (json.JSONDecodeError, OSError):
            return None, ""
        data = entry.get("raw_response")
        r_hash = entry.get("response_hash", "")
        if data is None or not r_hash:
            return None, ""
        return data, r_hash

    def put(
        self,
        igme_id: str,
        endpoint: str,
        data: list[dict[str, Any]],
        response_hash: str | None = None,
    ) -> dict[str, Any]:
        """Store a response atomically.

        Returns a result dict:
            {"stored": bool, "changed": bool, "response_hash": str}

        - If the key already holds identical content → no write, changed=False.
        - If the key holds different content → original preserved, new content
          stored under a changed-version file, changed=True.
        - Otherwise → main file written, changed=False.
        """
        if response_hash is None:
            response_hash = _hash_response(data)

        with self._lock:
            existing_hash = ""
            main_path = self._main_path(igme_id, endpoint)
            if main_path.exists():
                try:
                    with open(main_path) as f:
                        existing_hash = json.load(f).get("response_hash", "")
                except (json.JSONDecodeError, OSError):
                    existing_hash = ""

            if existing_hash == response_hash:
                return {"stored": False, "changed": False, "response_hash": response_hash}

            entry = self._entry(igme_id, endpoint, data, response_hash)
            text = json.dumps(entry, sort_keys=True, ensure_ascii=False, default=str)

            if existing_hash:
                # Never overwrite a different raw response silently.
                conflict = self._conflict_path(igme_id, endpoint, response_hash)
                entry["changed_from"] = existing_hash
                atomic_write_text(
                    conflict,
                    json.dumps(entry, sort_keys=True, ensure_ascii=False, default=str),
                )
                return {
                    "stored": True,
                    "changed": True,
                    "response_hash": response_hash,
                    "changed_from": existing_hash,
                }

            atomic_write_text(main_path, text)
            return {"stored": True, "changed": False, "response_hash": response_hash}

    def has(self, igme_id: str, endpoint: str) -> bool:
        return self._main_path(igme_id, endpoint).exists()

    # ── stats / integrity ────────────────────────────────────────────

    def _main_files(self) -> list[Path]:
        """Main (non-conflict) cache entry files."""
        return [p for p in self.cache_dir.glob("*/[!_]*.json") if "__changed_" not in p.name]

    def entry_count(self) -> int:
        return len(self._main_files())

    def cache_stats(self) -> dict[str, Any]:
        main = self._main_files()
        changed = list(self.cache_dir.glob("*/*__changed_*.json"))
        total_bytes = sum(p.stat().st_size for p in main + changed)
        return {
            "entries": len(main),
            "changed_versions": len(changed),
            "records": len(list(self.cache_dir.iterdir())) if self.cache_dir.exists() else 0,
            "size_bytes": total_bytes,
        }

    def verify_integrity(self) -> dict[str, Any]:
        """Re-hash every stored raw response and compare with its stored hash."""
        corrupted: list[str] = []
        total = 0
        for path in self._main_files():
            try:
                with open(path) as f:
                    entry = json.load(f)
            except (json.JSONDecodeError, OSError):
                corrupted.append(str(path))
                continue
            data = entry.get("raw_response")
            stored_hash = entry.get("response_hash", "")
            if data is None:
                corrupted.append(str(path))
                continue
            total += 1
            actual = _hash_response(data)
            if actual != stored_hash:
                corrupted.append(f"{path.name} ({stored_hash[:8]} != {actual[:8]})")
        return {
            "total_entries": total,
            "corrupted": len(corrupted),
            "corrupted_entries": corrupted,
            "valid": len(corrupted) == 0,
        }

    def all_keys(self) -> list[tuple[str, str]]:
        keys: list[tuple[str, str]] = []
        for path in self._main_files():
            parts = path.stem.split("@")
            if len(parts) != 2:
                continue
            keys.append((path.parent.name, parts[0]))
        return keys
