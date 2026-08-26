"""Gate 4: Immutable evidence cache (step 4).

The cache maps: IGME ID -> endpoint -> raw response -> hash -> normalized evidence.

Rules:
- Never overwrite a different raw response silently.
- If the same request later returns different content:
  preserve both versions and flag `source_response_changed`.
- SHA-256 hashing for every stored artifact.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aquascan.data.provenance import compute_sha256

logger = logging.getLogger(__name__)


class ResponseCache:
    """Immutable evidence cache.

    Maps: IGME ID → endpoint → raw response → hash.
    """

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_file = self.cache_dir / "enrichment_cache.json"
        self._storage: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)

        # Track source_response_changed events
        self._changed_ids: set[str] = set()
        self._changed_endpoints: dict[str, list[str]] = defaultdict(list)

        if self._cache_file.exists():
            self._load()

    def _load(self) -> None:
        """Load cache from disk."""
        try:
            with open(self._cache_file) as f:
                data = json.load(f)
            for igme_id, endpoints in data.items():
                self._storage[igme_id] = dict(endpoints)
            logger.info(
                "Loaded cache: %d records",
                len(self._storage),
            )
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Could not load cache: %s", e)
            self._storage = defaultdict(dict)

    def _save(self) -> None:
        """Persist cache to disk."""
        tmp = self._cache_file.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(dict(self._storage), f, indent=2, default=str, ensure_ascii=False)
        tmp.replace(self._cache_file)

    def save(self) -> None:
        """Public save (atomic). Call periodically or after a batch of puts."""
        self._save()

    def get(self, igme_id: str, endpoint: str) -> tuple[list[dict[str, Any]] | None, str]:
        """Retrieve cached response for an IGME ID + endpoint.

        Returns:
            Tuple of (data_list, sha256_hash) or (None, "") if not cached.
        """
        ep_cache = self._storage.get(igme_id, {}).get(endpoint)
        if not ep_cache:
            return None, ""
        return ep_cache.get("data"), ep_cache.get("sha256", "")

    def put(
        self,
        igme_id: str,
        endpoint: str,
        data: list[dict[str, Any]],
        new_hash: str,
    ) -> bool:
        """Store a response in the cache.

        If the endpoint already has cached data with a DIFFERENT hash:
        - Preserve the original under versioned keys
        - Flag source_response_changed

        Returns:
            True if this was a new cache entry, False if it replaced an existing one.
        """
        existing = self._storage.get(igme_id, {}).get(endpoint)
        if existing:
            old_hash = existing.get("sha256", "")
            if old_hash == new_hash:
                return True  # Same content, nothing to do

            # Different content — preserve old version
            existing["versions"] = existing.get("versions", [])
            existing["versions"].append(
                {
                    "sha256": old_hash,
                    "timestamp": existing.get("timestamp"),
                    "data": existing.get("data"),
                }
            )
            self._changed_ids.add(igme_id)
            self._changed_endpoints[igme_id].append(endpoint)
            logger.warning(
                "source_response_changed: %s/%s (old=%s new=%s)",
                igme_id,
                endpoint,
                old_hash[:16],
                new_hash[:16],
            )

        self._storage.setdefault(igme_id, {})[endpoint] = {
            "sha256": new_hash,
            "timestamp": datetime.now(UTC).isoformat(),
            "data": data,
            "versions": existing.get("versions", []) if existing else [],
        }
        # Note: caller should call save() to persist (avoids O(n^2) writes).
        return True

    @property
    def changed_records(self) -> dict[str, list[str]]:
        """Return records whose source responses changed."""
        return dict(self._changed_endpoints)

    @property
    def record_count(self) -> int:
        """Number of records in cache."""
        return len(self._storage)

    def has_record(self, igme_id: str) -> bool:
        """Check if a record is cached."""
        return igme_id in self._storage

    def record_endpoints(self, igme_id: str) -> list[str]:
        """List cached endpoints for a record."""
        return list(self._storage.get(igme_id, {}).keys())

    def verify_integrity(self) -> dict[str, Any]:
        """Verify cache integrity by re-hashing stored data."""
        corrupted: list[str] = []
        total_entries = 0

        for igme_id, endpoints in self._storage.items():
            for ep_name, ep_data in endpoints.items():
                stored_data = ep_data.get("data")
                stored_hash = ep_data.get("sha256", "")
                if stored_data is None or not stored_hash:
                    continue
                total_entries += 1

                raw = json.dumps(stored_data, sort_keys=True, ensure_ascii=False).encode("utf-8")
                actual_hash = compute_sha256(raw)
                if actual_hash != stored_hash:
                    corrupted.append(f"{igme_id}/{ep_name}")

        return {
            "total_entries": total_entries,
            "corrupted": len(corrupted),
            "corrupted_entries": corrupted,
            "valid": len(corrupted) == 0,
        }
