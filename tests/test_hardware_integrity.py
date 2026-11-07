"""Hardware integrity and claim-control tests.

Covers:
- RAW immutability + tamper detection (synthetic fixture)
- canonical RAW is never modified by tests
- public documentation contains no unsupported groundwater claims
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw" / "aq_test_001_immutability_test"

# Unqualified achievement claims. Qualified mentions (e.g. "does not detect
# groundwater" or "...not yet measured") are allowed and are guarded by
# test_required_disclaimers. The real risk is fabricated numbers, so the
# guard blocks numeric accuracy/success claims.
FORBIDDEN_PATTERNS = [
    re.compile(r"groundwater detection accuracy\s*(?:of|:)?\s*\d"),
    re.compile(r"success(?:ful|fully)? (?:detects|finds) groundwater"),
    re.compile(r"validated groundwater ai"),
    re.compile(r"\b\d{1,3}(?:\.\d+)?\s*%\s*(?:accuracy|success rate|detection rate)"),
    re.compile(r"has been proven to (?:detect|find) groundwater"),
]


def test_immutability_tamper_detection() -> None:
    """Copies verify; a tampered copy must fail integrity verification."""
    raw_file = RAW_DIR / "RAW_synthetic.bin"
    archive = RAW_DIR / "central_archive_copy.bin"
    assert raw_file.exists()
    raw_hash = hashlib.sha256(raw_file.read_bytes()).hexdigest()
    archive_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
    # The archive copy was deliberately tampered by the pipeline demo
    assert archive_hash != raw_hash, "tampered copy must not match RAW"


def test_canonical_raw_not_modified_by_test() -> None:
    """The immutability test only tampers a COPY — RAW stays pristine."""
    raw_file = RAW_DIR / "RAW_synthetic.bin"
    local = RAW_DIR / "local_nvme_copy.bin"
    assert raw_file.read_bytes() == local.read_bytes()


def test_public_docs_have_no_unsupported_claims() -> None:
    """README and docs must not claim real groundwater detection capability."""
    targets = [Path("README.md"), *sorted(Path("docs").glob("*.md"))]
    for path in targets:
        text = path.read_text().lower()
        for pattern in FORBIDDEN_PATTERNS:
            assert not pattern.search(text), (
                f"{path} contains unsupported numeric claim: {pattern.pattern}"
            )


def test_required_disclaimers_present() -> None:
    """The core disclaimers must remain in the README."""
    readme = Path("README.md").read_text().lower()
    assert 'does **not** "detect groundwater"' in readme
    assert "not yet performed" in readme
    assert "not validated" in readme


def test_synthetic_fixtures_clearly_labeled() -> None:
    """Synthetic fixtures must be labeled, never presented as field data."""
    from aquascan.physical.synthetic.fixtures import SYNTHETIC_WARNING

    assert "SYNTHETIC_ENGINEERING_FIXTURE" in SYNTHETIC_WARNING
    assert "NOT Ground Truth" in SYNTHETIC_WARNING
