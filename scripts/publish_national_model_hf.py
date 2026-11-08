#!/usr/bin/env python3
"""Publish the national model artifact directory to Hugging Face.

The canonical repository is ``angellopez/aquascan-open-science-model``. Only the
files in ``models/national_v1`` are uploaded: weights, preprocessing, policies,
aggregate metrics, model card, license and checksums. No record-level IGME
dataset is included.

Use ``--archive-current-as v0-5domain`` the first time the national model
replaces the earlier five-domain baseline, so the old model remains available
at a tag.

Usage:
    python scripts/publish_national_model_hf.py --archive-current-as v0-5domain
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = ROOT / "models/national_v1"
DEFAULT_REPO = "angellopez/aquascan-open-science-model"
CHECKSUM_PATTERN = re.compile(r"^([0-9a-f]{64})  \./(.+)$")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_local_checksums(folder: Path) -> dict[str, str]:
    sums_path = folder / "SHA256SUMS"
    if not sums_path.exists():
        raise FileNotFoundError(sums_path)
    expected: dict[str, str] = {}
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        match = CHECKSUM_PATTERN.match(line)
        if not match:
            raise ValueError(f"invalid SHA256SUMS line: {line!r}")
        digest, name = match.groups()
        expected[name] = digest
    if not expected:
        raise ValueError("SHA256SUMS is empty")
    for name, digest in expected.items():
        path = folder / name
        if not path.exists():
            raise FileNotFoundError(path)
        actual = _sha256_file(path)
        if actual != digest:
            raise ValueError(f"local hash mismatch for {name}: {actual} != {digest}")
    return expected


def _verify_remote_checksums(repo_id: str, expected: dict[str, str]) -> None:
    remote_sums = Path(hf_hub_download(repo_id, "SHA256SUMS", force_download=True))
    remote_expected: dict[str, str] = {}
    for line in remote_sums.read_text(encoding="utf-8").splitlines():
        match = CHECKSUM_PATTERN.match(line)
        if not match:
            raise ValueError(f"invalid remote SHA256SUMS line: {line!r}")
        digest, name = match.groups()
        remote_expected[name] = digest
    if remote_expected != expected:
        missing = sorted(set(expected) - set(remote_expected))
        extra = sorted(set(remote_expected) - set(expected))
        raise ValueError(f"remote SHA256SUMS differs: missing={missing} extra={extra}")
    for name, digest in remote_expected.items():
        path = Path(hf_hub_download(repo_id, name, force_download=True))
        actual = _sha256_file(path)
        if actual != digest:
            raise ValueError(f"remote hash mismatch for {name}: {actual} != {digest}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument(
        "--archive-current-as",
        default=None,
        metavar="TAG",
        help="Tag the current main commit before uploading (e.g. v0-5domain).",
    )
    parser.add_argument("--private", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    folder = args.folder.resolve()
    if not folder.is_dir():
        raise NotADirectoryError(folder)
    expected = _verify_local_checksums(folder)
    print(f"[verify] local SHA256SUMS OK ({len(expected)} files)")

    api = HfApi()
    identity = api.whoami()
    print(f"[auth] {identity.get('name', 'unknown')}")

    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )
    print(f"[repo] {args.repo_id} ready")

    if args.archive_current_as:
        api.create_tag(
            repo_id=args.repo_id,
            tag=args.archive_current_as,
            revision="main",
            exist_ok=True,
        )
        refs = api.list_repo_refs(args.repo_id)
        archived = {tag.name: tag.target_commit for tag in refs.tags}
        print(
            f"[tag] {args.archive_current_as} -> {archived.get(args.archive_current_as, 'unknown')}"
        )

    api.upload_folder(
        folder_path=str(folder),
        repo_id=args.repo_id,
        repo_type="model",
        commit_message=("Publish national depth-to-water model v1 as the canonical model"),
    )
    print("[upload] complete; verifying remote hashes ...")
    _verify_remote_checksums(args.repo_id, expected)

    info = api.model_info(args.repo_id, files_metadata=True)
    print(f"[verify] remote SHA256SUMS OK ({len(info.siblings)} files)")
    print(f"[done] https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
