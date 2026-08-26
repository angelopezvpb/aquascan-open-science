# Software License Review

This document compares candidate open-source licenses for the public release.
It does **not** select a license silently — the final choice requires human
confirmation.

## Candidates compared

| Criterion | MIT | Apache-2.0 |
|---|---|---|
| Redistribution | Yes | Yes |
| Attribution required | Yes | Yes |
| Patent grant | No | Explicit (sections 3–4) |
| Compatibility with physical-stack deps | Good | Good |
| Future commercial use | Allowed | Allowed |
| Contributor implications | Minimal | Explicit contribution license |

## Recommendation (pending human confirmation)
**Apache-2.0** is the current declared license in `pyproject.toml` and
`LICENSE`. It is recommended over MIT because the project anticipates
hardware/physical-method development where an explicit patent grant reduces
uncertainty for contributors and downstream users.

## Dependency compatibility check
All core dependencies (BSD-3, MIT, Apache-2.0, PSF) are compatible with
Apache-2.0 redistribution. See the dependency-license table produced by the
release build (and `reports/gate10_license_manifest_v0.csv`) for the full list.

> **Action for human reviewers:** confirm the license choice and update
> `LICENSE` / `CITATION.cff` / `pyproject.toml` consistently before the
> first official release tag.
