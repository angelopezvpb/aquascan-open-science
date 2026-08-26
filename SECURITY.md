# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in AquaScan Open Science, please report it responsibly.

**Do not open a public GitHub issue.**

Instead, contact the project maintainers directly.

## Scope

This security policy covers the public AquaScan Open Science repository and its code, dependencies, and data handling. Production infrastructure, proprietary models, and client data are out of scope for this public repository.

## What This Repository Does NOT Contain

By design, this repository never stores:

- API keys, tokens, or passwords
- Private client coordinates or borehole data
- Owner names or personal information
- Production model weights
- Proprietary calibration algorithms
- Authentication credentials of any kind

If you discover any of the above accidentally committed, report it immediately.

## Data Security

- RAW data is immutable by design
- All downloads are checksummed (SHA-256)
- The `data/` directory is in `.gitignore` for large binary files
- `.env` files are excluded from version control

## Best Practices for Contributors

- Never commit credentials, even in test files
- Use environment variables for any API access
- Do not commit data files larger than a few KB
- Verify checksums when downloading external data