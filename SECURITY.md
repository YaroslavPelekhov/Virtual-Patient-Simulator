# Security Policy

## Supported Version

This repository is an active research prototype. Security fixes are applied to the latest revision of the `main` branch; older commits are not maintained as supported releases.

## Reporting a Vulnerability

Do not disclose vulnerabilities, credentials, private dialogue data, or exploitable deployment details in a public issue.

Report the issue privately to the repository owner through the private contact channel listed on the owner's GitHub profile. Include:

- the affected commit or deployment version;
- a concise description of the issue;
- reproduction steps using synthetic data;
- the expected security impact;
- any suggested mitigation.

Do not access, copy, or modify data that does not belong to you while investigating a problem.

## High-Risk Areas

Please treat the following as security-sensitive:

- teacher-only case details;
- session histories and JSONL exports;
- API, Telegram, and speech-service credentials;
- unrestricted benchmark endpoints;
- upstream provider error messages;
- permissive CORS and unauthenticated development deployments.

The default configuration is intended for local development. Public deployments require the hardening steps described in `docs/DEPLOYMENT.md`.
