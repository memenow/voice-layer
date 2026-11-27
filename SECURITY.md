# Security Policy

VoiceLayer is a local-first voice composition layer for Ubuntu and macOS (Apple Silicon) desktops.
It runs as an unprivileged user-level daemon and processes microphone audio using local or optional remote models.
There is no managed VoiceLayer service, multi-tenant deployment, or public network endpoint operated by the project.

## Supported Versions

| Version | Status      |
| ------- | ----------- |
| 0.2.x   | Supported   |
| < 0.2   | Unsupported |

Releases are tagged from `main`. Security fixes target the latest `0.2.x` release without backports to older builds.

## Threat Model

Because execution is local, the practical security surface is concentrated in:

- Malformed audio inputs reaching a transcription, rewrite, or translation worker (parser bugs, decoder crashes, prompt injection embedded in transcripts).
- Untrusted model weights or worker binaries placed in local cache directories. Verify model artifacts against upstream checksums before loading them.
- Misconfigured host adapters that could inject text into an unintended window, terminal pane, or remote endpoint.
- Optional remote provider endpoints (when explicitly configured) that transmit captured audio or text off-host.

Issues requiring local root access or denial of service on a single-user workstation are outside this scope.

## Reporting a Vulnerability

Report suspected vulnerabilities through GitHub Private Vulnerability Reporting:

1. Open <https://github.com/memenow/voice-layer/security>.
2. Choose "Report a vulnerability".
3. Include a description, reproduction steps, affected version, and redacted logs.

Reports are acknowledged on a best-effort basis within seven days, with coordinated disclosure timelines agreed per report.
Do not open public issues or pull requests for unfixed vulnerabilities.

## Disclosure

Release notes for security fixes describe exposure and upgrade steps without publishing exploit steps against unpatched hosts.
