# Security Policy

VoiceLayer is a local-first voice composition layer for Ubuntu desktops. It
runs as an unprivileged user-level daemon and processes audio captured from
the local microphone with locally hosted or optional remote models. There is
no managed VoiceLayer service, no shared multi-tenant deployment, and no
public network endpoint operated by the project.

## Supported Versions

| Version | Status                  |
| ------- | ----------------------- |
| 0.1.x   | Supported (alpha)       |
| < 0.1   | Unsupported             |

VoiceLayer is in early development. Releases are tagged from `main`. Security
fixes target the latest 0.1.x line. Older preview builds will not receive
backports.

## Threat Model

Because execution is local, the practical threat surface is concentrated in:

- Malicious or malformed audio input that reaches a transcription, rewrite,
  or translation worker (parsing bugs, decoder crashes, prompt injection
  embedded in transcripts).
- Untrusted model weights or worker binaries dropped into the cache
  directories. Do not submit, distribute, or load model artefacts that have
  not been verified against the upstream publisher's checksum.
- Misconfigured host integrations that could leak transcripts to unintended
  windows, terminals, or remote endpoints.
- Optional remote provider integrations (when explicitly enabled) that may
  transmit captured audio or text off-host.

Issues outside this scope, such as denial of service on a contributor's
personal laptop or attacks that require root on the user's machine, are
generally out of scope.

## Reporting a Vulnerability

Please report suspected vulnerabilities through GitHub's Private
Vulnerability Reporting channel:

1. Open <https://github.com/memenow/voice-layer/security>.
2. Choose "Report a vulnerability".
3. Provide a description, reproduction steps, affected version, and any
   logs with secrets redacted.

We aim to acknowledge new reports on a best-effort basis within seven days
and to share a remediation or mitigation plan once the issue is confirmed.
Coordinated disclosure timelines are negotiated per report.

Please do not open a public issue, pull request, or discussion thread for
unfixed security findings.

## Disclosure

Once a fix ships, the affected release notes will describe the issue at a
level of detail that lets operators assess exposure without enabling
reproduction against unpatched hosts.
