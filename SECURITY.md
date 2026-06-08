# Security Policy

Wallie is designed to be **local-first**. It runs on your own machine, talks to the
providers you configure with **your own API keys**, and does not phone home. That
keeps the security surface small — but if you find a problem, we want to hear about
it.

## Supported versions

Wallie ships from `main`. Security fixes land on `main`; please test against the
latest commit before reporting.

## Reporting a vulnerability

**Please do not open a public issue, PR, or discussion for security problems.**

Report privately via GitHub's **"Report a vulnerability"** button under the repo's
**Security** tab (Security Advisories). Include:

- a description of the issue and its impact,
- steps to reproduce (a minimal case is ideal),
- the affected version/commit and your OS,
- any suggested fix, if you have one.

We aim to acknowledge reports within a few days and will keep you updated on the fix.
Responsible disclosure is appreciated — please give us a chance to ship a fix before
going public.

## Security model & good practices

How Wallie handles sensitive data, so you know what to expect:

- **API keys** live in a local `.env` file (created from `.env.example`), with
  restricted permissions (`chmod 600` on POSIX). They are never committed — `.env`
  is in `.gitignore`.
- **The dashboard never exposes raw keys** — only masked previews (e.g. `sk-•••xyz`).
- **The dashboard binds to `127.0.0.1` only** and is not reachable from the network
  by default. An optional PIN gate exists for when you do expose it.
- **Provider error messages are scrubbed** of key-shaped strings before display.
- **The allowed env variables are hard-coded** — the UI cannot write arbitrary keys.
- **Atomic writes** are used for `.env` to prevent corruption.

### If you self-expose the dashboard

Do **not** put the dashboard on a public network without a reverse proxy that adds
authentication and TLS. The built-in PIN is a convenience, not a hardened auth system.

### Your responsibility

- Keep your `.env` and `client_secret.json` private; never share screen recordings
  that show them.
- Use API keys scoped to the minimum you need, and rotate them if leaked.

## Out of scope

- Vulnerabilities in third-party providers (OpenAI, Anthropic, Fish, ElevenLabs, etc.)
  — report those to the respective vendors.
- Issues that require an already-compromised local machine.

Thanks for helping keep Wallie and its users safe.
