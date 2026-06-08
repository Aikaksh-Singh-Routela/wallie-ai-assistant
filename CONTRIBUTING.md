# Contributing to Wallie

Thanks for being here. Wallie is an open-source AI streamer framework, and it gets
better every time someone files a sharp bug report or sends a focused PR. Whether
you're fixing a typo, adding an LLM/TTS provider, or designing a new persona — you're
welcome.

## Ways to contribute

- **🐛 Bug reports** — the most valuable thing you can do. Include your OS, the
  profile/providers you used, and the relevant log lines.
- **✨ Features & providers** — new LLM/TTS adapters, chat platforms, avatar backends.
- **🎭 Personas** — a great character is content. Share a profile YAML in a PR or issue.
- **📝 Docs** — if something tripped you up, it'll trip up the next person too.
- **💡 Ideas** — open a discussion. See the Roadmap in the README for direction.

New here? Look for issues labeled **`good first issue`**.

## Dev setup

```bash
git clone https://github.com/Alradyin/wallie-V2.git
cd wallie-V2
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python wallie.py --dashboard        # opens http://127.0.0.1:8765
```

Run the test suite and the import check before opening a PR:

```bash
python -m pytest -q
python scripts/_import_check.py
```

## Ground rules (please read)

These keep the project sane — most were learned the hard way:

- **Code is English-only.** Comments, logs, variables, identifiers — all English.
  The UI can be localized.
- **Single pipeline, single history.** Don't add parallel generation paths.
  Everything flows through one orchestrator, one message list, one output path.
  That road has been walked; it leads to an AI that contradicts itself.
- **Lazy imports for optional deps.** A Groq user shouldn't need `anthropic`
  installed. Import provider/optional SDKs inside the function that uses them.
- **Keep it local-first & BYOK.** No telemetry, no phoning home, no hosted lock-in.
  Keys stay on the user's machine.
- **Match the surrounding style.** No new formatters/linters in a feature PR.

## Pull requests

1. Branch off `main`.
2. Keep the PR focused — one logical change. Smaller PRs get merged faster.
3. Describe **what** changed and **why**, and how you tested it.
4. Make sure `pytest` and `_import_check.py` pass.
5. If it changes behavior, update the README/docs in the same PR.

## Reporting security issues

Please **do not** open a public issue for security problems. See
[SECURITY.md](SECURITY.md) for private disclosure.

## License

By contributing, you agree your contributions are licensed under the project's
[MIT License](LICENSE).
