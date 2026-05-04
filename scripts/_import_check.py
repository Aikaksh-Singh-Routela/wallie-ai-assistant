"""Dev-time sanity check: all intra-project 'from X import Y' targets resolve."""
import ast
import pathlib
import sys

EXTERNAL_PREFIXES = (
    "openai", "anthropic", "google", "httpx", "sounddevice", "mss", "imagehash",
    "PIL", "numpy", "loguru", "pydantic", "dotenv", "fastapi", "uvicorn",
    "websockets", "yaml", "orjson", "asyncio", "typing", "dataclasses",
    "collections", "pathlib", "random", "time", "os", "io", "json", "base64",
    "re", "difflib", "threading", "datetime", "argparse", "sys",
    "googleapiclient", "google_auth_oauthlib",
)

INTRA_ROOTS = (
    "core", "llm", "tts", "audio", "vision", "chat", "dashboard", "utils",
    "config", "wallie",
)


def main() -> int:
    root = pathlib.Path(__file__).resolve().parent.parent
    files = [p for p in root.rglob("*.py") if ".venv" not in p.parts]
    modules: set[str] = set()
    for p in files:
        rel = p.relative_to(root).with_suffix("")
        parts = list(rel.parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        modules.add(".".join(parts))
        # Also register package path prefixes.
        for i in range(1, len(parts)):
            modules.add(".".join(parts[:i]))

    issues: list[str] = []
    for p in files:
        tree = ast.parse(p.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.level > 0:
                continue
            mod = node.module or ""
            if not mod:
                continue
            if mod.startswith(EXTERNAL_PREFIXES):
                continue
            if not mod.startswith(INTRA_ROOTS):
                continue
            if mod not in modules:
                issues.append(f"{p.relative_to(root)}: cannot resolve 'from {mod} import ...'")

    if issues:
        for i in issues:
            print(i)
        return 1
    print(f"OK — {len(files)} files, {len(modules)} modules, no dangling intra-project imports")
    return 0


if __name__ == "__main__":
    sys.exit(main())
