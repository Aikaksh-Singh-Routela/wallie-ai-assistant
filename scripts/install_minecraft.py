"""One-click Minecraft Play setup — downloads everything Wallie needs to PLAY Minecraft.

It installs the Fabric loader, pulls the public mods (Fabric API + Meteor Client) from Modrinth
for the target version, and drops in the bundled Wallie jars (smoothcam + Baritone) from
<repo>/dist/mods/. Run it once; then launch the Fabric 1.21.11 profile and start Wallie Play.

  python scripts/install_minecraft.py
  python scripts/install_minecraft.py --mc 1.21.11

Safe to re-run: it backs up your current mods folder first. Stdlib only (urllib), no extra deps.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

MC_VERSION = "1.21.11"
MODRINTH = "https://api.modrinth.com/v2"
PUBLIC_MODS = ["fabric-api"]          # pulled from Modrinth; Meteor/Baritone/smoothcam are bundled
UA = {"User-Agent": "WallieInstaller/1.0 (github.com/wallie)"}


def _log(msg: str) -> None:
    print(f"[install] {msg}", flush=True)


def _minecraft_dir() -> Path:
    if sys.platform == "win32":
        return Path(os.environ["APPDATA"]) / ".minecraft"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "minecraft"
    return Path.home() / ".minecraft"


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    _log(f"downloading {dest.name} ...")
    dest.write_bytes(_get(url))


def _modrinth_file(project: str, mc: str) -> tuple[str, str] | None:
    q = urllib.parse.urlencode({"game_versions": json.dumps([mc]), "loaders": json.dumps(["fabric"])})
    try:
        versions = json.loads(_get(f"{MODRINTH}/project/{project}/version?{q}"))
    except Exception as e:
        _log(f"  ! Modrinth lookup failed for {project}: {e}")
        return None
    if not versions:
        _log(f"  ! no {project} build for {mc} yet")
        return None
    files = versions[0].get("files", [])
    primary = next((f for f in files if f.get("primary")), files[0] if files else None)
    return (primary["url"], primary["filename"]) if primary else None


def _install_fabric(mc: str, mc_dir: Path) -> None:
    installer = mc_dir / "wallie-fabric-installer.jar"
    if shutil.which("java") is None:
        _log("  ! Java not found on PATH — install Java 21+ (the game needs it anyway), then re-run.")
        return
    try:
        meta = json.loads(_get("https://meta.fabricmc.net/v2/versions/installer"))
        url = meta[0]["url"]
        _download(url, installer)
        _log("running Fabric installer (client, no profile override) ...")
        subprocess.run(["java", "-jar", str(installer), "client",
                        "-mcversion", mc, "-dir", str(mc_dir), "-noprofile"],
                       check=False, timeout=180)
    except Exception as e:
        _log(f"  ! Fabric install step failed ({e}); install Fabric manually from fabricmc.net")
    finally:
        installer.unlink(missing_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Install the Wallie Minecraft Play stack")
    ap.add_argument("--mc", default=MC_VERSION, help="Minecraft version (default 1.21.11)")
    ap.add_argument("--skip-fabric", action="store_true", help="don't install the Fabric loader")
    args = ap.parse_args()

    mc_dir = _minecraft_dir()
    if not mc_dir.exists():
        _log(f"! .minecraft not found at {mc_dir} — launch the vanilla game once, then re-run.")
        return 1
    mods = mc_dir / "mods"

    if mods.exists() and any(mods.iterdir()):
        backup = mc_dir / f"mods.backup.{int(time.time())}"
        shutil.copytree(mods, backup)
        _log(f"backed up existing mods -> {backup.name}")
    mods.mkdir(exist_ok=True)

    if not args.skip_fabric:
        _install_fabric(args.mc, mc_dir)

    ok, miss = [], []
    for proj in PUBLIC_MODS:
        info = _modrinth_file(proj, args.mc)
        if info is None:
            miss.append(proj)
            continue
        url, fname = info
        try:
            _download(url, mods / fname)
            ok.append(fname)
        except Exception as e:
            _log(f"  ! failed {proj}: {e}")
            miss.append(proj)

    dist = Path(__file__).resolve().parent.parent / "dist" / "mods"
    if dist.is_dir():
        for jar in dist.glob("*.jar"):
            shutil.copy2(jar, mods / jar.name)
            ok.append(jar.name)
            _log(f"bundled {jar.name}")
    else:
        miss.append("smoothcam+baritone (bundled jars in dist/mods/ not found)")

    _log("")
    _log(f"DONE. mods folder: {mods}")
    _log(f"installed: {', '.join(ok) if ok else '(none)'}")
    if miss:
        _log(f"still needed: {', '.join(miss)}")
    _log("Next: open the Minecraft launcher, pick the 'fabric-loader-" + args.mc +
         "' profile, join a survival world, then start Wallie Play.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
