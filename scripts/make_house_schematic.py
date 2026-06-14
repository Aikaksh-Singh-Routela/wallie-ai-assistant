"""Generate a simple cobblestone starter-house schematic (Sponge .schem v2) that
Baritone can build with '#build wallie_house'. Cobblestone is mined directly (no
crafting), so the whole gather->build flow is fully autonomous.

  python scripts/make_house_schematic.py

Writes to %APPDATA%/.minecraft/schematics/wallie_house.schem
Then in-game (standing where you want it):  #build wallie_house ~ ~ ~
"""
from __future__ import annotations

import gzip
import os
import struct
from pathlib import Path

# ---- minimal NBT (big-endian) ----
T_BYTE, T_SHORT, T_INT, T_BYTEARRAY, T_STRING, T_COMPOUND, T_INTARRAY = 1, 2, 3, 7, 8, 10, 11


def _name(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack(">H", len(b)) + b


def _tag(t: int, name: str, payload: bytes) -> bytes:
    return bytes([t]) + _name(name) + payload


def _compound(entries: list[bytes]) -> bytes:
    return b"".join(entries) + bytes([0])      # entries already include their End-less payloads


def build_house(w: int = 5, l: int = 5, h: int = 4) -> tuple[int, int, int, dict, bytes]:
    AIR, COBBLE = 0, 1
    # wood shelter — built from planks the bot crafts early (no mining needed)
    palette = {"minecraft:air": AIR, "minecraft:oak_planks": COBBLE}
    data = bytearray(w * l * h)

    def idx(x: int, y: int, z: int) -> int:
        return (y * l + z) * w + x

    for x in range(w):
        for z in range(l):
            for y in range(h):
                solid = False
                if y == 0 or y == h - 1:                 # floor + roof
                    solid = True
                elif x == 0 or x == w - 1 or z == 0 or z == l - 1:   # walls
                    solid = True
                # door gap: front wall (z==0), middle x, lower two rows
                if z == 0 and x == w // 2 and 1 <= y <= 2:
                    solid = False
                data[idx(x, y, z)] = COBBLE if solid else AIR
    return w, h, l, palette, bytes(data)


def main() -> None:
    w, h, l, palette, blockdata = build_house()

    pal_entries = [_tag(T_INT, name, struct.pack(">i", i)) for name, i in palette.items()]
    root = _compound([
        _tag(T_INT, "Version", struct.pack(">i", 2)),
        _tag(T_INT, "DataVersion", struct.pack(">i", 3953)),  # 1.21.x; blocks are by-name so this is informational
        _tag(T_SHORT, "Width", struct.pack(">h", w)),
        _tag(T_SHORT, "Height", struct.pack(">h", h)),
        _tag(T_SHORT, "Length", struct.pack(">h", l)),
        _tag(T_INTARRAY, "Offset", struct.pack(">i", 3) + struct.pack(">iii", 0, 0, 0)),
        _tag(T_INT, "PaletteMax", struct.pack(">i", len(palette))),
        _tag(T_COMPOUND, "Palette", _compound(pal_entries)),
        _tag(T_BYTEARRAY, "BlockData", struct.pack(">i", len(blockdata)) + blockdata),
        _tag(T_COMPOUND, "Metadata", _compound([])),
    ])
    nbt = _tag(T_COMPOUND, "Schematic", root)

    out_dir = Path(os.environ["APPDATA"]) / ".minecraft" / "schematics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "wallie_house.schem"
    with gzip.open(out, "wb") as f:
        f.write(nbt)
    blocks = sum(1 for b in blockdata if b != 0)
    print(f"wrote {out}")
    print(f"size {w}x{h}x{l}, {blocks} cobblestone needed")
    print('build in-game with:  #build wallie_house ~ ~ ~')


if __name__ == "__main__":
    main()
