#!/usr/bin/env python3
"""The console's GLSL -> SPIR-V table, from a shader dump's xbox360/ folder (SHOCK2_SHADER_DUMP; xbox360/README.md,
"Shaders"): the Xbox 360 has no shaderc, so `engine::vk::shader::compile` looks each source up here instead.

    shader_table.py <dump>/xbox360 <out.bin>

Little-endian: b"S2ST", the entry count (u32), then each entry: the source's hash (u64, shader::source_hash), its
stage (u32: 0 vertex, 1 fragment), its word count (u32) and its words. Entries are sorted by hash and stage.
"""
import struct
import sys
from pathlib import Path

STAGES = {"vert": 0, "frag": 1}


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    entries = []
    for path in Path(sys.argv[1]).glob("*.compiled.spv"):
        source_hash, stage = path.name.split(".")[:2]
        words = path.read_bytes()
        entries.append((int(source_hash, 16), STAGES[stage], words))
    entries.sort(key=lambda e: e[:2])
    out = bytearray(b"S2ST" + struct.pack("<I", len(entries)))
    for source_hash, stage, words in entries:
        out += struct.pack("<QII", source_hash, stage, len(words) // 4) + words
    Path(sys.argv[2]).write_bytes(out)
    print(f"{len(entries)} sources, {len(out)} bytes")


if __name__ == "__main__":
    main()
