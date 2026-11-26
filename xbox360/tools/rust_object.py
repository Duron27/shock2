#!/usr/bin/env python3
"""Turns a Rust staticlib built for powerpc64-xsdk-ilp32 into one object for xsdk's linker.

    rust_object.py <prefix> <lib.a> <out.o> <entry> [<entry> ...]

<prefix> is the xsdk toolchain's, such as ~/xsdk-gcc/bin/powerpc-xsdk-elf-.

1. `ld -r` links the archive's members into one relocatable object, keeping only what the entries reach, so Rust's
   own runtime (core, alloc, compiler-builtins) resolves inside it.
2. Every global but the entries is made local, so nothing of Rust's - compiler-builtins' memcpy and memset among
   them - can stand in for newlib's or xsdk's, nor collide with them.
3. The ELF header gets EF_PPC_XSDK, the flag xsdk's binutils uses to keep calling conventions apart. xenon-forge's
   PPC64-ILP32 LLVM emits the console's convention, which -mcall-xsdk is too (xbox360/README.md), but LLVM doesn't
   know the flag.
"""
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

EF_PPC_XSDK = 0x100
E_FLAGS_OFFSET = 0x24  # ELF32
EM_PPC = 20


def main():
    if len(sys.argv) < 5:
        sys.exit(__doc__)
    prefix, archive, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    entries = sys.argv[4:]
    with tempfile.TemporaryDirectory() as tmp:
        linked = Path(tmp) / "linked.o"
        roots = [f"--undefined={e}" for e in entries]
        subprocess.run([f"{prefix}ld", "-r", "--gc-sections", *roots, "--whole-archive", archive, "-o", str(linked)],
                       check=True)
        keep = [f"--keep-global-symbol={e}" for e in entries]
        subprocess.run([f"{prefix}objcopy", *keep, str(linked), str(out)], check=True)

    data = bytearray(out.read_bytes())
    if data[:4] != b"\x7fELF" or data[4] != 1 or data[5] != 2:
        sys.exit(f"{out}: not a 32-bit big-endian ELF object")
    (machine,) = struct.unpack_from(">H", data, 0x12)
    if machine != EM_PPC:
        sys.exit(f"{out}: machine {machine}, not PowerPC")
    (flags,) = struct.unpack_from(">I", data, E_FLAGS_OFFSET)
    struct.pack_into(">I", data, E_FLAGS_OFFSET, flags | EF_PPC_XSDK)
    out.write_bytes(bytes(data))

    nm = subprocess.run([f"{prefix}nm", "-u", "--format=posix", str(out)], check=True, capture_output=True,
                        text=True).stdout
    calls = sorted({line.split()[0] for line in nm.splitlines() if line.strip()})
    print(f"rust_object: {out.name}: entries {', '.join(entries)}; calls {len(calls)} C functions: {', '.join(calls)}")


if __name__ == "__main__":
    main()
