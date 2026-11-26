#!/usr/bin/env python3
"""Turns a Rust staticlib built for powerpc-xsdk-none into one object xsdk's linker takes, behind the boundary.

    rust_object.py <prefix> <lib.a> <out.o> <entry> [<entry> ...]

<prefix> is the toolchain's, such as ~/xsdk-gcc/bin/powerpc-xsdk-elf-. The steps (xbox360/README.md, "The calling
convention boundary"):

1. `ld -r` links the archive's members into one relocatable object, keeping only what the entries reach, so Rust's
   own runtime (core, compiler-builtins and its memcpy) resolves inside it.
2. Every global but the entries is made local: nothing of Rust's - its memcpy among them - can stand in for newlib's
   or xsdk's.
3. Every symbol still undefined, X, is renamed __rs2c_X, which only bridge.c's thunks define: a Rust call into C that
   has no thunk fails the final link instead of running with the wrong calling convention.
4. The ELF header gets EF_PPC_XSDK, the flag xsdk's binutils uses to keep conventions apart. That is a claim about the
   object's *interface*, which steps 2 and 3 make true; the code inside stays SysV.
"""
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

EF_PPC_XSDK = 0x100
E_FLAGS_OFFSET = 0x24  # ELF32


def run(*args):
    subprocess.run(args, check=True)


def undefined_symbols(nm, obj):
    out = subprocess.run([nm, "-u", "--format=posix", obj], check=True, capture_output=True, text=True).stdout
    return sorted({line.split()[0] for line in out.splitlines() if line.strip()})


def main():
    if len(sys.argv) < 5:
        sys.exit(__doc__)
    prefix, archive, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    entries = sys.argv[4:]
    with tempfile.TemporaryDirectory() as tmp:
        linked = Path(tmp) / "linked.o"
        roots = [f"--undefined={e}" for e in entries]
        run(f"{prefix}ld", "-r", "--gc-sections", *roots, "--whole-archive", archive, "-o", str(linked))

        undefined = undefined_symbols(f"{prefix}nm", str(linked))
        renames = Path(tmp) / "renames.txt"
        renames.write_text("".join(f"{s} __rs2c_{s}\n" for s in undefined))
        keep = [f"--keep-global-symbol={e}" for e in entries]
        run(f"{prefix}objcopy", *keep, f"--redefine-syms={renames}", str(linked), str(out))

    data = bytearray(out.read_bytes())
    if data[:4] != b"\x7fELF" or data[4] != 1 or data[5] != 2:
        sys.exit(f"{out}: not a 32-bit big-endian ELF object")
    (flags,) = struct.unpack_from(">I", data, E_FLAGS_OFFSET)
    struct.pack_into(">I", data, E_FLAGS_OFFSET, flags | EF_PPC_XSDK)
    out.write_bytes(bytes(data))
    print(f"rust_object: {out.name}: entries {', '.join(entries)}; calls out through __rs2c_ thunks to: "
          + (", ".join(undefined) or "nothing"))


if __name__ == "__main__":
    main()
