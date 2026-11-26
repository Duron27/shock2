#!/usr/bin/env python3
"""Turns a Rust staticlib built for powerpc64-xsdk-ilp32 into one object for xsdk's linker.

    rust_object.py <prefix> <lib.a> <out.o> <entry> [<entry> ...]

<prefix> is the xsdk toolchain's, such as ~/xsdk-gcc/bin/powerpc-xsdk-elf-.

1. `ld -r` links the archive's members into one relocatable object, keeping only what the entries reach, so Rust's
   own runtime (core, alloc, compiler-builtins) resolves inside it.
2. Every global but the entries is made local, so nothing of Rust's - compiler-builtins' memcpy and memset among
   them - can stand in for newlib's or xsdk's, nor collide with them.
3. Kernel and XAM imports that xenon-forge's std declares (xbox360-sys: one 16-byte `.xthunks.*` section each, the
   import record and then `nop; nop; blr`, which its own `xex-link` collects) become xsdk's: the two record words
   the loader rewrites to load the export into r11, then `mtctr r11; bctr`, each with an `.xsdk.imports` record so
   `elf2xex.py` lists it in the import libraries header.
4. The ELF header gets EF_PPC_XSDK, the flag xsdk's binutils uses to keep calling conventions apart. xenon-forge's
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
# xbox360-sys's lib_index, and the names xsdk's imports.txt gives those libraries.
LIBRARIES = {0: "xboxkrnl.exe", 1: "xam.xex"}
RECORD_THUNK = 0x01000000
RECORD_THUNK_SECOND = 0x02000000
MTCTR_R11 = 0x7D6903A6
BCTR = 0x4E800420


def thunk_sections(prefix, obj):
    """(index, name, size) of each .xthunks.* section."""
    out = subprocess.run([f"{prefix}readelf", "-W", "-S", str(obj)], check=True, capture_output=True, text=True).stdout
    found = []
    for line in out.splitlines():
        fields = line.replace("[ ", "[").split()
        if len(fields) > 6 and fields[1].startswith(".xthunks"):
            found.append((fields[1], int(fields[5], 16)))
    return found


def convert_imports(prefix, obj, tmp):
    """Rewrites xbox360-sys's thunks in obj into xsdk's form; returns how many."""
    sections = thunk_sections(prefix, obj)
    if not sections:
        return 0
    args, symbols, records = [], [], []
    for n, (section, size) in enumerate(sections):
        if size != 16:
            sys.exit(f"{section}: {size} bytes, not one 16-byte thunk")
        dump = Path(tmp) / f"thunk{n}.bin"
        subprocess.run([f"{prefix}objcopy", "-O", "binary", f"--only-section={section}", str(obj), str(dump)],
                       check=True)
        (record,) = struct.unpack_from(">I", dump.read_bytes(), 0)
        kind, lib, ordinal = record >> 24, record >> 16 & 0xFF, record & 0xFFFF
        if kind != 1 or lib not in LIBRARIES:
            sys.exit(f"{section}: record {record:#010x} is not a thunk into a library xsdk knows")
        dump.write_bytes(struct.pack(">IIII", RECORD_THUNK | ordinal, RECORD_THUNK_SECOND | ordinal, MTCTR_R11, BCTR))
        symbol = f"__rust_import_{n}"
        renamed = f".text.rust_import.{n}"
        args += [f"--update-section={section}={dump}", f"--rename-section={section}={renamed},alloc,code,readonly,load,contents"]
        # A second pass, against the new name: --add-symbol splits at the first ':', and the old names have '::'.
        symbols.append(f"--add-symbol={symbol}={renamed}:0,global,function")
        records.append((symbol, LIBRARIES[lib], ordinal, section.rsplit("::", 1)[-1]))
    subprocess.run([f"{prefix}objcopy", *args, str(obj)], check=True)
    subprocess.run([f"{prefix}objcopy", *symbols, str(obj)], check=True)
    # The records, and each import's slot word, which elf2xex checks and the loader writes.
    asm = []
    for symbol, library, ordinal, name in records:
        asm += [f"// {library} {ordinal:#x}: {name}", f"\t.section .rodata.{symbol}, \"a\"", "\t.balign 4",
                f"{symbol}_slot:", f"\t.long {ordinal:#010x}", "\t.section .xsdk.imports, \"\"",
                f"\t.long {symbol}_slot", f"\t.long {symbol}", f"\t.long {ordinal:#x}", "\t.long 0",
                '\t.ascii "' + library + "\\0" * (16 - len(library)) + '"']
    source = Path(tmp) / "imports.S"
    source.write_text("\n".join(asm) + "\n")
    records_obj = Path(tmp) / "imports.o"
    subprocess.run([f"{prefix}gcc", "-mcpu=cell", "-c", str(source), "-o", str(records_obj)], check=True)
    merged = Path(tmp) / "merged.o"
    subprocess.run([f"{prefix}ld", "-r", str(obj), str(records_obj), "-o", str(merged)], check=True)
    obj.write_bytes(merged.read_bytes())
    return len(records)


def set_xsdk_flag(obj):
    data = bytearray(obj.read_bytes())
    if data[:4] != b"\x7fELF" or data[4] != 1 or data[5] != 2:
        sys.exit(f"{obj}: not a 32-bit big-endian ELF object")
    (machine,) = struct.unpack_from(">H", data, 0x12)
    if machine != EM_PPC:
        sys.exit(f"{obj}: machine {machine}, not PowerPC")
    (flags,) = struct.unpack_from(">I", data, E_FLAGS_OFFSET)
    struct.pack_into(">I", data, E_FLAGS_OFFSET, flags | EF_PPC_XSDK)
    obj.write_bytes(bytes(data))


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
        set_xsdk_flag(out)
        imports = convert_imports(prefix, out, tmp)

    nm = subprocess.run([f"{prefix}nm", "-u", "--format=posix", str(out)], check=True, capture_output=True,
                        text=True).stdout
    calls = sorted({line.split()[0] for line in nm.splitlines() if line.strip()})
    print(f"rust_object: {out.name}: entries {', '.join(entries)}; {imports} kernel/XAM imports of its own; "
          f"calls {len(calls)} C functions: {', '.join(calls)}")


if __name__ == "__main__":
    main()
