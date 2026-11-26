# Xbox 360 (xsdk): the Rust test title

Step 1 of [`XBOX360-XSDK.md`](../XBOX360-XSDK.md): a custom Rust target for the console and a tiny `no_std` title
on [xsdk](https://github.com/xenosworks/xsdk), packed into a `.xex` with xsdk's `tools/elf2xex.py`. It uses xsdk,
its GCC and the open-source pieces around it only: no Microsoft tooling or SDK material.

```
xbox360/build.sh        # -> xbox360/hello/build/rust_hello.xex
```

The build needs nightly Rust with `rust-src` (`rustup component add rust-src --toolchain nightly`), the xsdk
toolchain at `~/xsdk-gcc` (`XSDK_GCC` overrides it), and a built xsdk checkout (`make` in it). The default is
openmw-xenon's copy (`XSDK` overrides it).

The title clears the screen to a colour the controller sets: left stick x sets red, right stick y green, the right
trigger blue, and the left trigger dims all three. Holding A rumbles and Back ends the title, as xsdk's own `hello`
sample does. Like xsdk's samples, it reports on itself in `rust_hello.log` beside the `.xex`:

- `PASS: rust_hello: 64-bit arithmetic`: LLVM's 64-bit integer code on 32-bit PowerPC, which goes through
  compiler-builtins, matches a value `build.rs` computed on the PC.
- `PASS: rust_hello: float arithmetic`.
- `PASS: rust_hello: the CPU is big-endian`.
- `PASS: rust_hello: the first resolved frame holds the colour it was cleared to`: the first pixel, read back from the
  front buffer.
- `checks: N passed, M failed`, then xsdk's `exit: status 0`.

**Not yet run on the console.** It builds, links and packs, and the checks below were made on the build. Running it
means copying `rust_hello.xex` into a folder of its own on the console, such as `Hdd1:\xsdk\rust_hello\`, with
GETTING_STARTED.md section 6's `curl -T`, and launching it from Aurora. Then read the log, as section 7 describes.

## Files

| File | What it is |
|---|---|
| `powerpc-xsdk-none.json` | The target: 32-bit big-endian PowerPC, `os` none, static, `panic` = abort, AltiVec off, LLVM CPU `ppc` |
| `hello/src/lib.rs` | The title, a `no_std` `staticlib` |
| `hello/bridge.c` | The C side: `main`, the thunks both ways, and three helpers |
| `tools/rust_object.py` | Turns the Rust archive into one object that xsdk's linker takes |
| `build.sh` | cargo with `-Zbuild-std`, then `rust_object.py`, then xsdk's GCC link, `xenia_scan`, and `elf2xex.py` |

## The calling convention boundary

This is the finding that shapes the whole port. xsdk's GCC does not use the PowerPC SysV ABI. Its default,
`-mcall-xsdk`, is the console's own compiler's convention (`tools/toolchain/gcc-xsdk.patch`, `rs6000.h`):

- arguments go in 8-byte slots, AIX-style, and a 64-bit integer takes one register;
- every caller leaves a 64-byte parameter save area at 16(r1), after a 16-byte fixed area;
- r14–r31 are callee-saved **whole, 64 bits** (`std`/`ld`); v14–v31 are kept;
- the receiver widens narrow values, rather than the caller;
- binutils marks such objects `EF_PPC_XSDK` (0x100) and refuses to link them with any others.

LLVM can't emit this convention. Its 32-bit PowerPC is SysV: 64-bit integers in register pairs, an 8-byte frame
header, and r14–r31 saved with `stw` and restored with `lwz`, which zeroes the high words xsdk's code keeps there.
Titles run in 32-bit mode (MSR.SF clear, docs/HARDWARE.md), so addresses ignore high words. That's why SysV code can
run here at all; but the two conventions can't call each other directly. So the boundary is explicit:

1. **One way in.** `bridge.c`'s `main` calls `rust_title_main` through `enter_rust`, an assembly thunk that saves
   r14–r31 as 64-bit around the call.
2. **Only thunked ways out.** `rust_object.py` links the Rust archive into one relocatable object (`ld -r`, garbage
   collected from the entry). It makes every global but the entry local, so Rust's own `memcpy`/`memset` from
   compiler-builtins can't stand in for newlib's. It then renames every remaining undefined symbol `X` to `__rs2c_X`.
   `bridge.c` defines `__rs2c_X` only for the calls it allows. Each one calls `X` from an xsdk-shaped frame (16
   fixed bytes plus the 64-byte parameter save area), so a varargs or address-taking callee spills into the thunk's
   frame, not Rust's locals.
3. **The flag.** The object then gets `EF_PPC_XSDK`. That flag describes the object's interface, which steps 1 and 2
   make true. The code inside is still SysV.
4. **Types across it:** `i32`, `u32`, pointers and doubles only. No 64-bit integers, which go in different registers,
   and no `u8`/`u16`/`bool` results, because xsdk leaves them unwidened where LLVM assumes they're widened. That's why
   `XamInputSetState`'s struct of two `u16`s and xsdk's varargs log are wrapped in `bridge.c`.

Checked on this build:
- A Rust call with no thunk fails the link (`undefined reference to '__rs2c_xsdk_gpu_close'`, tried by removing that
  one thunk).
- `readelf` shows `xsdk call` on both `rust.o` and the `.elf`.
- In the Rust object's disassembly there is no r2 and no r13: LLVM's SVR4 reserves both, and on the console r13 is
  the thread's pointer, which xsdk's TLS and `errno` read. There are also no VMX instructions and no 64-bit
  instructions.
- `rust.o` defines `memcpy` and `memset` only as locals, and `rust_title_main` is its only global.
- `xenia_scan` flags one `mcrf` in `core::fmt`'s integer formatting, which Xenia skips and the console runs.

What remains unchecked until it runs on the console: CR fields 2–4, which SysV keeps callee-saved and xsdk's GCC never
allocates (`-ffixed-cr2`…`cr7`), so C can't disturb Rust's; f14–f31, which both conventions keep; and a float load at
an unaligned address. xsdk's GCC moves such floats through integer registers, but LLVM doesn't know to. Rust only
loads floats through aligned references, but `read_unaligned` of an `f32` is a risk to check.
