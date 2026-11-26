# Xbox 360 (xsdk): Rust test titles

The first steps of [`XBOX360-XSDK.md`](../XBOX360-XSDK.md): Rust titles on [xsdk](https://github.com/xenosworks/xsdk),
packed into `.xex` files with xsdk's `tools/elf2xex.py`. Only open-source pieces are used: xsdk, its GCC, vulkan4xenon,
and xenon-forge's compiler. There's no Microsoft tooling or SDK material in any of them.

```
xbox360/build.sh hello      # -> hello/build/rust_hello.xex: clears the screen to a colour the controller sets
xbox360/build.sh triangle   # -> triangle/build/rust_triangle.xex: a triangle through ash on vulkan4xenon
xbox360/build.sh stdcheck   # -> stdcheck/build/rust_stdcheck.xex: Rust's std (threads, locks, files, clocks) on the console
xbox360/ftp.sh push triangle; xbox360/ftp.sh log triangle
```

| Title | What it shows | Ends on |
|---|---|---|
| `rust_hello` | The screen's colour follows the sticks and triggers (xsdk's `hello`, in Rust); A rumbles | Back |
| `rust_triangle` | A turning triangle drawn with `ash` on vulkan4xenon: the left stick moves it, RT grows it, A rumbles | Back |
| `rust_stdcheck` | Nothing on screen: std's threads, `Mutex`/`Condvar`/channels, `thread_local!`, clocks, heap, `HashMap` and files in `game:\rust_stdcheck` | By itself |

Each title writes `rust_<title>.log` beside its `.xex`, with `PASS:`/`FAIL:` lines and `checks: N passed, M failed`,
as xsdk's samples do. On the console they live in `Hdd1\Games\Shock2`, the only folder `ftp.sh` touches. Launch them
from Aurora's file browser. Aurora's FTP is gone while a title runs, so read the log after going back to the
dashboard.

## What the build needs

- **xenon-forge's compiler** at `~/xenon-forge-toolchain` (`XENON_FORGE` overrides it): its prebuilt PPC64-ILP32
  Rust 1.98 and LLVM for Linux x86_64. The pinned archive is nightly `c9997c1d885d`:
  `https://github.com/landaire/xenon-forge/releases/download/toolchain-nightly-c9997c1d885d/xbox360-toolchain-x86_64-unknown-linux-gnu.tar.xz`,
  SHA-256 `720535e797486ab72dc208b65d2cc019b88fbd9d13b176f6d871481cbd73c0ae`; unpack it there. The copy on this PC
  came from openxbdm's Buck2 cache, which had downloaded that same archive.
- **Rust 1.98.0's library source**, for `-Zbuild-std`: `rustup toolchain install 1.98.0 --profile minimal --component
  rust-src`. The build uses that toolchain's cargo with xenon-forge's rustc.
- **xsdk** at `~/xsdk-gcc` (its toolchain) and **xsdk and vulkan4xenon** under `XENOSWORKS` (default: openmw-xenon's
  `sources/xenosworks`), each built with `make`.
- **glslang** for the triangle's shaders (the system's Khronos build is fine for GLSL; xsdk docs/GETTING_STARTED.md
  section 4).

## The calling convention

xsdk's GCC uses the console's own calling convention, `-mcall-xsdk` (`tools/toolchain/gcc-xsdk.patch`):
- arguments go in 8-byte slots, and a 64-bit integer takes one register;
- every caller leaves a 16-byte linkage area and then a 64-byte parameter save area;
- r14–r31 are saved whole, as 64 bits;
- narrow values are widened by the receiver;
- binutils marks such objects `EF_PPC_XSDK` and won't link them with others.

Stock LLVM can't emit this convention: its 32-bit PowerPC is SysV. The first version of `hello` got round that with
assembly thunks at every Rust/C crossing (commit 1e92aa77). That wouldn't have scaled to Vulkan, whose handles are
64-bit and which `ash` reaches through function pointers.

**xenon-forge's LLVM** ([landaire/xenon-forge](https://github.com/landaire/xenon-forge), MIT/Apache, the compiler
openxbdm is built with) adds a PPC64-ILP32 target for the console's convention: 64-bit registers, 32-bit pointers,
ELF32 output. Its `docs/llvm-xbox360-abi-evidence.md` records how its authors derived it. Compiling the same functions
with both compilers and comparing the code gave:

| | xsdk GCC | xenon-forge Rust |
|---|---|---|
| `u64` argument and result | one register (`sldi 3,3,1`) | one register (`sldi r3,r3,1`) |
| r14–r31 saves | `std`/`ld` | `std`/`ld` |
| A caller's frame | 16 linkage + 64 parameter bytes | 16 + 64, in 96-byte frames |
| Narrow results | widened by the caller | left unwidened (`addi r3,r3,1` for a `u8`) |

So Rust and xsdk's C call each other directly. `main` is Rust's, and xsdk's start-up calls it. `ash` calls
vulkan4xenon through `vkGetInstanceProcAddr`'s pointers. `tools/rust_object.py` still does two things:
- It links the Rust archive into one object whose only global is `main`, so compiler-builtins' `memcpy` can't replace
  newlib's.
- It sets `EF_PPC_XSDK`, since LLVM doesn't know the flag.

The target is `powerpc64-xsdk-ilp32.json`, xenon-forge's own `powerpc64-unknown-xbox360-ilp32` with two changes. It
leaves out VMX128, so these titles never touch vector registers. And it uses xsdk's newlib start-up instead of
xenon-forge's runtime.

Checked on these builds: neither Rust object uses r2 or r13 (r13 is the console's thread pointer, which xsdk's TLS
reads) or any vector instruction, and both are flagged `xsdk call`.

**Not yet checked on the console:**
- whether the two conventions really agree on vector registers, which these titles leave out;
- whether they agree on aggregates returned by value, which the titles don't use: `ash` returns `VkResult`, and its
  structs go through pointers;
- floats loaded from unaligned addresses, which fault on the console. xsdk's GCC avoids them, and xenon-forge's LLVM
  may not.

## std

`rust_stdcheck` is built with Rust's own std, using xenon-forge's Xbox 360 backend (`toolchain/rust-std-xbox360`,
MIT/Apache). That backend replaces std's `sys` modules (and core's atomics) for `target_os = "xbox360"`: threads on
`ExCreateThread`, locks on a wait-on-address table over kernel events, `thread_local!` on `KeTls*`, files on the NT
file API (`game:\x` is the object `\??\game:\x`), the heap on `ExAllocatePoolWithTag`, clocks, and `DbgPrint` for
`println!`. `tools/std_source.sh` lays it over Rust 1.98.0's library source (default `~/xenon-forge-std/library`; the
overlay on this PC is in openxbdm's Buck2 cache) and `build.sh stdcheck` builds std from that.

The backend reaches the kernel through `xbox360-sys`'s import thunks, which xenon-forge's own `xex-link` collects.
`rust_object.py` rewrites each one into xsdk's form: the two record words, then `mtctr r11; bctr`. It also adds an
`.xsdk.imports` record for each, so `elf2xex.py` lists them in the import table beside xsdk's own (28 imports for
`rust_stdcheck`). `elf2xex.py` already gives every title the 64 TLS slots std needs.

The PPC64-ILP32 LLVM emits atomics in the protected-reservation form the kernel accepts: `mtmsrd r13,1` (r13, the
processor block's address, has the interrupt-enable bit clear) before each `lwarx`, and the saved MSR restored after
`stwcx.`. That form only reads r13, which nothing writes.

On the console (2026-11-26) all 16 checks passed in 341 ms and the title exited with status 0: 4 threads summed 400000
locked additions in 109 ms, 1000 Condvar round trips took 6.6 ms, `sleep(100 ms)` took 100.08 ms, 64 MiB was allocated
and touched in 71 ms, and files were written, read, sought and listed in `game:\rust_stdcheck`.

Still open: whether all of xsdk's C (newlib's per-thread state, its pthreads) is safe to call from a thread std started
with `ExCreateThread`. `xsdk_log_write` was, from all four workers.

## Files

| File | What it is |
|---|---|
| `powerpc64-xsdk-ilp32.json` | The target |
| `xsdk_rt/` | What the titles share: the log, checks, controller and rumble, a heap over newlib's `memalign`, the panic handler |
| `hello/`, `triangle/` | The `no_std` titles, as `staticlib`s; the triangle's GLSL is in `triangle/shaders/` |
| `stdcheck/` | The std title |
| `tools/std_source.sh` | xenon-forge's std overlay over Rust's library source |
| `tools/rust_object.py` | The archive into one object for xsdk's linker, std's kernel imports in xsdk's form |
| `build.sh` | Shaders (the triangle), cargo with xenon-forge's rustc, `rust_object.py`, xsdk's GCC link, `xenia_scan`, `elf2xex.py` |
| `ftp.sh` | Push a title to `Hdd1\Games\Shock2`, read its log, list the folder |
