# Building for the Xbox 360 with xsdk (a note, 2026-11-26)

Possible, but harder than a C++ engine: xsdk (the Xbox 360 homebrew SDK that `~/AndroidStudioProjects/openmw-xenon`
uses) is a C/C++ toolchain, and this project is Rust. The flatscreen SDL runtime (`runtimes/sdl_runtime`) is the
part to port; the VR runtimes don't apply. No Microsoft tooling or leaked SDK material is involved, and none may be
added.

## The pieces (all already built on this PC for openmw-xenon)

- **xsdk** (`openmw-xenon/sources/xenosworks/xsdk`): GCC 15 for `powerpc-xsdk-elf` at `~/xsdk-gcc`, newlib, the
  system calls (threads, files, controllers, audio), and `tools/elf2xex.py` for the `.xex`. Read its
  `docs/GETTING_STARTED.md`, `docs/PITFALLS.md` and `docs/PARITY.md`.
- **vulkan4xenon** (`openmw-xenon/sources/xenosworks/vulkan4xenon`): Vulkan on the 360's GPU as a static library;
  `docs/SUPPORT.md` lists what it implements. The engine's renderer is Vulkan through `ash`, which helps.
- **SDL-xenon** (SDL3) and openmw-xenon's `xenon/sdl2compat` (SDL2's API over SDL3): for the `sdl2` crate.

## What a Rust port needs

1. **A Rust target for the console.** No built-in target fits: write a custom target JSON (32-bit big-endian
   PowerPC, `target-c-int-width` 32, `panic` = abort to start, linker = `~/xsdk-gcc/bin/powerpc-xsdk-elf-gcc` with
   xsdk's linker script and libraries) and build the standard library with `-Zbuild-std` on nightly. `std` on
   newlib has been done before for other consoles (Rust's `armv6k-nintendo-3ds` and `armv7-sony-vita-newlibeabihf`
   targets, through the `libc` crate's newlib support); xsdk would need the same `libc` bindings for its threads and
   files, or start `no_std` + `alloc` with a thin C shim over xsdk.
2. **Dependencies:** every crate with C code (`sdl2`, `engine_ffmpeg`/FFmpeg, anything with `cc`) must cross-compile
   with xsdk's GCC; `ash` must link `libvulkan4xenon.a` statically (its `linked` feature) instead of loading a
   library at run time. `glfw` and OpenXR are out.
3. **Big-endian.** Dark engine files (`.mis`, `.gam`, `.bin` models, `.crf` archives) are little-endian: every reader
   must use explicit `from_le_bytes`/`byteorder`, never transmute or `bytemuck` casts of file data.
4. **Memory:** 512 MB shared between CPU and GPU, about 450 MB for a title. The remaster's upgraded textures are
   large: plan on the originals or compressed copies.
5. **Shaders:** SPIR-V goes through xsdk's shader compiler (`xsc.h`) on the console or on the PC; test each one.

A cheaper first step: a small `no_std` Rust title on xsdk (clear the screen, read a controller) to prove the target
and the link, before the engine. **Done (2026-11-26), not yet run on the console:** `xbox360/` (its README): `rust_hello` and `rust_triangle`, a
triangle through `ash` on vulkan4xenon. xsdk's GCC uses the console's own calling convention (`-mcall-xsdk`), which
stock LLVM can't emit; xenon-forge's PPC64-ILP32 LLVM (the compiler openxbdm is built with) emits it, so Rust calls
xsdk and vulkan4xenon directly. Step 1 above is that compiler plus `powerpc64-xsdk-ilp32.json`, not a target of our own.
