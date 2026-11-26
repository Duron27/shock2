#!/bin/bash
# Builds a Rust test title for the Xbox 360 (xbox360/README.md):
#   build.sh hello      -> hello/build/rust_hello.xex
#   build.sh triangle   -> triangle/build/rust_triangle.xex
#   build.sh stdcheck   -> stdcheck/build/rust_stdcheck.xex
# Environment:
#   XENOSWORKS  the folder holding xsdk and vulkan4xenon, each built with `make` (default: openmw-xenon's)
#   XSDK_GCC    xsdk's toolchain PREFIX (default: ~/xsdk-gcc)
#   XENON_FORGE xenon-forge's prebuilt PPC64-ILP32 Rust and LLVM (default: ~/xenon-forge-toolchain)
#   RUST_SRC    the Rust 1.98.0 library source it builds core from (default: rustup's 1.98.0 rust-src)
#   STD_SRC     Rust 1.98.0's library with xenon-forge's std overlay, for stdcheck (tools/std_source.sh makes it;
#               default ~/xenon-forge-std/library)
#   GLSLANG     glslang for the triangle's shaders (default: the one on PATH)
set -euo pipefail

TITLE="${1:?usage: build.sh hello|triangle|stdcheck}"
HERE="$(cd "$(dirname "$0")" && pwd)"
XENOSWORKS="${XENOSWORKS:-$HOME/AndroidStudioProjects/openmw-xenon/sources/xenosworks}"
XSDK="$XENOSWORKS/xsdk"
V4X="$XENOSWORKS/vulkan4xenon"
XSDK_GCC="${XSDK_GCC:-$HOME/xsdk-gcc}"
XENON_FORGE="${XENON_FORGE:-$HOME/xenon-forge-toolchain}"
RUST_SRC="${RUST_SRC:-$HOME/.rustup/toolchains/1.98.0-x86_64-unknown-linux-gnu/lib/rustlib/src/rust/library}"
GLSLANG="${GLSLANG:-$(command -v glslang)}"
TC="$XSDK_GCC/bin/powerpc-xsdk-elf-"
OUT="$HERE/$TITLE/build"
NAME="rust_$TITLE"

case "$TITLE" in
hello) CRATE=xenon_hello ;;
triangle) CRATE=xenon_triangle ;;
stdcheck) CRATE=xenon_stdcheck ;;
*) echo "unknown title $TITLE" >&2; exit 1 ;;
esac
[ -x "$XENON_FORGE/bin/rustc" ] || { echo "no $XENON_FORGE/bin/rustc: see xbox360/README.md, 'The compiler'" >&2; exit 1; }
[ -f "$RUST_SRC/Cargo.lock" ] || { echo "no $RUST_SRC: rustup toolchain install 1.98.0 --component rust-src" >&2; exit 1; }
[ -f "$XSDK/build/libxsdk.a" ] || { echo "no $XSDK/build/libxsdk.a: run make in xsdk" >&2; exit 1; }
mkdir -p "$OUT"
BUILD_STD=core,alloc,compiler_builtins
if [ "$TITLE" = stdcheck ]; then
	RUST_SRC="${STD_SRC:-$HOME/xenon-forge-std/library}"
	[ -f "$RUST_SRC/std/src/sys/pal/xbox360/mod.rs" ] || { echo "no std overlay at $RUST_SRC: run tools/std_source.sh" >&2; exit 1; }
	BUILD_STD=std,panic_abort,core,alloc,compiler_builtins
fi

C_OBJECTS=()
LIBS=("$XSDK/build/libxsdk.a")
if [ "$TITLE" = triangle ]; then
	[ -f "$V4X/build/libvulkan4xenon.a" ] || { echo "no $V4X/build/libvulkan4xenon.a: run make in vulkan4xenon" >&2; exit 1; }
	# GLSL -> SPIR-V -> xost -> build/shaders.c: each shader's SPIR-V and the bundle vkCreateShaderModule looks in.
	python3 "$V4X/tools/build_shaders.py" --xost "$XSDK/xenos-toolchain" --glslang "$GLSLANG" \
		--output "$OUT/shaders" --prefix tri_ "$HERE/triangle/shaders/triangle.vert" "$HERE/triangle/shaders/triangle.frag"
	"${TC}gcc" -mcpu=cell -fno-pic -O2 -I"$V4X/include" -I"$V4X/third_party/vulkan-headers/include" -I"$XSDK/include" \
		-c "$OUT/shaders.c" -o "$OUT/shaders.o"
	C_OBJECTS+=("$OUT/shaders.o")
	LIBS=("$V4X/build/libvulkan4xenon.a" -lstdc++ "$XSDK/build/libxsdk.a")
fi

# 1. Rust, with xenon-forge's compiler: core, alloc and compiler-builtins from 1.98.0's source. compiler-builtins-mem
#    gives Rust its own memcpy and friends, made local in step 2.
(cd "$HERE" && __CARGO_TESTS_ONLY_SRC_ROOT="$RUST_SRC" RUSTC="$XENON_FORGE/bin/rustc" RUSTC_BOOTSTRAP=1 \
	cargo +1.98.0 build --release -p "$CRATE" --target "$HERE/powerpc64-xsdk-ilp32.json" \
	-Zbuild-std=$BUILD_STD -Zbuild-std-features=compiler-builtins-mem -Zjson-target-spec)
LIB="$HERE/target/powerpc64-xsdk-ilp32/release/lib$CRATE.a"

# 2. One object: Rust's main, calling C directly.
python3 "$HERE/tools/rust_object.py" "$TC" "$LIB" "$OUT/rust.o" main

# 3. The link, as xsdk.cmake links a title.
"${TC}gcc" -mcpu=cell -fno-pic -static -nostdlib -T "$XSDK/src/xsdk.ld" -Wl,-Map,"$OUT/$NAME.map" \
	"$OUT/rust.o" "${C_OBJECTS[@]}" \
	-Wl,--start-group "${LIBS[@]}" -lc -lm -lgcc -Wl,--end-group -o "$OUT/$NAME.elf"

# 4. What Xenia would skip, then the .xex.
"${TC}objdump" -d "$OUT/$NAME.elf" | python3 "$XSDK/tools/xenia_scan.py"
python3 "$XSDK/tools/elf2xex.py" --title-name "Shock2 Rust $TITLE" "$OUT/$NAME.elf" "$OUT/$NAME.xex"
ls -l "$OUT/$NAME.xex"
