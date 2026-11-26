#!/bin/bash
# Builds the Rust test title for the Xbox 360 into hello/build/rust_hello.xex (xbox360/README.md).
#   XSDK      xsdk's checkout, built with `make` (default: openmw-xenon's)
#   XSDK_GCC  the toolchain's PREFIX (default: ~/xsdk-gcc)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
XSDK="${XSDK:-$HOME/AndroidStudioProjects/openmw-xenon/sources/xenosworks/xsdk}"
XSDK_GCC="${XSDK_GCC:-$HOME/xsdk-gcc}"
TC="$XSDK_GCC/bin/powerpc-xsdk-elf-"
OUT="$HERE/hello/build"
NAME=rust_hello

[ -f "$XSDK/build/libxsdk.a" ] || { echo "no $XSDK/build/libxsdk.a: run make in xsdk first" >&2; exit 1; }
mkdir -p "$OUT"

# 1. Rust, for the custom target, with core and compiler-builtins built from source. compiler-builtins-mem gives
#    Rust its own memcpy and friends, so LLVM's calls to them stay inside Rust's object (tools/rust_object.py).
(cd "$HERE/hello" && cargo +nightly build --release \
	--target "$HERE/powerpc-xsdk-none.json" \
	-Zbuild-std=core,compiler_builtins -Zbuild-std-features=compiler-builtins-mem)
LIB="$HERE/hello/target/powerpc-xsdk-none/release/libxenon_hello.a"

# 2. One object, closed but for its entry and its thunked calls.
python3 "$HERE/tools/rust_object.py" "$TC" "$LIB" "$OUT/rust.o" rust_title_main

# 3. The bridge, as xsdk's CMake builds a title's C, then the link, as xsdk.cmake links a title.
"${TC}gcc" -mcpu=cell -fno-pic -O2 -Wall -Wextra -Werror -std=gnu17 -I"$XSDK/include" \
	-c "$HERE/hello/bridge.c" -o "$OUT/bridge.o"
"${TC}gcc" -mcpu=cell -fno-pic -static -nostdlib -T "$XSDK/src/xsdk.ld" -Wl,-Map,"$OUT/$NAME.map" \
	"$OUT/bridge.o" "$OUT/rust.o" \
	-Wl,--start-group "$XSDK/build/libxsdk.a" -lc -lm -lgcc -Wl,--end-group -o "$OUT/$NAME.elf"

# 4. What Xenia would skip, then the .xex.
"${TC}objdump" -d "$OUT/$NAME.elf" | python3 "$XSDK/tools/xenia_scan.py"
python3 "$XSDK/tools/elf2xex.py" --title-name "Shock2 Rust hello" "$OUT/$NAME.elf" "$OUT/$NAME.xex"
ls -l "$OUT/$NAME.xex"
