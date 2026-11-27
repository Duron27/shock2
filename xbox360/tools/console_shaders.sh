#!/bin/bash
# console_shaders.sh <out dir> [dump dir]: everything the console needs to draw with the engine's shaders, from a
# shader dump (made by shader_sweep.sh into <out>/dump unless one is given; needs the game's data and a desktop GPU):
#   <out>/audit.md             each module through xost; the script stops if xost refuses one
#   <out>/shock2_shaders.table the GLSL -> SPIR-V table for engine::vk::shader::set_console_table
#   <out>/shock2_shaders.{c,h} vulkan4xenon's bundle: each module's microcode, keyed by its SPIR-V's hash
# XENOSWORKS as build.sh takes it.
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$HERE/../.."
if [ -z "${XENOSWORKS:-}" ] && [ -f "$HERE/../deps/xsdk/Makefile" ]; then XENOSWORKS="$HERE/../deps"; fi
XENOSWORKS="${XENOSWORKS:-$HOME/AndroidStudioProjects/openmw-xenon/sources/xenosworks}"
OUT="$(realpath -m "$1")"
DUMP="$(realpath -m "${2:-$OUT/dump}")"
mkdir -p "$OUT"
if [ -z "${2:-}" ]; then
	(cd "$ROOT" && cargo build -p debug_runtime)
	rm -rf "$DUMP"
	"$HERE/shader_sweep.sh" "$DUMP"
fi
python3 "$HERE/shader_audit.py" "$XENOSWORKS/vulkan4xenon" "$XENOSWORKS/xsdk/xenos-toolchain" "$DUMP/xbox360" \
	"$OUT/audit.md" | head -4
if grep -q "^- \`" "$OUT/audit.md"; then echo "xost refused a module: see $OUT/audit.md"; exit 1; fi
python3 "$HERE/shader_table.py" "$DUMP/xbox360" "$OUT/shock2_shaders.table"
modules=()
for f in "$DUMP"/xbox360/*.spv; do case "$f" in *.compiled.spv) ;; *) modules+=("$f") ;; esac; done
# build_shaders.py works in build/ under the current directory.
(cd "$OUT" && python3 "$XENOSWORKS/vulkan4xenon/tools/build_shaders.py" --xost "$XENOSWORKS/xsdk/xenos-toolchain" \
	--optimize --prefix shock_ --output "$OUT/shock2_shaders" "${modules[@]}")
echo "${#modules[@]} modules in $OUT/shock2_shaders.c"
