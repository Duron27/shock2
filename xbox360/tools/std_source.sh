#!/bin/bash
# Lays xenon-forge's Xbox 360 std overlay (toolchain/rust-std-xbox360, MIT/Apache) over Rust 1.98.0's library source,
# into the folder build.sh's std titles build core, alloc and std from:
#   std_source.sh <overlay> <rust-src library> <out>
# <overlay> is xenon-forge's toolchain/rust-std-xbox360/library; the copy on this PC is in openxbdm's Buck2 cache.
set -euo pipefail
overlay="${1:?overlay library}"; src="${2:?rust-src library}"; out="${3:?out}"
[ -f "$overlay/std/src/sys/pal/xbox360/mod.rs" ] || { echo "$overlay is not the overlay's library" >&2; exit 1; }
[ -f "$src/Cargo.lock" ] || { echo "$src is not Rust's library source" >&2; exit 1; }
rm -rf "$out.tmp"; mkdir -p "$out.tmp"
cp -a "$src/." "$out.tmp/"
cp -a "$overlay/." "$out.tmp/"
# xbox360-sys from the overlay's vendor folder rather than crates.io.
grep -q '^\[patch.crates-io\]' "$out.tmp/Cargo.toml" || printf '\n[patch.crates-io]\n' >> "$out.tmp/Cargo.toml"
sed -i '/^\[patch.crates-io\]/a xbox360-sys = { path = "vendor/xbox360-sys-0.1.0" }' "$out.tmp/Cargo.toml"
rm -rf "$out"; mv "$out.tmp" "$out"
echo "std source: $out"
