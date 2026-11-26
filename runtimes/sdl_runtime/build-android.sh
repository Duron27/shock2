#!/usr/bin/env bash
# Build libshock2.so (and the libshaderc_shared.so its Vulkan backend loads)
# for alpha3, arm64-v8a.
#
#   runtimes/sdl_runtime/build-android.sh [--install <alpha3 checkout>]
#
# Output: target/android-dist/arm64-v8a/{libshock2.so,libshaderc_shared.so}
# and target/android-dist/shock2/assets (the engine's bundled files, which go
# in the game folder: /storage/emulated/0/Alpha3/shock2/assets).
# --install copies the two libraries into <alpha3>/androidApp/src/main/jniLibs.
#
# Env: ANDROID_NDK_HOME (default: newest NDK under ~/Android/Sdk/ndk),
#      ALPHA3 (default: ../alpha3 beside this repo) for its libSDL2.so.
set -euo pipefail

repo="$(cd "$(dirname "$0")/../.." && pwd)"
api=26
install_to=""
if [[ "${1:-}" == "--install" ]]; then
  install_to="${2:?--install needs the alpha3 checkout}"
fi

ndk="${ANDROID_NDK_HOME:-$(ls -d "$HOME"/Android/Sdk/ndk/* | sort -V | tail -1)}"
bin="$ndk/toolchains/llvm/prebuilt/linux-x86_64/bin"
alpha3="${ALPHA3:-$repo/../alpha3}"
sdl_dir="$alpha3/androidApp/src/main/jniLibs/arm64-v8a"
[[ -f "$sdl_dir/libSDL2.so" ]] || { echo "no libSDL2.so in $sdl_dir (set ALPHA3)" >&2; exit 1; }

export CC_aarch64_linux_android="$bin/aarch64-linux-android$api-clang"
export CXX_aarch64_linux_android="$bin/aarch64-linux-android$api-clang++"
export AR_aarch64_linux_android="$bin/llvm-ar"
export CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER="$bin/aarch64-linux-android$api-clang"
export SHOCK2_SDL2_LIB_DIR="$sdl_dir"

rustup target list --installed | grep -q aarch64-linux-android \
  || rustup target add aarch64-linux-android

cd "$repo"
cargo build -p sdl_runtime --lib --release --target aarch64-linux-android

out="$repo/target/android-dist"
mkdir -p "$out/arm64-v8a"
"$bin/llvm-strip" -o "$out/arm64-v8a/libshock2.so" \
  target/aarch64-linux-android/release/libshock2.so

# libshaderc: the engine compiles its GLSL at runtime and dlopens
# libshaderc_shared.so. The NDK ships shaderc's source; build it once.
shaderc="$repo/target/android-shaderc"
if [[ ! -f "$shaderc/libshaderc_shared.so" ]]; then
  mkdir -p "$shaderc"
  (cd "$shaderc" && "$ndk/ndk-build" -j"$(nproc)" NDK_PROJECT_PATH=. \
    APP_BUILD_SCRIPT="$ndk/sources/third_party/shaderc/Android.mk" \
    APP_STL=c++_static APP_ABI=arm64-v8a APP_PLATFORM=android-$api \
    libshaderc_combined)
  "$bin/aarch64-linux-android$api-clang++" -shared -o "$shaderc/libshaderc_shared.so" \
    -Wl,-soname,libshaderc_shared.so \
    -Wl,--whole-archive "$shaderc/obj/local/arm64-v8a/libshaderc_combined.a" \
    -Wl,--no-whole-archive -static-libstdc++ -Wl,--gc-sections -s
fi
cp "$shaderc/libshaderc_shared.so" "$out/arm64-v8a/"

rm -rf "$out/shock2/assets"
mkdir -p "$out/shock2"
cp -r "$repo/assets" "$out/shock2/assets"

echo "built:"
ls -la "$out/arm64-v8a"
"$bin/llvm-readelf" -d "$out/arm64-v8a/libshock2.so" | grep NEEDED

if [[ -n "$install_to" ]]; then
  cp "$out/arm64-v8a/"*.so "$install_to/androidApp/src/main/jniLibs/arm64-v8a/"
  echo "installed into $install_to/androidApp/src/main/jniLibs/arm64-v8a"
fi
