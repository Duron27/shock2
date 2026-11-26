# sdl_runtime

The flatscreen game on SDL2 + Vulkan. One loop, two hosts:

- **`shock2_sdl`**: a desktop binary (`cargo run -p sdl_runtime --bin shock2_sdl -- --mission medsci1.mis`),
  which tests the Android code path on Linux. `SHOCK2_VK_CAPTURE=out.png@400` saves frame 400.
- **`libshock2.so`**: the same code, loaded by alpha3's `SDLActivity` on Android.

## Building for Android

```bash
runtimes/sdl_runtime/build-android.sh                    # -> target/android-dist/
runtimes/sdl_runtime/build-android.sh --install ../alpha3  # also copies the .so files into alpha3's jniLibs
```

It links against alpha3's own `libSDL2.so` (`$ALPHA3/androidApp/src/main/jniLibs/arm64-v8a`),
and builds `libshaderc_shared.so` once from the NDK's shaderc source. The Vulkan backend compiles
its GLSL at runtime and dlopens that library, so it has to ship in the APK beside `libshock2.so`.

Output:

| File | Goes to |
| --- | --- |
| `arm64-v8a/libshock2.so` | alpha3 `jniLibs/arm64-v8a` |
| `arm64-v8a/libshaderc_shared.so` | alpha3 `jniLibs/arm64-v8a` |
| `shock2/assets/` | the game folder: `/storage/emulated/0/Alpha3/shock2/assets` |

FFmpeg is off on Android for now, so cutscene videos are skipped.

## What the library gives the host

| Symbol | Meaning |
| --- | --- |
| `SDL_main` | entry point for `nativeRunMain`. It takes the usual `--mission`, `--save-file` and `--experimental` arguments, and ignores any it doesn't know. |
| `SDLActivity.isMouseShown()` | 1 while the game shows a 2D cursor (main and pause menus, use mode, inventory, MFD panels, keypads), which is `App::wants_pointer`. 0 during play, when motion turns the view. |
| `SDLActivity.getMouseX()` / `getMouseY()` | the cursor in window pixels |

During play, mouse motion (relative or absolute, as alpha3 sends it) turns the view, LMB fires,
RMB uses, and keys follow the desktop bindings (`shock2vr::input::keyboard`). While
`isMouseShown` is 1, the mouse moves and clicks the game's cursor. The library never asks Android
for pointer capture, so the launcher's controls stay in charge.

A game controller works too: left stick moves, right stick looks, RT fires, LT uses, A jumps,
B crouches, X reloads, Y toggles use mode, Start pauses, Back opens the map, LB/RB cycle ammo and
fire mode, D-pad left/right leans, and D-pad up reads the last unread log. While a cursor is up,
the left stick moves the cursor, A clicks, and B right-clicks.

Paths: the game folder is `/storage/emulated/0/Alpha3/shock2`. If the host sets
`DARK_ASSET_PATH` before loading the library, that folder is used instead. Saves and
`user-settings.json` are written there. stdout and stderr go to logcat (tag `shock2`) and to
`shock2.log` in the game folder.

Backgrounding releases the Vulkan surface and recreates it on return. On a rotated phone, frames
are presented unrotated and the compositor turns them. Pre-rotating in the final copy would save
that composition pass, but that hasn't been done yet.

## Hooking it into alpha3

This mirrors the Lithtech entry:

```kotlin
// Variables.kt
const val SHOCK2_MAIN_LIB = "libshock2.so"

// LibraryLoader.loadLibraries()
"Shock2" -> base + "shock2"
// LibraryLoader.getMainSharedObject()
"Shock2" -> SHOCK2_MAIN_LIB

// StateManager.kt
val gameList = arrayOf(/* ... */, "Shock2")

// EngineActivity.kt, next to the UQM/OpenMW cases
if (UIStateManager.tempCodeGroup == "Shock2") {
    AutoMouseModeComposable()   // polls SDLActivity.isMouseShown()
}
```

The game folder needs the System Shock 2 data (`shock2.gam`, `*.mis`, `motiondb.bin`,
`res/*.crf`, or the 25th Anniversary archive) plus the `assets/` folder above.
