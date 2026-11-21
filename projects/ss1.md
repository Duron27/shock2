# System Shock (1994) on our engine

A Rust rewrite of System Shock 1 that draws its world with our renderer
(`engine/`) instead of the original 8-bit software 3D, while keeping the
original's UI layout and style (sharpened, not redesigned). The formats come
from Night Dive's GPL-3.0 source release,
[shockmac](https://github.com/NightDive-Studio/shockmac).

## Ground rules

- **No SS2 code is touched.** SS1 lives in its own crates (`ss1/`,
  `tools/ss1_query`, later an SS1 runtime) and depends only on `engine/`.
- **Licence:** shockmac is GPL-3.0 and this repo is GPLv2, which can't share a
  binary. The SS1 crates are GPL-3.0-or-later and must never link `dark`,
  `shock2vr` or any runtime of SS2; `engine/` (MIT) is fine to share.
- **Data:** the 1994 CD release (DOS data). Found via `SS1_DATA_PATH`
  (the install folder or its `data/`), else a `System Shock (1994)*` folder
  beside the repo.

## Roadmap

1. **Resource files** - `LG Res File v2` directory, LZW, compound ref tables.
   `cargo s1q res <file> [--id N]`, `cargo s1q check`. *Done.*
2. **Palette and bitmaps** - `gamepal.res`, RLE/flat 8-bit images, texture
   atlas export (`cargo s1q image <file> --sheet`). *Done.*
3. **Levels** - the tile map in `archive.dat` (heights, slopes, textures per
   tile), objects.
4. **3D view** - build level geometry from tiles and draw it with `engine/`
   (lighting, filtering, our materials) in an SS1 runtime with free-look.
5. **Objects and 3D models** - sprites and `obj3d.res` models.
6. **UI** - the original screen layout (`gamescr.res`, `mfdart.res`),
   upscaled and smoothed.
7. **Gameplay** - movement, doors, inventory, AI, ... (to be broken down).
