# Patched crates

Applied to the whole workspace through `[patch.crates-io]` in the root `Cargo.toml`. Each patch changes only code that
big-endian targets compile, so little-endian builds (desktop, Quest, Android) build the same code as upstream.

- `texture2ddecoder-0.1.2`: its big-endian code, which the Xbox 360 build is the first to compile.
  - `src/crunch/crn_unpacker.rs`: the `target_endian = "big"` blocks used names that don't exist (`pDst`,
    `pAlpha_selectors`, `self.m_color_selectors`), so the crate didn't compile. Renamed to the little-endian blocks'
    names. The engine doesn't decode crunch textures; the math there is upstream's, unchecked.
  - `src/color.rs`: `color()` packs `0xAARRGGBB` as a value on every CPU, but the big-endian `TRANSPARENT_MASK` and
    `TRANSPARENT_SHIFT` put alpha in the low byte, so BC1's transparent texels and BC2/BC3's alpha went into blue.
    Both now use the little-endian values.
