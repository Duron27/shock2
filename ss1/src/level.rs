//! Level maps from `archive.dat`.
//!
//! Each level `n` owns the resource block `4000 + 100 * n` (shockmac
//! `SAVELOAD.C`, `load_current_map`). The pieces read here:
//!
//! - `+4` the `FullMap` header: `i32` x size, y size, x shift, y shift,
//!   z shift, ...;
//! - `+5` the tiles, 16 bytes each, row-major from the south-west corner;
//! - `+7` the level's texture list (`loved_textures`), `u16` texture ids
//!   that the tiles' 5/6-bit texture fields index into.
//!
//! Tile layout (`MapElem`, v11, PC byte order):
//!
//! | byte | field |
//! |------|-------|
//! | 0 | tile type (`TILE_*`) |
//! | 1 | floor height (bits 0-4), floor texture rotation (bits 5-6) |
//! | 2 | ceiling depth below the top (bits 0-4), rotation (bits 5-6) |
//! | 3 | slope height (`param`) |
//! | 4-5 | first object in the tile |
//! | 6-7 | textures: wall (0-5), ceiling (6-10), floor (11-15) |
//! | 8-11 | flags: slope mode in bits 10-11, floor light 16-19, ceiling light 24-27 |
//! | 12-15 | renderer state |
//!
//! Heights are in `1 / 2^z_shift` tile units: 32 steps cover four tiles with
//! the usual shift of 3 (`fix_from_map_height`).

use crate::res::{ResError, ResFile};

pub const MAP_HEIGHTS: u8 = 32;
const TILE_LEN: usize = 16;

/// Corners of a tile, in the renderer's point order (`FRTABLES.C`).
pub const SW: usize = 0;
pub const NW: usize = 1;
pub const NE: usize = 2;
pub const SE: usize = 3;

/// `TILE_*` types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TileType {
    Solid,
    Open,
    /// Half solid along a diagonal; the named corner is the solid one.
    SolidDiagonal(usize),
    /// A sloped floor; `raised` corners sit `param` above the floor height.
    Slope {
        raised: [bool; 4],
        split: Split,
    },
    /// Any type this reader doesn't model yet; drawn as open.
    Other(u8),
}

/// Which diagonal a sloped tile's two triangles share.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Split {
    /// SW-NE.
    Rising,
    /// NW-SE.
    Falling,
}

impl TileType {
    pub fn from_u8(value: u8) -> TileType {
        let corners = |list: &[usize]| {
            let mut raised = [false; 4];
            for &c in list {
                raised[c] = true;
            }
            raised
        };
        // From `tile_floors`: basic slopes raise one edge; "zany" valleys
        // raise three corners and ridges one; diagonal splits raise a whole
        // triangle.
        let slope = |list: &[usize], split| TileType::Slope {
            raised: corners(list),
            split,
        };
        use Split::*;
        match value {
            0 => TileType::Solid,
            1 => TileType::Open,
            2 => TileType::SolidDiagonal(NW),
            3 => TileType::SolidDiagonal(NE),
            4 => TileType::SolidDiagonal(SE),
            5 => TileType::SolidDiagonal(SW),
            6 => slope(&[NW, NE], Rising),
            7 => slope(&[NE, SE], Rising),
            8 => slope(&[SE, SW], Rising),
            9 => slope(&[SW, NW], Rising),
            10 => slope(&[NW, NE, SW], Falling),
            11 => slope(&[NE, SE, NW], Rising),
            12 => slope(&[SE, SW, NE], Falling),
            13 => slope(&[SW, NW, SE], Rising),
            14 => slope(&[SE], Falling),
            15 => slope(&[SW], Rising),
            16 => slope(&[NW], Falling),
            17 => slope(&[NE], Rising),
            18 => slope(&[NW, SW, NE], Rising),
            19 => slope(&[NE, NW, SE], Falling),
            20 => slope(&[SW, NW, SE], Falling),
            21 => slope(&[SE, NE, SW], Rising),
            other => TileType::Other(other),
        }
    }

    pub fn is_solid(self) -> bool {
        self == TileType::Solid
    }
}

/// How a slope applies to the ceiling (`MAP_MATCH` .. `MAP_FFLAT`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlopeMode {
    /// Floor and ceiling slope together.
    Parallel,
    /// The ceiling slopes the other way.
    Mirror,
    /// Only the floor slopes.
    FloorOnly,
    /// Only the ceiling slopes.
    CeilingOnly,
}

#[derive(Debug, Clone, Copy)]
pub struct Tile {
    pub kind: TileType,
    pub raw_type: u8,
    pub floor: u8,
    /// Ceiling height above zero (`MAP_HEIGHTS - depth`): its highest point.
    pub ceiling: u8,
    pub slope: u8,
    pub floor_rotation: u8,
    pub ceiling_rotation: u8,
    pub object: u16,
    pub wall_texture: u8,
    pub ceiling_texture: u8,
    pub floor_texture: u8,
    pub flags: u32,
}

impl Tile {
    pub fn parse(b: &[u8]) -> Tile {
        let textures = u16::from_le_bytes([b[6], b[7]]);
        Tile {
            kind: TileType::from_u8(b[0]),
            raw_type: b[0],
            floor: b[1] & 0x1f,
            ceiling: MAP_HEIGHTS - (b[2] & 0x1f),
            slope: b[3],
            floor_rotation: (b[1] >> 5) & 3,
            ceiling_rotation: (b[2] >> 5) & 3,
            object: u16::from_le_bytes([b[4], b[5]]),
            wall_texture: (textures & 0x3f) as u8,
            ceiling_texture: ((textures >> 6) & 0x1f) as u8,
            floor_texture: (textures >> 11) as u8,
            flags: u32::from_le_bytes([b[8], b[9], b[10], b[11]]),
        }
    }

    pub fn slope_mode(&self) -> SlopeMode {
        match (self.flags >> 10) & 3 {
            0 => SlopeMode::Parallel,
            1 => SlopeMode::Mirror,
            2 => SlopeMode::FloorOnly,
            _ => SlopeMode::CeilingOnly,
        }
    }

    /// 0 (dark) to 15 (bright), as authored.
    pub fn floor_shade(&self) -> u8 {
        ((self.flags >> 16) & 0xf) as u8
    }

    pub fn ceiling_shade(&self) -> u8 {
        ((self.flags >> 24) & 0xf) as u8
    }

    fn raised(&self) -> [bool; 4] {
        match self.kind {
            TileType::Slope { raised, .. } => raised,
            _ => [false; 4],
        }
    }

    /// Floor height at each corner (`SW, NW, NE, SE`), in map height units.
    pub fn floor_corners(&self) -> [f32; 4] {
        let sloped = !matches!(self.slope_mode(), SlopeMode::CeilingOnly);
        self.raised()
            .map(|up| self.floor as f32 + if up && sloped { self.slope as f32 } else { 0.0 })
    }

    /// Ceiling height at each corner. The stored height is the ceiling's
    /// highest point, so a slope only ever brings corners down from it.
    pub fn ceiling_corners(&self) -> [f32; 4] {
        let top = self.ceiling as f32;
        let p = self.slope as f32;
        self.raised().map(|up| match self.slope_mode() {
            SlopeMode::Parallel | SlopeMode::CeilingOnly => top - if up { 0.0 } else { p },
            SlopeMode::Mirror => top - if up { p } else { 0.0 },
            SlopeMode::FloorOnly => top,
        })
    }
}

/// One level's map.
pub struct Level {
    pub number: u16,
    pub width: usize,
    pub height: usize,
    pub z_shift: u32,
    /// `tiles[y * width + x]`, `y` growing north.
    pub tiles: Vec<Tile>,
    /// Texture ids (`texture.res` 1000 + id at 128x128) the tiles index into.
    pub textures: Vec<u16>,
}

pub fn level_base(number: u16) -> u16 {
    4000 + 100 * number
}

impl Level {
    /// Levels present in an archive (those with a tile map).
    pub fn numbers(archive: &ResFile) -> Vec<u16> {
        (0..16)
            .filter(|&n| archive.entry(level_base(n) + 5).is_some())
            .collect()
    }

    pub fn load(archive: &ResFile, number: u16) -> Result<Level, ResError> {
        let base = level_base(number);
        let header = archive.load(base + 4)?;
        let i32_at = |at: usize| {
            header
                .data
                .get(at..at + 4)
                .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
                .ok_or(ResError::Corrupt {
                    id: base + 4,
                    reason: "short map header".into(),
                })
        };
        let width = i32_at(0)?.max(0) as usize;
        let height = i32_at(4)?.max(0) as usize;
        let z_shift = i32_at(16)?.clamp(0, 5) as u32;

        let map = archive.load(base + 5)?;
        if map.data.len() < width * height * TILE_LEN {
            return Err(ResError::Corrupt {
                id: base + 5,
                reason: format!("{} bytes for a {width}x{height} map", map.data.len()),
            });
        }
        let tiles = map
            .data
            .chunks_exact(TILE_LEN)
            .take(width * height)
            .map(Tile::parse)
            .collect();

        let textures = archive
            .load(base + 7)
            .map(|res| {
                res.data
                    .chunks_exact(2)
                    .map(|c| u16::from_le_bytes([c[0], c[1]]))
                    .collect()
            })
            .unwrap_or_default();

        Ok(Level {
            number,
            width,
            height,
            z_shift,
            tiles,
            textures,
        })
    }

    pub fn tile(&self, x: usize, y: usize) -> Option<&Tile> {
        (x < self.width && y < self.height).then(|| &self.tiles[y * self.width + x])
    }

    /// World units (tiles) per map height step.
    pub fn height_scale(&self) -> f32 {
        1.0 / (1u32 << self.z_shift) as f32
    }

    /// The texture id a tile's 5/6-bit texture field selects.
    pub fn texture_id(&self, index: u8) -> Option<u16> {
        self.textures.get(index as usize).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tile(kind: u8, floor: u8, depth: u8, slope: u8, mode: u32) -> Tile {
        let mut b = [0u8; 16];
        b[0] = kind;
        b[1] = floor;
        b[2] = depth;
        b[3] = slope;
        b[8..12].copy_from_slice(&(mode << 10).to_le_bytes());
        Tile::parse(&b)
    }

    #[test]
    fn slope_up_north_raises_the_north_corners() {
        let t = tile(6, 4, 8, 2, 0);
        assert_eq!(t.floor_corners(), [4.0, 6.0, 6.0, 4.0]);
        // Parallel: the ceiling's top is over the raised edge.
        assert_eq!(t.ceiling_corners(), [22.0, 24.0, 24.0, 22.0]);
        let mirrored = tile(6, 4, 8, 2, 1);
        assert_eq!(mirrored.ceiling_corners(), [24.0, 22.0, 22.0, 24.0]);
        let floor_only = tile(6, 4, 8, 2, 2);
        assert_eq!(floor_only.ceiling_corners(), [24.0; 4]);
        let ceiling_only = tile(6, 4, 8, 2, 3);
        assert_eq!(ceiling_only.floor_corners(), [4.0; 4]);
    }

    #[test]
    fn texture_fields_unpack() {
        let mut b = [0u8; 16];
        b[0] = 1;
        let packed: u16 = (21 << 11) | (9 << 6) | 37;
        b[6..8].copy_from_slice(&packed.to_le_bytes());
        let t = Tile::parse(&b);
        assert_eq!(
            (t.floor_texture, t.ceiling_texture, t.wall_texture),
            (21, 9, 37)
        );
        assert_eq!(t.kind, TileType::Open);
    }
}
