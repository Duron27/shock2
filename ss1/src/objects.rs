//! Objects: the per-level object table and the shared object properties.
//!
//! Every object is a *triple* (class, subclass, type). Triples are numbered
//! in class/subclass order (`OPTRIP`), 476 in all, and that number indexes
//! the common properties at the end of `objprop.dat` and the art in
//! `objart.res`.
//!
//! Level object records (`archive.dat`, level base + 8) are `Obj` structs of
//! 27 bytes (shockmac `OBJECTS.H`, `OBJAPP.H`): active, class, subclass,
//! specID, ref, next, prev, location (x, y as 8.8 tile coordinates, z, pitch,
//! heading, bank) and info (ph, type, hp, make info, frame, time, flags).

use std::path::Path;

use crate::level::level_base;
use crate::res::{ResError, ResFile};

/// Types per subclass, per class, from shockmac's `SubclassPropHeaders`.
pub const SUBCLASS_TYPES: [&[u8]; 15] = [
    &[5, 2, 2, 2, 3, 2],           // gun
    &[2, 2, 3, 2, 2, 2, 2],        // ammo
    &[6, 16, 2],                   // physics
    &[5, 3],                       // grenade
    &[7],                          // drug
    &[5, 10],                      // hardware
    &[7, 3, 4, 5, 3],              // software
    &[9, 10, 11, 4, 9, 8, 16, 10], // bigstuff
    &[8, 10, 15, 6, 12, 12, 9, 8], // smallstuff
    &[9, 7, 3, 11, 2, 3],          // fixture
    &[10, 9, 7, 5, 10],            // door
    &[9, 11, 14],                  // animating
    &[13, 1, 5],                   // trap
    &[3, 3, 4, 8, 13, 7, 8],       // container
    &[9, 12, 7, 7, 2],             // critter
];

pub const CLASS_NAMES: [&str; 15] = [
    "gun",
    "ammo",
    "physics",
    "grenade",
    "drug",
    "hardware",
    "software",
    "bigstuff",
    "smallstuff",
    "fixture",
    "door",
    "animating",
    "trap",
    "container",
    "critter",
];

pub const CLASS_DOOR: u8 = 10;
pub const CLASS_CRITTER: u8 = 14;

pub fn num_triples() -> usize {
    SUBCLASS_TYPES
        .iter()
        .map(|c| c.iter().map(|&n| n as usize).sum::<usize>())
        .sum()
}

/// `OPTRIP`: the property/art number of a triple.
pub fn triple_index(class: u8, subclass: u8, kind: u8) -> Option<usize> {
    let subclasses = SUBCLASS_TYPES.get(class as usize)?;
    if subclass as usize >= subclasses.len() || kind >= subclasses[subclass as usize] {
        return None;
    }
    let before_class: usize = SUBCLASS_TYPES[..class as usize]
        .iter()
        .map(|c| c.iter().map(|&n| n as usize).sum::<usize>())
        .sum();
    let before_subclass: usize = subclasses[..subclass as usize]
        .iter()
        .map(|&n| n as usize)
        .sum();
    Some(before_class + before_subclass + kind as usize)
}

/// How the original renderer draws an object type (`FAUBJ_*`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderType {
    Unknown,
    TextPoly,
    /// A camera-facing sprite.
    Bitmap,
    /// A 3D model from `obj3d.res`.
    TPoly,
    Critter,
    AnimPoly,
    Vox,
    NoObj,
    /// A flat sprite with a fixed facing (signs, wall decals).
    TexBitmap,
    FlatPoly,
    MultiView,
    Special,
    TlPoly,
    Other(u8),
}

impl RenderType {
    pub fn from_u8(v: u8) -> RenderType {
        use RenderType::*;
        match v {
            0 => Unknown,
            1 => TextPoly,
            2 => Bitmap,
            3 => TPoly,
            4 => Critter,
            5 => AnimPoly,
            6 => Vox,
            7 => NoObj,
            8 => TexBitmap,
            9 => FlatPoly,
            10 => MultiView,
            11 => Special,
            12 => TlPoly,
            o => Other(o),
        }
    }
}

/// The common part of an object type's properties (`ObjProp`, 27 bytes).
#[derive(Debug, Clone, Copy)]
pub struct ObjProp {
    pub mass: i32,
    pub hit_points: i16,
    pub render_type: RenderType,
    pub physics_model: u8,
    pub physics_xr: u8,
    pub flags: u16,
    pub mfd_id: i16,
    pub bitmap_3d: u16,
}

pub const OBJ_PROP_LEN: usize = 27;
const CRITTER_PROP_LEN: usize = 75;
/// Mutant, robot, cyborg, cyber and robobabe subclass properties.
const CRITTER_TAIL_LEN: usize = 9 + 12 * 2 + 7 * 2 + 7 * 6 + 2;
/// `RES_bmCritterStanding_0` in `objart2.res`.
pub const CRITTER_STANDING_BASE: u16 = 1585;
/// `MY_IM_LARGE`: the sprite draws at twice the usual scale.
pub const FLAG_LARGE: u16 = 0x0008;

impl ObjProp {
    fn parse(b: &[u8]) -> ObjProp {
        ObjProp {
            mass: i32::from_le_bytes([b[0], b[1], b[2], b[3]]),
            hit_points: i16::from_le_bytes([b[4], b[5]]),
            render_type: RenderType::from_u8(b[7]),
            physics_model: b[8],
            physics_xr: b[11],
            flags: u16::from_le_bytes([b[20], b[21]]),
            mfd_id: i16::from_le_bytes([b[22], b[23]]),
            bitmap_3d: u16::from_le_bytes([b[24], b[25]]),
        }
    }

    /// `FRAME_NUM_3D`: world frames beyond the first.
    pub fn extra_frames(&self) -> usize {
        ((self.bitmap_3d & 0x7000) >> 12) as usize
    }
}

/// All object types' common properties, and where their art starts.
pub struct ObjProps {
    pub props: Vec<ObjProp>,
    /// Per critter type: how many directions its sprites are drawn from.
    pub critter_views: Vec<u8>,
    /// Per triple: the `objart.res` item of its inventory picture; its world
    /// frames follow it (`obj_load_art`).
    pub art_start: Vec<usize>,
}

impl ObjProps {
    /// Read `objprop.dat`: the common properties are its last
    /// `476 * 27` bytes, after the class-specific tables.
    pub fn load(data: &Path) -> std::io::Result<ObjProps> {
        let bytes = std::fs::read(data.join("objprop.dat"))?;
        ObjProps::parse(&bytes).ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "objprop.dat too short")
        })
    }

    pub fn parse(bytes: &[u8]) -> Option<ObjProps> {
        let n = num_triples();
        let start = bytes.len().checked_sub(n * OBJ_PROP_LEN)?;
        // `CritterProp` records (75 bytes each) sit just before the critter
        // subclass tables (91 bytes) and the common properties; `views` is
        // byte 69 of each.
        let critters = SUBCLASS_TYPES[CLASS_CRITTER as usize]
            .iter()
            .map(|&n| n as usize)
            .sum::<usize>();
        let critter_start = start.checked_sub(CRITTER_TAIL_LEN + critters * CRITTER_PROP_LEN)?;
        let critter_views = (0..critters)
            .map(|i| bytes[critter_start + i * CRITTER_PROP_LEN + 69])
            .collect();
        let props: Vec<ObjProp> = bytes[start..]
            .chunks_exact(OBJ_PROP_LEN)
            .map(ObjProp::parse)
            .collect();
        // Item 0 is a placeholder; each type then owns its inventory
        // picture, `extra_frames + 1` world frames and an editor icon.
        let mut art_start = Vec::with_capacity(n);
        let mut at = 1;
        for p in &props {
            art_start.push(at);
            at += 3 + p.extra_frames();
        }
        Some(ObjProps {
            props,
            critter_views,
            art_start,
        })
    }

    /// The `objart2.res` resource of a critter's standing sprite from
    /// `view` (0..8), following `critter_id_table`.
    pub fn critter_standing(&self, critter: usize, view: usize) -> Option<u16> {
        let views = *self.critter_views.get(critter)? as usize;
        let before: usize = self.critter_views[..critter]
            .iter()
            .map(|&v| v as usize)
            .sum();
        let view = if views > 1 { view % views } else { 0 };
        Some(CRITTER_STANDING_BASE + (before + view) as u16)
    }

    /// The `objart.res` item of a type's world frame.
    pub fn world_art(&self, triple: usize, frame: usize) -> Option<usize> {
        let p = self.props.get(triple)?;
        Some(self.art_start[triple] + 1 + frame.min(p.extra_frames()))
    }
}

/// The view (0..8) of a critter at `(x, y)` with `heading` drawn for an eye
/// at `(eye_x, eye_y)` (tiles), exactly as the original's octant test.
pub fn critter_view(x: f32, y: f32, heading: u8, eye_x: f32, eye_y: f32) -> usize {
    const QTAB: [i32; 4] = [3, 0, 2, 1];
    let xd = ((x - eye_x) * 256.0) as i32;
    let yd = ((y - eye_y) * 256.0) as i32;
    let q = QTAB[(if xd > 0 { 2 } else { 0 }) + (if yd > 0 { 1 } else { 0 })];
    let (xd, yd) = (xd.abs(), yd.abs());
    let l = if xd >= yd << 1 {
        0
    } else if xd << 1 <= yd {
        2
    } else {
        1
    };
    let view = if q & 1 != 0 {
        (q << 1) + 2 - l
    } else {
        (q << 1) + l
    };
    (((heading as i32 - (view << 5) + 0x180) & 0xff) >> 5) as usize
}

#[derive(Debug, Clone, Copy)]
pub struct Obj {
    pub id: usize,
    pub class: u8,
    pub subclass: u8,
    pub spec_id: u16,
    /// Position in tiles (east, north).
    pub x: f32,
    pub y: f32,
    /// Raw height; [`Obj::height_tiles`] scales it.
    pub z: u8,
    pub pitch: u8,
    /// 0..256 is a full turn.
    pub heading: u8,
    pub bank: u8,
    pub kind: u8,
    pub frame: u8,
}

pub const OBJ_LEN: usize = 27;

impl Obj {
    pub fn triple(&self) -> Option<usize> {
        triple_index(self.class, self.subclass, self.kind)
    }

    /// Height in tiles: `fix_from_obj_height` is `z << (8 + 5 - z_shift)`
    /// in 16.16 fixed point.
    pub fn height_tiles(&self, z_shift: u32) -> f32 {
        self.z as f32 / (1u32 << (3 + z_shift)) as f32
    }
}

/// The active objects of a level.
/// Size and textures of a container (crate) drawn as a box.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CrateShape {
    /// Half extents east, north, and full height, in tiles.
    pub radius: [f32; 3],
    /// Side and top/bottom textures: bit 7 set picks a level texture
    /// (`loved textures` index), clear a custom material.
    pub side: u8,
    pub top: u8,
}

/// `ObjContainer` records (21 bytes) in the level's container table.
const CONTAINER_LEN: usize = 21;
pub const CLASS_CONTAINER: u8 = 13;

/// The box of a crate (`SML_CRT`, `LG_CRT`, `SECURE_CONTR`), as
/// `obj_get_model_data` computes it, or None for other objects.
pub fn crate_shape(table: &[u8], obj: &Obj) -> Option<CrateShape> {
    if obj.class != CLASS_CONTAINER || obj.subclass != 0 || obj.kind > 2 {
        return None;
    }
    let (dims, tex) = match obj.kind {
        0 => ([8u8, 8, 8], [0x0c, 0x0b]),
        1 => ([16, 16, 16], [0x0c, 0x0b]),
        _ => ([32, 32, 32], [0x0a, 0x0a]),
    };
    let at = obj.spec_id as usize * CONTAINER_LEN;
    let rec = table.get(at..at + CONTAINER_LEN)?;
    let pick = |v: u8, d: u8| if v == 0 { d } else { v };
    let data1 = u32::from_le_bytes([rec[17], rec[18], rec[19], rec[20]]);
    // `dim << CONTAINER_MODEL_SHF` (10) in 16.16 fixed point.
    let r = |v: u8| v as f32 * 1024.0 / 65536.0;
    Some(CrateShape {
        radius: [
            r(pick(rec[14], dims[0])),
            r(pick(rec[15], dims[1])),
            r(pick(rec[16], dims[2])),
        ],
        side: pick(data1 as u8, tex[0]),
        top: pick((data1 >> 8) as u8, tex[1]),
    })
}

pub const CLASS_BIGSTUFF: u8 = 7;
/// `ObjBigstuff` records: header, cosmetic value, data1, data2.
const BIGSTUFF_LEN: usize = 16;

/// A bigstuff object's `data2` (a screen's picture, a texture index).
pub fn bigstuff_data2(table: &[u8], obj: &Obj) -> Option<u32> {
    let at = obj.spec_id as usize * BIGSTUFF_LEN + 12;
    Some(u32::from_le_bytes(table.get(at..at + 4)?.try_into().ok()?))
}

/// What a wall screen (`SCREEN`, `TMAP`, `SUPERSCREEN`, `BIGSCREEN`) shows.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScreenArt {
    /// `texture.res` small texture.
    SmallTexture(u16),
    /// A custom material (`citmat.res` 2180+).
    Custom(u16),
    /// One of the level's textures.
    LevelTexture(u16),
}

/// A wall screen's half width in tiles and its picture, as the original's
/// `FAUBJ_TPOLY` case picks them, or None for anything else.
pub fn screen(table: &[u8], obj: &Obj) -> Option<(f32, ScreenArt)> {
    if obj.class != CLASS_BIGSTUFF || obj.subclass != 2 || !(6..=9).contains(&obj.kind) {
        return None;
    }
    let data2 = bigstuff_data2(table, obj)?;
    // `scale = n - wlog` makes the picture 2^n / 128 tiles across.
    let half = match obj.kind {
        6 => 0.25,
        8 => 1.0,
        _ => 0.5,
    };
    if obj.kind == 7 {
        return Some((half, ScreenArt::LevelTexture(data2 as u16)));
    }
    if data2 == 0 {
        return None;
    }
    let rep = (data2 + obj.frame as u32) & 0xfff;
    let index = (rep & 0x7f) as u16;
    match (rep >> 7) & 3 {
        0 => Some((half, ScreenArt::SmallTexture(index))),
        1 => Some((half, ScreenArt::Custom(index))),
        _ => None, // text and scrolling text: not drawn yet
    }
}

/// A class's per-object table from a level (`level_base + 10 + class`).
pub fn load_class_table(archive: &ResFile, level: u16, class: u8) -> Result<Vec<u8>, ResError> {
    Ok(archive.load(level_base(level) + 10 + class as u16)?.data)
}

pub fn load_level_objects(archive: &ResFile, level: u16) -> Result<Vec<Obj>, ResError> {
    let res = archive.load(level_base(level) + 8)?;
    let mut out = Vec::new();
    for (id, b) in res.data.chunks_exact(OBJ_LEN).enumerate() {
        if b[0] == 0 || id == 0 {
            continue;
        }
        let x = u16::from_le_bytes([b[11], b[12]]);
        let y = u16::from_le_bytes([b[13], b[14]]);
        out.push(Obj {
            id,
            class: b[1],
            subclass: b[2],
            spec_id: u16::from_le_bytes([b[3], b[4]]),
            x: x as f32 / 256.0,
            y: y as f32 / 256.0,
            z: b[15],
            pitch: b[16],
            heading: b[17],
            bank: b[18],
            kind: b[20],
            frame: b[24],
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn triples_number_in_class_order() {
        assert_eq!(num_triples(), 476);
        assert_eq!(triple_index(0, 0, 0), Some(0));
        assert_eq!(triple_index(0, 1, 0), Some(5));
        assert_eq!(triple_index(1, 0, 0), Some(16));
        assert_eq!(triple_index(14, 4, 1), Some(475));
        assert_eq!(triple_index(14, 4, 2), None);
    }

    /// Which of a critter's 8 views faces a viewer, as `FAUBJ_CRIT` picks it.
    #[test]
    fn critter_view_faces_the_viewer() {
        // A critter heading 0 seen from straight ahead of where it faces.
        let front = critter_view(0.0, 0.0, 0, 0.0, 5.0);
        let back = critter_view(0.0, 0.0, 0, 0.0, -5.0);
        assert_ne!(front, back);
        assert_eq!((front + 4) % 8, back);
    }
}
