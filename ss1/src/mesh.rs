//! Level geometry and baked lighting, engine-agnostic.
//!
//! Units are tiles: `x` east, `y` up, `z` south (so a map's north is `-z`,
//! keeping the world right-handed with `y` up). A tile's floor, ceiling and
//! the walls bounding its open space become [`Face`]s; each face is a quad
//! (a triangle repeats its last corner) whose first and third corners are the
//! diagonal it is split along.
//!
//! Lighting replaces SS1's flat per-tile shade with a smooth field: tile
//! shades are interpolated between tile centres, walls blend from the floor's
//! shade to the ceiling's, and every face picks up soft contact shadowing
//! where it meets a wall, floor or ceiling.

use crate::level::{Level, NE, NW, SE, SW, Tile, TileType};

pub type Vec3 = [f32; 3];

/// What a face is, for lighting.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Surface {
    Floor,
    Ceiling,
    /// A wall seen from inside tile `(x, y)`.
    Wall,
}

#[derive(Debug, Clone)]
pub struct Face {
    /// Texture id (`texture.res` 1000 + id), if the level names one.
    pub texture: Option<u16>,
    pub corners: [Vec3; 4],
    pub uvs: [[f32; 2]; 4],
    pub normal: Vec3,
    pub surface: Surface,
    /// The open tile the face belongs to.
    pub tile: (usize, usize),
}

impl Face {
    pub fn is_triangle(&self) -> bool {
        self.corners[2] == self.corners[3]
    }

    /// Bilinear point on the face for `s, t` in `0..=1` (`s` runs corner 0
    /// to 1, `t` corner 0 to 3).
    pub fn point(&self, s: f32, t: f32) -> Vec3 {
        let [a, b, c, d] = self.corners;
        let mut out = [0.0; 3];
        for i in 0..3 {
            let bottom = a[i] + (b[i] - a[i]) * s;
            let top = d[i] + (c[i] - d[i]) * s;
            out[i] = bottom + (top - bottom) * t;
        }
        out
    }
}

/// Corner positions in tile coordinates (east, north).
fn corner_xy(x: usize, y: usize, corner: usize) -> (f32, f32) {
    let (x, y) = (x as f32, y as f32);
    match corner {
        SW => (x, y),
        NW => (x, y + 1.0),
        NE => (x + 1.0, y + 1.0),
        _ => (x + 1.0, y),
    }
}

/// The world point of a tile corner at map height `h`.
fn point(level: &Level, x: usize, y: usize, corner: usize, h: f32) -> Vec3 {
    let (cx, cy) = corner_xy(x, y, corner);
    [cx, h * level.height_scale(), -cy]
}

/// Floor texture coordinates for a corner, turned by the tile's rotation.
fn floor_uv(corner: usize, rotation: u8) -> [f32; 2] {
    let base = match corner {
        SW => [0.0, 1.0],
        NW => [0.0, 0.0],
        NE => [1.0, 0.0],
        _ => [1.0, 1.0],
    };
    let mut uv = base;
    for _ in 0..rotation {
        uv = [1.0 - uv[1], uv[0]];
    }
    uv
}

/// Edges in order N, E, S, W: our two corners, the neighbour's matching
/// corners, the step to the neighbour and the wall normal facing into us.
const EDGES: [([usize; 2], [usize; 2], (i32, i32), Vec3); 4] = [
    ([NW, NE], [SW, SE], (0, 1), [0.0, 0.0, 1.0]),
    ([NE, SE], [NW, SW], (1, 0), [-1.0, 0.0, 0.0]),
    ([SE, SW], [NE, NW], (0, -1), [0.0, 0.0, -1.0]),
    ([SW, NW], [SE, NE], (-1, 0), [1.0, 0.0, 0.0]),
];

fn edge_has(corners: [usize; 2], corner: usize) -> bool {
    corners.contains(&corner)
}

/// Whether `tile`'s side made of `corners` is solid wall.
fn side_is_solid(tile: Option<&Tile>, corners: [usize; 2]) -> bool {
    match tile.map(|t| t.kind) {
        None | Some(TileType::Solid) => true,
        Some(TileType::SolidDiagonal(c)) => edge_has(corners, c),
        _ => false,
    }
}

fn neighbour(level: &Level, x: usize, y: usize, (dx, dy): (i32, i32)) -> Option<&Tile> {
    let nx = x as i32 + dx;
    let ny = y as i32 + dy;
    if nx < 0 || ny < 0 {
        return None;
    }
    level.tile(nx as usize, ny as usize)
}

fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn sub(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn normalize(v: Vec3) -> Vec3 {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-6);
    [v[0] / len, v[1] / len, v[2] / len]
}

/// Build every face of a level.
pub fn build(level: &Level) -> Vec<Face> {
    let mut faces = Vec::new();
    for y in 0..level.height {
        for x in 0..level.width {
            let tile = level.tile(x, y).unwrap();
            if tile.kind.is_solid() {
                continue;
            }
            add_floor_and_ceiling(level, x, y, tile, &mut faces);
            add_walls(level, x, y, tile, &mut faces);
        }
    }
    faces
}

fn add_floor_and_ceiling(level: &Level, x: usize, y: usize, tile: &Tile, faces: &mut Vec<Face>) {
    // Corner order for the quad: its first and third corners are the split.
    let order: [usize; 4] = match tile.kind {
        TileType::SolidDiagonal(solid) => {
            // The three open corners, in winding order, as a triangle.
            let ring = [SW, NW, NE, SE];
            let at = ring.iter().position(|&c| c == solid).unwrap();
            let a = ring[(at + 1) % 4];
            let b = ring[(at + 2) % 4];
            let c = ring[(at + 3) % 4];
            [a, b, c, c]
        }
        TileType::Slope {
            split: crate::level::Split::Falling,
            ..
        } => [NW, NE, SE, SW],
        _ => [SW, NW, NE, SE],
    };
    let floor = tile.floor_corners();
    let ceiling = tile.ceiling_corners();
    let floor_points = order.map(|c| point(level, x, y, c, floor[c]));
    let ceiling_points = order.map(|c| point(level, x, y, c, ceiling[c]));

    let floor_normal = normalize(cross(
        sub(floor_points[1], floor_points[0]),
        sub(floor_points[2], floor_points[0]),
    ));
    // Winding SW -> NW -> NE is clockwise seen from above, so the cross
    // product points down; flip to face up into the room.
    let floor_normal = if floor_normal[1] < 0.0 {
        floor_normal.map(|v| -v)
    } else {
        floor_normal
    };
    let ceiling_normal = normalize(cross(
        sub(ceiling_points[1], ceiling_points[0]),
        sub(ceiling_points[2], ceiling_points[0]),
    ));
    let ceiling_normal = if ceiling_normal[1] > 0.0 {
        ceiling_normal.map(|v| -v)
    } else {
        ceiling_normal
    };

    faces.push(Face {
        texture: level.texture_id(tile.floor_texture),
        corners: floor_points,
        uvs: order.map(|c| floor_uv(c, tile.floor_rotation)),
        normal: floor_normal,
        surface: Surface::Floor,
        tile: (x, y),
    });
    faces.push(Face {
        texture: level.texture_id(tile.ceiling_texture),
        corners: ceiling_points,
        uvs: order.map(|c| floor_uv(c, tile.ceiling_rotation)),
        normal: ceiling_normal,
        surface: Surface::Ceiling,
        tile: (x, y),
    });

    // The diagonal face of a half-solid tile.
    if let TileType::SolidDiagonal(_) = tile.kind {
        let (a, b) = (order[0], order[2]);
        let [a_floor, b_floor] = [floor[a], floor[b]];
        let [a_ceil, b_ceil] = [ceiling[a], ceiling[b]];
        let normal = {
            let pa = point(level, x, y, a, 0.0);
            let pb = point(level, x, y, b, 0.0);
            let along = sub(pb, pa);
            // Face the open corner (order[1]).
            let open = point(level, x, y, order[1], 0.0);
            let n = normalize([-along[2], 0.0, along[0]]);
            let to_open = sub(open, pa);
            if n[0] * to_open[0] + n[2] * to_open[2] < 0.0 {
                n.map(|v| -v)
            } else {
                n
            }
        };
        let width = std::f32::consts::SQRT_2;
        faces.push(wall_face(
            level,
            tile,
            (x, y),
            [(a, a_floor, a_ceil), (b, b_floor, b_ceil)],
            normal,
            width,
            level.texture_id(tile.wall_texture),
        ));
    }
}

/// A wall quad along our corners `a -> b`, bottom to top at each.
fn wall_face(
    level: &Level,
    tile: &Tile,
    (x, y): (usize, usize),
    [(a, a_bottom, a_top), (b, b_bottom, b_top)]: [(usize, f32, f32); 2],
    normal: Vec3,
    width: f32,
    texture: Option<u16>,
) -> Face {
    let scale = level.height_scale();
    // Textures repeat once per tile up the wall, anchored to the ceiling
    // like the original, nudged by the tile's vertical lock offset.
    let offset = (tile.flags & 0x1f) as f32 * scale;
    let top_ref = tile.ceiling as f32 * scale;
    let v = |h: f32| top_ref - h * scale + offset;
    Face {
        texture,
        corners: [
            point(level, x, y, a, a_bottom),
            point(level, x, y, b, b_bottom),
            point(level, x, y, b, b_top),
            point(level, x, y, a, a_top),
        ],
        uvs: [
            [0.0, v(a_bottom)],
            [width, v(b_bottom)],
            [width, v(b_top)],
            [0.0, v(a_top)],
        ],
        normal,
        surface: Surface::Wall,
        tile: (x, y),
    }
}

fn add_walls(level: &Level, x: usize, y: usize, tile: &Tile, faces: &mut Vec<Face>) {
    let floor = tile.floor_corners();
    let ceiling = tile.ceiling_corners();
    for (ours, theirs, step, normal) in EDGES {
        if let TileType::SolidDiagonal(solid) = tile.kind
            && edge_has(ours, solid)
        {
            continue;
        }
        let [a, b] = ours;
        let other = neighbour(level, x, y, step);
        // `MAP_FRIEND` borrows the neighbour's wall texture (`_fr_parse_wall`).
        let wall = match other {
            Some(o) if tile.flags & 0x100 != 0 => o.wall_texture,
            _ => tile.wall_texture,
        };
        let texture = level.texture_id(wall);
        if side_is_solid(other, theirs) {
            faces.push(wall_face(
                level,
                tile,
                (x, y),
                [(a, floor[a], ceiling[a]), (b, floor[b], ceiling[b])],
                normal,
                1.0,
                texture,
            ));
            continue;
        }
        let other = other.unwrap();
        let (their_floor, their_ceiling) = (other.floor_corners(), other.ceiling_corners());
        let [ta, tb] = theirs;
        // A step up to the neighbour's floor (capped at our ceiling).
        let step_a = their_floor[ta].min(ceiling[a]).max(floor[a]);
        let step_b = their_floor[tb].min(ceiling[b]).max(floor[b]);
        if step_a > floor[a] + 0.01 || step_b > floor[b] + 0.01 {
            faces.push(wall_face(
                level,
                tile,
                (x, y),
                [(a, floor[a], step_a), (b, floor[b], step_b)],
                normal,
                1.0,
                texture,
            ));
        }
        // A drop from our ceiling to the neighbour's lower one.
        let drop_a = their_ceiling[ta].max(floor[a].max(step_a)).min(ceiling[a]);
        let drop_b = their_ceiling[tb].max(floor[b].max(step_b)).min(ceiling[b]);
        if drop_a < ceiling[a] - 0.01 || drop_b < ceiling[b] - 0.01 {
            faces.push(wall_face(
                level,
                tile,
                (x, y),
                [(a, drop_a, ceiling[a]), (b, drop_b, ceiling[b])],
                normal,
                1.0,
                texture,
            ));
        }
    }
}

/// Smooth, original-derived lighting.
pub struct Lighting<'a> {
    level: &'a Level,
}

impl<'a> Lighting<'a> {
    pub fn new(level: &'a Level) -> Self {
        Lighting { level }
    }

    /// A tile's authored brightness, 0..1 (shade 0 is fully lit).
    fn tile_brightness(tile: &Tile, ceiling: bool) -> f32 {
        let shade = if ceiling {
            tile.ceiling_shade()
        } else {
            tile.floor_shade()
        };
        1.0 - shade as f32 / 16.0
    }

    /// Brightness at map point `(mx, my)` (tile units, north up),
    /// interpolated between the centres of the open tiles around it.
    fn field(&self, mx: f32, my: f32, ceiling: bool, home: (usize, usize)) -> f32 {
        let fx = mx - 0.5;
        let fy = my - 0.5;
        let (x0, y0) = (fx.floor(), fy.floor());
        let (tx, ty) = (fx - x0, fy - y0);
        let mut sum = 0.0;
        let mut weight = 0.0;
        for (dx, dy, w) in [
            (0, 0, (1.0 - tx) * (1.0 - ty)),
            (1, 0, tx * (1.0 - ty)),
            (0, 1, (1.0 - tx) * ty),
            (1, 1, tx * ty),
        ] {
            let (x, y) = (x0 as i32 + dx, y0 as i32 + dy);
            if x < 0 || y < 0 {
                continue;
            }
            if let Some(tile) = self.level.tile(x as usize, y as usize)
                && !tile.kind.is_solid()
                && w > 0.0
            {
                sum += Self::tile_brightness(tile, ceiling) * w;
                weight += w;
            }
        }
        if weight > 0.0 {
            sum / weight
        } else {
            let tile = self.level.tile(home.0, home.1).unwrap();
            Self::tile_brightness(tile, ceiling)
        }
    }

    /// Soft darkening near the walls bounding a floor or ceiling.
    fn edge_occlusion(&self, face: &Face, mx: f32, my: f32) -> f32 {
        let (x, y) = face.tile;
        let tile = self.level.tile(x, y).unwrap();
        let mut ao = 1.0;
        for (ours, theirs, step, _) in EDGES {
            let other = neighbour(self.level, x, y, step);
            let blocked = side_is_solid(other, theirs)
                || other.is_some_and(|o| match face.surface {
                    Surface::Floor => o.floor > tile.floor + 1,
                    _ => o.ceiling + 1 < tile.ceiling,
                });
            if !blocked {
                continue;
            }
            let _ = ours;
            let d = match step {
                (0, 1) => (y as f32 + 1.0) - my,
                (0, -1) => my - y as f32,
                (1, 0) => (x as f32 + 1.0) - mx,
                _ => mx - x as f32,
            }
            .max(0.0);
            ao *= 1.0 - 0.45 * (-d / 0.12).exp();
        }
        ao
    }

    /// Final light at a face point, 0..1.
    pub fn sample(&self, face: &Face, s: f32, t: f32) -> f32 {
        let p = face.point(s, t);
        let (mx, my) = (p[0], -p[2]);
        let scale = self.level.height_scale();
        let light = match face.surface {
            Surface::Floor => {
                self.field(mx, my, false, face.tile) * self.edge_occlusion(face, mx, my)
            }
            Surface::Ceiling => {
                self.field(mx, my, true, face.tile) * self.edge_occlusion(face, mx, my)
            }
            Surface::Wall => {
                // Nudge into the room so the field reads the open side.
                let (mx, my) = (mx + face.normal[0] * 0.05, my - face.normal[2] * 0.05);
                let tile = self.level.tile(face.tile.0, face.tile.1).unwrap();
                let floor = tile.floor as f32 * scale;
                let ceiling = tile.ceiling as f32 * scale;
                let span = (ceiling - floor).max(0.01);
                let f = ((p[1] - floor) / span).clamp(0.0, 1.0);
                let lit = self.field(mx, my, false, face.tile) * (1.0 - f)
                    + self.field(mx, my, true, face.tile) * f;
                let near_floor = 1.0 - 0.4 * (-(p[1] - floor).max(0.0) / 0.12).exp();
                let near_ceiling = 1.0 - 0.3 * (-(ceiling - p[1]).max(0.0) / 0.12).exp();
                lit * near_floor * near_ceiling
            }
        };
        light.clamp(0.0, 1.0)
    }
}

/// A baked lightmap atlas: one `PATCH`-texel square per face.
pub struct Lightmap {
    pub width: usize,
    pub height: usize,
    /// RGB.
    pub pixels: Vec<u8>,
    /// Per face: the atlas UV of each corner.
    pub uvs: Vec<[[f32; 2]; 4]>,
}

pub const PATCH: usize = 16;

pub fn bake(level: &Level, faces: &[Face], width: usize) -> Lightmap {
    let per_row = width / PATCH;
    let rows = faces.len().div_ceil(per_row).max(1);
    let height = (rows * PATCH).next_power_of_two();
    let mut pixels = vec![0u8; width * height * 3];
    let mut uvs = Vec::with_capacity(faces.len());
    let lighting = Lighting::new(level);
    let inner = (PATCH - 2) as f32;
    for (i, face) in faces.iter().enumerate() {
        let (px, py) = ((i % per_row) * PATCH, (i / per_row) * PATCH);
        for ty in 0..PATCH {
            for tx in 0..PATCH {
                // Texel centres; the border texels extend the face slightly
                // so bilinear filtering never reads a neighbouring patch.
                let s = ((tx as f32 + 0.5 - 1.0) / inner).clamp(0.0, 1.0);
                let t = ((ty as f32 + 0.5 - 1.0) / inner).clamp(0.0, 1.0);
                let v = (lighting.sample(face, s, t) * 255.0).round() as u8;
                let at = ((py + ty) * width + px + tx) * 3;
                pixels[at..at + 3].copy_from_slice(&[v, v, v]);
            }
        }
        let u0 = (px as f32 + 1.0) / width as f32;
        let u1 = (px as f32 + 1.0 + inner) / width as f32;
        let v0 = (py as f32 + 1.0) / height as f32;
        let v1 = (py as f32 + 1.0 + inner) / height as f32;
        uvs.push([[u0, v0], [u1, v0], [u1, v1], [u0, v1]]);
    }
    Lightmap {
        width,
        height,
        pixels,
        uvs,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn level(tiles: &[(usize, usize, u8, u8, u8)]) -> Level {
        // (x, y, type, floor, ceiling depth) on a 3x3 solid map.
        let mut raw = vec![[0u8; 16]; 9];
        for &(x, y, kind, floor, depth) in tiles {
            let t = &mut raw[y * 3 + x];
            t[0] = kind;
            t[1] = floor;
            t[2] = depth;
        }
        Level {
            number: 0,
            width: 3,
            height: 3,
            z_shift: 3,
            tiles: raw.iter().map(|b| Tile::parse(b)).collect(),
            textures: vec![7],
        }
    }

    #[test]
    fn a_lone_open_tile_is_a_closed_box() {
        let level = level(&[(1, 1, 1, 0, 16)]);
        let faces = build(&level);
        assert_eq!(faces.len(), 6, "floor, ceiling and four walls");
        let walls: Vec<_> = faces
            .iter()
            .filter(|f| f.surface == Surface::Wall)
            .collect();
        for wall in walls {
            // Floor height 0 to ceiling 16 units = 2 tiles.
            assert_eq!(wall.corners[0][1], 0.0);
            assert_eq!(wall.corners[3][1], 2.0);
            // Normals face the tile centre (1.5, -1.5).
            let mid = wall.point(0.5, 0.5);
            let to_centre = [1.5 - mid[0], -1.5 - mid[2]];
            assert!(wall.normal[0] * to_centre[0] + wall.normal[2] * to_centre[1] > 0.0);
        }
    }

    #[test]
    fn a_higher_neighbour_gets_a_step_wall_only() {
        let level = level(&[(1, 1, 1, 0, 16), (1, 2, 1, 4, 16)]);
        let faces = build(&level);
        let steps: Vec<_> = faces
            .iter()
            .filter(|f| {
                f.surface == Surface::Wall && f.tile == (1, 1) && f.normal == [0.0, 0.0, 1.0]
            })
            .collect();
        assert_eq!(steps.len(), 1);
        assert_eq!(steps[0].corners[0][1], 0.0);
        assert_eq!(steps[0].corners[3][1], 0.5);
    }

    #[test]
    fn diagonal_tiles_have_a_triangle_floor_and_a_diagonal_wall() {
        let level = level(&[(1, 1, 2, 0, 16)]);
        let faces = build(&level);
        let floor = faces.iter().find(|f| f.surface == Surface::Floor).unwrap();
        assert!(floor.is_triangle());
        // N and W edges belong to the solid half: two edge walls plus the diagonal.
        let walls = faces.iter().filter(|f| f.surface == Surface::Wall).count();
        assert_eq!(walls, 3);
    }

    #[test]
    fn lighting_follows_shade_and_darkens_corners() {
        let mut level = level(&[(1, 1, 1, 0, 16)]);
        let lit = Lighting::new(&level);
        let floor = build(&level)
            .into_iter()
            .find(|f| f.surface == Surface::Floor)
            .unwrap();
        let centre = lit.sample(&floor, 0.5, 0.5);
        let corner = lit.sample(&floor, 0.0, 0.0);
        assert!(centre > 0.9, "shade 0 is bright: {centre}");
        assert!(corner < centre * 0.6, "walls shade their foot: {corner}");
        level.tiles[4].flags |= 12 << 16;
        let dim = Lighting::new(&level).sample(&floor, 0.5, 0.5);
        assert!(dim < 0.3, "shade 12 is dark: {dim}");
    }
}
