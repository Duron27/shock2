//! 3D object models (`obj3d.res`, ids 2300 + model number).
//!
//! A model is bytecode for the original's 3D interpreter (`INTERP.C`):
//! opcodes define points (absolute or relative to another point), set a fill
//! colour or texture, and draw polygons, with `sortnorm`/`jnorm` choosing a
//! draw order or skipping back faces for the camera. We need every polygon
//! once, so the interpreter here walks both sides of each sort, ignores the
//! visibility jumps, and records polygons with their fill.
//!
//! Coordinates are 16.16 fixed point in the original's object space (y up),
//! where one unit is one tile.

use std::collections::HashSet;

/// The resource id of model `n`.
pub const MODEL_BASE: u16 = 2300;
/// The resource id of virtual texture `n` (`RES_materialMaps`, `citmat.res`).
pub const MATERIAL_BASE: u16 = 475;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Fill {
    /// A palette colour.
    Color(u8),
    /// One of the object's per-instance colours (`g3_set_vcolor`).
    VColor(u16),
    /// A palette colour passed by the caller; unknown statically.
    Param,
    /// A virtual texture, with per-vertex uvs.
    Texture(u16),
}

#[derive(Debug, Clone)]
pub struct Poly {
    pub fill: Fill,
    /// Object-space points (x, y up, z).
    pub points: Vec<[f32; 3]>,
    /// Texture coordinates for a textured polygon (1.0 spans the texture).
    pub uvs: Vec<[f32; 2]>,
}

#[derive(Debug, Clone, Default)]
pub struct Model {
    pub polys: Vec<Poly>,
}

impl Model {
    /// Object-space bounds (min, max).
    pub fn bounds(&self) -> Option<([f32; 3], [f32; 3])> {
        let mut points = self.polys.iter().flat_map(|p| p.points.iter());
        let first = *points.next()?;
        Some(points.fold((first, first), |(lo, hi), p| {
            (
                [lo[0].min(p[0]), lo[1].min(p[1]), lo[2].min(p[2])],
                [hi[0].max(p[0]), hi[1].max(p[1]), hi[2].max(p[2])],
            )
        }))
    }

    /// Every virtual texture the model draws with.
    pub fn textures(&self) -> Vec<u16> {
        let mut out: Vec<u16> = self
            .polys
            .iter()
            .filter_map(|p| match p.fill {
                Fill::Texture(t) => Some(t),
                _ => None,
            })
            .collect();
        out.sort_unstable();
        out.dedup();
        out
    }
}

#[derive(Clone, Copy, Default)]
struct Point {
    pos: [f32; 3],
    uv: [f32; 2],
}

struct Interp<'a> {
    code: &'a [u8],
    points: Vec<Option<Point>>,
    fill: Fill,
    polys: Vec<Poly>,
    visited: HashSet<usize>,
}

fn fix(v: i32) -> f32 {
    v as f32 / 65536.0
}

impl Interp<'_> {
    fn u16(&self, at: usize) -> Option<u16> {
        Some(u16::from_le_bytes(
            self.code.get(at..at + 2)?.try_into().ok()?,
        ))
    }
    fn i16(&self, at: usize) -> Option<i16> {
        self.u16(at).map(|v| v as i16)
    }
    fn i32(&self, at: usize) -> Option<i32> {
        Some(i32::from_le_bytes(
            self.code.get(at..at + 4)?.try_into().ok()?,
        ))
    }
    fn vector(&self, at: usize) -> Option<[f32; 3]> {
        Some([
            fix(self.i32(at)?),
            fix(self.i32(at + 4)?),
            fix(self.i32(at + 8)?),
        ])
    }
    fn point(&self, i: u16) -> Option<Point> {
        self.points.get(i as usize).copied().flatten()
    }
    fn set(&mut self, i: u16, point: Point) {
        let i = i as usize;
        if i >= self.points.len() {
            self.points.resize(i + 1, None);
        }
        self.points[i] = Some(point);
    }
    fn set_uv(&mut self, i: u16, u: i32, v: i32) {
        if let Some(Some(p)) = self.points.get_mut(i as usize) {
            p.uv = [fix(u), fix(v)];
        }
    }
    fn relative(&mut self, at: usize, axes: &[usize]) -> Option<()> {
        let (dest, src) = (self.u16(at + 2)?, self.u16(at + 4)?);
        let mut p = self.point(src)?;
        for (n, &axis) in axes.iter().enumerate() {
            p.pos[axis] += fix(self.i32(at + 6 + n * 4)?);
        }
        self.set(dest, p);
        Some(())
    }
    fn poly(&mut self, indices: Vec<u16>, fill: Fill) {
        let points: Option<Vec<Point>> = indices.iter().map(|&i| self.point(i)).collect();
        if let Some(points) = points.filter(|p| p.len() >= 3) {
            self.polys.push(Poly {
                fill,
                uvs: points.iter().map(|p| p.uv).collect(),
                points: points.iter().map(|p| p.pos).collect(),
            });
        }
    }

    /// Run from `at` until `eof`, as `interpreter_loop` does.
    fn run(&mut self, mut at: usize, depth: u32) -> Option<()> {
        if depth > 64 {
            return None;
        }
        loop {
            if !self.visited.insert(at) {
                return Some(());
            }
            let op = self.u16(at)?;
            at = match op {
                0 => return Some(()), // eof
                1 => at + 28,         // jnorm: draw the face whatever its facing
                2 => at + 6,          // lnres: a line
                3 => {
                    // multires: count points from index start
                    let (count, start) = (self.u16(at + 2)?, self.u16(at + 4)?);
                    for i in 0..count {
                        let pos = self.vector(at + 6 + i as usize * 12)?;
                        self.set(start + i, Point { pos, uv: [0.0; 2] });
                    }
                    at + 6 + count as usize * 12
                }
                4 => {
                    // polyres: a flat polygon in the current colour
                    let count = self.u16(at + 2)? as usize;
                    let indices = (0..count)
                        .map(|i| self.u16(at + 4 + i * 2))
                        .collect::<Option<Vec<_>>>()?;
                    self.poly(indices, self.fill);
                    at + 4 + count * 2
                }
                5 => {
                    self.fill = Fill::Color(self.u16(at + 2)? as u8);
                    at + 4
                }
                6 => {
                    // sortnorm: both subtrees, then carry on
                    let a = at as isize + self.i16(at + 26)? as isize;
                    let b = at as isize + self.i16(at + 28)? as isize;
                    for branch in [a, b] {
                        if branch > 0 {
                            self.run(branch as usize, depth + 1);
                        }
                    }
                    at + 30
                }
                7 => return Some(()), // debug: stops the original too
                8 => at + 4 + self.u16(at + 2)? as usize * 4, // setshade
                9 => {
                    // goursurf: a shaded fill from a base colour
                    self.fill = Fill::Color(self.u16(at + 2)? as u8);
                    at + 4
                }
                10 => {
                    self.relative(at, &[0])?;
                    at + 10
                }
                11 => {
                    self.relative(at, &[1])?;
                    at + 10
                }
                12 => {
                    self.relative(at, &[2])?;
                    at + 10
                }
                13 => {
                    self.relative(at, &[0, 1])?;
                    at + 14
                }
                14 => {
                    self.relative(at, &[0, 2])?;
                    at + 14
                }
                15 => {
                    self.relative(at, &[1, 2])?;
                    at + 14
                }
                // icall: a sub-object by absolute pointer; unreachable here.
                16..=18 => at + 20,
                20 => {
                    // sfcal: a subroutine at a relative offset
                    let target = at + self.u16(at + 2)? as usize;
                    self.run(target, depth + 1);
                    at + 4
                }
                21 => {
                    let i = self.u16(at + 2)?;
                    let pos = self.vector(at + 4)?;
                    self.set(i, Point { pos, uv: [0.0; 2] });
                    at + 16
                }
                22 => {
                    let i = self.u16(at + 2)?;
                    let pos = self.vector(at + 4)?;
                    self.set(i, Point { pos, uv: [0.0; 2] });
                    at + 18
                }
                23 | 24 => at + 8, // getparms
                25 => {
                    self.fill = Fill::Param;
                    at + 4
                }
                26 | 27 => {
                    self.fill = Fill::VColor(self.u16(at + 2)?);
                    at + 4
                }
                28 => {
                    self.fill = Fill::VColor(self.u16(at + 2)?);
                    at + 6
                }
                29 => at + 4 + self.u16(at + 2)? as usize * 10, // rgbshades
                30 => at + 4,                                   // draw_mode
                31 => {
                    self.fill = Fill::Param;
                    at + 4
                }
                32 => {
                    self.fill = Fill::Param;
                    at + 6
                }
                34 | 35 => at + 6, // vpnt: points passed by the caller
                36 => {
                    let i = self.u16(at + 2)?;
                    let (u, v) = (self.i32(at + 4)?, self.i32(at + 8)?);
                    self.set_uv(i, u, v);
                    at + 12
                }
                37 => {
                    let count = self.u16(at + 2)? as usize;
                    for n in 0..count {
                        let e = at + 4 + n * 10;
                        let i = self.u16(e)?;
                        let (u, v) = (self.i32(e + 2)?, self.i32(e + 6)?);
                        self.set_uv(i, u, v);
                    }
                    at + 4 + count * 10
                }
                38 => {
                    // tmap: a textured polygon
                    let texture = self.u16(at + 2)?;
                    let count = self.u16(at + 4)? as usize;
                    let indices = (0..count)
                        .map(|i| self.u16(at + 6 + i * 2))
                        .collect::<Option<Vec<_>>>()?;
                    self.poly(indices, Fill::Texture(texture));
                    at + 6 + count * 2
                }
                39 => at + 8, // dbg
                _ => return None,
            };
        }
    }
}

/// Decode a model's bytecode.
pub fn parse(code: &[u8]) -> Model {
    let mut interp = Interp {
        code,
        points: Vec::new(),
        fill: Fill::Color(0),
        polys: Vec::new(),
        visited: HashSet::new(),
    };
    interp.run(0, 0);
    Model {
        polys: interp.polys,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn op(out: &mut Vec<u8>, words: &[i32], wide: &[bool]) {
        for (&w, &is_wide) in words.iter().zip(wide) {
            if is_wide {
                out.extend_from_slice(&w.to_le_bytes());
            } else {
                out.extend_from_slice(&(w as u16).to_le_bytes());
            }
        }
    }

    #[test]
    fn a_coloured_triangle_from_relative_points() {
        let one = 0x10000;
        let mut code = Vec::new();
        // defres 0 at origin
        op(
            &mut code,
            &[21, 0, 0, 0, 0],
            &[false, false, true, true, true],
        );
        // x_rel 1 = 0 + x
        op(&mut code, &[10, 1, 0, one], &[false, false, false, true]);
        // y_rel 2 = 0 + y
        op(&mut code, &[11, 2, 0, one], &[false, false, false, true]);
        // setcolor 42
        op(&mut code, &[5, 42], &[false, false]);
        // polyres 3: 0 1 2
        op(&mut code, &[4, 3, 0, 1, 2], &[false; 5]);
        op(&mut code, &[0], &[false]);
        let model = parse(&code);
        assert_eq!(model.polys.len(), 1);
        let poly = &model.polys[0];
        assert_eq!(poly.fill, Fill::Color(42));
        assert_eq!(
            poly.points,
            vec![[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        );
    }
}
