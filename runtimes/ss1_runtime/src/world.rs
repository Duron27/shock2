//! A level turned into engine scene objects.

use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;

use anyhow::{Context, Result};
use cgmath::{Vector2, Vector3, Vector4, vec2, vec3};
use engine::materials::LightmapMaterial;
use engine::scene::{SceneObject, VertexPositionTextureLightmapAtlasNormal};
use engine::texture::{Texture, TextureFilter, TextureOptions, TextureTrait};
use engine::texture_format::{PixelFormat, RawTextureData};
use ss1::{Image, Level, Palette, ResFile, mesh};

/// World units per map tile. Roughly feet, so the engine's distance
/// darkness and gloss settle at the same scale as in System Shock 2.
pub const TILE: f32 = 8.0;
const LIGHTMAP_WIDTH: usize = 2048;

pub struct World {
    pub level: Level,
    pub objects: Vec<SceneObject>,
    pub face_count: usize,
}

/// The lightmap shader insets each atlas rect by a texel of a 4096 atlas and
/// wraps inside it; with a whole-texture rect, pre-invert that so the face
/// samples exactly the patch it was baked into.
fn lightmap_uv([u, v]: [f32; 2]) -> Vector2<f32> {
    let half = 0.5 / 4096.0;
    let size = 1.0 - 2.0 * half;
    vec2((u - half) / size, (v - half) / size)
}

fn rgba_texture(rgba: Vec<u8>, width: usize, height: usize, options: &TextureOptions) -> Texture {
    engine::texture::init_from_memory2(
        RawTextureData {
            bytes: rgba,
            width: width as u32,
            height: height as u32,
            format: PixelFormat::RGBA,
        },
        options,
    )
}

/// Load a wall/floor texture: the 128x128 version if there is one, else the
/// smaller mips the game ships for distance.
fn load_texture(textures: &ResFile, palette: &Palette, id: u16) -> Option<Rc<dyn TextureTrait>> {
    for base in [1000u16, 707, 321, 75] {
        let Ok(resource) = textures.load(base + id) else {
            continue;
        };
        let Some(Ok(image)) = resource.item(0).map(Image::parse) else {
            continue;
        };
        if image.width == 0 || image.height == 0 {
            continue;
        }
        let texture = rgba_texture(
            image.to_rgba(palette),
            image.width,
            image.height,
            &TextureOptions::default(),
        );
        return Some(Rc::new(texture));
    }
    None
}

fn v3([x, y, z]: mesh::Vec3) -> Vector3<f32> {
    vec3(x * TILE, y * TILE, z * TILE)
}

impl World {
    pub fn load(data: &Path, number: u16) -> Result<World> {
        let archive = ResFile::open(data.join("archive.dat")).context("archive.dat")?;
        let textures = ResFile::open(data.join("texture.res")).context("texture.res")?;
        let palette = Palette::load_game(data)?;
        let level = Level::load(&archive, number)?;

        let faces = mesh::build(&level);
        let lightmap = mesh::bake(&level, &faces, LIGHTMAP_WIDTH);
        let lightmap_texture = Rc::new(engine::texture::init_from_memory2(
            RawTextureData {
                bytes: lightmap.pixels,
                width: lightmap.width as u32,
                height: lightmap.height as u32,
                format: PixelFormat::RGB,
            },
            &TextureOptions {
                wrap: false,
                filter: TextureFilter::Linear,
                ..TextureOptions::default()
            },
        ));

        // Faces grouped by texture, one draw per texture.
        let mut groups: HashMap<Option<u16>, Vec<VertexPositionTextureLightmapAtlasNormal>> =
            HashMap::new();
        let atlas: Vector4<f32> = Vector4::new(0.0, 0.0, 1.0, 1.0);
        for (face, lm) in faces.iter().zip(&lightmap.uvs) {
            let vertex = |i: usize| VertexPositionTextureLightmapAtlasNormal {
                position: v3(face.corners[i]),
                uv: vec2(face.uvs[i][0], face.uvs[i][1]),
                lightmap_uv: lightmap_uv(lm[i]),
                lightmap_atlas: atlas,
                normal: vec3(face.normal[0], face.normal[1], face.normal[2]),
            };
            let out = groups.entry(face.texture).or_default();
            out.extend([vertex(0), vertex(1), vertex(2)]);
            if !face.is_triangle() {
                out.extend([vertex(0), vertex(2), vertex(3)]);
            }
        }

        let fallback: Rc<dyn TextureTrait> = Rc::new(rgba_texture(
            vec![128, 128, 128, 255],
            1,
            1,
            &TextureOptions::default(),
        ));
        let mut objects = Vec::new();
        for (texture, vertices) in groups {
            let diffuse = texture
                .and_then(|id| load_texture(&textures, &palette, id))
                .unwrap_or_else(|| fallback.clone());
            let material = LightmapMaterial::create(
                lightmap_texture.clone(),
                diffuse,
                vec3(0.015, 0.015, 0.02),
            );
            let geometry = engine::scene::mesh::create(vertices);
            objects.push(SceneObject::new(material, Box::new(geometry)));
        }

        Ok(World {
            face_count: faces.len(),
            level,
            objects,
        })
    }

    /// Floor height (world units) under a world point, if it is open.
    pub fn floor_at(&self, x: f32, z: f32) -> Option<f32> {
        let (mx, my) = (x / TILE, -z / TILE);
        if mx < 0.0 || my < 0.0 {
            return None;
        }
        let tile = self.level.tile(mx as usize, my as usize)?;
        if tile.kind.is_solid() {
            return None;
        }
        let (s, t) = (mx.fract(), my.fract());
        let [sw, nw, ne, se] = tile.floor_corners();
        let south = sw + (se - sw) * s;
        let north = nw + (ne - nw) * s;
        Some((south + (north - south) * t) * self.level.height_scale() * TILE)
    }

    /// Ceiling height (world units) at the middle of the tile under a point.
    pub fn ceiling_at(&self, x: f32, z: f32) -> Option<f32> {
        let (mx, my) = (x / TILE, -z / TILE);
        if mx < 0.0 || my < 0.0 {
            return None;
        }
        let tile = self.level.tile(mx as usize, my as usize)?;
        let lowest = tile.ceiling_corners().into_iter().fold(f32::MAX, f32::min);
        Some(lowest * self.level.height_scale() * TILE)
    }

    /// A reasonable place to stand: the open tile with the most open space
    /// around it, nearest the middle of the level.
    pub fn start_position(&self) -> Vector3<f32> {
        let level = &self.level;
        let mut best = (i32::MIN, 0usize, 0usize);
        let (mut cx, mut cy, mut n) = (0.0f32, 0.0f32, 0.0f32);
        for y in 0..level.height {
            for x in 0..level.width {
                if !level.tile(x, y).unwrap().kind.is_solid() {
                    cx += x as f32;
                    cy += y as f32;
                    n += 1.0;
                }
            }
        }
        let (cx, cy) = (cx / n.max(1.0), cy / n.max(1.0));
        for y in 1..level.height - 1 {
            for x in 1..level.width - 1 {
                let tile = level.tile(x, y).unwrap();
                if tile.kind != ss1::TileType::Open || tile.ceiling < tile.floor + 6 {
                    continue;
                }
                let mut open = 0;
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        let t = level
                            .tile((x as i32 + dx) as usize, (y as i32 + dy) as usize)
                            .unwrap();
                        if t.kind == ss1::TileType::Open && t.floor == tile.floor {
                            open += 1;
                        }
                    }
                }
                let distance = ((x as f32 - cx).powi(2) + (y as f32 - cy).powi(2)).sqrt();
                let score = open * 100 - distance as i32;
                if score > best.0 {
                    best = (score, x, y);
                }
            }
        }
        let (x, z) = ((best.1 as f32 + 0.5) * TILE, -(best.2 as f32 + 0.5) * TILE);
        vec3(x, self.floor_at(x, z).unwrap_or(0.0), z)
    }
}
