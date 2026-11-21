//! Level objects drawn as the original drew them: camera-facing sprites
//! (`FAUBJ_BITMAP`), fixed flat bitmaps such as doors and signs
//! (`FAUBJ_TEXBITMAP`), and critters (`FAUBJ_CRIT`), whose sprite is the one
//! of their (up to) eight views that faces the camera, and polygon models
//! (`obj3d.res`) as real meshes.
//!
//! Each object is lit by the baked lightmap texel of the floor under it, so
//! it sits in the same light as the room, and its art is Scale2x'd for
//! smoother edges.

use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;

use cgmath::{InnerSpace, Matrix4, Rad, Vector2, Vector3, Vector4, vec2, vec3};
use engine::materials::LightmapMaterial;
use engine::scene::render_pass::RenderPass;
use engine::scene::{SceneObject, VertexPositionTextureLightmapAtlasNormal};
use engine::texture::{Texture, TextureOptions, TextureTrait};
use engine::texture_format::{PixelFormat, RawTextureData};
use ss1::model::{Fill, MATERIAL_BASE, MODEL_BASE, Model, Poly};
use ss1::objects::{CLASS_CRITTER, FLAG_LARGE, Obj, ObjProps, RenderType, triple_index};
use ss1::{Image, Palette, ResFile, Resource};

use crate::world::TILE;

/// World units per sprite pixel: `g3_set_bitmap_scale(2048/3)`, i.e. a
/// tile is 96 pixels (48 for `MY_IM_LARGE`).
const SPRITE_SCALE: f32 = TILE / 96.0;
/// Flat bitmaps: `w << 9` half-extent in 16.16, i.e. 64 pixels per tile.
const FLAT_SCALE: f32 = TILE / 64.0;

pub enum Facing {
    /// Turns to face the camera about the vertical axis.
    Billboard,
    /// Keeps the object's heading.
    Fixed,
    /// Faces the camera, showing the view that matches its heading.
    Critter,
    /// A 3D model, oriented like a fixed object.
    Model,
}

pub struct Placed {
    /// One sprite or model (one object per texture), or one per critter view.
    views: Vec<Vec<SceneObject>>,
    position: Vector3<f32>,
    facing: Facing,
    /// Orientation of fixed objects: heading, pitch, bank (`ORDER_YXZ`).
    rotation: Matrix4<f32>,
    raw_heading: u8,
    /// Map position, in tiles.
    tile: (f32, f32),
}

impl Placed {
    /// The object to draw for a camera at `eye` (world) looking at `yaw`
    /// degrees, with its transform set.
    pub fn objects(&self, camera_yaw: f32, eye: Vector3<f32>) -> Vec<SceneObject> {
        let view = match self.facing {
            Facing::Critter if self.views.len() > 1 => ss1::objects::critter_view(
                self.tile.0,
                self.tile.1,
                self.raw_heading,
                eye.x / TILE,
                -eye.z / TILE,
            ),
            _ => 0,
        };
        let Some(parts) = self.views.get(view) else {
            return Vec::new();
        };
        let rotation = match self.facing {
            Facing::Fixed | Facing::Model => self.rotation,
            _ => Matrix4::from_angle_y(Rad(-camera_yaw.to_radians())),
        };
        let transform = Matrix4::from_translation(self.position) * rotation;
        parts
            .iter()
            .map(|part| {
                let mut part = part.clone();
                part.transform = transform;
                part
            })
            .collect()
    }
}

/// RGBA through the palette, with each clear pixel taking the colour of an
/// opaque neighbour so filtering doesn't fringe cut-out edges with black.
fn rgba_dilated(image: &Image, palette: &Palette) -> Vec<u8> {
    let mut rgba = image.to_rgba(palette);
    let (w, h) = (image.width, image.height);
    for _ in 0..2 {
        let source = rgba.clone();
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) * 4;
                if source[i + 3] != 0 {
                    continue;
                }
                let mut sum = [0u32; 3];
                let mut n = 0;
                for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                    let (nx, ny) = (x as i32 + dx, y as i32 + dy);
                    if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                        continue;
                    }
                    let j = (ny as usize * w + nx as usize) * 4;
                    if source[j + 3] != 0 || (source[j] | source[j + 1] | source[j + 2]) != 0 {
                        for c in 0..3 {
                            sum[c] += source[j + c] as u32;
                        }
                        n += 1;
                    }
                }
                if n > 0 {
                    for c in 0..3 {
                        rgba[i + c] = (sum[c] / n) as u8;
                    }
                }
            }
        }
    }
    rgba
}

#[derive(Clone)]
struct Sprite {
    texture: Rc<dyn TextureTrait>,
    width: f32,
    height: f32,
    anchor: [i16; 4],
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum ArtKey {
    /// An item of `objart.res`.
    Object(usize),
    /// A critter resource in `objart2.res` (its first frame).
    Critter(u16),
}

pub struct ObjectArt {
    props: ObjProps,
    art: Resource,
    critters: Option<ResFile>,
    models: Option<ResFile>,
    materials: Option<ResFile>,
    textures: Option<ResFile>,
    /// The level's texture ids, for crates that use them.
    level_textures: Vec<u16>,
    /// The level's container table, for crate sizes.
    containers: Option<Vec<u8>>,
    /// The palette as a 16x16 texture, for flat-coloured polygons.
    swatches: Rc<dyn TextureTrait>,
    material_cache: HashMap<u16, Option<Rc<dyn TextureTrait>>>,
    model_cache: HashMap<u16, Option<Rc<Model>>>,
    palette: Palette,
    cache: HashMap<ArtKey, Option<Sprite>>,
}

impl ObjectArt {
    pub fn new(
        props: ObjProps,
        art: Resource,
        data: &Path,
        palette: Palette,
        level_textures: Vec<u16>,
        containers: Option<Vec<u8>>,
    ) -> Self {
        let mut bytes = Vec::with_capacity(256 * 4);
        for i in 0..256 {
            let [r, g, b] = palette.0[i];
            bytes.extend_from_slice(&[r, g, b, 255]);
        }
        let swatches = engine::texture::init_from_memory2(
            RawTextureData {
                bytes,
                width: 16,
                height: 16,
                format: PixelFormat::RGBA,
            },
            &TextureOptions {
                wrap: false,
                ..TextureOptions::default()
            },
        );
        ObjectArt {
            props,
            art,
            critters: ResFile::open(data.join("objart2.res")).ok(),
            models: ResFile::open(data.join("obj3d.res")).ok(),
            materials: ResFile::open(data.join("citmat.res")).ok(),
            textures: ResFile::open(data.join("texture.res")).ok(),
            level_textures,
            containers,
            swatches: Rc::new(swatches),
            material_cache: HashMap::new(),
            model_cache: HashMap::new(),
            palette,
            cache: HashMap::new(),
        }
    }

    fn sprite(&mut self, key: ArtKey) -> Option<Sprite> {
        if let Some(hit) = self.cache.get(&key) {
            return hit.clone();
        }
        let image = match key {
            ArtKey::Object(item) => self.art.item(item).and_then(|b| Image::parse(b).ok()),
            ArtKey::Critter(id) => self
                .critters
                .as_ref()
                .and_then(|res| res.load(id).ok())
                .and_then(|r| r.item(0).and_then(|b| Image::parse(b).ok())),
        };
        let made = image
            .filter(|image| image.width > 0 && image.height > 0)
            .map(|image| {
                let big = ss1::scale::scale2x(&ss1::scale::scale2x(&image));
                let texture = engine::texture::init_from_memory2(
                    RawTextureData {
                        bytes: rgba_dilated(&big, &self.palette),
                        width: big.width as u32,
                        height: big.height as u32,
                        format: PixelFormat::RGBA,
                    },
                    &TextureOptions {
                        wrap: false,
                        ..TextureOptions::default()
                    },
                );
                Sprite {
                    texture: Rc::new(texture),
                    width: image.width as f32,
                    height: image.height as f32,
                    anchor: image.anchor,
                }
            });
        self.cache.insert(key, made.clone());
        made
    }

    /// Scene objects for every drawable object of a level.
    pub fn place(
        &mut self,
        objs: &[Obj],
        z_shift: u32,
        lightmap: &Rc<Texture>,
        light_uv: impl Fn(f32, f32) -> Option<Vector2<f32>>,
    ) -> Vec<Placed> {
        let first_critter = triple_index(CLASS_CRITTER, 0, 0).unwrap();
        let mut out = Vec::new();
        for obj in objs {
            let Some(triple) = obj.triple() else { continue };
            let prop = self.props.props[triple];
            let facing = match prop.render_type {
                RenderType::TextPoly | RenderType::AnimPoly | RenderType::FlatPoly => Facing::Model,
                RenderType::Special if self.crate_model(obj).is_some() => Facing::Model,
                RenderType::Bitmap => Facing::Billboard,
                RenderType::TexBitmap => Facing::Fixed,
                RenderType::Critter => Facing::Critter,
                _ => continue,
            };
            let Some(lm) = light_uv(obj.x, obj.y) else {
                continue;
            };
            let position = vec3(
                obj.x * TILE,
                obj.height_tiles(z_shift) * TILE,
                -obj.y * TILE,
            );
            if let Facing::Model = facing {
                let model = self
                    .crate_model(obj)
                    .map(Rc::new)
                    .or_else(|| self.model(prop.mfd_id.max(0) as u16));
                let parts = match model {
                    Some(model) => self.model_objects(&model, lightmap, lm),
                    None => Vec::new(),
                };
                if !parts.is_empty() {
                    out.push(Placed {
                        views: vec![parts],
                        position,
                        facing,
                        rotation: orientation(obj),
                        raw_heading: obj.heading,
                        tile: (obj.x, obj.y),
                    });
                }
                continue;
            }
            let keys: Vec<ArtKey> = match facing {
                Facing::Critter => {
                    let critter = triple - first_critter;
                    let views = self.props.critter_views.get(critter).copied().unwrap_or(1);
                    (0..views.max(1) as usize)
                        .filter_map(|v| self.props.critter_standing(critter, v))
                        .map(ArtKey::Critter)
                        .collect()
                }
                _ => match self.props.world_art(triple, obj.frame as usize) {
                    Some(item) => vec![ArtKey::Object(item)],
                    None => continue,
                },
            };
            let large = prop.flags & FLAG_LARGE != 0;
            let views: Vec<Vec<SceneObject>> = keys
                .into_iter()
                .map(|key| {
                    self.sprite(key)
                        .map(|sprite| make_object(&sprite, &facing, large, obj, lightmap, lm))
                        .into_iter()
                        .collect()
                })
                .collect();
            if views.iter().all(Vec::is_empty) {
                continue;
            }
            out.push(Placed {
                views,
                position,
                facing,
                rotation: orientation(obj),
                raw_heading: obj.heading,
                tile: (obj.x, obj.y),
            });
        }
        out
    }
}

impl ObjectArt {
    fn model(&mut self, number: u16) -> Option<Rc<Model>> {
        if let Some(hit) = self.model_cache.get(&number) {
            return hit.clone();
        }
        let model = self
            .models
            .as_ref()
            .and_then(|res| res.load(MODEL_BASE + number).ok())
            .map(|r| Rc::new(ss1::model::parse(r.item(0).unwrap_or_default())))
            .filter(|m| !m.polys.is_empty());
        self.model_cache.insert(number, model.clone());
        model
    }

    /// A crate as a six-sided box model, as `_fr_draw_parm_cube` builds it.
    fn crate_model(&self, obj: &Obj) -> Option<Model> {
        let shape = ss1::objects::crate_shape(self.containers.as_deref()?, obj)?;
        let [rx, ry, rz] = shape.radius;
        let h = rz * 2.0;
        // The original's space: y down, so the box rises to -h.
        let p = |x: f32, y: f32, z: f32| [x, y, z];
        let c = [
            p(-rx, 0.0, -ry),
            p(-rx, 0.0, ry),
            p(rx, 0.0, ry),
            p(rx, 0.0, -ry),
            p(-rx, -h, -ry),
            p(-rx, -h, ry),
            p(rx, -h, ry),
            p(rx, -h, -ry),
        ];
        let key = |t: u8| {
            if t & 0x80 != 0 {
                LEVEL_TEXTURE_KEY + (t & 0x7f) as u16
            } else {
                CUSTOM_MATERIAL_KEY + (t & 0x7f) as u16
            }
        };
        let quad = |a: usize, b: usize, cc: usize, d: usize, t: u8| Poly {
            fill: Fill::Texture(key(t)),
            points: vec![c[a], c[b], c[cc], c[d]],
            uvs: vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        };
        Some(Model {
            polys: vec![
                quad(0, 3, 2, 1, shape.top),
                quad(4, 5, 6, 7, shape.top),
                quad(4, 7, 3, 0, shape.side),
                quad(7, 6, 2, 3, shape.side),
                quad(6, 5, 1, 2, shape.side),
                quad(5, 4, 0, 1, shape.side),
            ],
        })
    }

    /// A texture by key: a model's virtual texture (`citmat.res` 475+),
    /// a custom material (2180+), or one of the level's own textures.
    fn material(&mut self, vtext: u16) -> Option<Rc<dyn TextureTrait>> {
        if let Some(hit) = self.material_cache.get(&vtext) {
            return hit.clone();
        }
        let (res, id) = if vtext >= LEVEL_TEXTURE_KEY {
            let index = (vtext - LEVEL_TEXTURE_KEY) as usize;
            (
                self.textures.as_ref(),
                self.level_textures
                    .get(index)
                    .map(|&t| LEVEL_TEXTURE_BASE + t),
            )
        } else if vtext >= CUSTOM_MATERIAL_KEY {
            (
                self.materials.as_ref(),
                Some(CUSTOM_MATERIAL_BASE + vtext - CUSTOM_MATERIAL_KEY),
            )
        } else {
            (self.materials.as_ref(), Some(MATERIAL_BASE + vtext))
        };
        let texture = res
            .zip(id)
            .and_then(|(res, id)| res.load(id).ok())
            .and_then(|r| r.item(0).and_then(|b| Image::parse(b).ok()))
            .filter(|i| i.width > 0 && i.height > 0)
            .map(|image| {
                let big = ss1::scale::scale2x(&image);
                let texture: Rc<dyn TextureTrait> = Rc::new(engine::texture::init_from_memory2(
                    RawTextureData {
                        bytes: rgba_dilated(&big, &self.palette),
                        width: big.width as u32,
                        height: big.height as u32,
                        format: PixelFormat::RGBA,
                    },
                    &TextureOptions::default(),
                ));
                texture
            });
        self.material_cache.insert(vtext, texture.clone());
        texture
    }

    /// A model's polygons as one scene object per texture (plus one for the
    /// flat colours), in object space: the original's y points down and its
    /// z north, so both flip.
    fn model_objects(
        &mut self,
        model: &Model,
        lightmap: &Rc<Texture>,
        lm: Vector2<f32>,
    ) -> Vec<SceneObject> {
        let mut groups: HashMap<Option<u16>, Vec<VertexPositionTextureLightmapAtlasNormal>> =
            HashMap::new();
        let atlas = Vector4::new(0.0, 0.0, 1.0, 1.0);
        for poly in &model.polys {
            let points: Vec<Vector3<f32>> = poly
                .points
                .iter()
                .map(|p| vec3(p[0], -p[1], -p[2]) * TILE)
                .collect();
            let normal = (points[1] - points[0]).cross(points[2] - points[0]);
            let normal = if normal.magnitude2() > 0.0 {
                normal.normalize()
            } else {
                vec3(0.0, 1.0, 0.0)
            };
            let (key, uvs): (Option<u16>, Vec<Vector2<f32>>) = match poly.fill {
                Fill::Texture(t) if self.material(t).is_some() => (
                    Some(t),
                    poly.uvs.iter().map(|uv| vec2(uv[0], uv[1])).collect(),
                ),
                fill => {
                    let index = match fill {
                        Fill::Color(c) => c,
                        // Unknown per-instance colours: a neutral metal grey.
                        _ => DEFAULT_COLOR,
                    };
                    let uv = vec2(
                        ((index % 16) as f32 + 0.5) / 16.0,
                        ((index / 16) as f32 + 0.5) / 16.0,
                    );
                    (None, vec![uv; points.len()])
                }
            };
            let group = groups.entry(key).or_default();
            for i in 1..points.len() - 1 {
                for j in [0, i, i + 1] {
                    group.push(VertexPositionTextureLightmapAtlasNormal {
                        position: points[j],
                        uv: uvs[j],
                        lightmap_uv: lm,
                        lightmap_atlas: atlas,
                        normal,
                    });
                }
            }
        }
        let mut out = Vec::new();
        for (key, vertices) in groups {
            let texture = match key {
                Some(t) => self.material(t).unwrap(),
                None => self.swatches.clone(),
            };
            let material =
                LightmapMaterial::create(lightmap.clone(), texture, vec3(0.02, 0.02, 0.025));
            out.push(SceneObject::new(
                material,
                Box::new(engine::scene::mesh::create(vertices)),
            ));
        }
        out
    }
}

/// Texture keys past the model materials: custom crate materials and
/// level textures.
const CUSTOM_MATERIAL_KEY: u16 = 0x1000;
const LEVEL_TEXTURE_KEY: u16 = 0x2000;
/// `RES_customTextureMaps` in `citmat.res`.
const CUSTOM_MATERIAL_BASE: u16 = 2180;
/// 64x64 level textures in `texture.res`.
const LEVEL_TEXTURE_BASE: u16 = 707;

/// The palette index drawn for colours the original takes per instance.
const DEFAULT_COLOR: u8 = 217;

/// An object's heading, pitch and bank. The original composes them
/// `ORDER_YXZ` in a space whose z points north; ours points south, which
/// mirrors heading and pitch and leaves bank alone.
fn orientation(obj: &Obj) -> Matrix4<f32> {
    let angle = |v: u8| v as f32 / 256.0 * std::f32::consts::TAU;
    Matrix4::from_angle_y(Rad(-angle(obj.heading)))
        * Matrix4::from_angle_x(Rad(-angle(obj.pitch)))
        * Matrix4::from_angle_z(Rad(angle(obj.bank)))
}

/// One textured quad in the object's local space, lit at lightmap `lm`.
fn make_object(
    sprite: &Sprite,
    facing: &Facing,
    large: bool,
    obj: &Obj,
    lightmap: &Rc<Texture>,
    lm: Vector2<f32>,
) -> SceneObject {
    let (w, h) = (sprite.width, sprite.height);
    // Local quad edges (left, right, bottom, top), in world units.
    let (x0, x1, y0, y1) = match facing {
        Facing::Fixed => {
            // SVGA door art is double resolution (`HIGHRES_DOORS`).
            let s = if obj.class == ss1::objects::CLASS_DOOR && w >= 128.0 {
                FLAT_SCALE / 2.0
            } else {
                FLAT_SCALE
            };
            (-w / 2.0 * s, w / 2.0 * s, -h / 2.0 * s, h / 2.0 * s)
        }
        _ => {
            let s = SPRITE_SCALE * if large { 2.0 } else { 1.0 };
            let (ax, ay) = (sprite.anchor[0] as f32, sprite.anchor[1] as f32);
            if ax > 0.0 || ay > 0.0 {
                // The anchor pixel sits on the object's position.
                (-ax * s, (w - ax) * s, -(h - ay) * s, ay * s)
            } else {
                (-w / 2.0 * s, w / 2.0 * s, -h / 2.0 * s, h / 2.0 * s)
            }
        }
    };
    let atlas = Vector4::new(0.0, 0.0, 1.0, 1.0);
    let vertex = |x: f32, y: f32, u: f32, v: f32| VertexPositionTextureLightmapAtlasNormal {
        position: vec3(x, y, 0.0),
        uv: vec2(u, v),
        lightmap_uv: lm,
        lightmap_atlas: atlas,
        normal: vec3(0.0, 0.0, 1.0),
    };
    let (a, b, c, d) = (
        vertex(x0, y0, 0.0, 1.0),
        vertex(x1, y0, 1.0, 1.0),
        vertex(x1, y1, 1.0, 0.0),
        vertex(x0, y1, 0.0, 0.0),
    );
    let geometry = engine::scene::mesh::create(vec![a.clone(), b, c.clone(), a, c, d]);
    let mut material = LightmapMaterial::create(
        lightmap.clone(),
        sprite.texture.clone(),
        vec3(0.02, 0.02, 0.025),
    );
    material.set_render_pass(RenderPass {
        alpha_test: true,
        ..RenderPass::default()
    });
    SceneObject::new(material, Box::new(geometry))
}
