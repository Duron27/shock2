//! Level objects drawn as the original drew them: camera-facing sprites
//! (`FAUBJ_BITMAP`), fixed flat bitmaps such as doors and signs
//! (`FAUBJ_TEXBITMAP`), and critters (`FAUBJ_CRIT`), whose sprite is the one
//! of their (up to) eight views that faces the camera. 3D models come later.
//!
//! Each object is lit by the baked lightmap texel of the floor under it, so
//! it sits in the same light as the room, and its art is Scale2x'd for
//! smoother edges.

use std::collections::HashMap;
use std::rc::Rc;

use cgmath::{Matrix4, Rad, Vector2, Vector3, Vector4, vec2, vec3};
use engine::materials::LightmapMaterial;
use engine::scene::render_pass::RenderPass;
use engine::scene::{SceneObject, VertexPositionTextureLightmapAtlasNormal};
use engine::texture::{Texture, TextureOptions, TextureTrait};
use engine::texture_format::{PixelFormat, RawTextureData};
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
}

pub struct Placed {
    /// One sprite, or one per critter view.
    views: Vec<Option<SceneObject>>,
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
    pub fn object(&self, camera_yaw: f32, eye: Vector3<f32>) -> Option<SceneObject> {
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
        let mut object = self.views.get(view)?.clone()?;
        let rotation = match self.facing {
            Facing::Fixed => self.rotation,
            _ => Matrix4::from_angle_y(Rad(-camera_yaw.to_radians())),
        };
        object.transform = Matrix4::from_translation(self.position) * rotation;
        Some(object)
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
    palette: Palette,
    cache: HashMap<ArtKey, Option<Sprite>>,
}

impl ObjectArt {
    pub fn new(
        props: ObjProps,
        art: Resource,
        critters: Option<ResFile>,
        palette: Palette,
    ) -> Self {
        ObjectArt {
            props,
            art,
            critters,
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
                RenderType::Bitmap => Facing::Billboard,
                RenderType::TexBitmap => Facing::Fixed,
                RenderType::Critter => Facing::Critter,
                _ => continue,
            };
            let Some(lm) = light_uv(obj.x, obj.y) else {
                continue;
            };
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
            let views: Vec<Option<SceneObject>> = keys
                .into_iter()
                .map(|key| {
                    let sprite = self.sprite(key)?;
                    Some(make_object(&sprite, &facing, large, obj, lightmap, lm))
                })
                .collect();
            if views.iter().all(Option::is_none) {
                continue;
            }
            out.push(Placed {
                views,
                position: vec3(
                    obj.x * TILE,
                    obj.height_tiles(z_shift) * TILE,
                    -obj.y * TILE,
                ),
                facing,
                rotation: orientation(obj),
                raw_heading: obj.heading,
                tile: (obj.x, obj.y),
            });
        }
        out
    }
}

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
