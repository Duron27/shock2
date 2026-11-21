//! Level objects drawn as the original drew them: camera-facing sprites
//! (`FAUBJ_BITMAP`) and fixed flat bitmaps such as doors and signs
//! (`FAUBJ_TEXBITMAP`). 3D models and critters come later.
//!
//! Each object is lit by the baked lightmap texel of the floor under it, so
//! it sits in the same light as the room, and its art is Scale2x'd for
//! smoother edges.

use std::collections::HashMap;
use std::rc::Rc;

use cgmath::{Matrix4, Rad, Vector3, Vector4, vec2, vec3};
use engine::materials::LightmapMaterial;
use engine::scene::render_pass::RenderPass;
use engine::scene::{SceneObject, VertexPositionTextureLightmapAtlasNormal};
use engine::texture::{TextureOptions, TextureTrait};
use engine::texture_format::{PixelFormat, RawTextureData};
use ss1::objects::{FLAG_LARGE, Obj, ObjProps, RenderType};
use ss1::{Image, Palette, Resource};

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
}

pub struct Placed {
    pub object: SceneObject,
    pub position: Vector3<f32>,
    pub facing: Facing,
    pub heading: f32,
}

impl Placed {
    /// The object's transform for a camera at `yaw` degrees.
    pub fn transform(&self, camera_yaw: f32) -> Matrix4<f32> {
        let angle = match self.facing {
            Facing::Billboard => -camera_yaw.to_radians(),
            Facing::Fixed => self.heading,
        };
        Matrix4::from_translation(self.position) * Matrix4::from_angle_y(Rad(angle))
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

pub struct ObjectArt {
    props: ObjProps,
    art: Resource,
    palette: Palette,
    cache: HashMap<usize, Option<(Rc<dyn TextureTrait>, usize, usize, [i16; 4], bool)>>,
}

impl ObjectArt {
    pub fn new(props: ObjProps, art: Resource, palette: Palette) -> Self {
        ObjectArt {
            props,
            art,
            palette,
            cache: HashMap::new(),
        }
    }

    /// Texture, original size, anchor and transparency of an art item.
    fn texture(
        &mut self,
        item: usize,
    ) -> Option<(Rc<dyn TextureTrait>, usize, usize, [i16; 4], bool)> {
        if let Some(hit) = self.cache.get(&item) {
            return hit.clone();
        }
        let made = self
            .art
            .item(item)
            .and_then(|bytes| Image::parse(bytes).ok())
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
                (
                    Rc::new(texture) as Rc<dyn TextureTrait>,
                    image.width,
                    image.height,
                    image.anchor,
                    image.is_transparent(),
                )
            });
        self.cache.insert(item, made.clone());
        made
    }

    /// Scene objects for every drawable object of a level.
    pub fn place(
        &mut self,
        objs: &[Obj],
        z_shift: u32,
        lightmap: &Rc<engine::texture::Texture>,
        light_uv: impl Fn(f32, f32) -> Option<cgmath::Vector2<f32>>,
    ) -> Vec<Placed> {
        let mut out = Vec::new();
        for obj in objs {
            let Some(triple) = obj.triple() else { continue };
            let prop = self.props.props[triple];
            let facing = match prop.render_type {
                RenderType::Bitmap => Facing::Billboard,
                RenderType::TexBitmap => Facing::Fixed,
                _ => continue,
            };
            let Some(item) = self.props.world_art(triple, obj.frame as usize) else {
                continue;
            };
            let Some((texture, w, h, anchor, _transparent)) = self.texture(item) else {
                continue;
            };
            let Some(lm) = light_uv(obj.x, obj.y) else {
                continue;
            };
            let (w, h) = (w as f32, h as f32);
            // Local quad corners (left, right, bottom, top), in world units.
            let (x0, x1, y0, y1) = match facing {
                Facing::Billboard => {
                    let s = SPRITE_SCALE
                        * if prop.flags & FLAG_LARGE != 0 {
                            2.0
                        } else {
                            1.0
                        };
                    let (ax, ay) = (anchor[0] as f32, anchor[1] as f32);
                    if ax > 0.0 || ay > 0.0 {
                        // The anchor pixel sits on the object's position.
                        (-ax * s, (w - ax) * s, -(h - ay) * s, ay * s)
                    } else {
                        (-w / 2.0 * s, w / 2.0 * s, -h / 2.0 * s, h / 2.0 * s)
                    }
                }
                Facing::Fixed => {
                    // SVGA door art is double resolution (`HIGHRES_DOORS`).
                    let s = if obj.class == ss1::objects::CLASS_DOOR && w >= 128.0 {
                        FLAT_SCALE / 2.0
                    } else {
                        FLAT_SCALE
                    };
                    (-w / 2.0 * s, w / 2.0 * s, -h / 2.0 * s, h / 2.0 * s)
                }
            };
            let atlas = Vector4::new(0.0, 0.0, 1.0, 1.0);
            let vertex =
                |x: f32, y: f32, u: f32, v: f32| VertexPositionTextureLightmapAtlasNormal {
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
            let mut material =
                LightmapMaterial::create(lightmap.clone(), texture, vec3(0.02, 0.02, 0.025));
            material.set_render_pass(RenderPass {
                alpha_test: true,
                ..RenderPass::default()
            });
            let position = vec3(
                obj.x * TILE,
                obj.height_tiles(z_shift) * TILE,
                -obj.y * TILE,
            );
            out.push(Placed {
                object: SceneObject::new(material, Box::new(geometry)),
                position,
                facing,
                heading: -(obj.heading as f32) / 256.0 * std::f32::consts::TAU,
            });
        }
        out
    }
}
