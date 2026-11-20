//! Point-light shadows: cube depth maps for a few lamps, packed in one atlas.
//!
//! The world is lit by baked lightmaps, which already hold world-on-world
//! shadows. What they cannot hold is anything that moves: objects lit by the
//! mission's lamps are shaded without occlusion, and cast nothing on the
//! floor. Each frame the host names up to [`MAX_SHADOW_LIGHTS`] lamps
//! ([`Scene::point_shadows`](super::scene::Scene)); the renderer draws depth
//! from each into six 90-degree cube faces, twice:
//!
//! - set 0, world geometry: objects test against it, so a lamp behind a wall
//!   no longer lights them (the original ray-cast object lighting against
//!   the world the same way). It doesn't move, so a row is drawn once per
//!   lamp and kept across frames;
//! - set 1, moving casters only ([`SceneObject::casts_shadow`], creatures),
//!   redrawn every frame:
//!   the world tests against it and darkens its baked light where one blocks
//!   the lamp. World casters stay out, or the shadows the lightmap already
//!   holds would be darkened twice; static props stay out of both, as the
//!   original's lighting ignored them (a fixture would black out its room).
//!
//! [`SceneObject::casts_shadow`]: super::scene_object::SceneObject::casts_shadow
//!
//! Tiles are packed `[face][set * MAX_SHADOW_LIGHTS + light]` in one depth
//! image. The receivers rebuild a face's view and projection analytically
//! from [`FACES`], which the caster pass also uses, so the two agree by
//! construction (see the tests).

use std::cell::RefCell;

use cgmath::{InnerSpace, Matrix3, Matrix4, Vector3, Vector4};

use super::{RenderLayer, SceneObject, light::LightArray, scene::Scene};
use crate::{engine::EngineRenderContext, gfx, renderer::Renderer};

/// Lamps that cast shadows in one frame.
pub const MAX_SHADOW_LIGHTS: usize = 4;
/// Texels along a cube face's side.
pub const FACE_SIZE: u32 = 512;
/// Shadow-map near plane: a caster closer to its lamp than this (the lamp's
/// own fixture, mostly) casts nothing.
pub const NEAR: f32 = 0.5;
/// The farthest a shadow reaches; an authored lamp's range is clamped to it
/// to keep depth precision.
pub const MAX_FAR: f32 = 60.0;
const SETS: usize = 2;
/// Units 0-1 hold diffuse/pass textures, 2-5 shine maps, 6 the self-depth
/// mask and 7 the environment cube.
const TEXTURE_UNIT: u32 = 8;

/// A lamp that casts shadows this frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShadowLight {
    pub position: Vector3<f32>,
    /// Shadows end here; at most [`MAX_FAR`].
    pub range: f32,
    /// RGB colour and intensity, as the objects' light sets carry it.
    pub color_intensity: Vector4<f32>,
}

impl ShadowLight {
    fn far(&self) -> f32 {
        self.range.clamp(NEAR + 1.0, MAX_FAR)
    }
}

/// Cube faces as (forward, up); right is `forward x up`. Mirrored in
/// [`GLSL`]'s `pointShadowFace`.
pub const FACES: [([f32; 3], [f32; 3]); 6] = [
    ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
    ([-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
    ([0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
    ([0.0, -1.0, 0.0], [0.0, 0.0, -1.0]),
    ([0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
    ([0.0, 0.0, -1.0], [0.0, 1.0, 0.0]),
];

fn face_basis(face: usize) -> (Vector3<f32>, Vector3<f32>, Vector3<f32>) {
    let (f, u) = FACES[face];
    let forward = Vector3::from(f);
    let up = Vector3::from(u);
    (forward, up, forward.cross(up))
}

/// GL view matrix looking down `face` from `position`.
pub fn face_view(position: Vector3<f32>, face: usize) -> Matrix4<f32> {
    let (forward, up, right) = face_basis(face);
    // Rows are the camera axes; GL cameras look down -z.
    let rotation = Matrix3::from_cols(right, up, -forward);
    let rotation = Matrix4::from(cgmath::Matrix::transpose(&rotation));
    rotation * Matrix4::from_translation(-position)
}

/// GL projection for one 90-degree face.
pub fn face_projection(far: f32) -> Matrix4<f32> {
    cgmath::perspective(cgmath::Deg(90.0), 1.0, NEAR, far)
}

/// The atlas's size in texels.
pub fn atlas_size() -> [u32; 2] {
    [6 * FACE_SIZE, (SETS * MAX_SHADOW_LIGHTS) as u32 * FACE_SIZE]
}

/// A tile's GL viewport (bottom-left origin) for `face` of `light` in `set`.
/// Its top edge in the image is row `(set * MAX + light) * FACE_SIZE`,
/// which is where [`GLSL`] reads it.
fn tile_viewport(set: usize, light: usize, face: usize) -> [i32; 4] {
    let r = FACE_SIZE as i32;
    let row = (set * MAX_SHADOW_LIGHTS + light) as i32;
    let height = atlas_size()[1] as i32;
    [face as i32 * r, height - (row + 1) * r, r, r]
}

/// The atlas this frame's receivers sample, and the lamp in each row
/// (`None`: unused, or its world depth isn't drawn yet).
struct Frame {
    texture: u32,
    lights: [Option<ShadowLight>; MAX_SHADOW_LIGHTS],
    strength: f32,
}

/// Which lamp's world depth each set-0 row holds. World geometry doesn't
/// move, so a row is redrawn only when its lamp changes; creatures, which
/// do, are redrawn into set 1 every frame.
#[derive(Default)]
struct Cache {
    rows: [Option<ShadowLight>; MAX_SHADOW_LIGHTS],
    /// World casters the rows were drawn from; a change (a level load)
    /// invalidates every row.
    world_casters: usize,
}

thread_local! {
    static FRAME: RefCell<Option<Frame>> = const { RefCell::new(None) };
    static CACHE: RefCell<Cache> = RefCell::new(Cache::default());
}

/// Set-0 rows rebuilt per frame: each redraws the level six times, so a
/// changed lamp set fills in over a few frames instead of in one hitch.
const WORLD_ROWS_PER_FRAME: usize = 1;

/// Draws the frame's atlas before the scene, and withdraws it after, so a
/// receiver drawn outside `Renderer::render` never samples a stale image.
pub(crate) struct FrameGuard;

impl Drop for FrameGuard {
    fn drop(&mut self) {
        FRAME.with(|f| f.borrow_mut().take());
    }
}

fn same_lamp(a: &ShadowLight, b: &ShadowLight) -> bool {
    (a.position - b.position).magnitude2() < 1.0e-4 && (a.far() - b.far()).abs() < 1.0e-3
}

/// Give each lamp a row, keeping lamps in the rows that already hold their
/// world depth. Returns the lamp per row.
fn assign_rows(
    cached: &[Option<ShadowLight>; MAX_SHADOW_LIGHTS],
    lights: &[ShadowLight],
) -> [Option<ShadowLight>; MAX_SHADOW_LIGHTS] {
    let mut rows = [None; MAX_SHADOW_LIGHTS];
    let mut pending = Vec::new();
    for light in lights.iter().take(MAX_SHADOW_LIGHTS) {
        match cached
            .iter()
            .position(|c| c.as_ref().is_some_and(|c| same_lamp(c, light)))
        {
            Some(row) if rows[row].is_none() => rows[row] = Some(*light),
            _ => pending.push(*light),
        }
    }
    for light in pending {
        // Prefer a row whose cached lamp no current lamp needs.
        if let Some(row) = (0..MAX_SHADOW_LIGHTS).find(|&r| rows[r].is_none()) {
            rows[row] = Some(light);
        }
    }
    rows
}

/// Bring the atlas up to date for this frame's lamps. Returns a guard that
/// keeps it bound for receivers until the frame is drawn.
pub(crate) fn render_atlas(
    engine: &Renderer,
    context: &EngineRenderContext,
    scene: &Scene,
) -> Option<FrameGuard> {
    let lights: Vec<ShadowLight> = scene
        .point_shadows
        .iter()
        .copied()
        .filter(|l| l.color_intensity.w > 0.0)
        .take(MAX_SHADOW_LIGHTS)
        .collect();
    if lights.is_empty() {
        return None;
    }
    let casters: Vec<_> = scene
        .objects_in_layer(RenderLayer::World)
        .filter(|o| !o.has_projection_override())
        .collect();
    let world: Vec<_> = casters.iter().filter(|o| o.lights().is_none()).collect();
    let movers: Vec<_> = casters.iter().filter(|o| o.casts_shadow).collect();

    CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        if cache.world_casters != world.len() {
            *cache = Cache {
                world_casters: world.len(),
                ..Cache::default()
            };
        }
        let rows = assign_rows(&cache.rows, &lights);
        let keep = cache.rows.iter().any(Option::is_some);
        let kept = gfx::vk_backend::begin_shadow_capture(atlas_size(), keep)?;
        if !kept {
            cache.rows = [None; MAX_SHADOW_LIGHTS];
        }
        let context = EngineRenderContext {
            self_depth: super::self_depth::Phase::Capture,
            ..*context
        };
        let no_lights = LightArray::new();
        let size = FACE_SIZE as i32;
        let draw_row = |set: usize, row: usize, light: &ShadowLight, objects: &[&&SceneObject]| {
            let context = EngineRenderContext {
                projection_matrix: face_projection(light.far()),
                ..context
            };
            for face in 0..6 {
                let [x, y, w, h] = tile_viewport(set, row, face);
                gfx::viewport(x, y, w, h);
                let view = face_view(light.position, face);
                for object in objects {
                    object.draw_opaque(engine, &context, &view, &no_lights);
                }
            }
        };
        // Scissored clears: the capture keeps everything else.
        let clear_rect = |x: i32, y: i32, w: i32, h: i32| {
            gfx::enable(gfx::Cap::ScissorTest);
            gfx::scissor(x, y, w, h);
            gfx::clear(false, true);
            gfx::disable(gfx::Cap::ScissorTest);
        };

        // World depth: only rows whose lamp changed, a few per frame.
        let mut rebuilt = 0;
        for row in 0..MAX_SHADOW_LIGHTS {
            let Some(light) = rows[row] else { continue };
            if cache.rows[row].is_some_and(|c| same_lamp(&c, &light)) {
                continue;
            }
            cache.rows[row] = None;
            if rebuilt == WORLD_ROWS_PER_FRAME {
                continue;
            }
            rebuilt += 1;
            let [_, y, _, _] = tile_viewport(0, row, 0);
            clear_rect(0, y, 6 * size, size);
            draw_row(0, row, &light, &world);
            cache.rows[row] = Some(light);
        }

        // Moving casters: every frame, into set 1 (all rows cleared).
        let [_, set1_y, _, _] = tile_viewport(1, MAX_SHADOW_LIGHTS - 1, 0);
        clear_rect(0, set1_y, 6 * size, MAX_SHADOW_LIGHTS as i32 * size);
        for (row, light) in rows.iter().enumerate() {
            let Some(light) = light else { continue };
            let near: Vec<_> = movers
                .iter()
                .filter(|o| (o.get_world_position() - light.position).magnitude() < light.far() + 8.0)
                .copied()
                .collect();
            if !near.is_empty() {
                draw_row(1, row, light, &near);
            }
        }

        let texture = gfx::vk_backend::end_depth_capture()?;
        // A lamp shadows nothing until its world depth is drawn: half a
        // shadow map would leak light through walls.
        let mut ready = [None; MAX_SHADOW_LIGHTS];
        for row in 0..MAX_SHADOW_LIGHTS {
            ready[row] = rows[row].filter(|l| cache.rows[row].is_some_and(|c| same_lamp(&c, l)));
        }
        FRAME.with(|f| {
            *f.borrow_mut() = Some(Frame {
                texture,
                lights: ready,
                strength: scene.point_shadow_strength,
            })
        });
        Some(FrameGuard)
    })
}

/// Shared receiver code. `lightShadowSlot[i]` is the shadow light lit slot
/// `i` matches, or -1.
pub(crate) const GLSL: &str = r#"
uniform bool pointShadowEnabled;
uniform highp sampler2D pointShadowAtlas;
// xyz position, w far plane (0 = slot unused).
uniform highp vec4 pointShadowLights[4];
uniform vec4 pointShadowColor[4];
uniform int lightShadowSlot[6];
uniform float pointShadowStrength;
const highp float POINT_SHADOW_NEAR = 0.5;
const int POINT_SHADOW_FACE = 512;
const int POINT_SHADOW_MAX = 4;

// Mirrors engine::scene::point_shadow::FACES.
int pointShadowFace(highp vec3 d, out highp vec3 forward, out highp vec3 up) {
    highp vec3 a = abs(d);
    if (a.x >= a.y && a.x >= a.z) {
        up = vec3(0.0, 1.0, 0.0);
        if (d.x >= 0.0) { forward = vec3(1.0, 0.0, 0.0); return 0; }
        forward = vec3(-1.0, 0.0, 0.0); return 1;
    }
    if (a.y >= a.z) {
        if (d.y >= 0.0) { forward = vec3(0.0, 1.0, 0.0); up = vec3(0.0, 0.0, 1.0); return 2; }
        forward = vec3(0.0, -1.0, 0.0); up = vec3(0.0, 0.0, -1.0); return 3;
    }
    up = vec3(0.0, 1.0, 0.0);
    if (d.z >= 0.0) { forward = vec3(0.0, 0.0, 1.0); return 4; }
    forward = vec3(0.0, 0.0, -1.0); return 5;
}

// 1 lit .. 0 shadowed, from `light`'s cube in `set` (0 every caster,
// 1 objects only).
highp float pointShadowVisibility(int light, int set, highp vec3 worldPos, vec3 normal) {
    if (!pointShadowEnabled || light < 0) return 1.0;
    highp vec4 l = pointShadowLights[light];
    highp float far = l.w;
    if (far <= 0.0) return 1.0;
    highp vec3 d = worldPos - l.xyz;
    highp float dist = length(d);
    if (dist >= far) return 1.0;
    // Normal offset by about a texel's footprint, toward the lamp's side:
    // keeps a lit surface from shadowing itself (acne).
    vec3 n = dot(normal, d) > 0.0 ? -normal : normal;
    d += n * (3.0 * dist / float(POINT_SHADOW_FACE));
    highp vec3 forward;
    highp vec3 up;
    int face = pointShadowFace(d, forward, up);
    highp float m = dot(d, forward);
    if (m <= POINT_SHADOW_NEAR) return 1.0;
    highp vec3 right = cross(forward, up);
    highp vec2 uv = vec2(dot(d, right), dot(d, up)) / m * 0.5 + 0.5;
    highp float size = float(POINT_SHADOW_FACE);
    highp vec2 tile = vec2(float(face), float(set * POINT_SHADOW_MAX + light)) * size;
    highp vec2 texel = tile + vec2(uv.x, 1.0 - uv.y) * size;
    highp float n2 = POINT_SHADOW_NEAR;
    highp float bias = 0.03 + 0.004 * m;
    highp float lit = 0.0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            highp vec2 t = clamp(texel + vec2(float(dx), float(dy)), tile + 0.5, tile + size - 0.5);
            highp float z = texelFetch(pointShadowAtlas, ivec2(t), 0).r * 2.0 - 1.0;
            // The occluder's distance along the face axis.
            highp float occluder = 2.0 * far * n2 / ((far + n2) - z * (far - n2));
            lit += m - bias > occluder ? 0.0 : 1.0;
        }
    }
    return lit / 9.0;
}
"#;

pub(crate) struct Uniforms {
    enabled: gfx::Uniform,
    atlas: gfx::Uniform,
    lights: gfx::Uniform,
    colors: gfx::Uniform,
    slots: gfx::Uniform,
    strength: gfx::Uniform,
}

impl Uniforms {
    pub fn new(program: &gfx::Program) -> Self {
        Self {
            enabled: program.uniform("pointShadowEnabled"),
            atlas: program.uniform("pointShadowAtlas"),
            lights: program.uniform("pointShadowLights"),
            colors: program.uniform("pointShadowColor"),
            slots: program.uniform("lightShadowSlot"),
            strength: program.uniform("pointShadowStrength"),
        }
    }

    /// Bind the frame's atlas and say which of `lights`' slots it shadows.
    pub fn bind(&self, lights: &LightArray, context: &EngineRenderContext) {
        FRAME.with(|frame| {
            let frame = frame.borrow();
            let frame = frame
                .as_ref()
                .filter(|_| !matches!(context.self_depth, super::self_depth::Phase::Capture));
            gfx::set_i32(self.enabled, i32::from(frame.is_some()));
            // Disabled, the sampler points at the diffuse unit, never at an
            // image being rendered into.
            gfx::set_i32(
                self.atlas,
                if frame.is_some() {
                    TEXTURE_UNIT as i32
                } else {
                    0
                },
            );
            let mut slots = [-1i32; 6];
            let mut positions = [[0.0f32; 4]; MAX_SHADOW_LIGHTS];
            let mut colors = [[0.0f32; 4]; MAX_SHADOW_LIGHTS];
            if let Some(frame) = frame {
                gfx::vk_backend::bind_texture_id(TEXTURE_UNIT, Some(frame.texture));
                for (i, l) in frame.lights.iter().enumerate() {
                    if let Some(l) = l {
                        positions[i] = [l.position.x, l.position.y, l.position.z, l.far()];
                        colors[i] = l.color_intensity.into();
                    }
                }
                for (slot, out) in slots.iter_mut().enumerate() {
                    if let Some(light) = lights.get_light(slot) {
                        *out = shadow_index(&frame.lights, light.position());
                    }
                }
            }
            gfx::set_vec4_array(self.lights, &positions);
            gfx::set_vec4_array(self.colors, &colors);
            gfx::set_i32_array(self.slots, &slots);
            gfx::set_f32(self.strength, frame.map_or(0.0, |f| f.strength));
        });
    }
}

/// The shadow light at `position`: the object's light set and the shadow
/// list are built from the same authored lights, so positions match exactly.
fn shadow_index(lights: &[Option<ShadowLight>], position: Vector3<f32>) -> i32 {
    lights
        .iter()
        .position(|l| l.is_some_and(|l| (l.position - position).magnitude2() < 1.0e-4))
        .map_or(-1, |i| i as i32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cgmath::{Vector4, vec3, vec4};

    /// The receiver's analytic lookup (as GLSL computes it) for direction d.
    fn analytic(d: Vector3<f32>, far: f32) -> (usize, f32, f32, f32) {
        let a = vec3(d.x.abs(), d.y.abs(), d.z.abs());
        let face = if a.x >= a.y && a.x >= a.z {
            if d.x >= 0.0 { 0 } else { 1 }
        } else if a.y >= a.z {
            if d.y >= 0.0 { 2 } else { 3 }
        } else if d.z >= 0.0 {
            4
        } else {
            5
        };
        let (forward, up, right) = face_basis(face);
        let m = d.dot(forward);
        let ndc_z = (far + NEAR) / (far - NEAR) - 2.0 * far * NEAR / ((far - NEAR) * m);
        (face, d.dot(right) / m, d.dot(up) / m, ndc_z)
    }

    #[test]
    fn analytic_lookup_matches_the_caster_matrices() {
        let light = vec3(3.0, -2.0, 7.5);
        let far = 40.0;
        let projection = face_projection(far);
        for d in [
            vec3(5.0, 1.0, -2.0),
            vec3(-4.0, 0.5, 3.0),
            vec3(0.3, 9.0, -1.0),
            vec3(1.0, -6.0, 2.0),
            vec3(-1.0, 2.0, 8.0),
            vec3(2.0, -3.0, -12.0),
        ] {
            let (face, x, y, z) = analytic(d, far);
            let p = light + d;
            let clip: Vector4<f32> =
                projection * face_view(light, face) * vec4(p.x, p.y, p.z, 1.0);
            let ndc = clip.truncate() / clip.w;
            assert!((ndc.x - x).abs() < 1e-4, "face {face} x {} vs {x}", ndc.x);
            assert!((ndc.y - y).abs() < 1e-4, "face {face} y {} vs {y}", ndc.y);
            assert!((ndc.z - z).abs() < 1e-4, "face {face} z {} vs {z}", ndc.z);
            assert!(ndc.x.abs() <= 1.0 && ndc.y.abs() <= 1.0, "inside its face");
        }
    }

    fn lamp(x: f32) -> ShadowLight {
        ShadowLight {
            position: vec3(x, 0.0, 0.0),
            range: 20.0,
            color_intensity: vec4(1.0, 1.0, 1.0, 1.0),
        }
    }

    #[test]
    fn lamps_keep_the_rows_holding_their_world_depth() {
        let cached = [Some(lamp(1.0)), Some(lamp(2.0)), Some(lamp(3.0)), None];
        // Reordered by strength, one lamp swapped for a new one.
        let rows = assign_rows(&cached, &[lamp(3.0), lamp(9.0), lamp(1.0)]);
        assert_eq!(rows[0], Some(lamp(1.0)));
        assert_eq!(rows[2], Some(lamp(3.0)));
        // The new lamp takes a free row, not a kept one.
        assert!(rows[1] == Some(lamp(9.0)) || rows[3] == Some(lamp(9.0)));
        assert_eq!(rows.iter().flatten().count(), 3);
    }

    #[test]
    fn faces_are_proper_rotations() {
        for face in 0..6 {
            let (f, u, r) = face_basis(face);
            assert!((r.cross(u) - (-f)).magnitude() < 1e-6, "right-handed face {face}");
        }
    }

    #[test]
    fn tiles_tile_the_atlas() {
        let [w, h] = atlas_size();
        let mut seen = std::collections::HashSet::new();
        for set in 0..SETS {
            for light in 0..MAX_SHADOW_LIGHTS {
                for face in 0..6 {
                    let [x, y, tw, th] = tile_viewport(set, light, face);
                    assert!(x >= 0 && y >= 0 && x + tw <= w as i32 && y + th <= h as i32);
                    assert!(seen.insert((x, y)));
                }
            }
        }
    }

    #[test]
    fn glsl_constants_match() {
        assert!(GLSL.contains(&format!("POINT_SHADOW_FACE = {FACE_SIZE};")));
        assert!(GLSL.contains(&format!("POINT_SHADOW_MAX = {MAX_SHADOW_LIGHTS};")));
        assert!(GLSL.contains(&format!("POINT_SHADOW_NEAR = {NEAR:?};")));
        assert!(GLSL.contains(&format!("pointShadowLights[{MAX_SHADOW_LIGHTS}]")));
    }
}
