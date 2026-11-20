use std::any::Any;
use std::ops::Deref;

use crate::engine::EngineRenderContext;
use crate::gfx::{self, Program, Uniform};
use crate::scene::Material;
use crate::scene::light::LightArray;
use crate::texture::TextureTrait;
use cgmath::{Matrix4, SquareMatrix, ortho};
use once_cell::sync::OnceCell;

// Simple vertex shader - no normals needed for UI elements
const VERTEX_SHADER_SOURCE: &str = r#"
        layout (location = 0) in vec3 inPos;
        layout (location = 1) in vec2 inTex;

        uniform mat4 world;
        uniform mat4 view;
        uniform mat4 projection;

        out vec2 texCoord;

        void main() {
            texCoord = inTex;
            gl_Position = projection * view * world * vec4(inPos, 1.0);
        }
"#;

// Simple fragment shader with horizontal clipping and color key
const FRAGMENT_SHADER_SOURCE: &str = r#"
        out vec4 fragColor;

        in vec2 texCoord;

        uniform sampler2D texture1;
        uniform float clipX;  // 0.0 to 1.0 - anything past this X coord is clipped

        void main() {
            // Clip anything past the clipX threshold
            if (texCoord.x > clipX) {
                discard;
            }

            // Sample texture and output full-bright (emissive)
            vec4 texColor = texture(texture1, texCoord);
            if (texColor.a < 0.1) discard;  // Alpha test

            // Color key filtering: discard cyan-like colors (handle variations like r=0, g=246, b=255)
            // More forgiving tolerance for slight color variations in assets
            if (texColor.r < 0.02 && texColor.g > 0.9 && texColor.b > 0.95) {
                discard;
            }

            fragColor = texColor;  // Full-bright output
        }
"#;

struct ClippedScreenUniforms {
    world: Uniform,
    view: Uniform,
    projection: Uniform,
    clip_x: Uniform,
}

static CLIPPED_SCREEN_SHADER: OnceCell<(Program, ClippedScreenUniforms)> = OnceCell::new();

pub struct ClippedScreenMaterial<T>
where
    T: Deref<Target = dyn TextureTrait + 'static>,
{
    has_initialized: bool,
    diffuse_texture: T,
    clip_percentage: f32, // 0.0 to 1.0
    // When true, draw in screen space: build an orthographic projection from the
    // render context's screen size (pixels, origin top-left) and ignore the 3D
    // camera. The world matrix then positions the quad directly in pixels.
    screen_space: bool,
}

impl<T> ClippedScreenMaterial<T>
where
    T: Deref<Target = dyn TextureTrait>,
{
    pub fn set_clip_percentage(&mut self, percentage: f32) {
        self.clip_percentage = percentage.clamp(0.0, 1.0);
    }

    pub fn draw_unified(
        &self,
        render_context: &EngineRenderContext,
        view_matrix: &Matrix4<f32>,
        world_matrix: &Matrix4<f32>,
    ) {
        let (program, uniforms) = CLIPPED_SCREEN_SHADER
            .get()
            .expect("clipped screen shader not compiled");
        self.diffuse_texture.bind0(render_context);
        program.bind();

        // In screen-space mode, substitute an orthographic projection (pixels,
        // origin top-left) and an identity view so the world matrix places the
        // quad directly in screen pixels - matching ScreenSpaceMaterial.
        let (projection, view) = if self.screen_space {
            let ortho_projection = ortho(
                0.0,
                render_context.screen_size.x,
                render_context.screen_size.y,
                0.0,
                0.0,
                1.0,
            );
            (ortho_projection, Matrix4::identity())
        } else {
            (render_context.projection_matrix, *view_matrix)
        };
        gfx::set_mat4(uniforms.world, world_matrix);
        gfx::set_mat4(uniforms.view, &view);
        gfx::set_mat4(uniforms.projection, &projection);
        gfx::set_f32(uniforms.clip_x, self.clip_percentage);
    }
}

impl<T> Material for ClippedScreenMaterial<T>
where
    T: Deref<Target = dyn TextureTrait> + 'static,
{
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn has_initialized(&self) -> bool {
        self.has_initialized
    }

    fn initialize(&mut self, _is_opengl_es: bool) {
        let _ = CLIPPED_SCREEN_SHADER.get_or_init(|| {
            let program = Program::new(VERTEX_SHADER_SOURCE, FRAGMENT_SHADER_SOURCE);
            let uniforms = ClippedScreenUniforms {
                world: program.uniform("world"),
                view: program.uniform("view"),
                projection: program.uniform("projection"),
                clip_x: program.uniform("clipX"),
            };
            (program, uniforms)
        });

        self.has_initialized = true;
    }

    fn draw_opaque(
        &self,
        render_context: &EngineRenderContext,
        view_matrix: &Matrix4<f32>,
        world_matrix: &Matrix4<f32>,
        _skinning_data: &[Matrix4<f32>],
        _lights: &LightArray,
    ) -> bool {
        self.draw_unified(render_context, view_matrix, world_matrix);
        true
    }

    fn draw_transparent(
        &self,
        render_context: &EngineRenderContext,
        view_matrix: &Matrix4<f32>,
        world_matrix: &Matrix4<f32>,
        _skinning_data: &[Matrix4<f32>],
        _lights: &LightArray,
    ) -> bool {
        // Clipped screen materials can be rendered as transparent for proper blending
        self.draw_unified(render_context, view_matrix, world_matrix);
        true
    }
}

pub fn create<T>(diffuse_texture: T, clip_percentage: f32) -> Box<dyn Material>
where
    T: Deref<Target = dyn TextureTrait> + 'static,
{
    Box::new(ClippedScreenMaterial {
        diffuse_texture,
        has_initialized: false,
        clip_percentage: clip_percentage.clamp(0.0, 1.0),
        screen_space: false,
    })
}

/// Like [`create`], but draws in screen space (orthographic, pixel coordinates,
/// origin top-left) rather than against the 3D camera. Used for the flat HUD's
/// horizontally-filling bars.
pub fn create_screen_space<T>(diffuse_texture: T, clip_percentage: f32) -> Box<dyn Material>
where
    T: Deref<Target = dyn TextureTrait> + 'static,
{
    Box::new(ClippedScreenMaterial {
        diffuse_texture,
        has_initialized: false,
        clip_percentage: clip_percentage.clamp(0.0, 1.0),
        screen_space: true,
    })
}
