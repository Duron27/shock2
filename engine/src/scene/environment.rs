//! The mission's captured environment cubemap: the direction mapping every
//! sampler shares, and a debug preview that paints it around the camera.
use crate::engine::EngineRenderContext;
use crate::gfx;
use crate::gfx::Program as ShaderProgram;
use crate::scene::Material;
use crate::texture::CubeTexture;

use cgmath::Matrix4;
use once_cell::sync::OnceCell;
use std::any::Any;
use std::rc::Rc;

/// Maps an engine direction into the 25AE capture's axes. Both are y-up; the
/// capture is turned a quarter about y and left-handed like a D3D/GL cube, so
/// text on the walls reads unmirrored. Matched against Earth's elevator doors,
/// which sit on engine +Z from the start and on the capture's -X face.
pub(crate) const GLSL: &str = r#"
vec3 environmentDirection(vec3 engineDirection) {
    return vec3(-engineDirection.z, engineDirection.y, -engineDirection.x);
}
"#;

const VERTEX_SHADER_SOURCE: &str = r#"
        layout (location = 0) in vec3 inPos;

        uniform mat4 world;
        uniform mat4 view;
        uniform mat4 projection;

        out vec3 worldPos;

        void main() {
            vec4 position = world * vec4(inPos, 1.0);
            worldPos = position.xyz;
            gl_Position = projection * view * position;
        }
"#;

const FRAGMENT_SHADER_SOURCE: &str = r#"
        out vec4 fragColor;

        in vec3 worldPos;

        uniform samplerCube environment;
        uniform vec3 environmentEye;

        void main() {
            vec3 direction = environmentDirection(normalize(worldPos - environmentEye));
            fragColor = vec4(textureLod(environment, direction, 0.0).rgb, 1.0);
        }
"#;

struct Uniforms {
    world: gfx::Uniform,
    view: gfx::Uniform,
    projection: gfx::Uniform,
    environment: gfx::Uniform,
    eye: gfx::Uniform,
}

static SHADER_PROGRAM: OnceCell<(ShaderProgram, Uniforms)> = OnceCell::new();

const ENVIRONMENT_UNIT: u32 = 0;

/// Draws the cubemap as seen from the eye, on whatever geometry surrounds it.
pub struct EnvironmentPreviewMaterial {
    has_initialized: bool,
    environment: Rc<CubeTexture>,
}

pub fn create_preview(environment: Rc<CubeTexture>) -> Box<dyn Material> {
    Box::new(EnvironmentPreviewMaterial {
        has_initialized: false,
        environment,
    })
}

impl Material for EnvironmentPreviewMaterial {
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
        let _ = SHADER_PROGRAM.get_or_init(|| {
            let vertex_shader = VERTEX_SHADER_SOURCE;
            let fragment_shader = &format!("{GLSL}\n{FRAGMENT_SHADER_SOURCE}");
            let shader = gfx::Program::new(&vertex_shader, &fragment_shader);
            let loc = |name: &str| shader.uniform(name);
            let uniforms = Uniforms {
                world: loc("world"),
                view: loc("view"),
                projection: loc("projection"),
                environment: loc("environment"),
                eye: loc("environmentEye"),
            };
            (shader, uniforms)
        });
        self.has_initialized = true;
    }

    fn draw_opaque(
        &self,
        render_context: &EngineRenderContext,
        view_matrix: &Matrix4<f32>,
        world_matrix: &Matrix4<f32>,
        _skinning_data: &[Matrix4<f32>],
        _lights: &crate::scene::light::LightArray,
    ) -> bool {
        let (program, uniforms) = SHADER_PROGRAM.get().expect("shader not compiled");
        program.bind();
        gfx::set_mat4(uniforms.world, &world_matrix);
        gfx::set_mat4(uniforms.view, &view_matrix);
        gfx::set_mat4(uniforms.projection, &render_context.projection_matrix);
        gfx::set_vec3(
            uniforms.eye,
            crate::scene::material::eye_position(view_matrix).into(),
        );
        gfx::set_i32(uniforms.environment, ENVIRONMENT_UNIT as i32);
        self.environment.bind_to(ENVIRONMENT_UNIT);

        true
    }

    fn draw_transparent(
        &self,
        _render_context: &EngineRenderContext,
        _view_matrix: &Matrix4<f32>,
        _world_matrix: &Matrix4<f32>,
        _skinning_data: &[Matrix4<f32>],
        _lights: &crate::scene::light::LightArray,
    ) -> bool {
        false
    }
}
