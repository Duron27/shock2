extern crate gl;

use crate::engine::EngineRenderContext;
use crate::gfx;
use crate::gfx::Program as ShaderProgram;
use crate::scene::Material;
use cgmath::Matrix4;
use once_cell::sync::OnceCell;
use std::any::Any;

// Debug material that visualizes normals as RGB colors for validation
const VERTEX_SHADER_SOURCE: &str = r#"
        layout (location = 0) in vec3 inPos;
        layout (location = 2) in vec3 inNormal;

        uniform mat4 world;
        uniform mat4 view;
        uniform mat4 projection;

        out vec3 worldNormal;

        void main() {
            mat3 normalMatrix = transpose(inverse(mat3(world)));
            worldNormal = normalize(normalMatrix * inNormal);
            gl_Position = projection * view * world * vec4(inPos, 1.0);
        }
"#;

const FRAGMENT_SHADER_SOURCE: &str = r#"
        out vec4 fragColor;

        in vec3 worldNormal;

        void main() {
            // Convert normal from [-1,1] to [0,1] range for RGB visualization
            // Red=X, Green=Y, Blue=Z components of normal vectors
            vec3 normalColor = worldNormal * 0.5 + 0.5;
            fragColor = vec4(normalColor, 1.0);
        }
"#;

static SHADER: OnceCell<ShaderProgram> = OnceCell::new();
static SKINNED_SHADER: OnceCell<ShaderProgram> = OnceCell::new();

pub struct DebugNormalMaterial {
    initialized: bool,
}

impl DebugNormalMaterial {
    pub fn new() -> DebugNormalMaterial {
        DebugNormalMaterial { initialized: false }
    }
}

impl Material for DebugNormalMaterial {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn has_initialized(&self) -> bool {
        self.initialized
    }

    fn initialize(&mut self, _is_opengl_es: bool) {
        SHADER.get_or_init(|| {
            let vertex_shader = VERTEX_SHADER_SOURCE;
            let fragment_shader = FRAGMENT_SHADER_SOURCE;
            gfx::Program::new(&vertex_shader, &fragment_shader)
        });
        self.initialized = true;
    }

    fn draw_opaque(
        &self,
        render_context: &EngineRenderContext,
        view_matrix: &Matrix4<f32>,
        world_matrix: &Matrix4<f32>,
        _skinning_data: &[Matrix4<f32>],
        _lights: &crate::scene::light::LightArray,
    ) -> bool {
        let shader = SHADER.get().expect("Shader should be initialized");

        shader.bind();

        // Set uniforms
        let world_loc = shader.uniform("world");
        let view_loc = shader.uniform("view");
        let projection_loc = shader.uniform("projection");

        gfx::set_mat4(world_loc, world_matrix);
        gfx::set_mat4(view_loc, &view_matrix);
        gfx::set_mat4(projection_loc, &render_context.projection_matrix);

        true
    }

    fn draw_transparent(
        &self,
        render_context: &EngineRenderContext,
        view_matrix: &Matrix4<f32>,
        world_matrix: &Matrix4<f32>,
        skinning_data: &[Matrix4<f32>],
        lights: &crate::scene::light::LightArray,
    ) -> bool {
        // Normal debug materials are always opaque
        self.draw_opaque(
            render_context,
            view_matrix,
            world_matrix,
            skinning_data,
            lights,
        )
    }
}

pub fn create() -> Box<dyn Material> {
    Box::new(DebugNormalMaterial::new())
}

const SKINNED_VERTEX_SHADER_SOURCE: &str = r#"
        layout (location = 0) in vec3 inPos;
        layout (location = 1) in vec2 inTex;
        layout (location = 2) in ivec4 bone_ids;
        layout (location = 3) in vec3 inNormal;

        uniform mat4 world;
        uniform mat4 view;
        uniform mat4 projection;
        // Matches the skinned material's palette (base joints + parent frames).
        uniform mat4 bone_matrices[80];

        out vec3 worldNormal;

        void main() {
            vec4 skinnedPos = bone_matrices[bone_ids.x] * vec4(inPos, 1.0);
            vec3 skinnedNormal = mat3(bone_matrices[bone_ids.x]) * inNormal;
            vec4 worldPosition = world * skinnedPos;
            worldNormal = normalize(mat3(world) * skinnedNormal);
            gl_Position = projection * view * worldPosition;
        }
"#;

const SKINNED_FRAGMENT_SHADER_SOURCE: &str = FRAGMENT_SHADER_SOURCE;

pub struct DebugNormalSkinnedMaterial {
    initialized: bool,
}

impl DebugNormalSkinnedMaterial {
    pub fn new() -> DebugNormalSkinnedMaterial {
        DebugNormalSkinnedMaterial { initialized: false }
    }
}

impl Material for DebugNormalSkinnedMaterial {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn has_initialized(&self) -> bool {
        self.initialized
    }

    fn initialize(&mut self, _is_opengl_es: bool) {
        SKINNED_SHADER.get_or_init(|| {
            let vertex_shader = SKINNED_VERTEX_SHADER_SOURCE;
            let fragment_shader = SKINNED_FRAGMENT_SHADER_SOURCE;
            gfx::Program::new(&vertex_shader, &fragment_shader)
        });
        self.initialized = true;
    }

    fn draw_opaque(
        &self,
        render_context: &EngineRenderContext,
        view_matrix: &Matrix4<f32>,
        world_matrix: &Matrix4<f32>,
        skinning_data: &[Matrix4<f32>],
        _lights: &crate::scene::light::LightArray,
    ) -> bool {
        let shader = SKINNED_SHADER.get().expect("Shader should be initialized");

        shader.bind();

        let world_loc = shader.uniform("world");
        let view_loc = shader.uniform("view");
        let projection_loc = shader.uniform("projection");

        gfx::set_mat4(world_loc, world_matrix);
        gfx::set_mat4(view_loc, &view_matrix);
        gfx::set_mat4(projection_loc, &render_context.projection_matrix);

        for i in 0..skinning_data.len() {
            let name = format!("bone_matrices[{i}]");
            let loc = shader.uniform(&name);
            gfx::set_mat4(loc, &skinning_data[i]);
        }

        true
    }

    fn draw_transparent(
        &self,
        render_context: &EngineRenderContext,
        view_matrix: &Matrix4<f32>,
        world_matrix: &Matrix4<f32>,
        skinning_data: &[Matrix4<f32>],
        lights: &crate::scene::light::LightArray,
    ) -> bool {
        self.draw_opaque(
            render_context,
            view_matrix,
            world_matrix,
            skinning_data,
            lights,
        )
    }
}

pub fn create_skinned() -> Box<dyn Material> {
    Box::new(DebugNormalSkinnedMaterial::new())
}
