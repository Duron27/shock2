extern crate gl;
use std::any::Any;
use std::ops::Deref;

use crate::engine::EngineRenderContext;
use crate::gfx;
use crate::gfx::Program as ShaderProgram;
use crate::scene::Material;

use crate::texture::TextureTrait;
use cgmath::Matrix4;
use cgmath::Vector3;

use once_cell::sync::OnceCell;

const VERTEX_SHADER_SOURCE: &str = r#"
        layout (location = 0) in vec3 inPos;
        layout (location = 1) in vec2 inTex;

        uniform mat4 world;
        uniform mat4 view;
        uniform mat4 projection;
        uniform float scale;
        out vec2 texCoord;

        void main() {
            texCoord = inTex;
            vec3 v_right = vec3(view[0].x, view[1].x, view[2].x);
            vec3 v_up = vec3(view[0].y, view[1].y, view[2].y);          
            vec3 billboard_center = waveWorldPosition(world[3]).xyz;

            vec3 adjusted_pos = billboard_center + inPos.x * v_right * scale + inPos.y * v_up * scale;
            gl_Position = projection * view * vec4(adjusted_pos, 1.0);
        }
"#;

const FRAGMENT_SHADER_SOURCE: &str = r#"
        out vec4 fragColor;

        in vec2 texCoord;

        uniform vec3 inColor;
        // texture sampler
        uniform sampler2D texture1;

        uniform float emissivity;
        uniform float transparency;

        void main() {
            vec4 texColor = texture(texture1, texCoord);
            if (texColor.a < 0.1) discard;
            fragColor = texColor * vec4(inColor, 1.0);
            // Emissive glow is tinted by inColor so it matches the particle color
            // (callers that want an untinted sprite pass inColor = white).
            fragColor.rgb += texColor.rgb * inColor * emissivity;
            fragColor.a *= 1.0 - transparency;
        }
"#;

struct Uniforms {
    world_wave: crate::scene::world_wave::Uniforms,
    world_loc: gfx::Uniform,
    view_loc: gfx::Uniform,
    projection_loc: gfx::Uniform,
    in_color_loc: gfx::Uniform,
    emissivity_loc: gfx::Uniform,
    transparency_loc: gfx::Uniform,
    scale_loc: gfx::Uniform,
}

static SHADER_PROGRAM: OnceCell<(ShaderProgram, Uniforms)> = OnceCell::new();

pub struct BillboardMaterial<T>
where
    T: Deref<Target = dyn TextureTrait> + 'static,
{
    has_initialized: bool,
    diffuse_texture: T,
    color: Vector3<f32>,
    emissivity: f32,
    transparency: f32,
    scale: f32,
}

impl<T> BillboardMaterial<T>
where
    T: Deref<Target = dyn TextureTrait> + 'static,
{
    pub fn create(
        diffuse_texture: T,
        color: Vector3<f32>,
        emissivity: f32,
        transparency: f32,
        scale: f32,
    ) -> Box<dyn Material> {
        Box::new(BillboardMaterial {
            diffuse_texture,
            has_initialized: false,
            color,
            emissivity,
            transparency,
            scale,
        })
    }

    pub fn is_transparent(&self) -> bool {
        self.transparency > 0.01
    }

    pub fn draw_common(
        &self,
        render_context: &EngineRenderContext,
        view_matrix: &Matrix4<f32>,
        world_matrix: &Matrix4<f32>,
    ) {
        let (shader_program, uniforms) = SHADER_PROGRAM.get().expect("shader not compiled");
        self.diffuse_texture.bind0(render_context);
        shader_program.bind();
        uniforms.world_wave.bind(render_context.world_wave);

        let projection = render_context.projection_matrix;

        gfx::set_mat4(uniforms.world_loc, &world_matrix);
        gfx::set_mat4(uniforms.view_loc, &view_matrix);
        gfx::set_mat4(uniforms.projection_loc, &projection);
        gfx::set_vec3(
            uniforms.in_color_loc,
            [self.color.x, self.color.y, self.color.z],
        );
        gfx::set_f32(uniforms.transparency_loc, self.transparency);
        gfx::set_f32(uniforms.emissivity_loc, self.emissivity);
        gfx::set_f32(uniforms.scale_loc, self.scale);
    }
}
impl<T> Material for BillboardMaterial<T>
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
        let _ = SHADER_PROGRAM.get_or_init(|| {
            // build and compile our shader program
            // ------------------------------------
            // vertex shader
            let vertex_shader = &[crate::scene::world_wave::GLSL, VERTEX_SHADER_SOURCE].concat();

            // fragment shader
            let fragment_shader = FRAGMENT_SHADER_SOURCE;
            // link shaders
            let shader = gfx::Program::new(&vertex_shader, &fragment_shader);

            let uniforms = Uniforms {
                world_wave: crate::scene::world_wave::Uniforms::new(&shader),
                world_loc: shader.uniform("world"),
                view_loc: shader.uniform("view"),
                in_color_loc: shader.uniform("inColor"),
                emissivity_loc: shader.uniform("emissivity"),
                transparency_loc: shader.uniform("transparency"),
                projection_loc: shader.uniform("projection"),
                scale_loc: shader.uniform("scale"),
            };
            (shader, uniforms)
        });

        // self.diffuse_texture_descriptor
        //     .borrow_mut()
        //     .initialize(storage);
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
        if !self.is_transparent() {
            self.draw_common(render_context, view_matrix, world_matrix);
            true
        } else {
            false
        }
    }

    fn draw_transparent(
        &self,
        render_context: &EngineRenderContext,
        view_matrix: &Matrix4<f32>,
        world_matrix: &Matrix4<f32>,
        _skinning_data: &[Matrix4<f32>],
        _lights: &crate::scene::light::LightArray,
    ) -> bool {
        if self.is_transparent() {
            self.draw_common(render_context, view_matrix, world_matrix);
            true
        } else {
            false
        }
    }
}
