use std::any::Any;
use std::rc::Rc;

use crate::engine::EngineRenderContext;
use crate::gfx::{self, Program, Uniform};
use crate::scene::Material;

use crate::texture::TextureTrait;
use cgmath::Matrix4;
use cgmath::Vector4;

use once_cell::sync::OnceCell;

const VERTEX_SHADER_SOURCE: &str = r#"
        layout (location = 0) in vec2 inPos;
        layout (location = 1) in vec2 inTex;

        uniform mat4 world;
        uniform mat4 projection;
        out vec2 texCoord;

        void main() {
            texCoord = inTex;
            gl_Position = projection * world * vec4(inPos.xy, 0.0, 1.0);
        }
"#;

const FRAGMENT_SHADER_SOURCE: &str = r#"
        out vec4 fragColor;

        in vec2 texCoord;

        uniform vec4 textColor;
        uniform sampler2D fontTexture;

        void main()
        {    
            vec4 sampled = texture(fontTexture, texCoord);
            fragColor = textColor * sampled;
        }  
"#;

struct Uniforms {
    projection: Uniform,
    world: Uniform,
    text_color: Uniform,
}

static SHADER_PROGRAM: OnceCell<(Program, Uniforms)> = OnceCell::new();

pub struct ScreenSpaceMaterial {
    diffuse_texture: Rc<dyn TextureTrait>,
    text_color: Vector4<f32>,
}

impl ScreenSpaceMaterial {
    pub fn create(
        diffuse_texture: Rc<dyn TextureTrait>,
        text_color: Vector4<f32>,
    ) -> Box<dyn Material> {
        Box::new(ScreenSpaceMaterial {
            diffuse_texture,
            text_color,
        })
    }

    pub fn draw_common(
        &self,
        render_context: &EngineRenderContext,
        _view_matrix: &Matrix4<f32>,
        world_matrix: &Matrix4<f32>,
    ) {
        let (program, uniforms) = SHADER_PROGRAM.get().expect("shader not compiled");
        self.diffuse_texture.bind0(render_context);
        program.bind();
        // Pixels, origin top-left.
        let projection = cgmath::ortho(
            0.0,
            render_context.screen_size.x,
            render_context.screen_size.y,
            0.0,
            0.0,
            1.0,
        );
        gfx::set_mat4(uniforms.world, world_matrix);
        gfx::set_mat4(uniforms.projection, &projection);
        gfx::set_vec4(uniforms.text_color, self.text_color.into());
    }
}

impl Material for ScreenSpaceMaterial {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn has_initialized(&self) -> bool {
        false
    }

    fn initialize(&mut self, _is_opengl_es: bool) {
        let _ = SHADER_PROGRAM.get_or_init(|| {
            let program = Program::new(VERTEX_SHADER_SOURCE, FRAGMENT_SHADER_SOURCE);
            let uniforms = Uniforms {
                text_color: program.uniform("textColor"),
                projection: program.uniform("projection"),
                world: program.uniform("world"),
            };
            (program, uniforms)
        });
    }

    fn draw_opaque(
        &self,
        _render_context: &EngineRenderContext,
        _view_matrix: &Matrix4<f32>,
        _world_matrix: &Matrix4<f32>,
        _skinning_data: &[Matrix4<f32>],
        _lights: &crate::scene::light::LightArray,
    ) -> bool {
        // no-op
        false
    }

    fn draw_transparent(
        &self,
        render_context: &EngineRenderContext,
        view_matrix: &Matrix4<f32>,
        world_matrix: &Matrix4<f32>,
        _skinning_data: &[Matrix4<f32>],
        _lights: &crate::scene::light::LightArray,
    ) -> bool {
        self.draw_common(render_context, view_matrix, world_matrix);
        true
    }
}
