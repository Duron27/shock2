//! Display gamma (the Options menu's Brightness) for the flat runtimes.
//!
//! The frame renders into an offscreen target the size of the window, then one
//! full-screen triangle copies it to the window through `pow(c, 1/gamma)` -
//! the same curve the original game's gamma slider applied. At gamma 1.0 the
//! pass is skipped entirely and the frame draws straight to the window.

use std::cell::RefCell;

use c_string::*;
use crate::scene::render_target::RenderTarget;
use crate::shader::{self, ShaderType};
use crate::shader_program;

const VERTEX: &str = r#"
    out vec2 uv;
    void main() {
        // One triangle covering the screen: (-1,-1), (3,-1), (-1,3).
        vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
        uv = p;
        gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
    }
"#;

const FRAGMENT: &str = r#"
    in vec2 uv;
    out vec4 fragColor;
    uniform sampler2D frame;
    uniform float invGamma;
    void main() {
        vec3 c = texture(frame, uv).rgb;
        fragColor = vec4(pow(max(c, vec3(0.0)), vec3(invGamma)), 1.0);
    }
"#;

struct Program {
    /// Owns the GL program: dropping a `ShaderProgram` deletes it.
    shader: shader_program::ShaderProgram,
    frame: i32,
    inv_gamma: i32,
    vao: u32,
}

#[derive(Default)]
pub struct GammaPass {
    target: RefCell<Option<RenderTarget>>,
    program: RefCell<Option<Program>>,
    active: RefCell<Option<f32>>,
}

impl GammaPass {
    /// Start a frame. With `gamma` != 1 the frame is redirected offscreen;
    /// pair every call with [`GammaPass::end`]. Needs the GL context current.
    pub fn begin(&self, gamma: f32, size: [i32; 2]) {
        let size = [size[0].max(1), size[1].max(1)];
        if (gamma - 1.0).abs() < 1.0e-3 {
            *self.active.borrow_mut() = None;
            return;
        }
        let mut target = self.target.borrow_mut();
        if target.as_ref().is_none_or(|t| t.size() != size) {
            *target = Some(RenderTarget::new(size));
        }
        let target = target.as_ref().unwrap();
        target.bind();
        unsafe {
            gl::ClearColor(0.0, 0.0, 0.0, 1.0);
            gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);
        }
        *self.active.borrow_mut() = Some(gamma);
    }

    /// Finish the frame: copy the offscreen frame to the window with gamma.
    pub fn end(&self) {
        let Some(gamma) = self.active.borrow_mut().take() else {
            return;
        };
        let target = self.target.borrow();
        let Some(target) = target.as_ref() else {
            return;
        };
        target.unbind();
        let mut program = self.program.borrow_mut();
        let program = program.get_or_insert_with(build_program);
        unsafe {
            gl::Disable(gl::DEPTH_TEST);
            gl::Disable(gl::BLEND);
            gl::Disable(gl::CULL_FACE);
            gl::UseProgram(program.shader.gl_id);
            gl::ActiveTexture(gl::TEXTURE0);
            gl::BindTexture(gl::TEXTURE_2D, target.color_texture());
            gl::Uniform1i(program.frame, 0);
            gl::Uniform1f(program.inv_gamma, 1.0 / gamma.max(0.1));
            gl::BindVertexArray(program.vao);
            gl::DrawArrays(gl::TRIANGLES, 0, 3);
            gl::BindVertexArray(0);
            gl::BindTexture(gl::TEXTURE_2D, 0);
            gl::Enable(gl::DEPTH_TEST);
            gl::Enable(gl::BLEND);
        }
    }
}

fn build_program() -> Program {
    let vertex = shader::build(VERTEX, ShaderType::Vertex, false);
    let fragment = shader::build(FRAGMENT, ShaderType::Fragment, false);
    let linked = shader_program::link(&vertex, &fragment);
    unsafe {
        let mut vao = 0;
        // Core profile needs a bound VAO even for attribute-less draws.
        gl::GenVertexArrays(1, &mut vao);
        let frame = gl::GetUniformLocation(linked.gl_id, c_str!("frame").as_ptr());
        let inv_gamma = gl::GetUniformLocation(linked.gl_id, c_str!("invGamma").as_ptr());
        Program {
            shader: linked,
            frame,
            inv_gamma,
            vao,
        }
    }
}
