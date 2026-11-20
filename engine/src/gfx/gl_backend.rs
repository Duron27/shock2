//! OpenGL implementation of [`super`]: each call is the GL call it names.
//! It exists only while materials move onto `gfx`; it goes when OpenGL does.

use std::ffi::CString;
use std::os::raw::c_void;

use cgmath::Matrix;

use super::{
    AttributeKind, BlendFactor, Cap, DepthFunc, Filter, FrontFace, PixelFormat, Sampling, Topology,
    VertexLayout,
};

pub struct GlProgram {
    program: crate::shader_program::ShaderProgram,
}

impl GlProgram {
    pub fn new(vertex: &str, fragment: &str) -> GlProgram {
        let vs = crate::shader::build(vertex, crate::shader::ShaderType::Vertex, false);
        let fs = crate::shader::build(fragment, crate::shader::ShaderType::Fragment, false);
        GlProgram {
            program: crate::shader_program::link(&vs, &fs),
        }
    }

    pub fn uniform(&self, name: &str) -> i32 {
        let name = CString::new(name).unwrap();
        unsafe { gl::GetUniformLocation(self.program.gl_id, name.as_ptr()) }
    }

    pub fn bind(&self) {
        unsafe { gl::UseProgram(self.program.gl_id) }
    }

    pub fn id(&self) -> u32 {
        self.program.gl_id
    }
}

pub fn set_f32(loc: i32, v: f32) {
    unsafe { gl::Uniform1f(loc, v) }
}
pub fn set_i32(loc: i32, v: i32) {
    unsafe { gl::Uniform1i(loc, v) }
}
pub fn set_vec2(loc: i32, v: [f32; 2]) {
    unsafe { gl::Uniform2f(loc, v[0], v[1]) }
}
pub fn set_vec3(loc: i32, v: [f32; 3]) {
    unsafe { gl::Uniform3f(loc, v[0], v[1], v[2]) }
}
pub fn set_vec4(loc: i32, v: [f32; 4]) {
    unsafe { gl::Uniform4f(loc, v[0], v[1], v[2], v[3]) }
}
pub fn set_mat3(loc: i32, m: &cgmath::Matrix3<f32>) {
    unsafe { gl::UniformMatrix3fv(loc, 1, gl::FALSE, m.as_ptr()) }
}
pub fn set_mat4(loc: i32, m: &cgmath::Matrix4<f32>) {
    unsafe { gl::UniformMatrix4fv(loc, 1, gl::FALSE, m.as_ptr()) }
}
pub fn set_f32_array(loc: i32, v: &[f32]) {
    unsafe { gl::Uniform1fv(loc, v.len() as i32, v.as_ptr()) }
}
pub fn set_i32_array(loc: i32, v: &[i32]) {
    unsafe { gl::Uniform1iv(loc, v.len() as i32, v.as_ptr()) }
}
pub fn set_vec3_array(loc: i32, v: &[[f32; 3]]) {
    unsafe { gl::Uniform3fv(loc, v.len() as i32, v.as_ptr() as *const f32) }
}
pub fn set_vec4_array(loc: i32, v: &[[f32; 4]]) {
    unsafe { gl::Uniform4fv(loc, v.len() as i32, v.as_ptr() as *const f32) }
}
pub fn set_mat4_array(loc: i32, v: &[cgmath::Matrix4<f32>]) {
    unsafe { gl::UniformMatrix4fv(loc, v.len() as i32, gl::FALSE, v.as_ptr() as *const f32) }
}

fn cap(cap: Cap) -> gl::types::GLenum {
    match cap {
        Cap::DepthTest => gl::DEPTH_TEST,
        Cap::Blend => gl::BLEND,
        Cap::CullFace => gl::CULL_FACE,
        Cap::PolygonOffsetFill => gl::POLYGON_OFFSET_FILL,
        Cap::ScissorTest => gl::SCISSOR_TEST,
    }
}

pub fn set_cap(c: Cap, on: bool) {
    unsafe {
        if on {
            gl::Enable(cap(c))
        } else {
            gl::Disable(cap(c))
        }
    }
}

pub fn depth_mask(write: bool) {
    unsafe { gl::DepthMask(if write { gl::TRUE } else { gl::FALSE }) }
}

pub fn depth_func(func: DepthFunc) {
    unsafe {
        gl::DepthFunc(match func {
            DepthFunc::Less => gl::LESS,
            DepthFunc::LessEqual => gl::LEQUAL,
            DepthFunc::Always => gl::ALWAYS,
        })
    }
}

fn factor(f: BlendFactor) -> gl::types::GLenum {
    match f {
        BlendFactor::Zero => gl::ZERO,
        BlendFactor::One => gl::ONE,
        BlendFactor::SrcColor => gl::SRC_COLOR,
        BlendFactor::OneMinusSrcColor => gl::ONE_MINUS_SRC_COLOR,
        BlendFactor::DstColor => gl::DST_COLOR,
        BlendFactor::OneMinusDstColor => gl::ONE_MINUS_DST_COLOR,
        BlendFactor::SrcAlpha => gl::SRC_ALPHA,
        BlendFactor::OneMinusSrcAlpha => gl::ONE_MINUS_SRC_ALPHA,
        BlendFactor::DstAlpha => gl::DST_ALPHA,
        BlendFactor::OneMinusDstAlpha => gl::ONE_MINUS_DST_ALPHA,
    }
}

pub fn blend_func_separate(s: BlendFactor, d: BlendFactor, sa: BlendFactor, da: BlendFactor) {
    unsafe { gl::BlendFuncSeparate(factor(s), factor(d), factor(sa), factor(da)) }
}

pub fn cull_back() {
    unsafe { gl::CullFace(gl::BACK) }
}

pub fn front_face(face: FrontFace) {
    unsafe {
        gl::FrontFace(match face {
            FrontFace::Clockwise => gl::CW,
            FrontFace::CounterClockwise => gl::CCW,
        })
    }
}

pub fn polygon_offset(factor: f32, units: f32) {
    unsafe { gl::PolygonOffset(factor, units) }
}

pub fn color_mask(r: bool, g: bool, b: bool, a: bool) {
    let f = |x| if x { gl::TRUE } else { gl::FALSE };
    unsafe { gl::ColorMask(f(r), f(g), f(b), f(a)) }
}

pub fn scissor(x: i32, y: i32, w: i32, h: i32) {
    unsafe { gl::Scissor(x, y, w, h) }
}

pub fn viewport(x: i32, y: i32, w: i32, h: i32) {
    unsafe { gl::Viewport(x, y, w, h) }
}

pub fn clear_color(c: [f32; 4]) {
    unsafe { gl::ClearColor(c[0], c[1], c[2], c[3]) }
}

pub fn clear(color: bool, depth: bool) {
    let mut bits = 0;
    if color {
        bits |= gl::COLOR_BUFFER_BIT;
    }
    if depth {
        bits |= gl::DEPTH_BUFFER_BIT;
    }
    if bits != 0 {
        unsafe { gl::Clear(bits) }
    }
}

pub struct GlMesh {
    vao: u32,
    vbo: u32,
    ebo: u32,
}

impl GlMesh {
    pub fn new(bytes: &[u8], layout: &VertexLayout, indices: Option<&[u32]>) -> GlMesh {
        let (mut vao, mut vbo, mut ebo) = (0, 0, 0);
        unsafe {
            gl::GenVertexArrays(1, &mut vao);
            gl::GenBuffers(1, &mut vbo);
            gl::BindVertexArray(vao);
            gl::BindBuffer(gl::ARRAY_BUFFER, vbo);
            gl::BufferData(
                gl::ARRAY_BUFFER,
                bytes.len() as isize,
                if bytes.is_empty() {
                    std::ptr::null()
                } else {
                    bytes.as_ptr() as *const c_void
                },
                gl::STATIC_DRAW,
            );
            if let Some(indices) = indices {
                gl::GenBuffers(1, &mut ebo);
                gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, ebo);
                gl::BufferData(
                    gl::ELEMENT_ARRAY_BUFFER,
                    std::mem::size_of_val(indices) as isize,
                    if indices.is_empty() {
                        std::ptr::null()
                    } else {
                        indices.as_ptr() as *const c_void
                    },
                    gl::STATIC_DRAW,
                );
            }
            let stride = layout.stride as i32;
            for (i, a) in layout.attributes.iter().enumerate() {
                let i = i as u32;
                let offset = a.offset as usize as *const c_void;
                match a.kind {
                    AttributeKind::Float | AttributeKind::NormalizedFloat => {
                        let normalized = if a.kind == AttributeKind::NormalizedFloat {
                            gl::TRUE
                        } else {
                            gl::FALSE
                        };
                        gl::VertexAttribPointer(
                            i,
                            a.components as i32,
                            gl::FLOAT,
                            normalized,
                            stride,
                            offset,
                        )
                    }
                    AttributeKind::Int => {
                        gl::VertexAttribIPointer(i, a.components as i32, gl::INT, stride, offset)
                    }
                }
                gl::EnableVertexAttribArray(i);
            }
            gl::BindVertexArray(0);
            gl::BindBuffer(gl::ARRAY_BUFFER, 0);
            gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, 0);
        }
        GlMesh { vao, vbo, ebo }
    }

    pub fn update_vertices(&self, bytes: &[u8]) {
        unsafe {
            gl::BindBuffer(gl::ARRAY_BUFFER, self.vbo);
            gl::BufferSubData(
                gl::ARRAY_BUFFER,
                0,
                bytes.len() as isize,
                bytes.as_ptr() as *const c_void,
            );
            gl::BindBuffer(gl::ARRAY_BUFFER, 0);
        }
    }

    pub fn draw(&self, topology: Topology, first: u32, count: u32, indexed: bool) {
        let mode = match topology {
            Topology::Triangles => gl::TRIANGLES,
            Topology::Lines => gl::LINES,
        };
        unsafe {
            gl::BindVertexArray(self.vao);
            if indexed {
                gl::DrawElements(
                    mode,
                    count as i32,
                    gl::UNSIGNED_INT,
                    (first as usize * 4) as *const c_void,
                );
            } else {
                gl::DrawArrays(mode, first as i32, count as i32);
            }
            gl::BindVertexArray(0);
        }
    }
}

impl Drop for GlMesh {
    fn drop(&mut self) {
        unsafe {
            if self.ebo != 0 {
                gl::DeleteBuffers(1, &self.ebo);
            }
            gl::DeleteBuffers(1, &self.vbo);
            gl::DeleteVertexArrays(1, &self.vao);
        }
    }
}

pub struct GlTexture {
    id: u32,
}

impl GlTexture {
    pub fn new(
        width: u32,
        height: u32,
        format: PixelFormat,
        pixels: &[u8],
        sampling: Sampling,
    ) -> GlTexture {
        let mut id = 0;
        unsafe {
            gl::GenTextures(1, &mut id);
            gl::BindTexture(gl::TEXTURE_2D, id);
            let wrap = if sampling.wrap {
                gl::REPEAT
            } else {
                gl::CLAMP_TO_EDGE
            };
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_S, wrap as i32);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_T, wrap as i32);
            let (min, mag) = match sampling.filter {
                Filter::Linear => (gl::LINEAR, gl::LINEAR),
                Filter::Nearest => (gl::NEAREST, gl::NEAREST),
                Filter::LinearMipmap => (gl::LINEAR_MIPMAP_LINEAR, gl::LINEAR),
            };
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, min as i32);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, mag as i32);
            let pixel_format = match format {
                PixelFormat::Rgb8 => gl::RGB,
                PixelFormat::Rgba8 => gl::RGBA,
            };
            // RGB rows are not 4-byte aligned for odd widths.
            gl::PixelStorei(gl::UNPACK_ALIGNMENT, 1);
            gl::TexImage2D(
                gl::TEXTURE_2D,
                0,
                pixel_format as i32,
                width as i32,
                height as i32,
                0,
                pixel_format,
                gl::UNSIGNED_BYTE,
                pixels.as_ptr() as *const c_void,
            );
            gl::PixelStorei(gl::UNPACK_ALIGNMENT, 4);
            if sampling.filter == Filter::LinearMipmap {
                gl::GenerateMipmap(gl::TEXTURE_2D);
            }
        }
        GlTexture { id }
    }

    pub fn update_rgb_region(&self, x: u32, y: u32, w: u32, h: u32, pixels: &[u8]) {
        unsafe {
            gl::BindTexture(gl::TEXTURE_2D, self.id);
            gl::PixelStorei(gl::UNPACK_ALIGNMENT, 1);
            gl::TexSubImage2D(
                gl::TEXTURE_2D,
                0,
                x as i32,
                y as i32,
                w as i32,
                h as i32,
                gl::RGB,
                gl::UNSIGNED_BYTE,
                pixels.as_ptr() as *const c_void,
            );
            gl::PixelStorei(gl::UNPACK_ALIGNMENT, 4);
        }
    }

    pub fn bind(&self, unit: u32) {
        unsafe {
            gl::ActiveTexture(gl::TEXTURE0 + unit);
            gl::BindTexture(gl::TEXTURE_2D, self.id);
        }
    }

    pub fn id(&self) -> u32 {
        self.id
    }
}

impl Drop for GlTexture {
    fn drop(&mut self) {
        unsafe { gl::DeleteTextures(1, &self.id) }
    }
}

pub struct GlCube {
    id: u32,
}

impl GlCube {
    pub fn new(size: u32, faces: &[u8]) -> GlCube {
        let face_bytes = (size * size * 4) as usize;
        let mut id = 0;
        unsafe {
            // GLES 3 always filters across face edges; desktop GL must opt in,
            // or blurred mips show the cube's seams.
            #[cfg(not(target_os = "android"))]
            gl::Enable(gl::TEXTURE_CUBE_MAP_SEAMLESS);
            gl::GenTextures(1, &mut id);
            gl::BindTexture(gl::TEXTURE_CUBE_MAP, id);
            for (index, face) in faces.chunks_exact(face_bytes).enumerate() {
                gl::TexImage2D(
                    gl::TEXTURE_CUBE_MAP_POSITIVE_X + index as u32,
                    0,
                    gl::RGBA as i32,
                    size as i32,
                    size as i32,
                    0,
                    gl::RGBA,
                    gl::UNSIGNED_BYTE,
                    face.as_ptr() as *const std::ffi::c_void,
                );
            }
            let target = gl::TEXTURE_CUBE_MAP;
            gl::TexParameteri(target, gl::TEXTURE_WRAP_S, gl::CLAMP_TO_EDGE as i32);
            gl::TexParameteri(target, gl::TEXTURE_WRAP_T, gl::CLAMP_TO_EDGE as i32);
            gl::TexParameteri(target, gl::TEXTURE_WRAP_R, gl::CLAMP_TO_EDGE as i32);
            gl::TexParameteri(
                target,
                gl::TEXTURE_MIN_FILTER,
                gl::LINEAR_MIPMAP_LINEAR as i32,
            );
            gl::TexParameteri(target, gl::TEXTURE_MAG_FILTER, gl::LINEAR as i32);
            gl::GenerateMipmap(target);
        }
        GlCube { id }
    }

    pub fn bind(&self, unit: u32) {
        unsafe {
            gl::ActiveTexture(gl::TEXTURE0 + unit);
            gl::BindTexture(gl::TEXTURE_CUBE_MAP, self.id);
        }
    }
}

impl Drop for GlCube {
    fn drop(&mut self) {
        unsafe { gl::DeleteTextures(1, &self.id) };
    }
}

pub fn set_mat3x4_array(loc: i32, values: &[f32]) {
    unsafe {
        gl::UniformMatrix3x4fv(loc, (values.len() / 12) as i32, gl::FALSE, values.as_ptr());
    }
}
