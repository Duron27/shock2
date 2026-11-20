//! The engine's graphics API: GL's state-machine semantics over either
//! OpenGL or Vulkan.
//!
//! Materials compile a [`Program`] from GLSL, set its uniforms by name, bind
//! textures to units and set blend/depth/cull state; geometry uploads
//! [`Mesh`]es and draws them. Under OpenGL each call is the GL call it names.
//! Under Vulkan the same calls are recorded: uniforms land in the program's
//! uniform blocks, and each draw resolves the current state into a cached
//! pipeline. Code written against this module renders identically on both,
//! which is what lets the renderer move to Vulkan one material at a time.
//!
//! Everything here runs on the render thread only.

mod gl_backend;
pub mod vk_backend;

use std::cell::Cell;

/// Which API the process renders with. Fixed at start-up, before any
/// resource is created.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    OpenGl,
    Vulkan,
}

thread_local! {
    static BACKEND: Cell<Backend> = const { Cell::new(Backend::OpenGl) };
}

pub fn backend() -> Backend {
    BACKEND.with(Cell::get)
}

pub fn is_vulkan() -> bool {
    backend() == Backend::Vulkan
}

/// Select the backend. Called once by a runtime before creating resources.
pub fn set_backend(backend: Backend) {
    BACKEND.with(|b| b.set(backend));
}

// ---------------------------------------------------------------------------
// Programs and uniforms
// ---------------------------------------------------------------------------

/// A linked vertex + fragment program. The sources are the engine's
/// version-less GLSL (as `shader::build` takes); each backend adds its own
/// header.
pub struct Program {
    inner: ProgramInner,
}

// Programs live in per-material `OnceCell` statics; render thread only.
unsafe impl Send for Program {}
unsafe impl Sync for Program {}

enum ProgramInner {
    Gl(gl_backend::GlProgram),
    Vk(u32),
}

/// A uniform's location in one program, resolved once by name. A name the
/// program does not use resolves to a uniform whose setters do nothing, as
/// GL's location -1 does.
#[derive(Clone, Copy, Debug)]
pub struct Uniform {
    gl: i32,
    vk: u32,
}

impl Program {
    pub fn new(vertex: &str, fragment: &str) -> Program {
        Program {
            inner: match backend() {
                Backend::OpenGl => ProgramInner::Gl(gl_backend::GlProgram::new(vertex, fragment)),
                Backend::Vulkan => ProgramInner::Vk(vk_backend::create_program(vertex, fragment)),
            },
        }
    }

    pub fn uniform(&self, name: &str) -> Uniform {
        match &self.inner {
            ProgramInner::Gl(p) => Uniform {
                gl: p.uniform(name),
                vk: vk_backend::NO_UNIFORM,
            },
            ProgramInner::Vk(p) => Uniform {
                gl: -1,
                vk: vk_backend::uniform(*p, name),
            },
        }
    }

    /// Make this the program the next uniform sets and draws use.
    pub fn bind(&self) {
        match &self.inner {
            ProgramInner::Gl(p) => p.bind(),
            ProgramInner::Vk(p) => vk_backend::bind_program(*p),
        }
    }

    /// The GL program object, for code not yet moved onto this module.
    pub fn gl_id(&self) -> u32 {
        match &self.inner {
            ProgramInner::Gl(p) => p.id(),
            ProgramInner::Vk(_) => 0,
        }
    }
}

/// Uniform values. Each sets `uniform` in the program bound last, like GL.
pub fn set_f32(uniform: Uniform, value: f32) {
    if is_vulkan() {
        vk_backend::set_floats(uniform.vk, &[value], 1)
    } else {
        gl_backend::set_f32(uniform.gl, value)
    }
}
pub fn set_i32(uniform: Uniform, value: i32) {
    if is_vulkan() {
        vk_backend::set_ints(uniform.vk, &[value])
    } else {
        gl_backend::set_i32(uniform.gl, value)
    }
}
pub fn set_vec2(uniform: Uniform, value: [f32; 2]) {
    if is_vulkan() {
        vk_backend::set_floats(uniform.vk, &value, 2)
    } else {
        gl_backend::set_vec2(uniform.gl, value)
    }
}
pub fn set_vec3(uniform: Uniform, value: [f32; 3]) {
    if is_vulkan() {
        vk_backend::set_floats(uniform.vk, &value, 3)
    } else {
        gl_backend::set_vec3(uniform.gl, value)
    }
}
pub fn set_vec4(uniform: Uniform, value: [f32; 4]) {
    if is_vulkan() {
        vk_backend::set_floats(uniform.vk, &value, 4)
    } else {
        gl_backend::set_vec4(uniform.gl, value)
    }
}
pub fn set_mat3(uniform: Uniform, value: &cgmath::Matrix3<f32>) {
    if is_vulkan() {
        vk_backend::set_matrices(uniform.vk, AsRef::<[f32; 9]>::as_ref(value), 3)
    } else {
        gl_backend::set_mat3(uniform.gl, value)
    }
}
pub fn set_mat4(uniform: Uniform, value: &cgmath::Matrix4<f32>) {
    if is_vulkan() {
        vk_backend::set_matrices(uniform.vk, AsRef::<[f32; 16]>::as_ref(value), 4)
    } else {
        gl_backend::set_mat4(uniform.gl, value)
    }
}
pub fn set_f32_array(uniform: Uniform, values: &[f32]) {
    if is_vulkan() {
        vk_backend::set_floats(uniform.vk, values, 1)
    } else {
        gl_backend::set_f32_array(uniform.gl, values)
    }
}
pub fn set_i32_array(uniform: Uniform, values: &[i32]) {
    if is_vulkan() {
        vk_backend::set_ints(uniform.vk, values)
    } else {
        gl_backend::set_i32_array(uniform.gl, values)
    }
}
pub fn set_vec3_array(uniform: Uniform, values: &[[f32; 3]]) {
    if is_vulkan() {
        vk_backend::set_floats(uniform.vk, values.as_flattened(), 3)
    } else {
        gl_backend::set_vec3_array(uniform.gl, values)
    }
}
pub fn set_vec4_array(uniform: Uniform, values: &[[f32; 4]]) {
    if is_vulkan() {
        vk_backend::set_floats(uniform.vk, values.as_flattened(), 4)
    } else {
        gl_backend::set_vec4_array(uniform.gl, values)
    }
}
pub fn set_mat4_array(uniform: Uniform, values: &[cgmath::Matrix4<f32>]) {
    if is_vulkan() {
        let flat: Vec<f32> = values
            .iter()
            .flat_map(|m| *AsRef::<[f32; 16]>::as_ref(m))
            .collect();
        vk_backend::set_matrices(uniform.vk, &flat, 4)
    } else {
        gl_backend::set_mat4_array(uniform.gl, values)
    }
}

// ---------------------------------------------------------------------------
// Fixed-function state
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Cap {
    DepthTest,
    Blend,
    CullFace,
    PolygonOffsetFill,
    ScissorTest,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DepthFunc {
    Less,
    LessEqual,
    Always,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BlendFactor {
    Zero,
    One,
    SrcColor,
    OneMinusSrcColor,
    DstColor,
    OneMinusDstColor,
    SrcAlpha,
    OneMinusSrcAlpha,
    DstAlpha,
    OneMinusDstAlpha,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FrontFace {
    Clockwise,
    CounterClockwise,
}

macro_rules! dispatch {
    ($vk:expr, $gl:expr) => {
        if is_vulkan() { $vk } else { $gl }
    };
}

pub fn enable(cap: Cap) {
    dispatch!(
        vk_backend::set_cap(cap, true),
        gl_backend::set_cap(cap, true)
    )
}
pub fn disable(cap: Cap) {
    dispatch!(
        vk_backend::set_cap(cap, false),
        gl_backend::set_cap(cap, false)
    )
}
pub fn depth_mask(write: bool) {
    dispatch!(vk_backend::depth_mask(write), gl_backend::depth_mask(write))
}
pub fn depth_func(func: DepthFunc) {
    dispatch!(vk_backend::depth_func(func), gl_backend::depth_func(func))
}
pub fn blend_func(src: BlendFactor, dst: BlendFactor) {
    blend_func_separate(src, dst, src, dst)
}
pub fn blend_func_separate(
    src: BlendFactor,
    dst: BlendFactor,
    src_alpha: BlendFactor,
    dst_alpha: BlendFactor,
) {
    dispatch!(
        vk_backend::blend_func_separate([src, dst, src_alpha, dst_alpha]),
        gl_backend::blend_func_separate(src, dst, src_alpha, dst_alpha)
    )
}
/// Cull back faces (the only face GL code here ever culls).
pub fn cull_back_faces() {
    // Vulkan bakes back-face culling into the pipeline.
    if !is_vulkan() {
        gl_backend::cull_back()
    }
}
pub fn front_face(face: FrontFace) {
    dispatch!(vk_backend::front_face(face), gl_backend::front_face(face))
}
pub fn polygon_offset(factor: f32, units: f32) {
    dispatch!(
        vk_backend::polygon_offset(factor, units),
        gl_backend::polygon_offset(factor, units)
    )
}
pub fn color_mask(r: bool, g: bool, b: bool, a: bool) {
    dispatch!(
        vk_backend::color_mask([r, g, b, a]),
        gl_backend::color_mask(r, g, b, a)
    )
}
/// Scissor rect in framebuffer pixels, origin bottom-left (GL's convention).
pub fn scissor(x: i32, y: i32, width: i32, height: i32) {
    dispatch!(
        vk_backend::scissor([x, y, width, height]),
        gl_backend::scissor(x, y, width, height)
    )
}
pub fn viewport(x: i32, y: i32, width: i32, height: i32) {
    dispatch!(
        vk_backend::viewport([x, y, width, height]),
        gl_backend::viewport(x, y, width, height)
    )
}
pub fn clear_color(rgba: [f32; 4]) {
    dispatch!(vk_backend::clear_color(rgba), gl_backend::clear_color(rgba))
}
pub fn clear(color: bool, depth: bool) {
    dispatch!(
        vk_backend::clear(color, depth),
        gl_backend::clear(color, depth)
    )
}

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AttributeKind {
    Float,
    NormalizedFloat,
    Int,
}

/// One vertex attribute; its index in [`VertexLayout::attributes`] is its
/// shader location.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Attribute {
    pub kind: AttributeKind,
    pub components: u32,
    pub offset: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct VertexLayout {
    pub stride: u32,
    pub attributes: Vec<Attribute>,
}

impl VertexLayout {
    pub fn of<T: crate::scene::Vertex>() -> VertexLayout {
        VertexLayout {
            stride: T::get_total_size() as u32,
            attributes: T::get_vertex_attributes()
                .iter()
                .map(|a| Attribute {
                    kind: match a.attribute_type {
                        crate::scene::VertexAttributeType::Float => AttributeKind::Float,
                        crate::scene::VertexAttributeType::NormalizedFloat => {
                            AttributeKind::NormalizedFloat
                        }
                        crate::scene::VertexAttributeType::Int => AttributeKind::Int,
                    },
                    components: a.size as u32,
                    offset: a.offset as u32,
                })
                .collect(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Topology {
    Triangles,
    Lines,
}

/// Vertices (and optionally `u32` indices) on the GPU, with their layout.
pub struct Mesh {
    inner: MeshInner,
    vertex_count: u32,
    index_count: Option<u32>,
}

// GPU handles are only touched on the render thread, but live in statics
// (shared quads, meshes cached in OnceCells) the way GL ids did.
unsafe impl Send for Mesh {}
unsafe impl Sync for Mesh {}

enum MeshInner {
    Gl(gl_backend::GlMesh),
    Vk(vk_backend::VkMesh),
}

impl Mesh {
    pub fn new<T: crate::scene::Vertex>(vertices: &[T], indices: Option<&[u32]>) -> Mesh {
        let layout = VertexLayout::of::<T>();
        let bytes = unsafe {
            std::slice::from_raw_parts(
                vertices.as_ptr() as *const u8,
                vertices.len() * layout.stride as usize,
            )
        };
        Mesh {
            inner: match backend() {
                Backend::OpenGl => MeshInner::Gl(gl_backend::GlMesh::new(bytes, &layout, indices)),
                Backend::Vulkan => MeshInner::Vk(vk_backend::VkMesh::new(bytes, &layout, indices)),
            },
            vertex_count: vertices.len() as u32,
            index_count: indices.map(|i| i.len() as u32),
        }
    }

    pub fn vertex_count(&self) -> u32 {
        self.vertex_count
    }

    /// Replace the first `vertices` of the buffer (same layout, no growth).
    pub fn update_vertices<T: crate::scene::Vertex>(&self, vertices: &[T]) {
        let bytes = unsafe {
            std::slice::from_raw_parts(
                vertices.as_ptr() as *const u8,
                std::mem::size_of_val(vertices),
            )
        };
        match &self.inner {
            MeshInner::Gl(m) => m.update_vertices(bytes),
            MeshInner::Vk(m) => m.update_vertices(bytes),
        }
    }

    /// Draw every vertex (or index) with the current program and state.
    pub fn draw(&self, topology: Topology) {
        let count = self.index_count.unwrap_or(self.vertex_count);
        self.draw_range(topology, 0, count);
    }

    /// Draw `count` vertices (or indices) from `first`.
    pub fn draw_range(&self, topology: Topology, first: u32, count: u32) {
        if count == 0 {
            return;
        }
        match &self.inner {
            MeshInner::Gl(m) => m.draw(topology, first, count, self.index_count.is_some()),
            MeshInner::Vk(m) => m.draw(topology, first, count),
        }
    }
}

// ---------------------------------------------------------------------------
// Textures
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PixelFormat {
    Rgb8,
    Rgba8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Filter {
    Nearest,
    Linear,
    /// Linear with a full mip chain, generated at upload.
    LinearMipmap,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Sampling {
    pub filter: Filter,
    /// Repeat outside 0..1; clamp to the edge otherwise.
    pub wrap: bool,
}

/// A 2D texture on the GPU.
pub struct Texture2D {
    inner: TextureInner,
    width: u32,
    height: u32,
}

unsafe impl Send for Texture2D {}
unsafe impl Sync for Texture2D {}

enum TextureInner {
    Gl(gl_backend::GlTexture),
    Vk(vk_backend::VkTexture),
}

impl Texture2D {
    /// Rows are top-first in `pixels`, as image files store them; v = 0 is
    /// the first row on both backends.
    pub fn new(
        width: u32,
        height: u32,
        format: PixelFormat,
        pixels: &[u8],
        sampling: Sampling,
    ) -> Texture2D {
        Texture2D {
            inner: match backend() {
                Backend::OpenGl => TextureInner::Gl(gl_backend::GlTexture::new(
                    width, height, format, pixels, sampling,
                )),
                Backend::Vulkan => TextureInner::Vk(vk_backend::VkTexture::new(
                    width, height, format, pixels, sampling,
                )),
            },
            width,
            height,
        }
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    /// Overwrite a region with tightly packed RGB8 pixels.
    pub fn update_rgb_region(&self, x: u32, y: u32, width: u32, height: u32, pixels: &[u8]) {
        match &self.inner {
            TextureInner::Gl(t) => t.update_rgb_region(x, y, width, height, pixels),
            TextureInner::Vk(t) => {
                t.update_rgb_region(x, y, width, height, pixels, (self.width, self.height))
            }
        }
    }

    /// Bind to texture unit `unit`; a sampler uniform set to `unit` reads it.
    pub fn bind(&self, unit: u32) {
        match &self.inner {
            TextureInner::Gl(t) => t.bind(unit),
            TextureInner::Vk(t) => t.bind(unit),
        }
    }

    /// The GL texture object, for code not yet moved onto this module.
    pub fn gl_id(&self) -> u32 {
        match &self.inner {
            TextureInner::Gl(t) => t.id(),
            TextureInner::Vk(_) => 0,
        }
    }
}
