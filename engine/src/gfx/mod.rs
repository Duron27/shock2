//! The engine's graphics API: GL's state-machine semantics, recorded into
//! Vulkan.
//!
//! Materials compile a [`Program`] from GLSL, set its uniforms by name, bind
//! textures to units and set blend/depth/cull state; geometry uploads
//! [`Mesh`]es and draws them. Uniforms land in the program's uniform
//! blocks, and each draw resolves the current state into a cached pipeline
//! (see `vk_backend`).
//!
//! Everything here runs on the render thread only.

pub mod vk_backend;

// ---------------------------------------------------------------------------
// Programs and uniforms
// ---------------------------------------------------------------------------

/// A linked vertex + fragment program. The sources are the engine's
/// version-less GLSL; `vk::shader` adds the Vulkan header.
pub struct Program {
    id: u32,
}

// Programs live in per-material `OnceCell` statics; render thread only.
unsafe impl Send for Program {}
unsafe impl Sync for Program {}

/// A uniform's location in one program, resolved once by name. A name the
/// program does not use resolves to a uniform whose setters do nothing, as
/// GL's location -1 does.
#[derive(Clone, Copy, Debug)]
pub struct Uniform {
    vk: u32,
}

impl Program {
    pub fn new(vertex: &str, fragment: &str) -> Program {
        Program {
            id: vk_backend::create_program(vertex, fragment),
        }
    }

    pub fn uniform(&self, name: &str) -> Uniform {
        Uniform {
            vk: vk_backend::uniform(self.id, name),
        }
    }

    /// Make this the program the next uniform sets and draws use.
    pub fn bind(&self) {
        vk_backend::bind_program(self.id)
    }
}

/// Uniform values. Each sets `uniform` in the program bound last, like GL.
pub fn set_f32(uniform: Uniform, value: f32) {
    vk_backend::set_floats(uniform.vk, &[value], 1)
}
pub fn set_i32(uniform: Uniform, value: i32) {
    vk_backend::set_ints(uniform.vk, &[value])
}
pub fn set_ivec2(uniform: Uniform, value: [i32; 2]) {
    vk_backend::set_ivec(uniform.vk, &value, 2)
}
pub fn set_vec2(uniform: Uniform, value: [f32; 2]) {
    vk_backend::set_floats(uniform.vk, &value, 2)
}
pub fn set_vec3(uniform: Uniform, value: [f32; 3]) {
    vk_backend::set_floats(uniform.vk, &value, 3)
}
pub fn set_vec4(uniform: Uniform, value: [f32; 4]) {
    vk_backend::set_floats(uniform.vk, &value, 4)
}
pub fn set_mat3(uniform: Uniform, value: &cgmath::Matrix3<f32>) {
    vk_backend::set_matrices(uniform.vk, AsRef::<[f32; 9]>::as_ref(value), 3)
}
pub fn set_mat4(uniform: Uniform, value: &cgmath::Matrix4<f32>) {
    vk_backend::set_matrices(uniform.vk, AsRef::<[f32; 16]>::as_ref(value), 4)
}
pub fn set_f32_array(uniform: Uniform, values: &[f32]) {
    vk_backend::set_floats(uniform.vk, values, 1)
}
pub fn set_i32_array(uniform: Uniform, values: &[i32]) {
    vk_backend::set_ints(uniform.vk, values)
}
pub fn set_vec3_array(uniform: Uniform, values: &[[f32; 3]]) {
    vk_backend::set_floats(uniform.vk, values.as_flattened(), 3)
}
pub fn set_vec4_array(uniform: Uniform, values: &[[f32; 4]]) {
    vk_backend::set_floats(uniform.vk, values.as_flattened(), 4)
}
/// `mat3x4` elements: three columns of four floats each, column-major.
pub fn set_mat3x4_array(uniform: Uniform, values: &[f32]) {
    vk_backend::set_mat3x4(uniform.vk, values)
}
pub fn set_mat4_array(uniform: Uniform, values: &[cgmath::Matrix4<f32>]) {
    let flat: Vec<f32> = values
        .iter()
        .flat_map(|m| *AsRef::<[f32; 16]>::as_ref(m))
        .collect();
    vk_backend::set_matrices(uniform.vk, &flat, 4)
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

pub fn enable(cap: Cap) {
    vk_backend::set_cap(cap, true)
}
pub fn disable(cap: Cap) {
    vk_backend::set_cap(cap, false)
}
pub fn depth_mask(write: bool) {
    vk_backend::depth_mask(write)
}
pub fn depth_func(func: DepthFunc) {
    vk_backend::depth_func(func)
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
    vk_backend::blend_func_separate([src, dst, src_alpha, dst_alpha])
}
/// Cull back faces (the only face this engine ever culls).
pub fn cull_back_faces() {
    // Back-face culling is baked into each pipeline; nothing to record.
}
pub fn front_face(face: FrontFace) {
    vk_backend::front_face(face)
}
pub fn polygon_offset(factor: f32, units: f32) {
    vk_backend::polygon_offset(factor, units)
}
pub fn color_mask(r: bool, g: bool, b: bool, a: bool) {
    vk_backend::color_mask([r, g, b, a])
}
/// Scissor rect in framebuffer pixels, origin bottom-left (GL's convention).
pub fn scissor(x: i32, y: i32, width: i32, height: i32) {
    vk_backend::scissor([x, y, width, height])
}
pub fn viewport(x: i32, y: i32, width: i32, height: i32) {
    vk_backend::viewport([x, y, width, height])
}
pub fn clear_color(rgba: [f32; 4]) {
    vk_backend::clear_color(rgba)
}
pub fn clear(color: bool, depth: bool) {
    vk_backend::clear(color, depth)
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
    inner: vk_backend::VkMesh,
    vertex_count: u32,
    index_count: Option<u32>,
}

// GPU handles are only touched on the render thread, but live in statics
// (shared quads, meshes cached in OnceCells) the way GL ids did.
unsafe impl Send for Mesh {}
unsafe impl Sync for Mesh {}

/// The Xbox 360's GPU fetches no integer vertex formats (vulkan4xenon has
/// none) and its shaders hold integers as floats, so there an integer
/// attribute's 32-bit words go up as `f32`s and the layout calls it `Float`.
/// `None` when the layout has no integer attribute.
fn ints_as_floats(bytes: &[u8], layout: &VertexLayout) -> Option<(Vec<u8>, VertexLayout)> {
    if !layout
        .attributes
        .iter()
        .any(|a| a.kind == AttributeKind::Int)
    {
        return None;
    }
    let mut out = bytes.to_vec();
    let stride = layout.stride as usize;
    for a in layout
        .attributes
        .iter()
        .filter(|a| a.kind == AttributeKind::Int)
    {
        for vertex in out.chunks_exact_mut(stride) {
            for c in 0..a.components as usize {
                let at = a.offset as usize + c * 4;
                let word: [u8; 4] = vertex[at..at + 4].try_into().unwrap();
                let value = i32::from_ne_bytes(word) as f32;
                vertex[at..at + 4].copy_from_slice(&value.to_ne_bytes());
            }
        }
    }
    let mut layout = layout.clone();
    for a in &mut layout.attributes {
        if a.kind == AttributeKind::Int {
            a.kind = AttributeKind::Float;
        }
    }
    Some((out, layout))
}

/// The bytes and layout as this target's GPU fetches them.
fn for_target<'a>(
    bytes: &'a [u8],
    layout: &VertexLayout,
) -> (std::borrow::Cow<'a, [u8]>, VertexLayout) {
    if cfg!(target_os = "xbox360") {
        if let Some((bytes, layout)) = ints_as_floats(bytes, layout) {
            return (std::borrow::Cow::Owned(bytes), layout);
        }
    }
    (std::borrow::Cow::Borrowed(bytes), layout.clone())
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
        let (bytes, layout) = for_target(bytes, &layout);
        Mesh {
            inner: vk_backend::VkMesh::new(&bytes, &layout, indices),
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
        let (bytes, _) = for_target(bytes, &VertexLayout::of::<T>());
        self.inner.update_vertices(&bytes)
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
        self.inner.draw(topology, first, count)
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
    inner: vk_backend::VkTexture,
    width: u32,
    height: u32,
}

unsafe impl Send for Texture2D {}
unsafe impl Sync for Texture2D {}

impl Texture2D {
    /// Rows are top-first in `pixels`, as image files store them; v = 0 is
    /// the first row.
    pub fn new(
        width: u32,
        height: u32,
        format: PixelFormat,
        pixels: &[u8],
        sampling: Sampling,
    ) -> Texture2D {
        Texture2D {
            inner: vk_backend::VkTexture::new(width, height, format, pixels, sampling),
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
        self.inner
            .update_rgb_region(x, y, width, height, pixels, (self.width, self.height))
    }

    /// Bind to texture unit `unit`; a sampler uniform set to `unit` reads it.
    pub fn bind(&self, unit: u32) {
        self.inner.bind(unit)
    }
}

/// A cube map: six square RGBA8 faces (+X, -X, +Y, -Y, +Z, -Z), clamped and
/// mipmapped (reflections sample the blurred mips).
pub struct TextureCube {
    inner: vk_backend::VkCube,
}

unsafe impl Send for TextureCube {}
unsafe impl Sync for TextureCube {}

impl TextureCube {
    /// `faces` holds the six faces in turn, `size * size * 4` bytes each.
    pub fn new(size: u32, faces: &[u8]) -> TextureCube {
        assert_eq!(faces.len(), (size * size * 4 * 6) as usize);
        TextureCube {
            inner: vk_backend::VkCube::new(size, faces),
        }
    }

    /// Bind to unit `unit`'s cube slot (a unit holds a 2D and a cube texture).
    pub fn bind(&self, unit: u32) {
        self.inner.bind(unit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integer_attributes_go_up_as_floats() {
        let layout = VertexLayout {
            stride: 12,
            attributes: vec![
                Attribute {
                    kind: AttributeKind::Float,
                    components: 1,
                    offset: 0,
                },
                Attribute {
                    kind: AttributeKind::Int,
                    components: 2,
                    offset: 4,
                },
            ],
        };
        let mut bytes = Vec::new();
        for (f, a, b) in [(0.5f32, 3u32, 39u32), (2.0, 0, 79)] {
            bytes.extend(f.to_ne_bytes());
            bytes.extend(a.to_ne_bytes());
            bytes.extend(b.to_ne_bytes());
        }
        let (out, converted) = ints_as_floats(&bytes, &layout).unwrap();
        let words: Vec<f32> = out
            .chunks_exact(4)
            .map(|w| f32::from_ne_bytes(w.try_into().unwrap()))
            .collect();
        assert_eq!(words, [0.5, 3.0, 39.0, 2.0, 0.0, 79.0]);
        assert!(
            converted
                .attributes
                .iter()
                .all(|a| a.kind == AttributeKind::Float)
        );
        assert_eq!(converted.stride, 12);
    }

    #[test]
    fn a_float_layout_is_left_alone() {
        let layout = VertexLayout {
            stride: 4,
            attributes: vec![Attribute {
                kind: AttributeKind::Float,
                components: 1,
                offset: 0,
            }],
        };
        assert!(ints_as_floats(&[0; 8], &layout).is_none());
    }
}
