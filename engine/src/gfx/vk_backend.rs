//! The Vulkan side of [`crate::gfx`]: GL's state machine, recorded.
//!
//! Fixed-function state is tracked the way GL tracks it; each draw resolves
//! it, with the bound program and the mesh's vertex layout, to a pipeline
//! cached by that combination. Uniform values live in each program's two
//! staging blocks (vertex, fragment) and are copied into a per-frame
//! uniform ring at every draw, which then gets a descriptor set from a
//! per-frame pool. Objects dropped while frames may still read them are
//! destroyed once those frames finish.
//!
//! Everything lives in one thread-local [`State`], installed by
//! [`crate::vk::VkRenderer`] for its device and torn down before the device.

#[cfg(not(target_os = "xbox360"))]
use std::cell::RefCell;
use std::collections::BTreeMap;
// Fx, not SipHash: these maps are read several times a draw (images,
// samplers, descriptor sets), which SipHash made about 3 ms a frame on the
// Xbox 360.
use rustc_hash::FxHashMap as HashMap;

use ash::vk;
use tracing::info;

use super::{
    BlendFactor, Cap, DepthFunc, Filter, FrontFace, PixelFormat, Sampling, Topology, VertexLayout,
};
use crate::vk::shader::{self, Member};

const FRAMES: usize = 2;
/// Sub-allocated, so on the Xbox 360 (where vulkan4xenon rounds every
/// allocation up to a 64 KiB page) small buffers still share pages. The
/// console's 512 MB is shared with the CPU, so its blocks are smaller: each
/// pool holds at most one block's slack.
const BLOCK_SIZE: u64 = if cfg!(target_os = "xbox360") {
    16 << 20
} else {
    64 << 20
};
const UBO_CHUNK: u64 = 4 << 20;
const POOL_SETS: u32 = 1024;
const MAX_UNITS: usize = 16;

// ---------------------------------------------------------------------------
// Memory: blocks sub-allocated first-fit (drivers cap allocations at ~4096)
// ---------------------------------------------------------------------------

struct Block {
    memory: vk::DeviceMemory,
    free: Vec<(u64, u64)>,
    mapped: *mut u8,
}

struct Pool {
    type_index: u32,
    linear: bool,
    blocks: Vec<Block>,
}

#[derive(Clone, Copy)]
struct Allocation {
    pool: usize,
    block: usize,
    offset: u64,
    size: u64,
    memory: vk::DeviceMemory,
    /// This allocation's own mapped bytes (null in device-local memory).
    mapped: *mut u8,
}

#[derive(Default)]
struct Allocator {
    pools: Vec<Pool>,
    /// Bytes handed out and not freed, for the log when a block is added.
    used: u64,
}

impl Allocator {
    fn alloc(
        &mut self,
        dev: &Dev,
        req: vk::MemoryRequirements,
        flags: vk::MemoryPropertyFlags,
        linear: bool,
    ) -> Allocation {
        let type_index = dev
            .memory_type(req.memory_type_bits, flags)
            .expect("no suitable Vulkan memory type");
        let pool = match self
            .pools
            .iter()
            .position(|p| p.type_index == type_index && p.linear == linear)
        {
            Some(i) => i,
            None => {
                self.pools.push(Pool {
                    type_index,
                    linear,
                    blocks: Vec::new(),
                });
                self.pools.len() - 1
            }
        };
        let align = req.alignment.max(1);
        self.used += req.size;
        let p = &mut self.pools[pool];
        for (b, block) in p.blocks.iter_mut().enumerate() {
            if let Some(offset) = take(&mut block.free, req.size, align) {
                return Allocation {
                    pool,
                    block: b,
                    offset,
                    size: req.size,
                    memory: block.memory,
                    mapped: at(block.mapped, offset),
                };
            }
        }
        let size = BLOCK_SIZE.max(req.size.next_multiple_of(1 << 20));
        tracing::debug!(
            "GPU memory: allocating {} MiB of memory type {type_index}",
            size >> 20
        );
        let memory = unsafe {
            dev.device.allocate_memory(
                &vk::MemoryAllocateInfo::builder()
                    .allocation_size(size)
                    .memory_type_index(type_index),
                None,
            )
        }
        .expect("vkAllocateMemory");
        let mapped = if flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE) {
            unsafe {
                dev.device
                    .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
            }
            .expect("vkMapMemory") as *mut u8
        } else {
            std::ptr::null_mut()
        };
        let mut block = Block {
            memory,
            free: vec![(0, size)],
            mapped,
        };
        let offset = take(&mut block.free, req.size, align).unwrap();
        p.blocks.push(block);
        let blocks: u64 = self.pools.iter().flat_map(|p| &p.blocks).count() as u64;
        info!(
            "GPU memory: block {blocks} ({} MiB, memory type {type_index}); {} MiB in use",
            size >> 20,
            self.used >> 20
        );
        let p = &mut self.pools[pool];
        Allocation {
            pool,
            block: p.blocks.len() - 1,
            offset,
            size: req.size,
            memory,
            mapped: at(mapped, offset),
        }
    }

    fn free(&mut self, a: Allocation) {
        self.used -= a.size;
        let free = &mut self.pools[a.pool].blocks[a.block].free;
        free.push((a.offset, a.size));
        free.sort_unstable();
        let mut merged: Vec<(u64, u64)> = Vec::with_capacity(free.len());
        for &(o, s) in free.iter() {
            match merged.last_mut() {
                Some(last) if last.0 + last.1 == o => last.1 += s,
                _ => merged.push((o, s)),
            }
        }
        *free = merged;
    }

    fn destroy(&mut self, device: &ash::Device) {
        for pool in self.pools.drain(..) {
            for block in pool.blocks {
                unsafe { device.free_memory(block.memory, None) };
            }
        }
    }
}

/// The mapped address `offset` bytes into a block (null when unmapped).
fn at(base: *mut u8, offset: u64) -> *mut u8 {
    if base.is_null() {
        base
    } else {
        unsafe { base.add(offset as usize) }
    }
}

/// First fit of `size` at `align` in a sorted free list.
fn take(free: &mut Vec<(u64, u64)>, size: u64, align: u64) -> Option<u64> {
    let i = free.iter().position(|&(o, s)| {
        let a = o.next_multiple_of(align);
        a + size <= o + s
    })?;
    let (o, s) = free.remove(i);
    let a = o.next_multiple_of(align);
    let mut at = i;
    if a > o {
        free.insert(at, (o, a - o));
        at += 1;
    }
    if a + size < o + s {
        free.insert(at, (a + size, o + s - a - size));
    }
    Some(a)
}

// ---------------------------------------------------------------------------
// Device and objects
// ---------------------------------------------------------------------------

struct Dev {
    device: ash::Device,
    queue: vk::Queue,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    ubo_align: u64,
    /// Anisotropy for mipmapped samplers, or `None` when the device lacks it.
    anisotropy: Option<f32>,
}

impl Dev {
    fn memory_type(&self, bits: u32, flags: vk::MemoryPropertyFlags) -> Option<u32> {
        (0..self.memory_properties.memory_type_count).find(|&i| {
            bits & (1 << i) != 0
                && self.memory_properties.memory_types[i as usize]
                    .property_flags
                    .contains(flags)
        })
    }
}

struct BufferObj {
    buffer: vk::Buffer,
    alloc: Allocation,
    size: u64,
    /// In host-visible memory (written in place) rather than device-local.
    host: bool,
}

struct ImageObj {
    image: vk::Image,
    view: vk::ImageView,
    alloc: Allocation,
    sampling: Sampling,
    mips: u32,
    /// 1, or 6 for a cube.
    layers: u32,
    size: (u32, u32),
}

#[derive(Default, Clone)]
struct Slot {
    /// (block 0 = vertex / 1 = fragment, layout) for each stage using it.
    members: Vec<(usize, Member)>,
    /// Index into `ProgramObj::samplers` when this name is a sampler.
    sampler: Option<usize>,
}

struct SamplerBinding {
    binding: u32,
    unit: u32,
    cube: bool,
}

struct ProgramObj {
    vertex: vk::ShaderModule,
    fragment: vk::ShaderModule,
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    /// Staged uniform values for the vertex and fragment blocks (empty when
    /// a stage has none).
    blocks: [Vec<u8>; 2],
    slots: Vec<Slot>,
    names: HashMap<String, u32>,
    samplers: Vec<SamplerBinding>,
    vertex_inputs: BTreeMap<u32, (bool, u32)>,
    /// Logged at its first draw.
    drawn: bool,
    /// The modules' SPIR-V and the set layout's (binding, type, stages), for
    /// the draw dump; empty off the console.
    spirv: [Vec<u32>; 2],
    layout_bindings: Vec<[u32; 3]>,
}

enum Garbage {
    Buffer(vk::Buffer, Allocation),
    Image(vk::Image, vk::ImageView, Allocation),
    /// A submitted upload command buffer, from the upload pool.
    Commands(vk::CommandBuffer),
    Framebuffer(vk::Framebuffer),
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct PipelineKey {
    program: u32,
    layout: u32,
    topology: Topology,
    blend: Option<[BlendFactor; 4]>,
    depth_test: bool,
    depth_write: bool,
    depth_func: DepthFunc,
    cull: Option<FrontFace>,
    depth_bias: bool,
    color_mask: [bool; 4],
    formats: (i32, i32),
    /// Rasterization samples; the frame's MSAA, or 1 in a depth capture.
    samples: u32,
}

/// GL's fixed-function state.
#[derive(Clone, Copy)]
struct GlState {
    depth_test: bool,
    blend: bool,
    cull: bool,
    polygon_offset: bool,
    scissor_test: bool,
    depth_write: bool,
    depth_func: DepthFunc,
    blend_factors: [BlendFactor; 4],
    front_face: FrontFace,
    offset: (f32, f32),
    color_mask: [bool; 4],
    scissor: [i32; 4],
    viewport: Option<[i32; 4]>,
    clear_color: [f32; 4],
}

impl Default for GlState {
    fn default() -> Self {
        GlState {
            depth_test: false,
            blend: false,
            cull: false,
            polygon_offset: false,
            scissor_test: false,
            depth_write: true,
            depth_func: DepthFunc::Less,
            blend_factors: [
                BlendFactor::One,
                BlendFactor::Zero,
                BlendFactor::One,
                BlendFactor::Zero,
            ],
            front_face: FrontFace::CounterClockwise,
            offset: (0.0, 0.0),
            color_mask: [true; 4],
            scissor: [0; 4],
            viewport: None,
            clear_color: [0.0; 4],
        }
    }
}

struct Recording {
    commands: vk::CommandBuffer,
    render_pass: vk::RenderPass,
    formats: (i32, i32),
    samples: vk::SampleCountFlags,
    extent: vk::Extent2D,
    /// The frame's framebuffer and the pass that resumes drawing into it
    /// after a depth capture paused it.
    framebuffer: vk::Framebuffer,
    resume_pass: vk::RenderPass,
    /// While capturing: the frame's pass, formats, extent and GL state.
    paused: Option<(
        vk::RenderPass,
        (i32, i32),
        vk::SampleCountFlags,
        vk::Extent2D,
        GlState,
    )>,
    slot: usize,
    pipeline: vk::Pipeline,
    viewport: Option<[i32; 4]>,
    scissor: Option<vk::Rect2D>,
    bias: Option<(f32, f32)>,
}

#[derive(Default)]
struct FrameRes {
    ubo: Vec<(vk::Buffer, Allocation)>,
    ubo_at: (usize, u64),
    pools: Vec<vk::DescriptorPool>,
    pool_at: usize,
    /// Sets taken from `pools[pool_at]` this frame.
    pool_used: u32,
    /// This frame's descriptor sets by what they hold: draws with the same
    /// program, uniform ring chunks and textures share one, their uniform
    /// offsets given when it is bound (dynamic uniform buffers). Allocating
    /// and writing one a draw cost the Xbox 360 about 3 ms a frame.
    sets: HashMap<SetKey, vk::DescriptorSet>,
}

/// What a draw's descriptor set holds: its layout, each uniform binding's
/// ring chunk, and each sampler binding's view and sampler.
#[derive(Clone, PartialEq, Eq, Hash)]
struct SetKey {
    layout: vk::DescriptorSetLayout,
    buffers: [(u32, vk::Buffer, u64); 2],
    images: Vec<(u32, vk::ImageView, vk::Sampler)>,
}

struct State {
    /// `dump_draws.txt`'s frame, and each buffer's bytes until it's written.
    dump: super::draw_dump::DrawDump,
    dump_bytes: HashMap<u32, Vec<u8>>,
    /// Draws per program since [`take_draw_counts`].
    draw_counts: BTreeMap<u32, u32>,
    /// Draws this frame, for `max_draws.txt`.
    frame_draws: u32,
    /// Region uploads since [`take_draw_counts`], by image size and mips.
    upload_counts: BTreeMap<(u32, u32, u32), u32>,
    dev: Dev,
    alloc: Allocator,
    upload_pool: vk::CommandPool,
    next_id: u32,
    buffers: HashMap<u32, BufferObj>,
    images: HashMap<u32, ImageObj>,
    programs: Vec<ProgramObj>,
    samplers: HashMap<Sampling, vk::Sampler>,
    pipelines: HashMap<PipelineKey, vk::Pipeline>,
    /// Render passes pipelines are built against, by (colour, depth) format;
    /// a rebuilt swapchain's pass of the same formats is compatible.
    layouts: Vec<VertexLayout>,
    layout_ids: HashMap<VertexLayout, u32>,
    zero_buffer: u32,
    white: u32,
    white_cube: u32,
    frames: Vec<FrameRes>,
    /// Frames submitted so far.
    submitted: u64,
    garbage: Vec<(u64, Garbage)>,
    /// Copies recorded since the last flush, and the staging they read.
    uploads: Option<vk::CommandBuffer>,
    staging: Vec<(vk::Buffer, Allocation)>,
    staging_bytes: u64,
    recording: Option<Recording>,
    gl: GlState,
    program: Option<u32>,
    /// Texture units, each holding a 2D and a cube binding as GL's do.
    units: [Option<u32>; MAX_UNITS],
    cube_units: [Option<u32>; MAX_UNITS],
    depth_capture: Option<DepthCapture>,
    /// Whether the open capture kept its image's previous contents.
    capture_kept: bool,
}

/// Depth-only capture targets, sampled as textures after a capture: the
/// self-depth scratch (resized with the viewport it captures) and the
/// point-light shadow atlas. They share one render pass.
struct DepthCapture {
    pass: vk::RenderPass,
    /// `pass` keeping what the image held: the shadow atlas is redrawn
    /// tile by tile across frames.
    load_pass: vk::RenderPass,
    images: [Option<(u32, vk::Framebuffer, vk::Extent2D)>; CAPTURE_TARGETS],
    /// Each target's unwritten colour image ([`CAPTURE_COLOR`]).
    colors: [Option<u32>; CAPTURE_TARGETS],
    /// The target the open capture draws into.
    active: usize,
}

/// vulkan4xenon's subpasses take one to four colour attachments, so a
/// depth capture there draws into a colour image too, which nothing writes
/// (colour masked off) or reads (docs/SUPPORT.md, "Render passes").
const CAPTURE_COLOR: Option<vk::Format> = if cfg!(target_os = "xbox360") {
    Some(vk::Format::B8G8R8A8_UNORM)
} else {
    None
};

/// Self depth, the shadow atlas, and the Xbox 360's second atlas
/// ([`begin_shadow_capture`]).
const CAPTURE_TARGETS: usize = 3;
const SELF_DEPTH_TARGET: usize = 0;
const SHADOW_TARGET: usize = 1;

/// Sampled depth: D32_SFLOAT supports sampling and depth attachment on
/// every desktop driver. vulkan4xenon's only depth format is D24S8, which
/// samples as depth in red as D32 does (its docs/SUPPORT.md, "Images").
const CAPTURE_DEPTH: vk::Format = if cfg!(target_os = "xbox360") {
    vk::Format::D24_UNORM_S8_UINT
} else {
    vk::Format::D32_SFLOAT
};

/// Per-second counters, printed with `SHOCK2_VK_STATS=1`.
#[derive(Default)]
struct Stats {
    frames: u32,
    draws: u32,
    pipelines: u32,
    descriptor_writes: u32,
    uniform_sets: u32,
    draw_time: std::time::Duration,
    upload_time: std::time::Duration,
    fence_wait: std::time::Duration,
    acquire_wait: std::time::Duration,
    since: Option<std::time::Instant>,
}

#[cfg(not(target_os = "xbox360"))]
thread_local! {
    static STATS: RefCell<Stats> = RefCell::new(Stats::default());
    static STATS_ON: bool = std::env::var_os("SHOCK2_VK_STATS").is_some();
}

#[cfg(not(target_os = "xbox360"))]
fn stat(f: impl FnOnce(&mut Stats)) {
    if STATS_ON.with(|on| *on) {
        STATS.with(|s| f(&mut s.borrow_mut()));
    }
}

// Always on the Xbox 360, which has no environment; its log is the net log.
// The backend's thread owns them, without a TLS lookup (hardware_threads::OwnedCell).
#[cfg(target_os = "xbox360")]
static STATS: crate::hardware_threads::OwnedCell<Option<Stats>> =
    crate::hardware_threads::OwnedCell::new(None);

#[cfg(target_os = "xbox360")]
fn stat(f: impl FnOnce(&mut Stats)) {
    STATS.with(|s| f(s.get_or_insert_with(Stats::default)));
}

/// Time the renderer spent waiting for a frame slot's fence and for the
/// next swapchain image (vsync shows up as the latter).
pub fn note_waits(fence: std::time::Duration, acquire: std::time::Duration) {
    stat(|s| {
        s.fence_wait += fence;
        s.acquire_wait += acquire;
    });
}

/// Count a frame; once a second, print and reset.
fn stats_frame() {
    stat(|s| {
        s.frames += 1;
        let since = *s.since.get_or_insert_with(std::time::Instant::now);
        let elapsed = since.elapsed().as_secs_f32();
        if elapsed >= 1.0 {
            let f = s.frames.max(1) as f32;
            let line = format!(
                "SHOCK2_VK_STATS {:.1} fps | per frame: {:.0} draws, {:.0} descriptor writes, \
                 {:.2} ms recording draws, {:.2} ms uploads, {:.2} ms waiting on the GPU, \
                 {:.2} ms acquiring | {} pipelines built, {:.0} uniform sets",
                f / elapsed,
                s.draws as f32 / f,
                s.descriptor_writes as f32 / f,
                s.draw_time.as_secs_f32() * 1000.0 / f,
                s.upload_time.as_secs_f32() * 1000.0 / f,
                s.fence_wait.as_secs_f32() * 1000.0 / f,
                s.acquire_wait.as_secs_f32() * 1000.0 / f,
                s.pipelines,
                s.uniform_sets as f32 / f,
            );
            if cfg!(target_os = "xbox360") {
                info!("{line}");
            } else {
                println!("{line}");
            }
            *s = Stats {
                since: Some(std::time::Instant::now()),
                ..Default::default()
            };
        }
    });
}

#[cfg(not(target_os = "xbox360"))]
thread_local! {
    static STATE: RefCell<Option<State>> = const { RefCell::new(None) };
}

#[cfg(not(target_os = "xbox360"))]
fn with<R>(f: impl FnOnce(&mut State) -> R) -> Option<R> {
    STATE.with(|s| s.borrow_mut().as_mut().map(f))
}

/// The installing thread's, as the thread local is elsewhere, without its
/// TLS lookup on every gfx call (hardware_threads::OwnedCell).
#[cfg(target_os = "xbox360")]
static STATE: crate::hardware_threads::OwnedCell<Option<State>> =
    crate::hardware_threads::OwnedCell::new(None);

#[cfg(target_os = "xbox360")]
fn with<R>(f: impl FnOnce(&mut State) -> R) -> Option<R> {
    STATE.with(|s| s.as_mut().map(f)).flatten()
}

fn with_state<R>(f: impl FnOnce(&mut State) -> R) -> R {
    with(f).expect("gfx: the Vulkan backend is not installed")
}

/// Install the backend on `ctx`'s device and switch gfx to it.
pub fn install(ctx: &crate::vk::VkContext) {
    let dev = Dev {
        device: ctx.device.clone(),
        queue: ctx.queue,
        memory_properties: ctx.memory_properties,
        ubo_align: ctx.limits.min_uniform_buffer_offset_alignment.max(16),
        // Floors seen at a grazing angle are where plain trilinear blurs
        // most; 8x keeps them sharp for a small bandwidth cost.
        anisotropy: ctx
            .sampler_anisotropy
            .then(|| ctx.limits.max_sampler_anisotropy.min(8.0))
            .filter(|&a| a > 1.0),
    };
    let upload_pool = unsafe {
        dev.device.create_command_pool(
            &vk::CommandPoolCreateInfo::builder()
                .flags(vk::CommandPoolCreateFlags::TRANSIENT)
                .queue_family_index(ctx.queue_family),
            None,
        )
    }
    .expect("vkCreateCommandPool");
    let mut state = State {
        dump: super::draw_dump::DrawDump::from_file(),
        dump_bytes: HashMap::default(),
        draw_counts: BTreeMap::new(),
        frame_draws: 0,
        upload_counts: BTreeMap::new(),
        dev,
        alloc: Allocator::default(),
        upload_pool,
        next_id: 1,
        buffers: HashMap::default(),
        images: HashMap::default(),
        programs: Vec::new(),
        samplers: HashMap::default(),
        pipelines: HashMap::default(),
        layouts: Vec::new(),
        layout_ids: HashMap::default(),
        zero_buffer: 0,
        white: 0,
        white_cube: 0,
        frames: (0..FRAMES).map(|_| FrameRes::default()).collect(),
        submitted: 0,
        garbage: Vec::new(),
        uploads: None,
        staging: Vec::new(),
        staging_bytes: 0,
        recording: None,
        gl: GlState::default(),
        program: None,
        units: [None; MAX_UNITS],
        cube_units: [None; MAX_UNITS],
        depth_capture: None,
        capture_kept: false,
    };
    state.zero_buffer = state.create_buffer(&[0u8; 16], vk::BufferUsageFlags::VERTEX_BUFFER);
    state.white = state.create_texture(
        1,
        1,
        PixelFormat::Rgba8,
        &[255; 4],
        Sampling {
            filter: Filter::Nearest,
            wrap: true,
        },
    );
    state.white_cube = state.create_cube(1, &[255; 24]);
    #[cfg(not(target_os = "xbox360"))]
    STATE.with(|s| *s.borrow_mut() = Some(state));
    #[cfg(target_os = "xbox360")]
    {
        let claimed = STATE.claim() && STATS.claim();
        assert!(claimed, "gfx: the Vulkan backend belongs to another thread");
        STATE.with(|s| *s = Some(state));
    }
}

/// Destroy every object the backend made. The device must be idle.
pub fn shutdown() {
    #[cfg(not(target_os = "xbox360"))]
    let taken = STATE.with(|s| s.borrow_mut().take());
    #[cfg(target_os = "xbox360")]
    let taken = STATE.with(|s| s.take()).flatten();
    let Some(mut s) = taken else {
        return;
    };
    s.flush_uploads();
    let d = s.dev.device.clone();
    unsafe {
        if let Some(capture) = s.depth_capture.take() {
            for (id, framebuffer, _) in capture.images.into_iter().flatten() {
                d.destroy_framebuffer(framebuffer, None);
                if let Some(i) = s.images.remove(&id) {
                    d.destroy_image_view(i.view, None);
                    d.destroy_image(i.image, None);
                }
            }
            d.destroy_render_pass(capture.pass, None);
            d.destroy_render_pass(capture.load_pass, None);
        }
        for (_, g) in s.garbage.drain(..) {
            destroy_garbage(&d, s.upload_pool, g);
        }
        for (_, b) in s.buffers.drain() {
            d.destroy_buffer(b.buffer, None);
        }
        for (_, i) in s.images.drain() {
            d.destroy_image_view(i.view, None);
            d.destroy_image(i.image, None);
        }
        for f in &s.frames {
            for &(b, _) in &f.ubo {
                d.destroy_buffer(b, None);
            }
            for &p in &f.pools {
                d.destroy_descriptor_pool(p, None);
            }
        }
        for (_, p) in s.pipelines.drain() {
            d.destroy_pipeline(p, None);
        }
        for p in s.programs.drain(..) {
            d.destroy_shader_module(p.vertex, None);
            d.destroy_shader_module(p.fragment, None);
            d.destroy_pipeline_layout(p.layout, None);
            d.destroy_descriptor_set_layout(p.set_layout, None);
        }
        for (_, sm) in s.samplers.drain() {
            d.destroy_sampler(sm, None);
        }
        d.destroy_command_pool(s.upload_pool, None);
    }
    s.alloc.destroy(&d);
}

/// Destroy `g`, returning the memory it held to free.
unsafe fn destroy_garbage(
    d: &ash::Device,
    upload_pool: vk::CommandPool,
    g: Garbage,
) -> Option<Allocation> {
    unsafe {
        match g {
            Garbage::Buffer(b, a) => {
                d.destroy_buffer(b, None);
                Some(a)
            }
            Garbage::Image(i, v, a) => {
                d.destroy_image_view(v, None);
                d.destroy_image(i, None);
                Some(a)
            }
            Garbage::Commands(cmd) => {
                d.free_command_buffers(upload_pool, &[cmd]);
                None
            }
            Garbage::Framebuffer(f) => {
                d.destroy_framebuffer(f, None);
                None
            }
        }
    }
}

/// Start recording gfx draws into `commands`, inside an open render pass.
/// `slot`'s previous frame has finished (its fence was waited on).
/// `framebuffer` and `resume_pass` let a self-depth capture pause the
/// frame's `render_pass` and pick it up again.
#[allow(clippy::too_many_arguments)]
pub fn begin_frame(
    commands: vk::CommandBuffer,
    render_pass: vk::RenderPass,
    resume_pass: vk::RenderPass,
    framebuffer: vk::Framebuffer,
    formats: (vk::Format, vk::Format),
    samples: vk::SampleCountFlags,
    extent: vk::Extent2D,
    slot: usize,
) {
    with(|s| {
        // On the Xbox 360 an update into an image the last frame still reads
        // (the lightmap atlas's flicker, every frame) waits there for the GPU
        // to finish that frame: end_frame submits them instead, after this
        // frame's recording has given the GPU time to get there, and still
        // ahead of the frame that draws with them.
        if !cfg!(target_os = "xbox360") {
            s.submit_uploads();
        }
        // Frames before `submitted - (FRAMES - 1)` are complete.
        let done_before = s.submitted.saturating_sub(FRAMES as u64 - 1);
        let mut keep = Vec::new();
        for (tag, g) in s.garbage.drain(..) {
            if tag <= done_before {
                if let Some(a) = unsafe { destroy_garbage(&s.dev.device, s.upload_pool, g) } {
                    s.alloc.free(a);
                }
            } else {
                keep.push((tag, g));
            }
        }
        s.garbage = keep;
        s.frame_draws = 0;
        s.dump.begin_frame(
            extent.width,
            extent.height,
            [
                formats.0.as_raw() as u32,
                formats.1.as_raw() as u32,
                samples.as_raw(),
            ],
        );
        let frame = &mut s.frames[slot];
        frame.ubo_at = (0, 0);
        frame.pool_at = 0;
        frame.pool_used = 0;
        frame.sets.clear();
        for &pool in &frame.pools {
            unsafe {
                s.dev
                    .device
                    .reset_descriptor_pool(pool, vk::DescriptorPoolResetFlags::empty())
            }
            .expect("vkResetDescriptorPool");
        }
        s.gl.viewport = None;
        s.recording = Some(Recording {
            commands,
            render_pass,
            formats: (formats.0.as_raw(), formats.1.as_raw()),
            samples,
            extent,
            framebuffer,
            resume_pass,
            paused: None,
            slot,
            pipeline: vk::Pipeline::null(),
            viewport: None,
            scissor: None,
            bias: None,
        });
    });
}

/// Stop recording; the renderer submits the frame next.
pub fn end_frame() {
    with(|s| {
        // Textures and meshes made while recording are drawn this frame.
        s.submit_uploads();
        if s.recording.take().is_some() {
            stats_frame();
            s.submitted += 1;
        }
        s.dump.end_frame();
        if !s.dump.pending() && !s.dump_bytes.is_empty() {
            s.dump_bytes = HashMap::default();
        }
    });
}

impl State {
    fn id(&mut self) -> u32 {
        self.next_id += 1;
        self.next_id
    }

    /// The frame number a resource dropped now may still be read by.
    fn retire_tag(&self) -> u64 {
        self.submitted + self.recording.is_some() as u64
    }

    fn retire(&mut self, g: Garbage) {
        let tag = self.retire_tag();
        self.garbage.push((tag, g));
    }

    fn raw_buffer(&mut self, size: u64, usage: vk::BufferUsageFlags) -> (vk::Buffer, Allocation) {
        self.raw_buffer_in(
            size,
            usage,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
    }

    fn raw_buffer_in(
        &mut self,
        size: u64,
        usage: vk::BufferUsageFlags,
        flags: vk::MemoryPropertyFlags,
    ) -> (vk::Buffer, Allocation) {
        let d = &self.dev.device;
        let buffer = unsafe {
            d.create_buffer(
                &vk::BufferCreateInfo::builder()
                    // Room past the end: the Xbox 360's vertex fetch reads past a buffer's last vertex.
                    .size(size.max(16) + if cfg!(target_os = "xbox360") { 4096 } else { 0 })
                    .usage(usage)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )
        }
        .expect("vkCreateBuffer");
        let req = unsafe { d.get_buffer_memory_requirements(buffer) };
        let alloc = self.alloc.alloc(&self.dev, req, flags, true);
        unsafe { d.bind_buffer_memory(buffer, alloc.memory, alloc.offset) }
            .expect("vkBindBufferMemory");
        (buffer, alloc)
    }

    /// A host-visible buffer holding `bytes`, for data rewritten often.
    fn create_buffer(&mut self, bytes: &[u8], usage: vk::BufferUsageFlags) -> u32 {
        let size = bytes.len() as u64;
        let (buffer, alloc) = self.raw_buffer(size, usage | vk::BufferUsageFlags::TRANSFER_DST);
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), alloc.mapped, bytes.len());
        }
        let id = self.id();
        if self.dump.pending() {
            self.dump_bytes.insert(id, bytes.to_vec());
        }
        self.buffers.insert(
            id,
            BufferObj {
                buffer,
                alloc,
                size,
                host: true,
            },
        );
        id
    }

    /// A device-local buffer holding `bytes`, copied in by the next flush.
    fn create_static_buffer(&mut self, bytes: &[u8], usage: vk::BufferUsageFlags) -> u32 {
        // The Xbox 360's memory is one pool, so device-local buys nothing,
        // and vulkan4xenon makes a copy into a buffer wait for every draw
        // before it (about 4.6 ms a frame): its buffers are the host's.
        if cfg!(target_os = "xbox360") {
            return self.create_buffer(bytes, usage);
        }
        let size = bytes.len() as u64;
        let (buffer, alloc) = self.raw_buffer_in(
            size,
            usage | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );
        if size > 0 {
            let staging = self.stage(bytes);
            let cmd = self.upload_commands();
            unsafe {
                self.dev.device.cmd_copy_buffer(
                    cmd,
                    staging,
                    buffer,
                    &[vk::BufferCopy {
                        src_offset: 0,
                        dst_offset: 0,
                        size,
                    }],
                );
            }
        }
        let id = self.id();
        if self.dump.pending() {
            self.dump_bytes.insert(id, bytes.to_vec());
        }
        self.buffers.insert(
            id,
            BufferObj {
                buffer,
                alloc,
                size,
                host: false,
            },
        );
        self.maybe_flush();
        id
    }

    /// A staging buffer holding `bytes`, freed by the next flush.
    fn stage(&mut self, bytes: &[u8]) -> vk::Buffer {
        let (buffer, alloc) =
            self.raw_buffer(bytes.len() as u64, vk::BufferUsageFlags::TRANSFER_SRC);
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), alloc.mapped, bytes.len()) };
        self.staging.push((buffer, alloc));
        self.staging_bytes += bytes.len() as u64;
        buffer
    }

    /// The open upload command buffer, begun if need be.
    fn upload_commands(&mut self) -> vk::CommandBuffer {
        if let Some(cmd) = self.uploads {
            return cmd;
        }
        let d = &self.dev.device;
        let cmd = unsafe {
            let cmd = d
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::builder()
                        .command_pool(self.upload_pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
                .expect("vkAllocateCommandBuffers")[0];
            d.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::builder()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .expect("vkBeginCommandBuffer");
            cmd
        };
        self.uploads = Some(cmd);
        cmd
    }

    /// Keep loading from piling up unbounded staging memory.
    fn maybe_flush(&mut self) {
        if self.staging_bytes > 64 << 20 {
            self.flush_uploads();
        }
    }

    /// Submit the recorded uploads ahead of the frame being recorded. Queue
    /// order (and the closing barrier) makes the frame read the new data, so
    /// nothing waits: the staging and command buffer are freed with the frame.
    fn submit_uploads(&mut self) {
        self.flush(false);
    }

    /// Submit the recorded uploads and wait for them (loading, outside any
    /// frame, where nothing would otherwise free the staging).
    fn flush_uploads(&mut self) {
        self.flush(true);
    }

    fn flush(&mut self, wait: bool) {
        let Some(cmd) = self.uploads.take() else {
            return;
        };
        let start = std::time::Instant::now();
        let d = &self.dev.device;
        unsafe {
            // Uploaded data is visible to every later command on the queue.
            let barrier = vk::MemoryBarrier::builder()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::MEMORY_READ)
                .build();
            d.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
            d.end_command_buffer(cmd).expect("vkEndCommandBuffer");
            let cmds = [cmd];
            let submit = [vk::SubmitInfo::builder().command_buffers(&cmds).build()];
            if wait {
                let fence = d
                    .create_fence(&vk::FenceCreateInfo::default(), None)
                    .expect("vkCreateFence");
                tracing::debug!("upload: submitting and waiting");
                d.queue_submit(self.dev.queue, &submit, fence)
                    .expect("vkQueueSubmit");
                d.wait_for_fences(&[fence], true, u64::MAX)
                    .expect("vkWaitForFences");
                tracing::debug!("upload: done");
                d.destroy_fence(fence, None);
                d.free_command_buffers(self.upload_pool, &cmds);
            } else {
                d.queue_submit(self.dev.queue, &submit, vk::Fence::null())
                    .expect("vkQueueSubmit");
            }
        }
        let staging = std::mem::take(&mut self.staging);
        if wait {
            for (buffer, alloc) in staging {
                unsafe { self.dev.device.destroy_buffer(buffer, None) };
                self.alloc.free(alloc);
            }
        } else {
            // A fence signal covers every earlier submission on the queue.
            // Submitted ahead of the next frame, so done when it is.
            let tag = self.submitted + 1;
            for (buffer, alloc) in staging {
                self.garbage.push((tag, Garbage::Buffer(buffer, alloc)));
            }
            self.garbage.push((tag, Garbage::Commands(cmd)));
        }
        self.staging_bytes = 0;
        stat(|s| s.upload_time += start.elapsed());
    }

    fn drop_buffer(&mut self, id: u32) {
        if let Some(b) = self.buffers.remove(&id) {
            self.retire(Garbage::Buffer(b.buffer, b.alloc));
        }
    }

    fn create_texture(
        &mut self,
        width: u32,
        height: u32,
        format: PixelFormat,
        pixels: &[u8],
        sampling: Sampling,
    ) -> u32 {
        tracing::debug!("texture: {width}x{height}");
        let rgba = to_rgba(format, pixels);
        self.create_image(width, height, 1, &rgba, sampling)
    }

    /// A cube from six RGBA8 faces (+X, -X, +Y, -Y, +Z, -Z), mipmapped and
    /// clamped as GL's cube maps here are.
    fn create_cube(&mut self, size: u32, faces: &[u8]) -> u32 {
        let sampling = Sampling {
            filter: Filter::LinearMipmap,
            wrap: false,
        };
        self.create_image(size, size, 6, faces, sampling)
    }

    /// `layers` 1 (a 2D texture) or 6 (a cube); `rgba` holds each layer in
    /// turn, rows top-first.
    fn create_image(
        &mut self,
        width: u32,
        height: u32,
        layers: u32,
        rgba: &[u8],
        sampling: Sampling,
    ) -> u32 {
        let cube = layers == 6;
        let power_of_two = width.is_power_of_two() && height.is_power_of_two();
        let mips = if sampling.filter == Filter::LinearMipmap
            && (power_of_two || !cfg!(target_os = "xbox360"))
        {
            32 - width.max(height).max(1).leading_zeros()
        } else {
            1
        };
        let d = self.dev.device.clone();
        let image = unsafe {
            d.create_image(
                &vk::ImageCreateInfo::builder()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(vk::Format::R8G8B8A8_UNORM)
                    .extent(vk::Extent3D {
                        width,
                        height,
                        depth: 1,
                    })
                    .mip_levels(mips)
                    .array_layers(layers)
                    .flags(if cube {
                        vk::ImageCreateFlags::CUBE_COMPATIBLE
                    } else {
                        vk::ImageCreateFlags::empty()
                    })
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(
                        vk::ImageUsageFlags::SAMPLED
                            | vk::ImageUsageFlags::TRANSFER_DST
                            | vk::ImageUsageFlags::TRANSFER_SRC,
                    )
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .initial_layout(vk::ImageLayout::UNDEFINED),
                None,
            )
        }
        .expect("vkCreateImage");
        let req = unsafe { d.get_image_memory_requirements(image) };
        let alloc = self
            .alloc
            .alloc(&self.dev, req, vk::MemoryPropertyFlags::DEVICE_LOCAL, false);
        unsafe { d.bind_image_memory(image, alloc.memory, alloc.offset) }
            .expect("vkBindImageMemory");
        let view = unsafe {
            d.create_image_view(
                &vk::ImageViewCreateInfo::builder()
                    .image(image)
                    .view_type(if cube {
                        vk::ImageViewType::CUBE
                    } else {
                        vk::ImageViewType::TYPE_2D
                    })
                    .format(vk::Format::R8G8B8A8_UNORM)
                    .subresource_range(range(0, mips)),
                None,
            )
        }
        .expect("vkCreateImageView");
        let id = self.id();
        self.images.insert(
            id,
            ImageObj {
                image,
                view,
                alloc,
                sampling,
                mips,
                layers,
                size: (width, height),
            },
        );
        self.upload(id, 0, 0, width, height, rgba, true, (width, height));
        id
    }

    /// Copy tightly packed RGBA8 into a region of mip 0; `fresh` images
    /// come from UNDEFINED and get their mip chain built.
    #[allow(clippy::too_many_arguments)]
    fn upload(
        &mut self,
        id: u32,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        rgba: &[u8],
        fresh: bool,
        size: (u32, u32),
    ) {
        let staging = self.stage(rgba);
        let (image, mips, faces) = {
            let i = &self.images[&id];
            (i.image, i.mips, i.layers)
        };
        // The console's frame log counts uploads by size and levels
        // (`take_draw_counts`); images made since the last count carry 1000
        // on their levels.
        *self
            .upload_counts
            .entry((size.0, size.1, mips + if fresh { 1000 } else { 0 }))
            .or_default() += 1;
        let cmd = self.upload_commands();
        let d = &self.dev.device;
        unsafe {
            let from = if fresh {
                vk::ImageLayout::UNDEFINED
            } else {
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
            };
            barrier(
                d,
                cmd,
                image,
                range(0, mips),
                from,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            );
            d.cmd_copy_buffer_to_image(
                cmd,
                staging,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[vk::BufferImageCopy {
                    buffer_offset: 0,
                    buffer_row_length: 0,
                    buffer_image_height: 0,
                    image_subresource: layers(0, faces),
                    image_offset: vk::Offset3D {
                        x: x as i32,
                        y: y as i32,
                        z: 0,
                    },
                    image_extent: vk::Extent3D {
                        width,
                        height,
                        depth: 1,
                    },
                }],
            );
            // Each level from the one above it, as glGenerateMipmap does.
            let (mut w, mut h) = (size.0 as i32, size.1 as i32);
            for level in 1..mips {
                barrier(
                    d,
                    cmd,
                    image,
                    range(level - 1, 1),
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                );
                let (nw, nh) = ((w / 2).max(1), (h / 2).max(1));
                d.cmd_blit_image(
                    cmd,
                    image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[vk::ImageBlit {
                        src_subresource: layers(level - 1, faces),
                        src_offsets: [vk::Offset3D::default(), vk::Offset3D { x: w, y: h, z: 1 }],
                        dst_subresource: layers(level, faces),
                        dst_offsets: [vk::Offset3D::default(), vk::Offset3D { x: nw, y: nh, z: 1 }],
                    }],
                    vk::Filter::LINEAR,
                );
                barrier(
                    d,
                    cmd,
                    image,
                    range(level - 1, 1),
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                );
                (w, h) = (nw, nh);
            }
            barrier(
                d,
                cmd,
                image,
                range(mips - 1, 1),
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            );
        }
        self.maybe_flush();
    }

    fn sampler(&mut self, sampling: Sampling) -> vk::Sampler {
        if let Some(&s) = self.samplers.get(&sampling) {
            return s;
        }
        let (filter, mut mip, mut max_lod) = match sampling.filter {
            Filter::Nearest => (vk::Filter::NEAREST, vk::SamplerMipmapMode::NEAREST, 0.0),
            Filter::Linear => (vk::Filter::LINEAR, vk::SamplerMipmapMode::NEAREST, 0.0),
            Filter::LinearMipmap => (vk::Filter::LINEAR, vk::SamplerMipmapMode::LINEAR, 1000.0),
        };
        let knobs = sampler_knobs();
        let aniso = self
            .dev
            .anisotropy
            .filter(|_| sampling.filter == Filter::LinearMipmap && !knobs.no_aniso);
        if let Some(lod) = knobs.max_lod {
            max_lod = f32::min(max_lod, lod);
        }
        if knobs.nearest_mip {
            mip = vk::SamplerMipmapMode::NEAREST;
        }
        let wrap = if sampling.wrap {
            vk::SamplerAddressMode::REPEAT
        } else {
            vk::SamplerAddressMode::CLAMP_TO_EDGE
        };
        let s = unsafe {
            self.dev.device.create_sampler(
                &vk::SamplerCreateInfo::builder()
                    .mag_filter(filter)
                    .min_filter(filter)
                    .mipmap_mode(mip)
                    .address_mode_u(wrap)
                    .address_mode_v(wrap)
                    .address_mode_w(wrap)
                    .anisotropy_enable(aniso.is_some())
                    .max_anisotropy(aniso.unwrap_or(1.0))
                    .max_lod(max_lod),
                None,
            )
        }
        .expect("vkCreateSampler");
        self.samplers.insert(sampling, s);
        s
    }

    fn layout_id(&mut self, layout: &VertexLayout) -> u32 {
        if let Some(&id) = self.layout_ids.get(layout) {
            return id;
        }
        let id = self.layouts.len() as u32;
        self.layouts.push(layout.clone());
        self.layout_ids.insert(layout.clone(), id);
        id
    }

    fn pipeline(&mut self, key: PipelineKey, render_pass: vk::RenderPass) -> vk::Pipeline {
        if let Some(&p) = self.pipelines.get(&key) {
            return p;
        }
        let program = &self.programs[key.program as usize];
        let layout = &self.layouts[key.layout as usize];
        let mut bindings = vec![vk::VertexInputBindingDescription {
            binding: 0,
            stride: layout.stride,
            input_rate: vk::VertexInputRate::VERTEX,
        }];
        let mut attributes = Vec::new();
        let mut needs_zero = false;
        for (&location, &(int, _)) in &program.vertex_inputs {
            match layout.attributes.get(location as usize) {
                Some(a) => attributes.push(vk::VertexInputAttributeDescription {
                    location,
                    binding: 0,
                    format: attribute_format(a.kind == super::AttributeKind::Int, a.components),
                    offset: a.offset,
                }),
                None => {
                    // GL reads a missing attribute as a constant; so does
                    // a zero-stride buffer.
                    needs_zero = true;
                    attributes.push(vk::VertexInputAttributeDescription {
                        location,
                        binding: 1,
                        format: attribute_format(int, 4),
                        offset: 0,
                    });
                }
            }
        }
        if needs_zero {
            bindings.push(vk::VertexInputBindingDescription {
                binding: 1,
                stride: 0,
                input_rate: vk::VertexInputRate::VERTEX,
            });
        }
        let entry = c"main";
        let stages = [
            vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(program.vertex)
                .name(entry)
                .build(),
            vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(program.fragment)
                .name(entry)
                .build(),
        ];
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_binding_descriptions(&bindings)
            .vertex_attribute_descriptions(&attributes);
        let assembly =
            vk::PipelineInputAssemblyStateCreateInfo::builder().topology(match key.topology {
                Topology::Triangles => vk::PrimitiveTopology::TRIANGLE_LIST,
                Topology::Lines => vk::PrimitiveTopology::LINE_LIST,
            });
        let viewport = vk::PipelineViewportStateCreateInfo::builder()
            .viewport_count(1)
            .scissor_count(1);
        // The vertex prologue flips y on the swapchain, which keeps GL's
        // winding: a triangle GL calls counter-clockwise still is.
        let raster = vk::PipelineRasterizationStateCreateInfo::builder()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(if key.cull.is_some() {
                vk::CullModeFlags::BACK
            } else {
                vk::CullModeFlags::NONE
            })
            .front_face(match key.cull {
                Some(FrontFace::Clockwise) => vk::FrontFace::CLOCKWISE,
                _ => vk::FrontFace::COUNTER_CLOCKWISE,
            })
            .depth_bias_enable(key.depth_bias)
            .line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::builder()
            .rasterization_samples(vk::SampleCountFlags::from_raw(key.samples));
        let depth = vk::PipelineDepthStencilStateCreateInfo::builder()
            .depth_test_enable(key.depth_test)
            .depth_write_enable(key.depth_write)
            .depth_compare_op(match key.depth_func {
                DepthFunc::Less => vk::CompareOp::LESS,
                DepthFunc::LessEqual => vk::CompareOp::LESS_OR_EQUAL,
                DepthFunc::Always => vk::CompareOp::ALWAYS,
            });
        let mut mask = vk::ColorComponentFlags::empty();
        for (on, bit) in key.color_mask.iter().zip([
            vk::ColorComponentFlags::R,
            vk::ColorComponentFlags::G,
            vk::ColorComponentFlags::B,
            vk::ColorComponentFlags::A,
        ]) {
            if *on {
                mask |= bit;
            }
        }
        let f = key.blend.unwrap_or([
            BlendFactor::One,
            BlendFactor::Zero,
            BlendFactor::One,
            BlendFactor::Zero,
        ]);
        let blend_attachment = [vk::PipelineColorBlendAttachmentState::builder()
            .blend_enable(key.blend.is_some())
            .src_color_blend_factor(blend_factor(f[0]))
            .dst_color_blend_factor(blend_factor(f[1]))
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(blend_factor(f[2]))
            .dst_alpha_blend_factor(blend_factor(f[3]))
            .alpha_blend_op(vk::BlendOp::ADD)
            .color_write_mask(mask)
            .build()];
        // A depth-only pass has no colour attachment to blend into.
        let blend = vk::PipelineColorBlendStateCreateInfo::builder().attachments(
            if key.formats.0 == vk::Format::UNDEFINED.as_raw() {
                &[]
            } else {
                &blend_attachment[..]
            },
        );
        let dynamic_states = [
            vk::DynamicState::VIEWPORT,
            vk::DynamicState::SCISSOR,
            vk::DynamicState::DEPTH_BIAS,
        ];
        let dynamic = vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(&dynamic_states);
        let info = vk::GraphicsPipelineCreateInfo::builder()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&assembly)
            .viewport_state(&viewport)
            .rasterization_state(&raster)
            .multisample_state(&multisample)
            .depth_stencil_state(&depth)
            .color_blend_state(&blend)
            .dynamic_state(&dynamic)
            .layout(program.layout)
            .render_pass(render_pass)
            .subpass(0)
            .build();
        let pipeline = unsafe {
            self.dev
                .device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[info], None)
        }
        .map_err(|(_, e)| e)
        .expect("vkCreateGraphicsPipelines")[0];
        self.pipelines.insert(key, pipeline);
        stat(|s| s.pipelines += 1);
        pipeline
    }

    /// Copy `bytes` into this frame's uniform ring.
    fn push_uniforms(&mut self, slot: usize, bytes: &[u8]) -> (vk::Buffer, u64) {
        let align = self.dev.ubo_align;
        let size = bytes.len() as u64;
        assert!(size <= UBO_CHUNK, "uniform block of {size} bytes");
        loop {
            let frame = &mut self.frames[slot];
            let (chunk, at) = frame.ubo_at;
            if chunk < frame.ubo.len() {
                let offset = at.next_multiple_of(align);
                if offset + size <= UBO_CHUNK {
                    let (buffer, alloc) = frame.ubo[chunk];
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            bytes.as_ptr(),
                            alloc.mapped.add(offset as usize),
                            bytes.len(),
                        )
                    };
                    frame.ubo_at = (chunk, offset + size);
                    return (buffer, offset);
                }
                frame.ubo_at = (chunk + 1, 0);
            } else {
                let b = self.raw_buffer(UBO_CHUNK, vk::BufferUsageFlags::UNIFORM_BUFFER);
                self.frames[slot].ubo.push(b);
            }
        }
    }

    fn descriptor_set(
        &mut self,
        slot: usize,
        layout: vk::DescriptorSetLayout,
    ) -> vk::DescriptorSet {
        loop {
            let frame = &mut self.frames[slot];
            // Move on before the pool runs out rather than probe a full one.
            if frame.pool_used >= POOL_SETS {
                frame.pool_at += 1;
                frame.pool_used = 0;
            }
            let fresh = frame.pool_at == frame.pools.len();
            if fresh {
                let sizes = [
                    vk::DescriptorPoolSize {
                        ty: vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
                        descriptor_count: POOL_SETS * 2,
                    },
                    vk::DescriptorPoolSize {
                        ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                        // Lit materials bind up to ~9 (diffuse, pass, shine,
                        // self-depth, environment, shadow atlas).
                        descriptor_count: POOL_SETS * 10,
                    },
                ];
                let pool = unsafe {
                    self.dev.device.create_descriptor_pool(
                        &vk::DescriptorPoolCreateInfo::builder()
                            .max_sets(POOL_SETS)
                            .pool_sizes(&sizes),
                        None,
                    )
                }
                .expect("vkCreateDescriptorPool");
                frame.pools.push(pool);
            }
            let layouts = [layout];
            match unsafe {
                self.dev.device.allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::builder()
                        .descriptor_pool(frame.pools[frame.pool_at])
                        .set_layouts(&layouts),
                )
            } {
                Ok(sets) => {
                    frame.pool_used += 1;
                    return sets[0];
                }
                // A Vulkan 1.0 device without maintenance1 may report an
                // exhausted pool as any allocation failure (out of device
                // memory, typically), so any failure moves on to the next
                // pool - unless the pool was new, which no retry would fix.
                Err(e) if !fresh => {
                    let _ = e;
                    frame.pool_at += 1;
                    frame.pool_used = 0;
                }
                Err(e) => panic!("vkAllocateDescriptorSets: {e}"),
            }
        }
    }

    fn draw(
        &mut self,
        vertices: u32,
        indices: Option<u32>,
        layout: u32,
        topology: Topology,
        first: u32,
        count: u32,
    ) {
        let Some(program_id) = self.program else {
            return;
        };
        // Diagnostic: `maxverts=N` in sampler.txt cuts each draw to N vertices.
        let count = match sampler_knobs().max_verts {
            Some(n) if topology == Topology::Triangles => count.min(n / 3 * 3),
            _ => count,
        };
        if skipped_programs().contains(&program_id) {
            return;
        }
        if let Some((min, max)) = max_draws() {
            let index = self.frame_draws;
            self.frame_draws += 1;
            if index < min || index >= max {
                return;
            }
            let textures: Vec<String> = self
                .units
                .iter()
                .chain(self.cube_units.iter())
                .flatten()
                .filter_map(|id| self.images.get(id))
                .map(|i| format!("{}x{}/{}x{}", i.size.0, i.size.1, i.mips, i.layers))
                .collect();
            tracing::info!(
                "draw {}: program {program_id}, {} from {first}, indexed {}, textures {}",
                index,
                count,
                indices.is_some(),
                textures.join(" ")
            );
        }
        *self.draw_counts.entry(program_id).or_default() += 1;
        if let Some(program) = self.programs.get_mut(program_id as usize)
            && !program.drawn
        {
            program.drawn = true;
            tracing::debug!("first draw with program {program_id}");
        }
        let Some(rec) = self.recording.as_ref() else {
            return;
        };
        let (cmd, slot, render_pass, formats, extent) = (
            rec.commands,
            rec.slot,
            rec.render_pass,
            rec.formats,
            rec.extent,
        );
        let samples = rec.samples.as_raw();
        let gl = self.gl;
        let key = PipelineKey {
            program: program_id,
            layout,
            topology,
            blend: gl.blend.then_some(gl.blend_factors),
            depth_test: gl.depth_test,
            depth_write: gl.depth_write,
            depth_func: gl.depth_func,
            cull: gl.cull.then_some(gl.front_face),
            depth_bias: gl.polygon_offset,
            color_mask: gl.color_mask,
            formats,
            samples,
        };
        let pipeline = self.pipeline(key, render_pass);

        // Uniform blocks into the ring, then this draw's descriptor set.
        let blocks = std::mem::take(&mut self.programs[program_id as usize].blocks);
        // Each uniform binding's ring chunk and size, and its offset there.
        let mut buffers = [(u32::MAX, vk::Buffer::null(), 0u64); 2];
        let mut dynamic_offsets = [0u32; 2];
        let mut dynamic_count = 0;
        for (binding, bytes) in blocks.iter().enumerate() {
            if !bytes.is_empty() {
                let (buffer, offset) = self.push_uniforms(slot, bytes);
                buffers[dynamic_count] = (binding as u32, buffer, bytes.len() as u64);
                dynamic_offsets[dynamic_count] = offset as u32;
                dynamic_count += 1;
            }
        }
        let blocks = if self.dump.active() {
            let copy = blocks.clone();
            self.programs[program_id as usize].blocks = blocks;
            copy
        } else {
            self.programs[program_id as usize].blocks = blocks;
            Default::default()
        };
        let sampler_units: Vec<(u32, u32, bool)> = self.programs[program_id as usize]
            .samplers
            .iter()
            .map(|s| (s.binding, s.unit, s.cube))
            .collect();
        let mut image_writes = Vec::new();
        let mut dump_samplers: Vec<[u32; 7]> = Vec::new();
        for (binding, unit, cube) in sampler_units {
            let (units, fallback) = if cube {
                (&self.cube_units, self.white_cube)
            } else {
                (&self.units, self.white)
            };
            let image = units
                .get(unit as usize)
                .copied()
                .flatten()
                .filter(|id| self.images.contains_key(id))
                .unwrap_or(fallback);
            // Diagnostic: `nobig` in sampler.txt samples white for the lightmap atlas.
            let image = if sampler_knobs().no_big && self.images[&image].size.0 > 2048 {
                fallback
            } else {
                image
            };
            let (view, sampling) = {
                let i = &self.images[&image];
                if self.dump.active() {
                    dump_samplers.push([
                        binding,
                        i.size.0,
                        i.size.1,
                        i.mips,
                        i.layers,
                        match i.sampling.filter {
                            Filter::Nearest => 0,
                            Filter::Linear => 1,
                            Filter::LinearMipmap => 2,
                        },
                        i.sampling.wrap as u32,
                    ]);
                }
                (i.view, i.sampling)
            };
            let sampler = self.sampler(sampling);
            image_writes.push((
                binding,
                vk::DescriptorImageInfo {
                    sampler,
                    image_view: view,
                    image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                },
            ));
        }
        let program = &self.programs[program_id as usize];
        let (set_layout, pipeline_layout) = (program.set_layout, program.layout);
        let set_key = SetKey {
            layout: set_layout,
            buffers,
            images: image_writes
                .iter()
                .map(|(binding, info)| (*binding, info.image_view, info.sampler))
                .collect(),
        };
        let set = match self.frames[slot].sets.get(&set_key) {
            Some(&set) => set,
            None => {
                let set = self.descriptor_set(slot, set_layout);
                let buffer_infos: Vec<(u32, vk::DescriptorBufferInfo)> = buffers[..dynamic_count]
                    .iter()
                    .map(|&(binding, buffer, range)| {
                        (
                            binding,
                            vk::DescriptorBufferInfo {
                                buffer,
                                offset: 0,
                                range,
                            },
                        )
                    })
                    .collect();
                let writes: Vec<vk::WriteDescriptorSet> = buffer_infos
                    .iter()
                    .map(|(binding, info)| {
                        vk::WriteDescriptorSet::builder()
                            .dst_set(set)
                            .dst_binding(*binding)
                            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                            .buffer_info(std::slice::from_ref(info))
                            .build()
                    })
                    .chain(image_writes.iter().map(|(binding, info)| {
                        vk::WriteDescriptorSet::builder()
                            .dst_set(set)
                            .dst_binding(*binding)
                            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                            .image_info(std::slice::from_ref(info))
                            .build()
                    }))
                    .collect();
                unsafe {
                    if !writes.is_empty() {
                        self.dev.device.update_descriptor_sets(&writes, &[]);
                    }
                }
                stat(|s| s.descriptor_writes += writes.len() as u32);
                self.frames[slot].sets.insert(set_key, set);
                set
            }
        };
        let d = &self.dev.device;

        // GL rects are bottom-left; the swapchain is top-left.
        let h = extent.height as i32;
        let vp = gl
            .viewport
            .unwrap_or([0, 0, extent.width as i32, extent.height as i32]);
        let viewport = vk::Viewport {
            x: vp[0] as f32,
            y: (h - vp[1] - vp[3]) as f32,
            width: vp[2] as f32,
            height: vp[3] as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let scissor = if gl.scissor_test {
            clip_rect(gl.scissor, extent)
        } else {
            vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent,
            }
        };
        let zero = self.buffers[&self.zero_buffer].buffer;
        let vertex_buffer = self.buffers.get(&vertices).map(|b| b.buffer);
        let index_buffer = indices.and_then(|i| self.buffers.get(&i)).map(|b| b.buffer);
        let Some(vertex_buffer) = vertex_buffer else {
            return;
        };
        if self.dump.active() {
            let mut dump = std::mem::take(&mut self.dump);
            let bias = if gl.polygon_offset {
                gl.offset
            } else {
                (0.0, 0.0)
            };
            let shape = DumpShape {
                vertices,
                indices,
                key,
                first,
                count,
                viewport,
                scissor,
                bias,
            };
            dump_draw(self, &mut dump, &shape, &blocks, &dump_samplers);
            self.dump = dump;
        }
        let rec = self.recording.as_mut().unwrap();
        unsafe {
            if rec.pipeline != pipeline {
                d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
                rec.pipeline = pipeline;
            }
            if rec.viewport != Some(vp) {
                d.cmd_set_viewport(cmd, 0, &[viewport]);
                rec.viewport = Some(vp);
            }
            if rec.scissor != Some(scissor) {
                d.cmd_set_scissor(cmd, 0, &[scissor]);
                rec.scissor = Some(scissor);
            }
            let bias = if gl.polygon_offset {
                gl.offset
            } else {
                (0.0, 0.0)
            };
            if rec.bias != Some(bias) {
                d.cmd_set_depth_bias(cmd, bias.1, 0.0, bias.0);
                rec.bias = Some(bias);
            }
            d.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline_layout,
                0,
                &[set],
                &dynamic_offsets[..dynamic_count],
            );
            d.cmd_push_constants(
                cmd,
                pipeline_layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                &(-1.0f32).to_ne_bytes(),
            );
            d.cmd_bind_vertex_buffers(cmd, 0, &[vertex_buffer, zero], &[0, 0]);
            match index_buffer {
                Some(ib) => {
                    d.cmd_bind_index_buffer(cmd, ib, 0, vk::IndexType::UINT32);
                    d.cmd_draw_indexed(cmd, count, 1, first, 0, 0);
                }
                None => d.cmd_draw(cmd, count, 1, first, 0),
            }
        }
    }
}

/// What [`dump_draw`] needs of a draw beyond the backend's state.
struct DumpShape {
    vertices: u32,
    indices: Option<u32>,
    key: PipelineKey,
    first: u32,
    count: u32,
    viewport: vk::Viewport,
    scissor: vk::Rect2D,
    bias: (f32, f32),
}

/// One draw into the draw dump, as [`State::pipeline`] and the draw issue it.
fn dump_draw(
    s: &State,
    dump: &mut super::draw_dump::DrawDump,
    d: &DumpShape,
    blocks: &[Vec<u8>; 2],
    samplers: &[[u32; 7]],
) {
    let key = d.key;
    let program = &s.programs[key.program as usize];
    let layout = &s.layouts[key.layout as usize];
    dump.program(key.program, &program.spirv, &program.layout_bindings);
    let attributes: Vec<[u32; 4]> = program
        .vertex_inputs
        .iter()
        .map(
            |(&location, &(int, _))| match layout.attributes.get(location as usize) {
                Some(a) => [
                    location,
                    0,
                    attribute_format(a.kind == super::AttributeKind::Int, a.components).as_raw()
                        as u32,
                    a.offset,
                ],
                None => [location, 1, attribute_format(int, 4).as_raw() as u32, 0],
            },
        )
        .collect();
    let empty = Vec::new();
    let bytes = s.dump_bytes.get(&d.vertices).unwrap_or(&empty);
    let stride = layout.stride as usize;
    let (vertices, indices): (&[u8], &[u8]) = match d.indices {
        Some(i) => {
            let index_bytes = s.dump_bytes.get(&i).unwrap_or(&empty);
            let from = (d.first as usize * 4).min(index_bytes.len());
            let to = ((d.first + d.count) as usize * 4).min(index_bytes.len());
            (bytes, &index_bytes[from..to])
        }
        None => {
            let from = (d.first as usize * stride).min(bytes.len());
            let to = ((d.first + d.count) as usize * stride).min(bytes.len());
            (&bytes[from..to], &[])
        }
    };
    if bytes.is_empty() {
        tracing::warn!("draw dump: no bytes kept for buffer {}", d.vertices);
    }
    let factors = key.blend.unwrap_or([
        BlendFactor::One,
        BlendFactor::Zero,
        BlendFactor::One,
        BlendFactor::Zero,
    ]);
    let mask = key
        .color_mask
        .iter()
        .enumerate()
        .fold(0, |m, (i, on)| m | ((*on as u32) << i));
    let v = d.viewport;
    let r = d.scissor;
    dump.draw(&super::draw_dump::Draw {
        program: key.program,
        topology: match key.topology {
            Topology::Triangles => 0,
            Topology::Lines => 1,
        },
        count: d.count,
        stride: layout.stride,
        attributes: &attributes,
        cull: match key.cull {
            Some(face) => [
                vk::CullModeFlags::BACK.as_raw(),
                match face {
                    FrontFace::Clockwise => vk::FrontFace::CLOCKWISE,
                    _ => vk::FrontFace::COUNTER_CLOCKWISE,
                }
                .as_raw() as u32,
            ],
            None => [0, vk::FrontFace::COUNTER_CLOCKWISE.as_raw() as u32],
        },
        depth_test: key.depth_test,
        depth_write: key.depth_write,
        depth_compare: match key.depth_func {
            DepthFunc::Less => vk::CompareOp::LESS,
            DepthFunc::LessEqual => vk::CompareOp::LESS_OR_EQUAL,
            DepthFunc::Always => vk::CompareOp::ALWAYS,
        }
        .as_raw() as u32,
        blend: key.blend.is_some(),
        blend_factors: factors.map(|f| blend_factor(f).as_raw() as u32),
        color_mask: mask,
        depth_bias: key.depth_bias,
        bias: [d.bias.1, d.bias.0],
        formats: [key.formats.0 as u32, key.formats.1 as u32, key.samples],
        viewport: [v.x, v.y, v.width, v.height, v.min_depth, v.max_depth],
        scissor: [
            r.offset.x,
            r.offset.y,
            r.extent.width as i32,
            r.extent.height as i32,
        ],
        vertices,
        indices,
        blocks: [&blocks[0], &blocks[1]],
        samplers,
    });
}

fn clip_rect(r: [i32; 4], extent: vk::Extent2D) -> vk::Rect2D {
    let (w, h) = (extent.width as i32, extent.height as i32);
    let x0 = r[0].clamp(0, w);
    let x1 = (r[0] + r[2]).clamp(0, w);
    let top = (h - r[1] - r[3]).clamp(0, h);
    let bottom = (h - r[1]).clamp(0, h);
    vk::Rect2D {
        offset: vk::Offset2D { x: x0, y: top },
        extent: vk::Extent2D {
            width: (x1 - x0) as u32,
            height: (bottom - top) as u32,
        },
    }
}

fn to_rgba(format: PixelFormat, pixels: &[u8]) -> Vec<u8> {
    match format {
        PixelFormat::Rgba8 => pixels.to_vec(),
        PixelFormat::Rgb8 => pixels
            .chunks_exact(3)
            .flat_map(|p| [p[0], p[1], p[2], 255])
            .collect(),
    }
}

fn attribute_format(int: bool, components: u32) -> vk::Format {
    // The Xbox 360 fetches no integer formats; its integers are floats
    // (`gfx::ints_as_floats`).
    let int = int && !cfg!(target_os = "xbox360");
    match (int, components) {
        (false, 1) => vk::Format::R32_SFLOAT,
        (false, 2) => vk::Format::R32G32_SFLOAT,
        (false, 3) => vk::Format::R32G32B32_SFLOAT,
        (false, _) => vk::Format::R32G32B32A32_SFLOAT,
        (true, 1) => vk::Format::R32_SINT,
        (true, 2) => vk::Format::R32G32_SINT,
        (true, 3) => vk::Format::R32G32B32_SINT,
        (true, _) => vk::Format::R32G32B32A32_SINT,
    }
}

fn blend_factor(f: BlendFactor) -> vk::BlendFactor {
    match f {
        BlendFactor::Zero => vk::BlendFactor::ZERO,
        BlendFactor::One => vk::BlendFactor::ONE,
        BlendFactor::SrcColor => vk::BlendFactor::SRC_COLOR,
        BlendFactor::OneMinusSrcColor => vk::BlendFactor::ONE_MINUS_SRC_COLOR,
        BlendFactor::DstColor => vk::BlendFactor::DST_COLOR,
        BlendFactor::OneMinusDstColor => vk::BlendFactor::ONE_MINUS_DST_COLOR,
        BlendFactor::SrcAlpha => vk::BlendFactor::SRC_ALPHA,
        BlendFactor::OneMinusSrcAlpha => vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
        BlendFactor::DstAlpha => vk::BlendFactor::DST_ALPHA,
        BlendFactor::OneMinusDstAlpha => vk::BlendFactor::ONE_MINUS_DST_ALPHA,
    }
}

fn range(base_mip: u32, mips: u32) -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: base_mip,
        level_count: mips,
        base_array_layer: 0,
        layer_count: vk::REMAINING_ARRAY_LAYERS,
    }
}

fn layers(mip: u32, count: u32) -> vk::ImageSubresourceLayers {
    vk::ImageSubresourceLayers {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        mip_level: mip,
        base_array_layer: 0,
        layer_count: count,
    }
}

unsafe fn barrier(
    d: &ash::Device,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    range: vk::ImageSubresourceRange,
    from: vk::ImageLayout,
    to: vk::ImageLayout,
) {
    let b = vk::ImageMemoryBarrier::builder()
        .src_access_mask(vk::AccessFlags::MEMORY_WRITE | vk::AccessFlags::SHADER_READ)
        .dst_access_mask(
            vk::AccessFlags::TRANSFER_READ
                | vk::AccessFlags::TRANSFER_WRITE
                | vk::AccessFlags::SHADER_READ,
        )
        .old_layout(from)
        .new_layout(to)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(range)
        .build();
    unsafe {
        d.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[b],
        );
    }
}

// ---------------------------------------------------------------------------
// The gfx entry points
// ---------------------------------------------------------------------------

pub fn create_program(vertex: &str, fragment: &str) -> u32 {
    let (vertex_source, fragment_source) = (vertex, fragment);
    let source_hash = {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        (vertex, fragment).hash(&mut hasher);
        hasher.finish()
    };
    let vs = shader::compile(vertex, shader::Stage::Vertex)
        .unwrap_or_else(|e| panic!("vertex shader: {e}"));
    let fs = shader::compile(fragment, shader::Stage::Fragment)
        .unwrap_or_else(|e| panic!("fragment shader: {e}"));
    let linked = shader::link(vs, fs);
    with_state(|s| {
        let d = &s.dev.device;
        let module = |code: &[u32]| unsafe {
            d.create_shader_module(&vk::ShaderModuleCreateInfo::builder().code(code), None)
                .expect("vkCreateShaderModule")
        };
        shader::record_program(vertex, fragment, &linked);
        shader::record_console_program(vertex, fragment, true);
        shader::record_module(&linked.vertex, shader::Stage::Vertex);
        shader::record_module(&linked.fragment, shader::Stage::Fragment);
        let vertex = module(&linked.vertex);
        let fragment = module(&linked.fragment);

        let mut slots: Vec<Slot> = Vec::new();
        let mut names: HashMap<String, u32> = HashMap::default();
        let mut slot_for = |name: &str, slots: &mut Vec<Slot>| -> usize {
            *names.entry(name.to_string()).or_insert_with(|| {
                slots.push(Slot::default());
                slots.len() as u32 - 1
            }) as usize
        };
        let mut blocks: [Vec<u8>; 2] = [Vec::new(), Vec::new()];
        let mut set_bindings = Vec::new();
        for (index, block) in [&linked.vertex_block, &linked.fragment_block]
            .into_iter()
            .enumerate()
        {
            if let Some((size, members)) = block
                && *size > 0
            {
                blocks[index] = vec![0; *size as usize];
                for (name, member) in members {
                    // SHOCK_COUNT's scale, which stays 1 (shader::to_vulkan_glsl_for).
                    if name == shader::LOOP_SCALE
                        && let Some(dst) = blocks[index]
                            .get_mut(member.offset as usize..member.offset as usize + 4)
                    {
                        dst.copy_from_slice(&1.0f32.to_ne_bytes());
                    }
                    let i = slot_for(name, &mut slots);
                    slots[i].members.push((index, *member));
                }
                set_bindings.push(
                    vk::DescriptorSetLayoutBinding::builder()
                        .binding(index as u32)
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                        .descriptor_count(1)
                        .stage_flags(if index == 0 {
                            vk::ShaderStageFlags::VERTEX
                        } else {
                            vk::ShaderStageFlags::FRAGMENT
                        })
                        .build(),
                );
            }
        }
        let mut samplers = Vec::new();
        for (name, use_) in &linked.samplers {
            let binding = use_.binding;
            let mut stages = vk::ShaderStageFlags::empty();
            if use_.vertex {
                stages |= vk::ShaderStageFlags::VERTEX;
            }
            if use_.fragment {
                stages |= vk::ShaderStageFlags::FRAGMENT;
            }
            let i = slot_for(name, &mut slots);
            slots[i].sampler = Some(samplers.len());
            samplers.push(SamplerBinding {
                binding,
                unit: 0,
                cube: use_.cube,
            });
            set_bindings.push(
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(binding)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(1)
                    .stage_flags(stages)
                    .build(),
            );
        }
        let set_layout = unsafe {
            d.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::builder().bindings(&set_bindings),
                None,
            )
        }
        .expect("vkCreateDescriptorSetLayout");
        let push = [vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::VERTEX,
            offset: 0,
            size: 4,
        }];
        let set_layouts = [set_layout];
        let layout = unsafe {
            d.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::builder()
                    .set_layouts(&set_layouts)
                    .push_constant_ranges(&push),
                None,
            )
        }
        .expect("vkCreatePipelineLayout");
        s.programs.push(ProgramObj {
            vertex,
            fragment,
            set_layout,
            layout,
            blocks,
            slots,
            names,
            samplers,
            vertex_inputs: linked.vertex_inputs,
            drawn: false,
            spirv: if cfg!(target_os = "xbox360") {
                [linked.vertex.clone(), linked.fragment.clone()]
            } else {
                [Vec::new(), Vec::new()]
            },
            layout_bindings: if cfg!(target_os = "xbox360") {
                set_bindings
                    .iter()
                    .map(|b| {
                        [
                            b.binding,
                            b.descriptor_type.as_raw() as u32,
                            b.stage_flags.as_raw(),
                        ]
                    })
                    .collect()
            } else {
                Vec::new()
            },
        });
        let id = s.programs.len() as u32 - 1;
        tracing::debug!(
            "program {id}: {source_hash:016x} (vertex {:016x}, fragment {:016x})",
            shader::source_hash(vertex_source),
            shader::source_hash(fragment_source)
        );
        id
    })
}

pub const NO_UNIFORM: u32 = u32::MAX;

/// Draws per program since the last call, as `program:count` pairs.
pub fn take_draw_counts() -> String {
    with(|s| {
        let draws = std::mem::take(&mut s.draw_counts)
            .into_iter()
            .map(|(program, count)| format!("{program}:{count}"))
            .collect::<Vec<_>>()
            .join(" ");
        let uploads = std::mem::take(&mut s.upload_counts)
            .into_iter()
            .map(|((w, h, mips), count)| format!("{count}x {w}x{h}/{mips}"))
            .collect::<Vec<_>>()
            .join(" ");
        format!("{draws}; uploads {uploads}")
    })
    .unwrap_or_default()
}

/// Programs whose draws are skipped: on the Xbox 360, the numbers in
/// `game:\\skip_programs.txt` (numbers and `a-b` ranges), to find a draw
/// that hangs its GPU. Empty elsewhere.
fn skipped_programs() -> &'static [u32] {
    static SKIPPED: std::sync::OnceLock<Vec<u32>> = std::sync::OnceLock::new();
    SKIPPED.get_or_init(|| {
        if !cfg!(target_os = "xbox360") {
            return Vec::new();
        }
        let text = std::fs::read_to_string(r"game:\skip_programs.txt").unwrap_or_default();
        let skipped = parse_program_list(&text);
        if !skipped.is_empty() {
            tracing::info!("skipping draws of programs {skipped:?}");
        }
        skipped
    })
}

/// `max_draws.txt` beside the console's `.xex`: draw only the first N draws
/// of each frame, and log each, to find a draw that hangs the GPU.
/// `n` keeps draws 0..n, `a-b` keeps draws a..=b.
/// `sampler.txt` beside the console's `.xex`, to look for a GPU hang in
/// texture sampling: `noaniso`, `nearestmip` and `maxlod=<f>`, by whitespace.
/// A console diagnostic: nothing reads it elsewhere.
#[derive(Default, Clone, Copy)]
struct SamplerKnobs {
    no_aniso: bool,
    nearest_mip: bool,
    max_lod: Option<f32>,
    max_verts: Option<u32>,
    no_big: bool,
}

fn sampler_knobs() -> SamplerKnobs {
    static KNOBS: std::sync::OnceLock<SamplerKnobs> = std::sync::OnceLock::new();
    *KNOBS.get_or_init(|| {
        if !cfg!(target_os = "xbox360") {
            return SamplerKnobs::default();
        }
        let text = std::fs::read_to_string(r"game:\sampler.txt").unwrap_or_default();
        let mut knobs = SamplerKnobs::default();
        for word in text.split_whitespace() {
            match word {
                "noaniso" => knobs.no_aniso = true,
                "nearestmip" => knobs.nearest_mip = true,
                "nobig" => knobs.no_big = true,
                _ if word.starts_with("maxverts=") => knobs.max_verts = word[9..].parse().ok(),
                _ => {
                    knobs.max_lod = word
                        .strip_prefix("maxlod=")
                        .and_then(|v| v.parse().ok())
                        .or(knobs.max_lod)
                }
            }
        }
        tracing::info!(
            "sampler knobs: noaniso {} nearestmip {} maxlod {:?} maxverts {:?} nobig {}",
            knobs.no_aniso,
            knobs.nearest_mip,
            knobs.max_lod,
            knobs.max_verts,
            knobs.no_big
        );
        knobs
    })
}

fn max_draws() -> Option<(u32, u32)> {
    static MAX: std::sync::OnceLock<Option<(u32, u32)>> = std::sync::OnceLock::new();
    *MAX.get_or_init(|| {
        if !cfg!(target_os = "xbox360") {
            return None;
        }
        let text = std::fs::read_to_string(r"game:\max_draws.txt").ok()?;
        let text = text.trim();
        let range = match text.split_once('-') {
            Some((a, b)) => Some((a.parse().ok()?, b.parse::<u32>().ok()? + 1)),
            None => text.parse().ok().map(|n| (0, n)),
        };
        if let Some((min, max)) = range {
            tracing::info!("drawing only draws {min}..{max} of each frame");
        }
        range
    })
}

fn parse_program_list(text: &str) -> Vec<u32> {
    text.split(|c: char| c == ',' || c.is_whitespace())
        .filter_map(|item| match item.split_once('-') {
            Some((a, b)) => Some(a.parse().ok()?..=b.parse().ok()?),
            None => item.parse().ok().map(|n| n..=n),
        })
        .flatten()
        .collect()
}

/// A uniform handle: the name's slot, and in the top 8 bits the array
/// element it starts at (GL's `name[3]` locations).
pub fn uniform(program: u32, name: &str) -> u32 {
    let (base, element) = match name.strip_suffix(']').and_then(|n| n.rsplit_once('[')) {
        Some((base, index)) => match index.parse::<u32>() {
            Ok(i) if i < 256 => (base, i),
            _ => return NO_UNIFORM,
        },
        None => (name, 0),
    };
    with(|s| s.programs[program as usize].names.get(base).copied())
        .flatten()
        .map_or(NO_UNIFORM, |slot| slot | (element << 24))
}

pub fn bind_program(program: u32) {
    with(|s| s.program = Some(program));
}

/// Write `values` (32-bit words, `columns` columns of `rows` each per
/// element, `elements` elements) into the bound program's uniform.
/// Write `values` (each turned into its word by `word`) into the bound
/// program's blocks. No allocation: a frame makes thousands of these.
fn write_uniform<T: Copy>(
    uniform: u32,
    values: &[T],
    word: impl Fn(T) -> u32,
    rows: usize,
    columns: usize,
) {
    if uniform == NO_UNIFORM {
        return;
    }
    let words = values;
    with(|s| {
        let Some(program) = s.program else { return };
        let p = &mut s.programs[program as usize];
        let first = (uniform >> 24) as usize;
        let Some(slot) = p.slots.get((uniform & 0xff_ffff) as usize) else {
            return;
        };
        for &(block, m) in &slot.members {
            let bytes = &mut p.blocks[block];
            let per_element = rows * columns;
            let room = (m.count as usize).saturating_sub(first);
            for (e, element) in words.chunks(per_element).enumerate().take(room) {
                let base = m.offset as usize + (first + e) * m.array_stride as usize;
                for c in 0..columns {
                    let at = base + c * m.matrix_stride as usize;
                    for r in 0..rows {
                        let o = at + r * 4;
                        if let (Some(dst), Some(v)) =
                            (bytes.get_mut(o..o + 4), element.get(c * rows + r))
                        {
                            dst.copy_from_slice(&word(*v).to_ne_bytes());
                        }
                    }
                }
            }
        }
    });
    stat(|s| s.uniform_sets += 1);
}

pub fn set_floats(uniform: u32, values: &[f32], components: usize) {
    write_uniform(uniform, values, f32::to_bits, components, 1);
}

pub fn set_matrices(uniform: u32, values: &[f32], size: usize) {
    write_uniform(uniform, values, f32::to_bits, size, size);
}

/// An int or bool uniform's word. On the Xbox 360 its shaders declare these
/// as floats (`shader::Target::Xbox360`), so the value goes in as one.
fn int_word(i: i32) -> u32 {
    if cfg!(target_os = "xbox360") {
        (i as f32).to_bits()
    } else {
        i as u32
    }
}

pub fn set_ivec(uniform: u32, values: &[i32], components: usize) {
    write_uniform(uniform, values, int_word, components, 1);
}

pub fn set_mat3x4(uniform: u32, values: &[f32]) {
    write_uniform(uniform, values, f32::to_bits, 4, 3);
}

pub fn set_ints(uniform: u32, values: &[i32]) {
    if uniform == NO_UNIFORM {
        return;
    }
    // A sampler's value is the texture unit it reads.
    let sampler = with(|s| {
        let program = s.program?;
        let p = &mut s.programs[program as usize];
        let i = p.slots.get((uniform & 0xff_ffff) as usize)?.sampler?;
        p.samplers[i].unit = *values.first()? as u32;
        Some(())
    })
    .flatten();
    if sampler.is_none() {
        write_uniform(uniform, values, int_word, 1, 1);
    }
}

pub fn set_cap(cap: Cap, on: bool) {
    with(|s| match cap {
        Cap::DepthTest => s.gl.depth_test = on,
        Cap::Blend => s.gl.blend = on,
        Cap::CullFace => s.gl.cull = on,
        Cap::PolygonOffsetFill => s.gl.polygon_offset = on,
        Cap::ScissorTest => s.gl.scissor_test = on,
    });
}

pub fn depth_mask(write: bool) {
    with(|s| s.gl.depth_write = write);
}
pub fn depth_func(func: DepthFunc) {
    with(|s| s.gl.depth_func = func);
}
pub fn blend_func_separate(factors: [BlendFactor; 4]) {
    with(|s| s.gl.blend_factors = factors);
}
pub fn front_face(face: FrontFace) {
    with(|s| s.gl.front_face = face);
}
pub fn polygon_offset(factor: f32, units: f32) {
    with(|s| s.gl.offset = (factor, units));
}
pub fn color_mask(mask: [bool; 4]) {
    with(|s| s.gl.color_mask = mask);
}
pub fn scissor(rect: [i32; 4]) {
    with(|s| s.gl.scissor = rect);
}

/// The GL-style state, kept to be put back with [`restore_state`] after a
/// draw that changes it (the Xbox 360's lamp-shadow pass).
pub struct SavedState(GlState);

pub fn save_state() -> Option<SavedState> {
    with(|s| SavedState(s.gl))
}

pub fn restore_state(saved: SavedState) {
    with(|s| s.gl = saved.0);
}

/// The viewport draws go to (GL's, bottom-left origin): the frame's whole
/// extent unless one was set.
pub fn current_viewport() -> Option<[i32; 4]> {
    with(|s| {
        s.gl.viewport.or_else(|| {
            s.recording
                .as_ref()
                .map(|r| [0, 0, r.extent.width as i32, r.extent.height as i32])
        })
    })
    .flatten()
}
pub fn viewport(rect: [i32; 4]) {
    with(|s| s.gl.viewport = Some(rect));
}
pub fn clear_color(rgba: [f32; 4]) {
    with(|s| s.gl.clear_color = rgba);
}

/// glClear inside the render pass: the scissor rect when scissoring, else
/// the whole framebuffer.
pub fn clear(color: bool, depth: bool) {
    with(|s| {
        let Some(rec) = s.recording.as_ref() else {
            return;
        };
        let rect = if s.gl.scissor_test {
            clip_rect(s.gl.scissor, rec.extent)
        } else {
            vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent: rec.extent,
            }
        };
        let mut attachments = Vec::new();
        if color {
            attachments.push(vk::ClearAttachment {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                color_attachment: 0,
                clear_value: vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: s.gl.clear_color,
                    },
                },
            });
        }
        if depth {
            attachments.push(vk::ClearAttachment {
                aspect_mask: vk::ImageAspectFlags::DEPTH,
                color_attachment: 0,
                clear_value: vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: 0,
                    },
                },
            });
        }
        if attachments.is_empty() || rect.extent.width == 0 || rect.extent.height == 0 {
            return;
        }
        let clear_color = s.gl.clear_color;
        s.dump.clear(
            color,
            depth,
            clear_color,
            [
                rect.offset.x,
                rect.offset.y,
                rect.extent.width as i32,
                rect.extent.height as i32,
            ],
        );
        unsafe {
            s.dev.device.cmd_clear_attachments(
                rec.commands,
                &attachments,
                &[vk::ClearRect {
                    rect,
                    base_array_layer: 0,
                    layer_count: 1,
                }],
            );
        }
    });
}

/// Pause the frame's render pass and capture depth only, into the self-depth
/// image, at the size of the current viewport; GL's "bind a depth FBO and
/// draw". Returns where that viewport's corner is in fragment coordinates
/// (Vulkan's are top-left, so a mask lines up with `gl_FragCoord`).
pub fn begin_depth_capture() -> Option<[i32; 2]> {
    with(|s| s.begin_depth_capture(SELF_DEPTH_TARGET, None, false)).flatten()
}

/// Pause the frame's render pass and capture depth only into the persistent
/// shadow atlas of `size` texels. The viewport starts as the whole atlas;
/// draws pick their tile with [`viewport`]. With `keep` the atlas holds what
/// earlier frames drew (clear tiles with a scissored [`clear`]); without it,
/// or when the atlas was just (re)made, it starts at the far plane. Returns
/// whether the old contents were kept.
pub fn begin_shadow_capture(size: [u32; 2], keep: bool) -> Option<bool> {
    with(|s| {
        // An atlas per frame slot on the Xbox 360, which keeps no rows from
        // frame to frame: drawing into the one the frame before still
        // samples made vulkan4xenon wait out that frame's GPU work, so the
        // CPU and GPU stopped overlapping (18 fps where 42 without shadows).
        let target = match &s.recording {
            Some(rec) if cfg!(target_os = "xbox360") && !keep => SHADOW_TARGET + rec.slot % 2,
            _ => SHADOW_TARGET,
        };
        s.begin_depth_capture(
            target,
            Some(vk::Extent2D {
                width: size[0].max(1),
                height: size[1].max(1),
            }),
            keep,
        )
        .map(|_| s.capture_kept)
    })
    .flatten()
}

/// Finish a capture and resume the frame. Returns the depth image's texture
/// id, for [`bind_texture_id`].
pub fn end_depth_capture() -> Option<u32> {
    with(|s| s.end_depth_capture()).flatten()
}

/// Bind a backend texture by id (the self-depth image) to a 2D unit.
pub fn bind_texture_id(unit: u32, id: Option<u32>) {
    with(|s| {
        if let Some(u) = s.units.get_mut(unit as usize) {
            *u = id;
        }
    });
}

impl State {
    fn begin_depth_capture(
        &mut self,
        target: usize,
        fixed_size: Option<vk::Extent2D>,
        keep: bool,
    ) -> Option<[i32; 2]> {
        // Only shadow maps on the Xbox 360: nothing there samples a self
        // depth mask (no VR held items), and its capture costs a pass.
        if cfg!(target_os = "xbox360") && target == SELF_DEPTH_TARGET {
            return None;
        }
        let rec = self.recording.as_ref()?;
        if rec.paused.is_some() {
            return None;
        }
        let (cmd, frame_extent) = (rec.commands, rec.extent);
        let h = frame_extent.height as i32;
        let vp = self
            .gl
            .viewport
            .unwrap_or([0, 0, frame_extent.width as i32, h]);
        let size = fixed_size.unwrap_or(vk::Extent2D {
            width: vp[2].max(1) as u32,
            height: vp[3].max(1) as u32,
        });
        let clear_pass = self.depth_capture_pass();
        let existed = self
            .depth_capture
            .as_ref()
            .and_then(|c| c.images[target])
            .is_some_and(|(_, _, extent)| extent == size);
        let (id, framebuffer) = self.depth_capture_image(target, clear_pass, size);
        self.capture_kept = keep && existed;
        let pass = match self.depth_capture.as_mut() {
            Some(c) => {
                c.active = target;
                if self.capture_kept {
                    c.load_pass
                } else {
                    c.pass
                }
            }
            None => clear_pass,
        };
        // Never sample the image being drawn into.
        for unit in self.units.iter_mut() {
            if *unit == Some(id) {
                *unit = None;
            }
        }
        let depth_clear = vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth: 1.0,
                stencil: 0,
            },
        };
        let clears = [vk::ClearValue::default(), depth_clear];
        let clears = if CAPTURE_COLOR.is_some() {
            &clears[..]
        } else {
            &clears[1..]
        };
        unsafe {
            let d = &self.dev.device;
            d.cmd_end_render_pass(cmd);
            d.cmd_begin_render_pass(
                cmd,
                &vk::RenderPassBeginInfo::builder()
                    .render_pass(pass)
                    .framebuffer(framebuffer)
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D::default(),
                        extent: size,
                    })
                    .clear_values(clears),
                vk::SubpassContents::INLINE,
            );
        }
        self.dump
            .capture_begin(size.width, size.height, self.capture_kept);
        let gl = self.gl;
        let rec = self.recording.as_mut().unwrap();
        rec.paused = Some((rec.render_pass, rec.formats, rec.samples, rec.extent, gl));
        rec.render_pass = pass;
        rec.formats = (
            CAPTURE_COLOR.unwrap_or(vk::Format::UNDEFINED).as_raw(),
            CAPTURE_DEPTH.as_raw(),
        );
        rec.samples = vk::SampleCountFlags::TYPE_1;
        rec.extent = size;
        rec.pipeline = vk::Pipeline::null();
        rec.viewport = None;
        rec.scissor = None;
        rec.bias = None;
        // The GL path's state for the capture: depth only, nearest wins.
        self.gl.viewport = Some([0, 0, size.width as i32, size.height as i32]);
        self.gl.scissor_test = false;
        self.gl.depth_test = true;
        self.gl.depth_write = true;
        self.gl.depth_func = DepthFunc::Less;
        self.gl.color_mask = [false; 4];
        Some([vp[0], h - vp[1] - vp[3]])
    }

    fn end_depth_capture(&mut self) -> Option<u32> {
        let rec = self.recording.as_mut()?;
        let (pass, formats, samples, extent, gl) = rec.paused.take()?;
        rec.render_pass = pass;
        rec.formats = formats;
        rec.samples = samples;
        rec.extent = extent;
        rec.pipeline = vk::Pipeline::null();
        rec.viewport = None;
        rec.scissor = None;
        rec.bias = None;
        let (cmd, resume, framebuffer) = (rec.commands, rec.resume_pass, rec.framebuffer);
        self.gl = gl;
        self.dump.capture_end();
        unsafe {
            let d = &self.dev.device;
            d.cmd_end_render_pass(cmd);
            d.cmd_begin_render_pass(
                cmd,
                &vk::RenderPassBeginInfo::builder()
                    .render_pass(resume)
                    .framebuffer(framebuffer)
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D::default(),
                        extent,
                    }),
                vk::SubpassContents::INLINE,
            );
        }
        self.depth_capture
            .as_ref()
            .and_then(|c| c.images[c.active])
            .map(|(id, _, _)| id)
    }

    fn depth_capture_pass(&mut self) -> vk::RenderPass {
        if let Some(c) = &self.depth_capture {
            return c.pass;
        }
        let pass = self.make_capture_pass(false);
        let load_pass = self.make_capture_pass(true);
        self.depth_capture = Some(DepthCapture {
            pass,
            load_pass,
            images: [None; CAPTURE_TARGETS],
            colors: [None; CAPTURE_TARGETS],
            active: SELF_DEPTH_TARGET,
        });
        pass
    }

    /// The depth-only capture pass: clearing, or (`load`) keeping the
    /// image's contents from the last capture, which left it shader-readable.
    fn make_capture_pass(&self, load: bool) -> vk::RenderPass {
        let (load_op, initial) = if load {
            (
                vk::AttachmentLoadOp::LOAD,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            )
        } else {
            (vk::AttachmentLoadOp::CLEAR, vk::ImageLayout::UNDEFINED)
        };
        let depth = vk::AttachmentDescription::builder()
            .format(CAPTURE_DEPTH)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(load_op)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(initial)
            .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .build();
        let color = vk::AttachmentDescription::builder()
            .format(CAPTURE_COLOR.unwrap_or(vk::Format::UNDEFINED))
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::DONT_CARE)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .build();
        let with_color = [color, depth];
        let attachments = if CAPTURE_COLOR.is_some() {
            &with_color[..]
        } else {
            &with_color[1..]
        };
        let depth_ref = vk::AttachmentReference {
            attachment: u32::from(CAPTURE_COLOR.is_some()),
            layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        };
        let color_ref = [vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        }];
        let mut subpass = vk::SubpassDescription::builder()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .depth_stencil_attachment(&depth_ref);
        if CAPTURE_COLOR.is_some() {
            subpass = subpass.color_attachments(&color_ref);
        }
        let subpass = [subpass.build()];
        let tests = vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
            | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS;
        let dependencies = [
            // An earlier mask's sampling finishes before this one clears it.
            vk::SubpassDependency::builder()
                .src_subpass(vk::SUBPASS_EXTERNAL)
                .dst_subpass(0)
                .src_stage_mask(vk::PipelineStageFlags::FRAGMENT_SHADER | tests)
                .dst_stage_mask(tests)
                .src_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE)
                .dst_access_mask(
                    vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                        | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                )
                .build(),
            // The mask is sampled by the draws that follow.
            vk::SubpassDependency::builder()
                .src_subpass(0)
                .dst_subpass(vk::SUBPASS_EXTERNAL)
                .src_stage_mask(tests)
                .dst_stage_mask(vk::PipelineStageFlags::FRAGMENT_SHADER)
                .src_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .build(),
        ];
        unsafe {
            self.dev.device.create_render_pass(
                &vk::RenderPassCreateInfo::builder()
                    .attachments(attachments)
                    .subpasses(&subpass)
                    .dependencies(&dependencies),
                None,
            )
        }
        .expect("vkCreateRenderPass")
    }

    /// The scratch image and framebuffer at `size`, rebuilt when it changes.
    fn depth_capture_image(
        &mut self,
        target: usize,
        pass: vk::RenderPass,
        size: vk::Extent2D,
    ) -> (u32, vk::Framebuffer) {
        if let Some((id, framebuffer, extent)) =
            self.depth_capture.as_ref().and_then(|c| c.images[target])
        {
            if extent == size {
                return (id, framebuffer);
            }
            if let Some(i) = self.images.remove(&id) {
                self.retire(Garbage::Image(i.image, i.view, i.alloc));
            }
            if let Some(color) = self
                .depth_capture
                .as_mut()
                .and_then(|c| c.colors[target].take())
                && let Some(i) = self.images.remove(&color)
            {
                self.retire(Garbage::Image(i.image, i.view, i.alloc));
            }
            self.retire(Garbage::Framebuffer(framebuffer));
        }
        let d = self.dev.device.clone();
        let image = unsafe {
            d.create_image(
                &vk::ImageCreateInfo::builder()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(CAPTURE_DEPTH)
                    .extent(vk::Extent3D {
                        width: size.width,
                        height: size.height,
                        depth: 1,
                    })
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(
                        vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                            | vk::ImageUsageFlags::SAMPLED,
                    )
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .initial_layout(vk::ImageLayout::UNDEFINED),
                None,
            )
        }
        .expect("vkCreateImage");
        let req = unsafe { d.get_image_memory_requirements(image) };
        let alloc = self
            .alloc
            .alloc(&self.dev, req, vk::MemoryPropertyFlags::DEVICE_LOCAL, false);
        unsafe { d.bind_image_memory(image, alloc.memory, alloc.offset) }
            .expect("vkBindImageMemory");
        let view = unsafe {
            d.create_image_view(
                &vk::ImageViewCreateInfo::builder()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(CAPTURE_DEPTH)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::DEPTH,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    }),
                None,
            )
        }
        .expect("vkCreateImageView");
        let color = CAPTURE_COLOR.map(|format| self.capture_color_image(format, size));
        let with_color = [color.map_or(vk::ImageView::null(), |(_, v)| v), view];
        let attachments = if color.is_some() {
            &with_color[..]
        } else {
            &with_color[1..]
        };
        let framebuffer = unsafe {
            d.create_framebuffer(
                &vk::FramebufferCreateInfo::builder()
                    .render_pass(pass)
                    .attachments(attachments)
                    .width(size.width)
                    .height(size.height)
                    .layers(1),
                None,
            )
        }
        .expect("vkCreateFramebuffer");
        let id = self.id();
        self.images.insert(
            id,
            ImageObj {
                image,
                view,
                alloc,
                sampling: Sampling {
                    filter: Filter::Nearest,
                    wrap: false,
                },
                mips: 1,
                layers: 1,
                size: (size.width, size.height),
            },
        );
        if let Some(c) = self.depth_capture.as_mut() {
            c.images[target] = Some((id, framebuffer, size));
            c.colors[target] = color.map(|(id, _)| id);
        }
        (id, framebuffer)
    }

    /// A capture's colour image ([`CAPTURE_COLOR`]): its id and view.
    fn capture_color_image(
        &mut self,
        format: vk::Format,
        size: vk::Extent2D,
    ) -> (u32, vk::ImageView) {
        let d = self.dev.device.clone();
        let image = unsafe {
            d.create_image(
                &vk::ImageCreateInfo::builder()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(format)
                    .extent(vk::Extent3D {
                        width: size.width,
                        height: size.height,
                        depth: 1,
                    })
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .initial_layout(vk::ImageLayout::UNDEFINED),
                None,
            )
        }
        .expect("vkCreateImage");
        let req = unsafe { d.get_image_memory_requirements(image) };
        let alloc = self
            .alloc
            .alloc(&self.dev, req, vk::MemoryPropertyFlags::DEVICE_LOCAL, false);
        unsafe { d.bind_image_memory(image, alloc.memory, alloc.offset) }
            .expect("vkBindImageMemory");
        let view = unsafe {
            d.create_image_view(
                &vk::ImageViewCreateInfo::builder()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    }),
                None,
            )
        }
        .expect("vkCreateImageView");
        let id = self.id();
        self.images.insert(
            id,
            ImageObj {
                image,
                view,
                alloc,
                sampling: Sampling {
                    filter: Filter::Nearest,
                    wrap: false,
                },
                mips: 1,
                layers: 1,
                size: (size.width, size.height),
            },
        );
        (id, view)
    }
}

/// A mesh's buffers. Dropping it retires them.
pub struct VkMesh {
    vertices: std::cell::Cell<u32>,
    indices: Option<u32>,
    layout: u32,
}

impl VkMesh {
    pub fn new(bytes: &[u8], layout: &VertexLayout, indices: Option<&[u32]>) -> VkMesh {
        with_state(|s| {
            // Device-local until the first update shows the mesh is dynamic.
            let vertices = s.create_static_buffer(bytes, vk::BufferUsageFlags::VERTEX_BUFFER);
            let indices = indices.map(|i| {
                let bytes = unsafe {
                    std::slice::from_raw_parts(i.as_ptr() as *const u8, std::mem::size_of_val(i))
                };
                s.create_static_buffer(bytes, vk::BufferUsageFlags::INDEX_BUFFER)
            });
            VkMesh {
                vertices: std::cell::Cell::new(vertices),
                indices,
                layout: s.layout_id(layout),
            }
        })
    }

    /// Frames in flight may still read the old buffer, so the update goes
    /// into a new one (GL's buffer orphaning).
    pub fn update_vertices(&self, bytes: &[u8]) {
        with(|s| {
            let old = self.vertices.get();
            let Some(b) = s.buffers.get(&old) else { return };
            let (size, old_host, old_buffer, old_mapped) =
                (b.size as usize, b.host, b.buffer, b.alloc.mapped);
            let n = bytes.len().min(size);
            let mut contents = vec![0u8; size];
            if n < size && old_host {
                unsafe { std::ptr::copy_nonoverlapping(old_mapped, contents.as_mut_ptr(), size) };
            }
            contents[..n].copy_from_slice(&bytes[..n]);
            let new = s.create_buffer(&contents, vk::BufferUsageFlags::VERTEX_BUFFER);
            if n < size && !old_host {
                // Keep the tail the update does not cover, GPU to GPU.
                let new_buffer = s.buffers[&new].buffer;
                let cmd = s.upload_commands();
                unsafe {
                    s.dev.device.cmd_copy_buffer(
                        cmd,
                        old_buffer,
                        new_buffer,
                        &[vk::BufferCopy {
                            src_offset: n as u64,
                            dst_offset: n as u64,
                            size: (size - n) as u64,
                        }],
                    );
                }
            }
            s.drop_buffer(old);
            self.vertices.set(new);
        });
    }

    pub fn draw(&self, topology: Topology, first: u32, count: u32) {
        let start = std::time::Instant::now();
        with(|s| {
            s.draw(
                self.vertices.get(),
                self.indices,
                self.layout,
                topology,
                first,
                count,
            )
        });
        stat(|s| {
            s.draws += 1;
            s.draw_time += start.elapsed();
        });
    }
}

impl Drop for VkMesh {
    fn drop(&mut self) {
        let _ = STATE.try_with(|state| {
            if let Ok(mut state) = state.try_borrow_mut()
                && let Some(s) = state.as_mut()
            {
                s.drop_buffer(self.vertices.get());
                if let Some(i) = self.indices {
                    s.drop_buffer(i);
                }
            }
        });
    }
}

/// A cube map's image. Dropping it retires it.
pub struct VkCube {
    id: u32,
}

impl VkCube {
    pub fn new(size: u32, faces: &[u8]) -> VkCube {
        VkCube {
            id: with_state(|s| s.create_cube(size, faces)),
        }
    }

    pub fn bind(&self, unit: u32) {
        with(|s| {
            if let Some(u) = s.cube_units.get_mut(unit as usize) {
                *u = Some(self.id);
            }
        });
    }
}

impl Drop for VkCube {
    fn drop(&mut self) {
        retire_image(self.id);
    }
}

fn retire_image(id: u32) {
    let _ = STATE.try_with(|state| {
        if let Ok(mut state) = state.try_borrow_mut()
            && let Some(s) = state.as_mut()
            && let Some(i) = s.images.remove(&id)
        {
            s.retire(Garbage::Image(i.image, i.view, i.alloc));
        }
    });
}

/// A texture's image. Dropping it retires it.
pub struct VkTexture {
    id: u32,
}

impl VkTexture {
    pub fn new(
        width: u32,
        height: u32,
        format: PixelFormat,
        pixels: &[u8],
        sampling: Sampling,
    ) -> VkTexture {
        VkTexture {
            id: with_state(|s| s.create_texture(width, height, format, pixels, sampling)),
        }
    }

    pub fn update_rgb_region(
        &self,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        pixels: &[u8],
        size: (u32, u32),
    ) {
        let rgba = to_rgba(PixelFormat::Rgb8, pixels);
        // The mip chain is rebuilt from the new level 0.
        with(|s| s.upload(self.id, x, y, width, height, &rgba, false, size));
    }

    pub fn bind(&self, unit: u32) {
        with(|s| {
            if let Some(u) = s.units.get_mut(unit as usize) {
                *u = Some(self.id);
            }
        });
    }
}

impl Drop for VkTexture {
    fn drop(&mut self) {
        retire_image(self.id);
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn program_lists_take_numbers_and_ranges() {
        assert_eq!(parse_program_list("3, 5-7\n12 x"), [3, 5, 6, 7, 12]);
        assert!(parse_program_list("").is_empty());
    }
    use super::{parse_program_list, take};

    #[test]
    fn first_fit_splits_and_aligns() {
        let mut free = vec![(0, 100)];
        assert_eq!(take(&mut free, 10, 1), Some(0));
        assert_eq!(take(&mut free, 10, 16), Some(16));
        assert_eq!(free, vec![(10, 6), (26, 74)]);
        assert_eq!(take(&mut free, 200, 1), None);
    }
}
