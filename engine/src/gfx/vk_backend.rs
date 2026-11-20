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

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};

use ash::vk;

use super::{
    BlendFactor, Cap, DepthFunc, Filter, FrontFace, PixelFormat, Sampling, Topology, VertexLayout,
};
use crate::vk::shader::{self, Member};

const FRAMES: usize = 2;
const BLOCK_SIZE: u64 = 64 << 20;
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
}

struct ImageObj {
    image: vk::Image,
    view: vk::ImageView,
    alloc: Allocation,
    sampling: Sampling,
    mips: u32,
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
}

enum Garbage {
    Buffer(vk::Buffer, Allocation),
    Image(vk::Image, vk::ImageView, Allocation),
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
    extent: vk::Extent2D,
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
}

struct State {
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
    frames: Vec<FrameRes>,
    /// Frames submitted so far.
    submitted: u64,
    garbage: Vec<(u64, Garbage)>,
    recording: Option<Recording>,
    gl: GlState,
    program: Option<u32>,
    units: [Option<u32>; MAX_UNITS],
}

thread_local! {
    static STATE: RefCell<Option<State>> = const { RefCell::new(None) };
}

fn with<R>(f: impl FnOnce(&mut State) -> R) -> Option<R> {
    STATE.with(|s| s.borrow_mut().as_mut().map(f))
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
        dev,
        alloc: Allocator::default(),
        upload_pool,
        next_id: 1,
        buffers: HashMap::new(),
        images: HashMap::new(),
        programs: Vec::new(),
        samplers: HashMap::new(),
        pipelines: HashMap::new(),
        layouts: Vec::new(),
        layout_ids: HashMap::new(),
        zero_buffer: 0,
        white: 0,
        frames: (0..FRAMES).map(|_| FrameRes::default()).collect(),
        submitted: 0,
        garbage: Vec::new(),
        recording: None,
        gl: GlState::default(),
        program: None,
        units: [None; MAX_UNITS],
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
    STATE.with(|s| *s.borrow_mut() = Some(state));
    super::set_backend(super::Backend::Vulkan);
}

/// Destroy every object the backend made. The device must be idle.
pub fn shutdown() {
    let Some(mut s) = STATE.with(|s| s.borrow_mut().take()) else {
        return;
    };
    let d = s.dev.device.clone();
    unsafe {
        for (_, g) in s.garbage.drain(..) {
            destroy_garbage(&d, g);
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
    super::set_backend(super::Backend::OpenGl);
}

unsafe fn destroy_garbage(d: &ash::Device, g: Garbage) {
    unsafe {
        match g {
            Garbage::Buffer(b, _) => d.destroy_buffer(b, None),
            Garbage::Image(i, v, _) => {
                d.destroy_image_view(v, None);
                d.destroy_image(i, None);
            }
        }
    }
}

/// Start recording gfx draws into `commands`, inside an open render pass.
/// `slot`'s previous frame has finished (its fence was waited on).
pub fn begin_frame(
    commands: vk::CommandBuffer,
    render_pass: vk::RenderPass,
    formats: (vk::Format, vk::Format),
    extent: vk::Extent2D,
    slot: usize,
) {
    with(|s| {
        // Frames before `submitted - (FRAMES - 1)` are complete.
        let done_before = s.submitted.saturating_sub(FRAMES as u64 - 1);
        let mut keep = Vec::new();
        for (tag, g) in s.garbage.drain(..) {
            if tag <= done_before {
                let a = match &g {
                    Garbage::Buffer(_, a) | Garbage::Image(_, _, a) => *a,
                };
                unsafe { destroy_garbage(&s.dev.device, g) };
                s.alloc.free(a);
            } else {
                keep.push((tag, g));
            }
        }
        s.garbage = keep;
        let frame = &mut s.frames[slot];
        frame.ubo_at = (0, 0);
        frame.pool_at = 0;
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
            extent,
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
        if s.recording.take().is_some() {
            s.submitted += 1;
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
        let d = &self.dev.device;
        let buffer = unsafe {
            d.create_buffer(
                &vk::BufferCreateInfo::builder()
                    .size(size.max(16))
                    .usage(usage)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )
        }
        .expect("vkCreateBuffer");
        let req = unsafe { d.get_buffer_memory_requirements(buffer) };
        let alloc = self.alloc.alloc(
            &self.dev,
            req,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            true,
        );
        unsafe { d.bind_buffer_memory(buffer, alloc.memory, alloc.offset) }
            .expect("vkBindBufferMemory");
        (buffer, alloc)
    }

    fn create_buffer(&mut self, bytes: &[u8], usage: vk::BufferUsageFlags) -> u32 {
        let size = bytes.len() as u64;
        let (buffer, alloc) = self.raw_buffer(size, usage);
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), alloc.mapped, bytes.len());
        }
        let id = self.id();
        self.buffers.insert(
            id,
            BufferObj {
                buffer,
                alloc,
                size,
            },
        );
        id
    }

    fn drop_buffer(&mut self, id: u32) {
        if let Some(b) = self.buffers.remove(&id) {
            self.retire(Garbage::Buffer(b.buffer, b.alloc));
        }
    }

    /// Record and run commands now, waiting for them to finish.
    fn one_shot(&mut self, record: impl FnOnce(&ash::Device, vk::CommandBuffer)) {
        let d = &self.dev.device;
        unsafe {
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
            record(d, cmd);
            d.end_command_buffer(cmd).expect("vkEndCommandBuffer");
            let fence = d
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .expect("vkCreateFence");
            let cmds = [cmd];
            d.queue_submit(
                self.dev.queue,
                &[vk::SubmitInfo::builder().command_buffers(&cmds).build()],
                fence,
            )
            .expect("vkQueueSubmit");
            d.wait_for_fences(&[fence], true, u64::MAX)
                .expect("vkWaitForFences");
            d.destroy_fence(fence, None);
            d.free_command_buffers(self.upload_pool, &cmds);
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
        let rgba = to_rgba(format, pixels);
        let mips = if sampling.filter == Filter::LinearMipmap {
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
                    .array_layers(1)
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
                    .view_type(vk::ImageViewType::TYPE_2D)
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
            },
        );
        self.upload(id, 0, 0, width, height, &rgba, true, (width, height));
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
        let (staging, staging_alloc) =
            self.raw_buffer(rgba.len() as u64, vk::BufferUsageFlags::TRANSFER_SRC);
        unsafe { std::ptr::copy_nonoverlapping(rgba.as_ptr(), staging_alloc.mapped, rgba.len()) };
        let (image, mips) = {
            let i = &self.images[&id];
            (i.image, i.mips)
        };
        self.one_shot(|d, cmd| unsafe {
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
                    image_subresource: layers(0),
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
                        src_subresource: layers(level - 1),
                        src_offsets: [vk::Offset3D::default(), vk::Offset3D { x: w, y: h, z: 1 }],
                        dst_subresource: layers(level),
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
        });
        unsafe { self.dev.device.destroy_buffer(staging, None) };
        self.alloc.free(staging_alloc);
    }

    fn sampler(&mut self, sampling: Sampling) -> vk::Sampler {
        if let Some(&s) = self.samplers.get(&sampling) {
            return s;
        }
        let (filter, mip, max_lod) = match sampling.filter {
            Filter::Nearest => (vk::Filter::NEAREST, vk::SamplerMipmapMode::NEAREST, 0.0),
            Filter::Linear => (vk::Filter::LINEAR, vk::SamplerMipmapMode::NEAREST, 0.0),
            Filter::LinearMipmap => (vk::Filter::LINEAR, vk::SamplerMipmapMode::LINEAR, 1000.0),
        };
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
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
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
        let blend = vk::PipelineColorBlendStateCreateInfo::builder().attachments(&blend_attachment);
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
            if frame.pool_at == frame.pools.len() {
                let sizes = [
                    vk::DescriptorPoolSize {
                        ty: vk::DescriptorType::UNIFORM_BUFFER,
                        descriptor_count: POOL_SETS * 2,
                    },
                    vk::DescriptorPoolSize {
                        ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                        descriptor_count: POOL_SETS * 4,
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
                Ok(sets) => return sets[0],
                Err(vk::Result::ERROR_OUT_OF_POOL_MEMORY | vk::Result::ERROR_FRAGMENTED_POOL) => {
                    frame.pool_at += 1
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
        let (Some(program_id), Some(rec)) = (self.program, self.recording.as_ref()) else {
            return;
        };
        let (cmd, slot, render_pass, formats, extent) = (
            rec.commands,
            rec.slot,
            rec.render_pass,
            rec.formats,
            rec.extent,
        );
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
        };
        let pipeline = self.pipeline(key, render_pass);

        // Uniform blocks into the ring, then this draw's descriptor set.
        let blocks: [Vec<u8>; 2] = self.programs[program_id as usize].blocks.clone();
        let mut buffer_writes = Vec::new();
        for (binding, bytes) in blocks.iter().enumerate() {
            if !bytes.is_empty() {
                let (buffer, offset) = self.push_uniforms(slot, bytes);
                buffer_writes.push((
                    binding as u32,
                    vk::DescriptorBufferInfo {
                        buffer,
                        offset,
                        range: bytes.len() as u64,
                    },
                ));
            }
        }
        let sampler_units: Vec<(u32, u32)> = self.programs[program_id as usize]
            .samplers
            .iter()
            .map(|s| (s.binding, s.unit))
            .collect();
        let mut image_writes = Vec::new();
        for (binding, unit) in sampler_units {
            let image = self
                .units
                .get(unit as usize)
                .copied()
                .flatten()
                .filter(|id| self.images.contains_key(id))
                .unwrap_or(self.white);
            let (view, sampling) = {
                let i = &self.images[&image];
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
        let set = self.descriptor_set(slot, set_layout);
        let writes: Vec<vk::WriteDescriptorSet> = buffer_writes
            .iter()
            .map(|(binding, info)| {
                vk::WriteDescriptorSet::builder()
                    .dst_set(set)
                    .dst_binding(*binding)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
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
        let d = &self.dev.device;
        unsafe {
            if !writes.is_empty() {
                d.update_descriptor_sets(&writes, &[]);
            }
        }

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
                &[],
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
        layer_count: 1,
    }
}

fn layers(mip: u32) -> vk::ImageSubresourceLayers {
    vk::ImageSubresourceLayers {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        mip_level: mip,
        base_array_layer: 0,
        layer_count: 1,
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
        let vertex = module(&linked.vertex);
        let fragment = module(&linked.fragment);

        let mut slots: Vec<Slot> = Vec::new();
        let mut names: HashMap<String, u32> = HashMap::new();
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
                    let i = slot_for(name, &mut slots);
                    slots[i].members.push((index, *member));
                }
                set_bindings.push(
                    vk::DescriptorSetLayoutBinding::builder()
                        .binding(index as u32)
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
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
        for (name, &(binding, in_vertex, in_fragment)) in &linked.samplers {
            let mut stages = vk::ShaderStageFlags::empty();
            if in_vertex {
                stages |= vk::ShaderStageFlags::VERTEX;
            }
            if in_fragment {
                stages |= vk::ShaderStageFlags::FRAGMENT;
            }
            let i = slot_for(name, &mut slots);
            slots[i].sampler = Some(samplers.len());
            samplers.push(SamplerBinding { binding, unit: 0 });
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
        });
        s.programs.len() as u32 - 1
    })
}

pub const NO_UNIFORM: u32 = u32::MAX;

pub fn uniform(program: u32, name: &str) -> u32 {
    // GL names array uniforms both `name` and `name[0]`.
    let name = name.strip_suffix("[0]").unwrap_or(name);
    with(|s| s.programs[program as usize].names.get(name).copied())
        .flatten()
        .unwrap_or(NO_UNIFORM)
}

pub fn bind_program(program: u32) {
    with(|s| s.program = Some(program));
}

/// Write `values` (32-bit words, `columns` columns of `rows` each per
/// element, `elements` elements) into the bound program's uniform.
fn write_uniform(uniform: u32, words: &[u32], rows: usize, columns: usize) {
    if uniform == NO_UNIFORM {
        return;
    }
    with(|s| {
        let Some(program) = s.program else { return };
        let p = &mut s.programs[program as usize];
        let Some(slot) = p.slots.get(uniform as usize) else {
            return;
        };
        for &(block, m) in &slot.members {
            let bytes = &mut p.blocks[block];
            let per_element = rows * columns;
            for (e, element) in words.chunks(per_element).enumerate().take(m.count as usize) {
                let base = m.offset as usize + e * m.array_stride as usize;
                for c in 0..columns {
                    let at = base + c * m.matrix_stride as usize;
                    for r in 0..rows {
                        let o = at + r * 4;
                        if let (Some(dst), Some(v)) =
                            (bytes.get_mut(o..o + 4), element.get(c * rows + r))
                        {
                            dst.copy_from_slice(&v.to_ne_bytes());
                        }
                    }
                }
            }
        }
    });
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|f| f.to_bits()).collect()
}

pub fn set_floats(uniform: u32, values: &[f32], components: usize) {
    write_uniform(uniform, &bits(values), components, 1);
}

pub fn set_matrices(uniform: u32, values: &[f32], size: usize) {
    write_uniform(uniform, &bits(values), size, size);
}

pub fn set_ints(uniform: u32, values: &[i32]) {
    if uniform == NO_UNIFORM {
        return;
    }
    // A sampler's value is the texture unit it reads.
    let sampler = with(|s| {
        let program = s.program?;
        let p = &mut s.programs[program as usize];
        let i = p.slots.get(uniform as usize)?.sampler?;
        p.samplers[i].unit = *values.first()? as u32;
        Some(())
    })
    .flatten();
    if sampler.is_none() {
        let words: Vec<u32> = values.iter().map(|&i| i as u32).collect();
        write_uniform(uniform, &words, 1, 1);
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

/// A mesh's buffers. Dropping it retires them.
pub struct VkMesh {
    vertices: std::cell::Cell<u32>,
    indices: Option<u32>,
    layout: u32,
}

impl VkMesh {
    pub fn new(bytes: &[u8], layout: &VertexLayout, indices: Option<&[u32]>) -> VkMesh {
        with_state(|s| {
            let vertices = s.create_buffer(bytes, vk::BufferUsageFlags::VERTEX_BUFFER);
            let indices = indices.map(|i| {
                let bytes = unsafe {
                    std::slice::from_raw_parts(i.as_ptr() as *const u8, std::mem::size_of_val(i))
                };
                s.create_buffer(bytes, vk::BufferUsageFlags::INDEX_BUFFER)
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
            let size = b.size as usize;
            let mut contents = vec![0u8; size];
            if bytes.len() < size {
                unsafe {
                    std::ptr::copy_nonoverlapping(b.alloc.mapped, contents.as_mut_ptr(), size)
                };
            }
            let n = bytes.len().min(size);
            contents[..n].copy_from_slice(&bytes[..n]);
            let new = s.create_buffer(&contents, vk::BufferUsageFlags::VERTEX_BUFFER);
            s.drop_buffer(old);
            self.vertices.set(new);
        });
    }

    pub fn draw(&self, topology: Topology, first: u32, count: u32) {
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
        let _ = STATE.try_with(|state| {
            if let Ok(mut state) = state.try_borrow_mut()
                && let Some(s) = state.as_mut()
                && let Some(i) = s.images.remove(&self.id)
            {
                s.retire(Garbage::Image(i.image, i.view, i.alloc));
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::take;

    #[test]
    fn first_fit_splits_and_aligns() {
        let mut free = vec![(0, 100)];
        assert_eq!(take(&mut free, 10, 1), Some(0));
        assert_eq!(take(&mut free, 10, 16), Some(16));
        assert_eq!(free, vec![(10, 6), (26, 74)]);
        assert_eq!(take(&mut free, 200, 1), None);
    }
}
