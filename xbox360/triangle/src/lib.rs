//! A Rust title for the Xbox 360: a triangle drawn through `ash` on
//! vulkan4xenon, the whole Vulkan path the engine's renderer would take.
//!
//! It turns by itself; the left stick moves it, the right trigger grows it,
//! A rumbles and Back ends. It reports each step in `rust_triangle.log`
//! beside the `.xex` as `PASS:`/`FAIL:` lines.
//!
//! `ash` calls vulkan4xenon's C through the function pointers its
//! `vkGetInstanceProcAddr` hands out. That is sound because this crate is
//! compiled by xenon-forge's PPC64-ILP32 LLVM, whose calling convention is
//! the console's own, as xsdk's GCC builds vulkan4xenon (xbox360/README.md).

#![no_std]

extern crate alloc;

use alloc::vec::Vec;
use ash::vk;
use xsdk_rt::{Checks, log};

mod shader_sizes {
    include!(concat!(env!("OUT_DIR"), "/shader_sizes.rs"));
}

// Vulkan's own ABI ("system"), which on PowerPC is the C one.
unsafe extern "system" {
    // libvulkan4xenon.a's loader entry: everything else comes through it.
    fn vkGetInstanceProcAddr(
        instance: vk::Instance,
        name: *const core::ffi::c_char,
    ) -> vk::PFN_vkVoidFunction;
}

unsafe extern "C" {
    // build/shaders.c, from tools/build_shaders.py.
    static tri_triangle_vert_spirv: u32;
    static tri_triangle_frag_spirv: u32;
}

const SPLASH_MS: u32 = 2000;
const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;
const RUMBLE_SPEED: u16 = 0x8000;
/// Frames after which the title counts as presenting, for its check.
const PRESENT_CHECK_FRAMES: u32 = 120;
/// How long a frame may wait for the GPU before the title gives up.
const FENCE_TIMEOUT_NS: u64 = 1_000_000_000;

/// position (x, y), colour (r, g, b): three vertices, 5 floats each.
const VERTICES: [[f32; 5]; 3] = [
    [0.0, -0.6, 1.0, 0.2, 0.2],
    [0.52, 0.3, 0.2, 1.0, 0.2],
    [-0.52, 0.3, 0.2, 0.3, 1.0],
];
const VERTEX_STRIDE: u32 = 5 * 4;

/// Push constants, matching `triangle.vert`'s `Transform`.
#[repr(C)]
struct Transform {
    rotation: [f32; 4],
    offset: [f32; 4],
}

/// `result?` with a log line naming the call that failed.
trait Logged<T> {
    fn logged(self, call: &str) -> Result<T, ()>;
}

impl<T> Logged<T> for Result<T, vk::Result> {
    fn logged(self, call: &str) -> Result<T, ()> {
        self.map_err(|e| log!("rust_triangle: {} returned {:?} ({})", call, e, e.as_raw()))
    }
}

fn spirv(start: &'static u32, bytes: usize) -> &'static [u32] {
    unsafe { core::slice::from_raw_parts(start, bytes / 4) }
}

struct Gpu {
    instance: ash::Instance,
    device: ash::Device,
    queue: vk::Queue,
    swapchain_fn: ash::khr::swapchain::Device,
    swapchain: vk::SwapchainKHR,
    render_pass: vk::RenderPass,
    framebuffers: Vec<vk::Framebuffer>,
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    vertices: vk::Buffer,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
    acquired: vk::Semaphore,
    rendered: vk::Semaphore,
}

fn setup(checks: &mut Checks) -> Result<Gpu, ()> {
    let entry = unsafe {
        ash::Entry::from_static_fn(ash::StaticFn {
            get_instance_proc_addr: vkGetInstanceProcAddr,
        })
    };
    let app = vk::ApplicationInfo::default()
        .application_name(c"rust_triangle")
        .api_version(vk::API_VERSION_1_0);
    let instance_extensions = [ash::khr::surface::NAME.as_ptr(), ash::khr::display::NAME.as_ptr()];
    let instance = unsafe {
        entry.create_instance(
            &vk::InstanceCreateInfo::default()
                .application_info(&app)
                .enabled_extension_names(&instance_extensions),
            None,
        )
    }
    .logged("vkCreateInstance")?;
    let physical = *unsafe { instance.enumerate_physical_devices() }
        .logged("vkEnumeratePhysicalDevices")?
        .first()
        .ok_or_else(|| log!("rust_triangle: no physical device"))?;
    let properties = unsafe { instance.get_physical_device_properties(physical) };
    log!(
        "rust_triangle: device {:?}, Vulkan {}.{}",
        properties.device_name_as_c_str().unwrap_or(c"?"),
        vk::api_version_major(properties.api_version),
        vk::api_version_minor(properties.api_version)
    );

    let priority = [1.0f32];
    let queues = [vk::DeviceQueueCreateInfo::default()
        .queue_family_index(0)
        .queue_priorities(&priority)];
    let device_extensions = [ash::khr::swapchain::NAME.as_ptr()];
    let device = unsafe {
        instance.create_device(
            physical,
            &vk::DeviceCreateInfo::default()
                .queue_create_infos(&queues)
                .enabled_extension_names(&device_extensions),
            None,
        )
    }
    .logged("vkCreateDevice")?;
    let queue = unsafe { device.get_device_queue(0, 0) };
    checks.check("rust_triangle: instance and device", true);

    // The display's one mode on plane 0, through VK_KHR_display (vulkan4xenon
    // docs/SUPPORT.md, "Display and swapchain").
    let display_fn = ash::khr::display::Instance::new(&entry, &instance);
    let display = *unsafe { display_fn.get_physical_device_display_properties(physical) }
        .logged("vkGetPhysicalDeviceDisplayPropertiesKHR")?
        .first()
        .ok_or_else(|| log!("rust_triangle: no display"))?;
    let mode = *unsafe { display_fn.get_display_mode_properties(physical, display.display) }
        .logged("vkGetDisplayModePropertiesKHR")?
        .first()
        .ok_or_else(|| log!("rust_triangle: no display mode"))?;
    log!(
        "rust_triangle: display {}x{} at {} mHz",
        mode.parameters.visible_region.width,
        mode.parameters.visible_region.height,
        mode.parameters.refresh_rate
    );
    let surface = unsafe {
        display_fn.create_display_plane_surface(
            &vk::DisplaySurfaceCreateInfoKHR::default()
                .display_mode(mode.display_mode)
                .plane_index(0)
                .transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
                .alpha_mode(vk::DisplayPlaneAlphaFlagsKHR::OPAQUE)
                .image_extent(mode.parameters.visible_region),
            None,
        )
    }
    .logged("vkCreateDisplayPlaneSurfaceKHR")?;

    let extent = vk::Extent2D {
        width: WIDTH,
        height: HEIGHT,
    };
    let swapchain_fn = ash::khr::swapchain::Device::new(&instance, &device);
    let swapchain = unsafe {
        swapchain_fn.create_swapchain(
            &vk::SwapchainCreateInfoKHR::default()
                .surface(surface)
                .min_image_count(2)
                .image_format(vk::Format::B8G8R8A8_UNORM)
                .image_color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR)
                .image_extent(extent)
                .image_array_layers(1)
                .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(vk::PresentModeKHR::FIFO)
                .clipped(true),
            None,
        )
    }
    .logged("vkCreateSwapchainKHR")?;
    let images = unsafe { swapchain_fn.get_swapchain_images(swapchain) }
        .logged("vkGetSwapchainImagesKHR")?;
    checks.check("rust_triangle: display surface and swapchain", true);

    let attachments = [vk::AttachmentDescription::default()
        .format(vk::Format::B8G8R8A8_UNORM)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)];
    let colour_refs = [vk::AttachmentReference {
        attachment: 0,
        layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    }];
    let subpasses = [vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&colour_refs)];
    let render_pass = unsafe {
        device.create_render_pass(
            &vk::RenderPassCreateInfo::default()
                .attachments(&attachments)
                .subpasses(&subpasses),
            None,
        )
    }
    .logged("vkCreateRenderPass")?;

    let mut framebuffers = Vec::new();
    for &image in &images {
        let view = unsafe {
            device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(vk::Format::B8G8R8A8_UNORM)
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
        .logged("vkCreateImageView")?;
        let views = [view];
        framebuffers.push(
            unsafe {
                device.create_framebuffer(
                    &vk::FramebufferCreateInfo::default()
                        .render_pass(render_pass)
                        .attachments(&views)
                        .width(WIDTH)
                        .height(HEIGHT)
                        .layers(1),
                    None,
                )
            }
            .logged("vkCreateFramebuffer")?,
        );
    }

    // Shaders: the SPIR-V that build/shaders.c holds, which vulkan4xenon
    // looks up in the bundle beside it, already compiled for Xenos.
    let vert = unsafe {
        device.create_shader_module(
            &vk::ShaderModuleCreateInfo::default().code(spirv(
                &tri_triangle_vert_spirv,
                shader_sizes::TRI_TRIANGLE_VERT_SPIRV_BYTES,
            )),
            None,
        )
    }
    .logged("vkCreateShaderModule (vertex)")?;
    let frag = unsafe {
        device.create_shader_module(
            &vk::ShaderModuleCreateInfo::default().code(spirv(
                &tri_triangle_frag_spirv,
                shader_sizes::TRI_TRIANGLE_FRAG_SPIRV_BYTES,
            )),
            None,
        )
    }
    .logged("vkCreateShaderModule (fragment)")?;
    checks.check("rust_triangle: shader modules from the bundle", true);

    let push_ranges = [vk::PushConstantRange {
        stage_flags: vk::ShaderStageFlags::VERTEX,
        offset: 0,
        size: core::mem::size_of::<Transform>() as u32,
    }];
    let layout = unsafe {
        device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&push_ranges),
            None,
        )
    }
    .logged("vkCreatePipelineLayout")?;
    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert)
            .name(c"main"),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag)
            .name(c"main"),
    ];
    let bindings = [vk::VertexInputBindingDescription {
        binding: 0,
        stride: VERTEX_STRIDE,
        input_rate: vk::VertexInputRate::VERTEX,
    }];
    let attributes = [
        vk::VertexInputAttributeDescription {
            location: 0,
            binding: 0,
            format: vk::Format::R32G32_SFLOAT,
            offset: 0,
        },
        vk::VertexInputAttributeDescription {
            location: 1,
            binding: 0,
            format: vk::Format::R32G32B32_SFLOAT,
            offset: 8,
        },
    ];
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&bindings)
        .vertex_attribute_descriptions(&attributes);
    let assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewports = [vk::Viewport {
        x: 0.0,
        y: 0.0,
        width: WIDTH as f32,
        height: HEIGHT as f32,
        min_depth: 0.0,
        max_depth: 1.0,
    }];
    let scissors = [vk::Rect2D {
        offset: vk::Offset2D { x: 0, y: 0 },
        extent,
    }];
    let viewport = vk::PipelineViewportStateCreateInfo::default()
        .viewports(&viewports)
        .scissors(&scissors);
    let raster = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);
    let multisample =
        vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let blend_attachments = [vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(vk::ColorComponentFlags::RGBA)];
    let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachments);
    let pipeline_info = [vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&assembly)
        .viewport_state(&viewport)
        .rasterization_state(&raster)
        .multisample_state(&multisample)
        .color_blend_state(&blend)
        .layout(layout)
        .render_pass(render_pass)
        .subpass(0)];
    let pipeline = unsafe {
        device.create_graphics_pipelines(vk::PipelineCache::null(), &pipeline_info, None)
    }
    .map_err(|(_, e)| e)
    .logged("vkCreateGraphicsPipelines")?[0];
    checks.check("rust_triangle: graphics pipeline", true);

    // The vertices, in memory the CPU writes and the GPU reads.
    let size = core::mem::size_of_val(&VERTICES) as u64;
    let vertices = unsafe {
        device.create_buffer(
            &vk::BufferCreateInfo::default()
                .size(size)
                .usage(vk::BufferUsageFlags::VERTEX_BUFFER),
            None,
        )
    }
    .logged("vkCreateBuffer")?;
    let requirements = unsafe { device.get_buffer_memory_requirements(vertices) };
    let memory_properties = unsafe { instance.get_physical_device_memory_properties(physical) };
    let wanted = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
    let memory_type = (0..memory_properties.memory_type_count)
        .find(|&i| {
            requirements.memory_type_bits & (1 << i) != 0
                && memory_properties.memory_types[i as usize].property_flags.contains(wanted)
        })
        .ok_or_else(|| log!("rust_triangle: no host-visible coherent memory"))?;
    let memory = unsafe {
        device.allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(requirements.size)
                .memory_type_index(memory_type),
            None,
        )
    }
    .logged("vkAllocateMemory")?;
    unsafe { device.bind_buffer_memory(vertices, memory, 0) }.logged("vkBindBufferMemory")?;
    let mapped = unsafe { device.map_memory(memory, 0, size, vk::MemoryMapFlags::empty()) }
        .logged("vkMapMemory")? as *mut u32;
    // A word at a time: GPU memory may be write-combined, where the 64-bit
    // stores a plain copy can merge into fault unless 8-byte aligned (xsdk
    // docs/PITFALLS.md, "Plain stores to GPU memory").
    for (i, value) in VERTICES.iter().flatten().enumerate() {
        unsafe { mapped.add(i).write_volatile(value.to_bits()) };
    }
    unsafe { device.unmap_memory(memory) };

    let pool = unsafe {
        device.create_command_pool(
            &vk::CommandPoolCreateInfo::default()
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                .queue_family_index(0),
            None,
        )
    }
    .logged("vkCreateCommandPool")?;
    let command_buffer = unsafe {
        device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1),
        )
    }
    .logged("vkAllocateCommandBuffers")?[0];
    let fence = unsafe {
        device.create_fence(
            &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
            None,
        )
    }
    .logged("vkCreateFence")?;
    let acquired = unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None) }
        .logged("vkCreateSemaphore")?;
    let rendered = unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None) }
        .logged("vkCreateSemaphore")?;

    Ok(Gpu {
        instance,
        device,
        queue,
        swapchain_fn,
        swapchain,
        render_pass,
        framebuffers,
        pipeline,
        layout,
        vertices,
        command_buffer,
        fence,
        acquired,
        rendered,
    })
}

/// cos and sin of `angle`, which is kept in -pi..pi by the caller.
fn rotation(angle: f32) -> (f32, f32) {
    // libm-free: a short Taylor series is plenty for a turning triangle, and
    // core has no sin/cos without std.
    let a2 = angle * angle;
    let sin = angle * (1.0 - a2 / 6.0 * (1.0 - a2 / 20.0 * (1.0 - a2 / 42.0 * (1.0 - a2 / 72.0))));
    let cos = 1.0 - a2 / 2.0 * (1.0 - a2 / 12.0 * (1.0 - a2 / 30.0 * (1.0 - a2 / 56.0 * (1.0 - a2 / 90.0))));
    (cos, sin)
}

fn draw_frame(gpu: &Gpu, transform: &Transform) -> Result<(), ()> {
    let d = &gpu.device;
    unsafe { d.wait_for_fences(&[gpu.fence], true, FENCE_TIMEOUT_NS) }.logged("vkWaitForFences")?;
    unsafe { d.reset_fences(&[gpu.fence]) }.logged("vkResetFences")?;
    let (index, _) = unsafe {
        gpu.swapchain_fn
            .acquire_next_image(gpu.swapchain, u64::MAX, gpu.acquired, vk::Fence::null())
    }
    .logged("vkAcquireNextImageKHR")?;

    let cb = gpu.command_buffer;
    unsafe {
        d.reset_command_buffer(cb, vk::CommandBufferResetFlags::empty())
            .logged("vkResetCommandBuffer")?;
        d.begin_command_buffer(cb, &vk::CommandBufferBeginInfo::default())
            .logged("vkBeginCommandBuffer")?;
        let clear = [vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.05, 0.06, 0.09, 1.0],
            },
        }];
        d.cmd_begin_render_pass(
            cb,
            &vk::RenderPassBeginInfo::default()
                .render_pass(gpu.render_pass)
                .framebuffer(gpu.framebuffers[index as usize])
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: vk::Extent2D {
                        width: WIDTH,
                        height: HEIGHT,
                    },
                })
                .clear_values(&clear),
            vk::SubpassContents::INLINE,
        );
        d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, gpu.pipeline);
        d.cmd_bind_vertex_buffers(cb, 0, &[gpu.vertices], &[0]);
        let bytes = core::slice::from_raw_parts(
            transform as *const Transform as *const u8,
            core::mem::size_of::<Transform>(),
        );
        d.cmd_push_constants(cb, gpu.layout, vk::ShaderStageFlags::VERTEX, 0, bytes);
        d.cmd_draw(cb, 3, 1, 0, 0);
        d.cmd_end_render_pass(cb);
        d.end_command_buffer(cb).logged("vkEndCommandBuffer")?;
    }

    let waits = [gpu.acquired];
    let stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
    let cbs = [cb];
    let signals = [gpu.rendered];
    let submit = [vk::SubmitInfo::default()
        .wait_semaphores(&waits)
        .wait_dst_stage_mask(&stages)
        .command_buffers(&cbs)
        .signal_semaphores(&signals)];
    unsafe { d.queue_submit(gpu.queue, &submit, gpu.fence) }.logged("vkQueueSubmit")?;
    let swapchains = [gpu.swapchain];
    let indices = [index];
    unsafe {
        gpu.swapchain_fn.queue_present(
            gpu.queue,
            &vk::PresentInfoKHR::default()
                .wait_semaphores(&signals)
                .swapchains(&swapchains)
                .image_indices(&indices),
        )
    }
    .logged("vkQueuePresentKHR")?;
    Ok(())
}

fn run(gpu: &Gpu, checks: &mut Checks) {
    let mut angle = 0.0f32;
    let mut offset = [0.0f32; 2];
    let mut rumbling = false;
    let mut last_result = xsdk_rt::ERROR_SUCCESS;
    let mut frames = 0u32;
    loop {
        let (result, pad) = xsdk_rt::pad(0);
        if result != last_result {
            log!("rust_triangle: XamInputGetState returned 0x{:08X}", result);
        }
        last_result = result;
        if pad.buttons & xsdk_rt::BUTTON_BACK != 0 {
            break;
        }
        let a = pad.buttons & xsdk_rt::BUTTON_A != 0;
        if a != rumbling {
            let speed = if a { RUMBLE_SPEED } else { 0 };
            xsdk_rt::rumble(0, speed, speed);
        }
        rumbling = a;

        // About a turn every 4 s at 60 Hz, kept in -pi..pi for `rotation`.
        angle += core::f32::consts::TAU / 240.0;
        if angle > core::f32::consts::PI {
            angle -= core::f32::consts::TAU;
        }
        let stick = |v: i16| {
            let v = f32::from(v) / 32767.0;
            if v.abs() < 0.2 { 0.0 } else { v }
        };
        // Stick +y is up; Vulkan's clip-space +y is down.
        offset[0] = (offset[0] + stick(pad.thumb_lx) * 0.02).clamp(-0.8, 0.8);
        offset[1] = (offset[1] - stick(pad.thumb_ly) * 0.02).clamp(-0.8, 0.8);
        let scale = 1.0 + f32::from(pad.right_trigger) / 255.0;
        let (cos, sin) = rotation(angle);
        let transform = Transform {
            rotation: [cos, sin, scale, 0.0],
            offset: [offset[0], offset[1], 0.0, 0.0],
        };
        if draw_frame(gpu, &transform).is_err() {
            checks.check("rust_triangle: every frame drew and presented", false);
            break;
        }
        frames += 1;
        if frames == PRESENT_CHECK_FRAMES {
            checks.check("rust_triangle: 120 frames drawn and presented", true);
        }
    }
    if rumbling {
        xsdk_rt::rumble(0, 0, 0);
    }
    log!("rust_triangle: {} frames", frames);
}

#[unsafe(no_mangle)]
pub extern "C" fn main(_argc: i32, _argv: *const *const u8) -> i32 {
    let mut checks = Checks::default();
    log!("rust_triangle: a triangle through ash on vulkan4xenon; left stick moves, RT grows, A rumbles, Back ends");
    let splash = unsafe { xsdk_rt::xsdk_splash_show(SPLASH_MS) };
    if splash < 0 {
        log!("rust_triangle: xsdk_splash_show failed: 0x{:08X}", splash as u32);
    }
    let Ok(gpu) = setup(&mut checks) else {
        checks.check("rust_triangle: Vulkan set-up", false);
        return checks.finish();
    };
    run(&gpu, &mut checks);
    unsafe {
        let _ = gpu.device.device_wait_idle();
        gpu.swapchain_fn.destroy_swapchain(gpu.swapchain, None);
    }
    let _ = &gpu.instance;
    log!("rust_triangle: Back pressed, done");
    checks.finish()
}
