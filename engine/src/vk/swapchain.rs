//! The window's swapchain with its depth buffer, render pass and
//! framebuffers. Rebuilt whole on resize: nothing outside holds its handles
//! across a frame.

use ash::vk;

use super::context::VkContext;

/// The colour format the frame is drawn in. UNORM, not sRGB: the GL renderer
/// wrote shader output straight to a linear default framebuffer, and the
/// palette art and lighting are tuned for that.
const PREFERRED_FORMATS: [vk::Format; 2] = [vk::Format::B8G8R8A8_UNORM, vk::Format::R8G8B8A8_UNORM];
const DEPTH_FORMATS: [vk::Format; 3] = [
    vk::Format::D32_SFLOAT,
    vk::Format::D24_UNORM_S8_UINT,
    vk::Format::D16_UNORM,
];

pub struct DepthBuffer {
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
    pub view: vk::ImageView,
}

pub struct VkSwapchain {
    pub handle: vk::SwapchainKHR,
    pub format: vk::Format,
    pub depth_format: vk::Format,
    pub extent: vk::Extent2D,
    pub images: Vec<vk::Image>,
    pub views: Vec<vk::ImageView>,
    pub depth: DepthBuffer,
    pub render_pass: vk::RenderPass,
    pub framebuffers: Vec<vk::Framebuffer>,
    /// The images can be copied from (screenshots).
    pub transfer_src: bool,
}

impl VkSwapchain {
    pub fn new(ctx: &VkContext, size: (u32, u32), vsync: bool) -> Result<VkSwapchain, vk::Result> {
        let pd = ctx.physical_device;
        let caps = unsafe {
            ctx.surface_loader
                .get_physical_device_surface_capabilities(pd, ctx.surface)
        }?;
        let formats = unsafe {
            ctx.surface_loader
                .get_physical_device_surface_formats(pd, ctx.surface)
        }?;
        let surface_format = PREFERRED_FORMATS
            .iter()
            .find_map(|&wanted| formats.iter().find(|f| f.format == wanted))
            .copied()
            .unwrap_or(formats[0]);
        let modes = unsafe {
            ctx.surface_loader
                .get_physical_device_surface_present_modes(pd, ctx.surface)
        }?;
        let present_mode = if vsync {
            vk::PresentModeKHR::FIFO
        } else {
            [vk::PresentModeKHR::MAILBOX, vk::PresentModeKHR::IMMEDIATE]
                .into_iter()
                .find(|m| modes.contains(m))
                .unwrap_or(vk::PresentModeKHR::FIFO)
        };
        let extent = if caps.current_extent.width != u32::MAX {
            caps.current_extent
        } else {
            vk::Extent2D {
                width: size
                    .0
                    .clamp(caps.min_image_extent.width, caps.max_image_extent.width),
                height: size
                    .1
                    .clamp(caps.min_image_extent.height, caps.max_image_extent.height),
            }
        };
        let mut image_count = caps.min_image_count + 1;
        if caps.max_image_count > 0 {
            image_count = image_count.min(caps.max_image_count);
        }
        // TRANSFER_SRC lets a screenshot copy the presented image back.
        let mut usage = vk::ImageUsageFlags::COLOR_ATTACHMENT;
        if caps
            .supported_usage_flags
            .contains(vk::ImageUsageFlags::TRANSFER_SRC)
        {
            usage |= vk::ImageUsageFlags::TRANSFER_SRC;
        }
        let composite = if caps
            .supported_composite_alpha
            .contains(vk::CompositeAlphaFlagsKHR::OPAQUE)
        {
            vk::CompositeAlphaFlagsKHR::OPAQUE
        } else {
            vk::CompositeAlphaFlagsKHR::INHERIT
        };
        let handle = unsafe {
            ctx.swapchain_loader.create_swapchain(
                &vk::SwapchainCreateInfoKHR::builder()
                    .surface(ctx.surface)
                    .min_image_count(image_count)
                    .image_format(surface_format.format)
                    .image_color_space(surface_format.color_space)
                    .image_extent(extent)
                    .image_array_layers(1)
                    .image_usage(usage)
                    .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .pre_transform(caps.current_transform)
                    .composite_alpha(composite)
                    .present_mode(present_mode)
                    .clipped(true),
                None,
            )
        }?;
        let images = unsafe { ctx.swapchain_loader.get_swapchain_images(handle) }?;
        let views = images
            .iter()
            .map(|&image| color_view(ctx, image, surface_format.format))
            .collect::<Result<Vec<_>, _>>()?;

        let depth_format = DEPTH_FORMATS
            .into_iter()
            .find(|&f| {
                unsafe { ctx.instance.get_physical_device_format_properties(pd, f) }
                    .optimal_tiling_features
                    .contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT)
            })
            .expect("no depth format");
        let depth = create_depth(ctx, depth_format, extent)?;
        let render_pass = create_render_pass(ctx, surface_format.format, depth_format)?;
        let framebuffers = views
            .iter()
            .map(|&view| {
                let attachments = [view, depth.view];
                unsafe {
                    ctx.device.create_framebuffer(
                        &vk::FramebufferCreateInfo::builder()
                            .render_pass(render_pass)
                            .attachments(&attachments)
                            .width(extent.width)
                            .height(extent.height)
                            .layers(1),
                        None,
                    )
                }
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(VkSwapchain {
            handle,
            format: surface_format.format,
            depth_format,
            extent,
            images,
            views,
            depth,
            render_pass,
            framebuffers,
            transfer_src: usage.contains(vk::ImageUsageFlags::TRANSFER_SRC),
        })
    }

    /// Destroy every handle. The caller waits for the device to go idle first.
    pub fn destroy(&mut self, ctx: &VkContext) {
        unsafe {
            for &fb in &self.framebuffers {
                ctx.device.destroy_framebuffer(fb, None);
            }
            ctx.device.destroy_render_pass(self.render_pass, None);
            ctx.device.destroy_image_view(self.depth.view, None);
            ctx.device.destroy_image(self.depth.image, None);
            ctx.device.free_memory(self.depth.memory, None);
            for &view in &self.views {
                ctx.device.destroy_image_view(view, None);
            }
            ctx.swapchain_loader.destroy_swapchain(self.handle, None);
        }
    }
}

fn color_view(
    ctx: &VkContext,
    image: vk::Image,
    format: vk::Format,
) -> Result<vk::ImageView, vk::Result> {
    unsafe {
        ctx.device.create_image_view(
            &vk::ImageViewCreateInfo::builder()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format)
                .subresource_range(subresource(vk::ImageAspectFlags::COLOR)),
            None,
        )
    }
}

pub(crate) fn subresource(aspect: vk::ImageAspectFlags) -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: aspect,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    }
}

fn create_depth(
    ctx: &VkContext,
    format: vk::Format,
    extent: vk::Extent2D,
) -> Result<DepthBuffer, vk::Result> {
    let image = unsafe {
        ctx.device.create_image(
            &vk::ImageCreateInfo::builder()
                .image_type(vk::ImageType::TYPE_2D)
                .format(format)
                .extent(vk::Extent3D {
                    width: extent.width,
                    height: extent.height,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .initial_layout(vk::ImageLayout::UNDEFINED),
            None,
        )
    }?;
    let requirements = unsafe { ctx.device.get_image_memory_requirements(image) };
    let memory_type = ctx
        .memory_type(
            requirements.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
        .expect("no device-local memory for the depth buffer");
    let memory = unsafe {
        ctx.device.allocate_memory(
            &vk::MemoryAllocateInfo::builder()
                .allocation_size(requirements.size)
                .memory_type_index(memory_type),
            None,
        )
    }?;
    unsafe { ctx.device.bind_image_memory(image, memory, 0) }?;
    let mut aspect = vk::ImageAspectFlags::DEPTH;
    if format == vk::Format::D24_UNORM_S8_UINT {
        aspect |= vk::ImageAspectFlags::STENCIL;
    }
    let view = unsafe {
        ctx.device.create_image_view(
            &vk::ImageViewCreateInfo::builder()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format)
                .subresource_range(subresource(aspect)),
            None,
        )
    }?;
    Ok(DepthBuffer {
        image,
        memory,
        view,
    })
}

fn create_render_pass(
    ctx: &VkContext,
    color: vk::Format,
    depth: vk::Format,
) -> Result<vk::RenderPass, vk::Result> {
    let attachments = [
        vk::AttachmentDescription::builder()
            .format(color)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .build(),
        vk::AttachmentDescription::builder()
            .format(depth)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .stencil_load_op(vk::AttachmentLoadOp::CLEAR)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            .build(),
    ];
    let color_ref = [vk::AttachmentReference {
        attachment: 0,
        layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    }];
    let depth_ref = vk::AttachmentReference {
        attachment: 1,
        layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    };
    let subpass = [vk::SubpassDescription::builder()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color_ref)
        .depth_stencil_attachment(&depth_ref)
        .build()];
    // The previous frame's use of the colour and depth attachments must finish
    // before this frame clears them.
    let stages = vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
        | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
        | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS;
    let dependency = [vk::SubpassDependency::builder()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(stages)
        .dst_stage_mask(stages)
        .src_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE)
        .dst_access_mask(
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
        )
        .build()];
    unsafe {
        ctx.device.create_render_pass(
            &vk::RenderPassCreateInfo::builder()
                .attachments(&attachments)
                .subpasses(&subpass)
                .dependencies(&dependency),
            None,
        )
    }
}
