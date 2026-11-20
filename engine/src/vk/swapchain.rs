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

/// An image the frame draws into that isn't the swapchain's own: the depth
/// buffer, and with MSAA the multisampled colour the pass resolves from.
pub struct AttachmentImage {
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
    pub depth: AttachmentImage,
    /// Samples per pixel of the scene attachments (1 = no MSAA).
    pub samples: vk::SampleCountFlags,
    /// With MSAA, the multisampled colour the pass draws into and resolves
    /// to the presented image at its end.
    pub msaa_color: Option<AttachmentImage>,
    pub render_pass: vk::RenderPass,
    /// `render_pass` loading instead of clearing (see `create_render_pass`).
    pub resume_pass: vk::RenderPass,
    pub framebuffers: Vec<vk::Framebuffer>,
    /// The images can be copied from (screenshots).
    pub transfer_src: bool,
}

impl VkSwapchain {
    pub fn new(
        ctx: &VkContext,
        size: (u32, u32),
        vsync: bool,
        samples: vk::SampleCountFlags,
    ) -> Result<VkSwapchain, vk::Result> {
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
        let depth = create_attachment(
            ctx,
            depth_format,
            extent,
            samples,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            depth_aspect(depth_format),
        )?;
        let msaa_color = if samples == vk::SampleCountFlags::TYPE_1 {
            None
        } else {
            Some(create_attachment(
                ctx,
                surface_format.format,
                extent,
                samples,
                vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSIENT_ATTACHMENT,
                vk::ImageAspectFlags::COLOR,
            )?)
        };
        let render_pass = create_render_pass(
            ctx,
            surface_format.format,
            depth_format,
            samples,
            false,
            false,
        )?;
        let resume_pass = create_render_pass(
            ctx,
            surface_format.format,
            depth_format,
            samples,
            false,
            true,
        )?;
        let framebuffers = views
            .iter()
            .map(|&view| {
                let attachments = scene_attachments(view, &depth, msaa_color.as_ref());
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
            samples,
            msaa_color,
            render_pass,
            resume_pass,
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
            ctx.device.destroy_render_pass(self.resume_pass, None);
            ctx.device.destroy_image_view(self.depth.view, None);
            ctx.device.destroy_image(self.depth.image, None);
            ctx.device.free_memory(self.depth.memory, None);
            if let Some(color) = self.msaa_color.take() {
                color.destroy(ctx);
            }
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

impl AttachmentImage {
    fn destroy(self, ctx: &VkContext) {
        unsafe {
            ctx.device.destroy_image_view(self.view, None);
            ctx.device.destroy_image(self.image, None);
            ctx.device.free_memory(self.memory, None);
        }
    }
}

/// A framebuffer's attachments in `create_render_pass`'s order: the colour
/// drawn into, depth, then (with MSAA) the single-sample image it resolves to.
pub(crate) fn scene_attachments(
    target: vk::ImageView,
    depth: &AttachmentImage,
    msaa_color: Option<&AttachmentImage>,
) -> Vec<vk::ImageView> {
    match msaa_color {
        Some(color) => vec![color.view, depth.view, target],
        None => vec![target, depth.view],
    }
}

fn depth_aspect(format: vk::Format) -> vk::ImageAspectFlags {
    if format == vk::Format::D24_UNORM_S8_UINT {
        vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
    } else {
        vk::ImageAspectFlags::DEPTH
    }
}

/// The highest sample count up to `wanted` that colour and depth attachments
/// both support.
pub(crate) fn supported_samples(ctx: &VkContext, wanted: u32) -> vk::SampleCountFlags {
    let limits = unsafe {
        ctx.instance
            .get_physical_device_properties(ctx.physical_device)
    }
    .limits;
    let supported =
        limits.framebuffer_color_sample_counts & limits.framebuffer_depth_sample_counts;
    [
        vk::SampleCountFlags::TYPE_8,
        vk::SampleCountFlags::TYPE_4,
        vk::SampleCountFlags::TYPE_2,
    ]
    .into_iter()
    .find(|&s| s.as_raw() <= wanted && supported.contains(s))
    .unwrap_or(vk::SampleCountFlags::TYPE_1)
}

fn create_attachment(
    ctx: &VkContext,
    format: vk::Format,
    extent: vk::Extent2D,
    samples: vk::SampleCountFlags,
    usage: vk::ImageUsageFlags,
    aspect: vk::ImageAspectFlags,
) -> Result<AttachmentImage, vk::Result> {
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
                .samples(samples)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(usage)
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
        .expect("no device-local memory for a frame attachment");
    let memory = unsafe {
        ctx.device.allocate_memory(
            &vk::MemoryAllocateInfo::builder()
                .allocation_size(requirements.size)
                .memory_type_index(memory_type),
            None,
        )
    }?;
    unsafe { ctx.device.bind_image_memory(image, memory, 0) }?;
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
    Ok(AttachmentImage {
        image,
        memory,
        view,
    })
}

/// The frame's render pass: colour cleared and stored, depth cleared. With
/// `sampled` the colour ends ready for a fragment shader to read (an
/// offscreen frame, see `gamma`) instead of ready to present. Both versions
/// are render-pass compatible, so pipelines built for one draw in the other.
/// With `resume` the pass loads what an earlier pass over the same
/// framebuffer left (a frame paused for a self-depth capture) instead of
/// clearing it.
///
/// With `samples` above 1 the colour and depth are multisampled and a third,
/// single-sample attachment receives the resolve at the end of the pass; it
/// then plays the part the colour attachment does without MSAA (presented or
/// sampled). A paused frame resolves early, harmlessly: the resumed pass
/// loads the multisampled colour and resolves again when it ends.
pub(crate) fn create_render_pass(
    ctx: &VkContext,
    color: vk::Format,
    depth: vk::Format,
    samples: vk::SampleCountFlags,
    sampled: bool,
    resume: bool,
) -> Result<vk::RenderPass, vk::Result> {
    let msaa = samples != vk::SampleCountFlags::TYPE_1;
    let color_final = if sampled {
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
    } else {
        vk::ImageLayout::PRESENT_SRC_KHR
    };
    // With MSAA the drawn-into colour is never presented or sampled.
    let drawn_final = if msaa {
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
    } else {
        color_final
    };
    let (load, color_initial, depth_initial) = if resume {
        (
            vk::AttachmentLoadOp::LOAD,
            drawn_final,
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        )
    } else {
        (
            vk::AttachmentLoadOp::CLEAR,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::UNDEFINED,
        )
    };
    let mut attachments = vec![
        vk::AttachmentDescription::builder()
            .format(color)
            .samples(samples)
            .load_op(load)
            // Kept with MSAA too: a paused frame resumes on it.
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(color_initial)
            .final_layout(drawn_final)
            .build(),
        vk::AttachmentDescription::builder()
            .format(depth)
            .samples(samples)
            .load_op(load)
            // Kept: a paused frame resumes on it.
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(depth_initial)
            .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            .build(),
    ];
    if msaa {
        // Every texel is overwritten by the resolve.
        attachments.push(
            vk::AttachmentDescription::builder()
                .format(color)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::DONT_CARE)
                .store_op(vk::AttachmentStoreOp::STORE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(color_final)
                .build(),
        );
    }
    let color_ref = [vk::AttachmentReference {
        attachment: 0,
        layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    }];
    let depth_ref = vk::AttachmentReference {
        attachment: 1,
        layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    };
    let resolve_ref = [vk::AttachmentReference {
        attachment: 2,
        layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    }];
    let mut subpass = vk::SubpassDescription::builder()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color_ref)
        .depth_stencil_attachment(&depth_ref);
    if msaa {
        subpass = subpass.resolve_attachments(&resolve_ref);
    }
    let subpass = [subpass.build()];
    // The previous frame's use of the colour and depth attachments must finish
    // before this frame clears them; an offscreen colour was last read by the
    // copy's fragment shader.
    //
    // The dependencies are the same whether or not the pass is `sampled`:
    // they are part of render-pass compatibility, and pipelines cached
    // against one pass draw in the other.
    let stages = vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
        | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
        | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS;
    let read_stage = vk::PipelineStageFlags::FRAGMENT_SHADER;
    let dependencies = [
        vk::SubpassDependency::builder()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(stages | read_stage)
            .dst_stage_mask(stages)
            .src_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            )
            .dst_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_READ
                    | vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            )
            .build(),
        // An offscreen frame's copy samples what this pass wrote.
        vk::SubpassDependency::builder()
            .src_subpass(0)
            .dst_subpass(vk::SUBPASS_EXTERNAL)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_stage_mask(vk::PipelineStageFlags::FRAGMENT_SHADER)
            .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .build(),
    ];
    unsafe {
        ctx.device.create_render_pass(
            &vk::RenderPassCreateInfo::builder()
                .attachments(&attachments)
                .subpasses(&subpass)
                .dependencies(&dependencies),
            None,
        )
    }
}
