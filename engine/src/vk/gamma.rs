//! Display gamma (the Options menu's Brightness).
//!
//! The frame renders into an offscreen image the size of the swapchain,
//! then one full-screen triangle copies it to the swapchain image through
//! `pow(c, 1/gamma)`. At gamma 1.0 the renderer skips this and draws
//! straight to the swapchain.
//!
//! The offscreen image has the swapchain's format and shares its depth
//! buffer, so its render pass is compatible with the swapchain's and every
//! pipeline gfx cached draws into either. The copy reads the texel under
//! each fragment (`texelFetch` at `gl_FragCoord`), so it is an exact 1:1
//! copy whatever y-flip the scene was drawn with.

use super::{VkContext, VkSwapchain, shader, swapchain};
use ash::vk;

const VERTEX: &str = r#"
void main() {
    // One triangle covering the screen: (-1,-1), (3,-1), (-1,3).
    vec2 p = vec2(float((gl_VertexIndex << 1) & 2), float(gl_VertexIndex & 2));
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"#;

const FRAGMENT: &str = r#"
layout(set = 0, binding = 0) uniform sampler2D frame;
// Offset 0 is the vertex stage's flip_y (see shader::to_vulkan_glsl).
layout(push_constant) uniform GammaPush { layout(offset = 4) float invGamma; } gamma;
layout(location = 0) out vec4 fragColor;
void main() {
    vec3 c = texelFetch(frame, ivec2(gl_FragCoord.xy), 0).rgb;
    fragColor = vec4(pow(max(c, vec3(0.0)), vec3(gamma.invGamma)), 1.0);
}
"#;

/// The offscreen frame and the copy that applies gamma to it. Built for one
/// swapchain; rebuild it with the swapchain.
pub(crate) struct GammaTarget {
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    /// Clears and draws the scene into `image`, leaving it ready to sample.
    pub render_pass: vk::RenderPass,
    /// `render_pass` loading instead of clearing, to resume a paused frame.
    pub resume_pass: vk::RenderPass,
    pub framebuffer: vk::Framebuffer,
    sampler: vk::Sampler,
    set_layout: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
    layout: vk::PipelineLayout,
    /// Draws the copy inside the swapchain's render pass.
    pipeline: vk::Pipeline,
}

impl GammaTarget {
    pub fn new(ctx: &VkContext, swapchain: &VkSwapchain) -> Result<GammaTarget, String> {
        unsafe { Self::build(ctx, swapchain) }.map_err(|e| format!("gamma target: {e}"))
    }

    unsafe fn build(ctx: &VkContext, swapchain: &VkSwapchain) -> Result<GammaTarget, String> {
        let d = &ctx.device;
        let extent = swapchain.extent;
        let vk_err = |what: &str| {
            let what = what.to_owned();
            move |e: vk::Result| format!("{what}: {e}")
        };
        unsafe {
            let image = d
                .create_image(
                    &vk::ImageCreateInfo::builder()
                        .image_type(vk::ImageType::TYPE_2D)
                        .format(swapchain.format)
                        .extent(vk::Extent3D {
                            width: extent.width,
                            height: extent.height,
                            depth: 1,
                        })
                        .mip_levels(1)
                        .array_layers(1)
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .tiling(vk::ImageTiling::OPTIMAL)
                        .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE)
                        .initial_layout(vk::ImageLayout::UNDEFINED),
                    None,
                )
                .map_err(vk_err("vkCreateImage"))?;
            let req = d.get_image_memory_requirements(image);
            let memory_type = ctx
                .memory_type(req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)
                .ok_or("no device-local memory for the offscreen frame")?;
            let memory = d
                .allocate_memory(
                    &vk::MemoryAllocateInfo::builder()
                        .allocation_size(req.size)
                        .memory_type_index(memory_type),
                    None,
                )
                .map_err(vk_err("vkAllocateMemory"))?;
            d.bind_image_memory(image, memory, 0)
                .map_err(vk_err("vkBindImageMemory"))?;
            let view = d
                .create_image_view(
                    &vk::ImageViewCreateInfo::builder()
                        .image(image)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(swapchain.format)
                        .subresource_range(swapchain::subresource(vk::ImageAspectFlags::COLOR)),
                    None,
                )
                .map_err(vk_err("vkCreateImageView"))?;
            let render_pass = swapchain::create_render_pass(
                ctx,
                swapchain.format,
                swapchain.depth_format,
                true,
                false,
            )
            .map_err(vk_err("vkCreateRenderPass"))?;
            let resume_pass = swapchain::create_render_pass(
                ctx,
                swapchain.format,
                swapchain.depth_format,
                true,
                true,
            )
            .map_err(vk_err("vkCreateRenderPass"))?;
            let attachments = [view, swapchain.depth.view];
            let framebuffer = d
                .create_framebuffer(
                    &vk::FramebufferCreateInfo::builder()
                        .render_pass(render_pass)
                        .attachments(&attachments)
                        .width(extent.width)
                        .height(extent.height)
                        .layers(1),
                    None,
                )
                .map_err(vk_err("vkCreateFramebuffer"))?;

            let sampler = d
                .create_sampler(
                    &vk::SamplerCreateInfo::builder()
                        .mag_filter(vk::Filter::NEAREST)
                        .min_filter(vk::Filter::NEAREST)
                        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                        .max_lod(0.0),
                    None,
                )
                .map_err(vk_err("vkCreateSampler"))?;
            let bindings = [vk::DescriptorSetLayoutBinding::builder()
                .binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                .build()];
            let set_layout = d
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings),
                    None,
                )
                .map_err(vk_err("vkCreateDescriptorSetLayout"))?;
            let sizes = [vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: 1,
            }];
            let pool = d
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::builder()
                        .max_sets(1)
                        .pool_sizes(&sizes),
                    None,
                )
                .map_err(vk_err("vkCreateDescriptorPool"))?;
            let set_layouts = [set_layout];
            let set = d
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::builder()
                        .descriptor_pool(pool)
                        .set_layouts(&set_layouts),
                )
                .map_err(vk_err("vkAllocateDescriptorSets"))?[0];
            // Written once: the image never changes and is only ever read
            // in SHADER_READ_ONLY_OPTIMAL.
            let image_info = [vk::DescriptorImageInfo {
                sampler,
                image_view: view,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            }];
            d.update_descriptor_sets(
                &[vk::WriteDescriptorSet::builder()
                    .dst_set(set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&image_info)
                    .build()],
                &[],
            );
            let push = [vk::PushConstantRange {
                stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                offset: 0,
                size: 8,
            }];
            let layout = d
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::builder()
                        .set_layouts(&set_layouts)
                        .push_constant_ranges(&push),
                    None,
                )
                .map_err(vk_err("vkCreatePipelineLayout"))?;
            let pipeline = build_pipeline(d, layout, swapchain.render_pass)?;
            Ok(GammaTarget {
                image,
                memory,
                view,
                render_pass,
                resume_pass,
                framebuffer,
                sampler,
                set_layout,
                pool,
                set,
                layout,
                pipeline,
            })
        }
    }

    /// Record the copy into the open swapchain render pass.
    pub fn record_copy(
        &self,
        d: &ash::Device,
        cmd: vk::CommandBuffer,
        extent: vk::Extent2D,
        gamma: f32,
    ) {
        // flip_y is irrelevant (the triangle covers the screen either way).
        let push = [1.0f32, 1.0 / gamma.max(0.1)];
        let bytes: Vec<u8> = push.iter().flat_map(|f| f.to_ne_bytes()).collect();
        unsafe {
            d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
            d.cmd_set_viewport(
                cmd,
                0,
                &[vk::Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: extent.width as f32,
                    height: extent.height as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                }],
            );
            d.cmd_set_scissor(
                cmd,
                0,
                &[vk::Rect2D {
                    offset: vk::Offset2D::default(),
                    extent,
                }],
            );
            d.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.layout,
                0,
                &[self.set],
                &[],
            );
            d.cmd_push_constants(
                cmd,
                self.layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                0,
                &bytes,
            );
            d.cmd_draw(cmd, 3, 1, 0, 0);
        }
    }

    /// Destroy every handle. The caller waits for the device to go idle first.
    pub fn destroy(&mut self, d: &ash::Device) {
        unsafe {
            d.destroy_pipeline(self.pipeline, None);
            d.destroy_pipeline_layout(self.layout, None);
            d.destroy_descriptor_pool(self.pool, None);
            d.destroy_descriptor_set_layout(self.set_layout, None);
            d.destroy_sampler(self.sampler, None);
            d.destroy_framebuffer(self.framebuffer, None);
            d.destroy_render_pass(self.render_pass, None);
            d.destroy_render_pass(self.resume_pass, None);
            d.destroy_image_view(self.view, None);
            d.destroy_image(self.image, None);
            d.free_memory(self.memory, None);
        }
    }
}

unsafe fn build_pipeline(
    d: &ash::Device,
    layout: vk::PipelineLayout,
    render_pass: vk::RenderPass,
) -> Result<vk::Pipeline, String> {
    let vertex = shader::compile(VERTEX, shader::Stage::Vertex)?;
    let fragment = shader::compile(FRAGMENT, shader::Stage::Fragment)?;
    unsafe {
        let module = |code: &[u32]| {
            d.create_shader_module(&vk::ShaderModuleCreateInfo::builder().code(code), None)
                .map_err(|e| format!("vkCreateShaderModule: {e}"))
        };
        let vs = module(&vertex)?;
        let fs = module(&fragment)?;
        let entry = c"main";
        let stages = [
            vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vs)
                .name(entry)
                .build(),
            vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(fs)
                .name(entry)
                .build(),
        ];
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
        let assembly = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let viewport = vk::PipelineViewportStateCreateInfo::builder()
            .viewport_count(1)
            .scissor_count(1);
        let raster = vk::PipelineRasterizationStateCreateInfo::builder()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::builder()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let depth = vk::PipelineDepthStencilStateCreateInfo::default();
        let blend_attachment = [vk::PipelineColorBlendAttachmentState::builder()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .build()];
        let blend = vk::PipelineColorBlendStateCreateInfo::builder().attachments(&blend_attachment);
        let dynamic = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic = vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(&dynamic);
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
            .layout(layout)
            .render_pass(render_pass)
            .subpass(0)
            .build();
        let result = d.create_graphics_pipelines(vk::PipelineCache::null(), &[info], None);
        d.destroy_shader_module(vs, None);
        d.destroy_shader_module(fs, None);
        result
            .map(|p| p[0])
            .map_err(|(_, e)| format!("vkCreateGraphicsPipelines: {e}"))
    }
}
