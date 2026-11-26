//! Display gamma (the Options menu's Brightness) and the screen grade
//! (vignette, film grain, contrast, saturation).
//!
//! The frame renders into an offscreen image the size of the swapchain,
//! then one full-screen triangle copies it to the swapchain image through
//! [`Grade`]. With a neutral grade the renderer skips this and draws
//! straight to the swapchain.
//!
//! The offscreen image has the swapchain's format and shares its depth
//! buffer (and with MSAA its multisampled colour, resolving into the
//! offscreen image), so its render pass is compatible with the swapchain's and every
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
layout(push_constant) uniform GammaPush {
    layout(offset = 4) float invGamma;
    float vignette;
    float grain;
    float time;
    float contrast;
    float saturation;
    float danger;
    float bloom;
    float streak;
} grade;
layout(location = 0) out vec4 fragColor;

// Integer hash: stable per pixel and frame, no texture.
float hash(uvec3 v) {
    v = v * 1664525u + 1013904223u;
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    v ^= v >> 16u;
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    return float(v.x & 0xffffu) / 65535.0;
}

const vec3 LUMA = vec3(0.299, 0.587, 0.114);
// The frame is display-referred (8-bit), so "bright" means near white:
// lamps, screens, flashes. A knee from 0.65 to 1.0: lamps glow, while a
// lit silver gun or a pale wall barely does.
// Taps read mip 2 (a 4x4 average), so the sparse spiral sees smooth light
// rather than single texels - no speckle, and thin bright edges such as a
// lit gun's rim average away while a lamp's face survives.
vec3 brightPart(vec2 at, vec2 size) {
    vec3 s = textureLod(frame, at / size, 2.0).rgb;
    return s * smoothstep(0.55, 1.0, dot(s, LUMA));
}

// The streak takes only near-white points (bulbs, flashes); a large bright
// surface would otherwise smear sideways into ghost copies of itself.
vec3 hotPart(vec2 at, vec2 size) {
    vec3 s = textureLod(frame, at / size, 1.0).rgb;
    return s * smoothstep(0.92, 1.0, dot(s, LUMA));
}

// Glare: a soft disc of light around bright sources (golden-angle spiral,
// bilinear taps so each covers four texels), plus a faint horizontal
// streak like light scattered in a lens. One pass, no extra targets.
vec3 glare(vec2 size) {
    vec3 sum = vec3(0.0);
    if (grade.bloom > 0.0) {
        float radius = size.y * 0.08;
        float weights = 0.0;
        // Turn the spiral per pixel: sparse taps then dither into fine
        // noise (hidden under the grain) instead of visible rings.
        // White noise: fine speckle the film grain hides (a structured
        // pattern such as interleaved gradient noise shows as hatching).
        float spin = hash(uvec3(uvec2(gl_FragCoord.xy), 7u)) * 6.28318;
        for (int i = 0; i < 40; i++) {
            float t = (float(i) + 0.5) / 40.0;
            float a = float(i) * 2.39996 + spin;
            vec2 offset = vec2(cos(a), sin(a)) * sqrt(t) * radius;
            float w = 1.0 - 0.6 * t;
            sum += brightPart(gl_FragCoord.xy + offset, size) * w * grade.bloom;
            weights += w;
        }
        sum /= weights / 2.0;
    }
    if (grade.streak > 0.0) {
        vec3 streak = vec3(0.0);
        float reach = size.x * 0.12;
        // Jitter the tap spacing per pixel so the taps blur into a smooth
        // line instead of stepping.
        float jitter = hash(uvec3(uvec2(gl_FragCoord.xy), 11u));
        for (int i = 0; i < 16; i++) {
            float t = (float(i) + jitter) / 16.0;
            float w = exp(-3.0 * t) * 0.6;
            vec2 offset = vec2(t * reach, 0.0);
            streak += (hotPart(gl_FragCoord.xy + offset, size)
                + hotPart(gl_FragCoord.xy - offset, size)) * w;
        }
        // A cool tint, like the coating on a lens.
        sum += streak * vec3(0.75, 0.85, 1.0) * grade.streak;
    }
    return sum;
}

void main() {
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    vec2 size = vec2(textureSize(frame, 0));
    vec3 c = texelFetch(frame, pixel, 0).rgb + glare(size);
    c = pow(max(c, vec3(0.0)), vec3(grade.invGamma));

    float luma = dot(c, vec3(0.299, 0.587, 0.114));
    // Low health drains the colour and closes in the edges with a
    // heartbeat: a strong beat then a softer one, about 70 bpm.
    float saturation = grade.saturation * (1.0 - 0.4 * grade.danger);
    c = max(mix(vec3(luma), c, saturation), vec3(0.0));
    // Contrast pivots low: midtones keep their level while shadows sink,
    // which is where a dark corridor reads as dark.
    const float pivot = 0.3;
    c = max((c - pivot) * grade.contrast + pivot, vec3(0.0));

    vec2 centred = (gl_FragCoord.xy / size - 0.5) * vec2(size.x / size.y, 1.0);
    float edge = smoothstep(0.35, 1.05, length(centred));
    float beatPhase = fract(grade.time * 1.15);
    float beat = exp(-pow((beatPhase - 0.05) * 18.0, 2.0))
        + 0.6 * exp(-pow((beatPhase - 0.27) * 18.0, 2.0));
    float vignette = min(grade.vignette + grade.danger * (0.3 + 0.25 * beat), 0.95);
    c *= 1.0 - vignette * edge;
    c = mix(c, c * vec3(1.35, 0.35, 0.3), grade.danger * edge * (0.5 + 0.5 * beat));

    // Grain shows most in the dark, as film and sensor noise do.
    float n = hash(uvec3(uvec2(pixel), uint(grade.time * 60.0))) - 0.5;
    c += grade.grain * n * (1.0 - 0.6 * min(luma, 1.0));

    fragColor = vec4(clamp(c, 0.0, 1.0), 1.0);
}
"#;

/// The screen grade applied by the copy. [`Grade::NEUTRAL`] leaves pixels
/// as drawn (and lets the renderer skip the offscreen copy).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Grade {
    pub gamma: f32,
    /// Edge darkening, 0..1.
    pub vignette: f32,
    /// Film grain amplitude, 0..~0.1.
    pub grain: f32,
    /// Slope around a low pivot; 1 is unchanged.
    pub contrast: f32,
    /// 1 is unchanged, 0 is greyscale.
    pub saturation: f32,
    /// Seconds, animating the grain and the low-health heartbeat.
    pub time: f32,
    /// Low-health state, 0..1: drains colour and pulses the edges red.
    pub danger: f32,
    /// Glow around bright sources, 0..~1.5.
    pub bloom: f32,
    /// Horizontal lens streak through bright sources, 0..~1.5.
    pub streak: f32,
}

impl Grade {
    pub const NEUTRAL: Grade = Grade {
        gamma: 1.0,
        vignette: 0.0,
        grain: 0.0,
        contrast: 1.0,
        saturation: 1.0,
        time: 0.0,
        danger: 0.0,
        bloom: 0.0,
        streak: 0.0,
    };

    pub fn is_neutral(&self) -> bool {
        (self.gamma - 1.0).abs() < 1.0e-3
            && self.vignette.abs() < 1.0e-3
            && self.grain.abs() < 1.0e-4
            && (self.contrast - 1.0).abs() < 1.0e-3
            && (self.saturation - 1.0).abs() < 1.0e-3
            && self.danger <= 0.0
            && self.bloom <= 0.0
            && self.streak <= 0.0
    }
}

/// The offscreen frame and the copy that applies gamma to it. Built for one
/// swapchain; rebuild it with the swapchain.
/// Mip levels of the offscreen frame. Level 0 is the frame; the rest are
/// rebuilt every frame (see [`GammaTarget::record_mips`]) so the glare taps
/// read pre-blurred light instead of speckling on single texels.
const LEVELS: u32 = 4;

pub(crate) struct GammaTarget {
    image: vk::Image,
    memory: vk::DeviceMemory,
    /// Level 0 only: the scene pass renders into it.
    view: vk::ImageView,
    /// Every level: what the copy samples.
    sampled_view: vk::ImageView,
    extent: vk::Extent2D,
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
                        .mip_levels(LEVELS)
                        .array_layers(1)
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .tiling(vk::ImageTiling::OPTIMAL)
                        .usage(
                            vk::ImageUsageFlags::COLOR_ATTACHMENT
                                | vk::ImageUsageFlags::SAMPLED
                                | vk::ImageUsageFlags::TRANSFER_SRC
                                | vk::ImageUsageFlags::TRANSFER_DST,
                        )
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
            let sampled_view = d
                .create_image_view(
                    &vk::ImageViewCreateInfo::builder()
                        .image(image)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(swapchain.format)
                        .subresource_range(vk::ImageSubresourceRange {
                            level_count: LEVELS,
                            ..swapchain::subresource(vk::ImageAspectFlags::COLOR)
                        }),
                    None,
                )
                .map_err(vk_err("vkCreateImageView"))?;
            let render_pass = swapchain::create_render_pass(
                ctx,
                swapchain.format,
                swapchain.depth_format,
                swapchain.samples,
                true,
                false,
            )
            .map_err(vk_err("vkCreateRenderPass"))?;
            let resume_pass = swapchain::create_render_pass(
                ctx,
                swapchain.format,
                swapchain.depth_format,
                swapchain.samples,
                true,
                true,
            )
            .map_err(vk_err("vkCreateRenderPass"))?;
            let attachments =
                swapchain::scene_attachments(view, &swapchain.depth, swapchain.msaa_color.as_ref());
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
                        // Linear for the glare taps; the copy itself uses
                        // texelFetch, which ignores filtering.
                        .mag_filter(vk::Filter::LINEAR)
                        .min_filter(vk::Filter::LINEAR)
                        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                        .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                        .max_lod(LEVELS as f32),
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
                image_view: sampled_view,
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
                size: 40,
            }];
            let layout = d
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::builder()
                        .set_layouts(&set_layouts)
                        .push_constant_ranges(&push),
                    None,
                )
                .map_err(vk_err("vkCreatePipelineLayout"))?;
            let pipeline = build_pipeline(d, layout, swapchain.render_pass, swapchain.samples)?;
            Ok(GammaTarget {
                image,
                memory,
                view,
                sampled_view,
                extent,
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

    /// Rebuild levels 1.. from the finished frame (level 0, left in
    /// SHADER_READ_ONLY by the scene pass), leaving every level ready to
    /// sample. Recorded between the scene pass and the swapchain pass.
    pub fn record_mips(&self, d: &ash::Device, cmd: vk::CommandBuffer) {
        let range = |level: u32, count: u32| vk::ImageSubresourceRange {
            base_mip_level: level,
            level_count: count,
            ..swapchain::subresource(vk::ImageAspectFlags::COLOR)
        };
        let barrier = |level: u32,
                       count: u32,
                       from: vk::ImageLayout,
                       to: vk::ImageLayout,
                       src: (vk::PipelineStageFlags, vk::AccessFlags),
                       dst: (vk::PipelineStageFlags, vk::AccessFlags)| unsafe {
            d.cmd_pipeline_barrier(
                cmd,
                src.0,
                dst.0,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[vk::ImageMemoryBarrier::builder()
                    .src_access_mask(src.1)
                    .dst_access_mask(dst.1)
                    .old_layout(from)
                    .new_layout(to)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(self.image)
                    .subresource_range(range(level, count))
                    .build()],
            );
        };
        let transfer = (
            vk::PipelineStageFlags::TRANSFER,
            vk::AccessFlags::TRANSFER_WRITE,
        );
        let read = (
            vk::PipelineStageFlags::TRANSFER,
            vk::AccessFlags::TRANSFER_READ,
        );
        barrier(
            0,
            1,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            (
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            ),
            read,
        );
        let size = |level: u32| {
            [
                (self.extent.width >> level).max(1) as i32,
                (self.extent.height >> level).max(1) as i32,
            ]
        };
        for level in 1..LEVELS {
            // Last frame's contents are dead: start from UNDEFINED.
            barrier(
                level,
                1,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                (
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::AccessFlags::empty(),
                ),
                transfer,
            );
            let [sw, sh] = size(level - 1);
            let [dw, dh] = size(level);
            let layers = |level: u32| vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: level,
                base_array_layer: 0,
                layer_count: 1,
            };
            unsafe {
                d.cmd_blit_image(
                    cmd,
                    self.image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    self.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[vk::ImageBlit {
                        src_subresource: layers(level - 1),
                        src_offsets: [vk::Offset3D::default(), vk::Offset3D { x: sw, y: sh, z: 1 }],
                        dst_subresource: layers(level),
                        dst_offsets: [vk::Offset3D::default(), vk::Offset3D { x: dw, y: dh, z: 1 }],
                    }],
                    vk::Filter::LINEAR,
                );
            }
            barrier(
                level,
                1,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                transfer,
                read,
            );
        }
        barrier(
            0,
            LEVELS,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            read,
            (
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::AccessFlags::SHADER_READ,
            ),
        );
    }

    /// Record the copy into the open swapchain render pass.
    pub fn record_copy(
        &self,
        d: &ash::Device,
        cmd: vk::CommandBuffer,
        extent: vk::Extent2D,
        grade: &Grade,
    ) {
        // flip_y is irrelevant (the triangle covers the screen either way).
        let push = [
            1.0f32,
            1.0 / grade.gamma.max(0.1),
            grade.vignette,
            grade.grain,
            grade.time,
            grade.contrast,
            grade.saturation,
            grade.danger,
            grade.bloom,
            grade.streak,
        ];
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
            d.destroy_image_view(self.sampled_view, None);
            d.destroy_image(self.image, None);
            d.free_memory(self.memory, None);
        }
    }
}

unsafe fn build_pipeline(
    d: &ash::Device,
    layout: vk::PipelineLayout,
    render_pass: vk::RenderPass,
    samples: vk::SampleCountFlags,
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
        let multisample =
            vk::PipelineMultisampleStateCreateInfo::builder().rasterization_samples(samples);
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
