//! The Vulkan 1.0 renderer.
//!
//! [`VkRenderer`] owns the device and the window's swapchain and runs the
//! frame loop: [`VkRenderer::begin_frame`] waits for a free frame slot,
//! acquires a swapchain image and opens its render pass on a fresh command
//! buffer; [`VkRenderer::end_frame`] closes it, submits and presents. Two
//! frames are in flight, so the CPU records one while the GPU draws the other.

mod context;
mod gamma;
pub mod shader;
mod swapchain;

pub use ash;
pub use context::{VkContext, validation_error_count};
pub use swapchain::VkSwapchain;

use ash::vk;
use tracing::warn;

const FRAMES_IN_FLIGHT: usize = 2;

struct FrameSlot {
    pool: vk::CommandPool,
    commands: vk::CommandBuffer,
    image_available: vk::Semaphore,
    in_flight: vk::Fence,
}

/// One frame being recorded: draw into `commands` inside the open render
/// pass, then hand it back to [`VkRenderer::end_frame`].
pub struct Frame {
    pub commands: vk::CommandBuffer,
    pub extent: vk::Extent2D,
    image_index: u32,
    /// Drawn offscreen, copied to the swapchain through this gamma at the end.
    gamma: Option<gamma::Grade>,
}

pub struct VkRenderer {
    // Declared first so it drops before the context it was created from.
    swapchain: Option<VkSwapchain>,
    slots: Vec<FrameSlot>,
    /// One per swapchain image: a present may still be reading the semaphore
    /// of an image whose frame slot has already come round again.
    render_finished: Vec<vk::Semaphore>,
    slot: usize,
    size: (u32, u32),
    vsync: bool,
    /// Requested MSAA samples per pixel; the swapchain uses the highest the
    /// device supports up to this (1 = off).
    msaa: u32,
    rebuild: bool,
    rebuilds: u32,
    capture: bool,
    captured: Option<Capture>,
    grade: gamma::Grade,
    /// Built on first use for the current swapchain.
    gamma_target: Option<gamma::GammaTarget>,
    pub ctx: VkContext,
}

/// 4x MSAA: smooths the geometry edges the 1999 art is full of (grates,
/// railings, pipes) at a cost desktop GPUs absorb. `SHOCK2_MSAA` overrides it
/// (1 disables) for comparing captures.
const DEFAULT_MSAA: u32 = 4;

/// A presented frame read back: tightly packed RGBA8, top row first.
pub struct Capture {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

impl VkRenderer {
    pub fn new(ctx: VkContext, size: (u32, u32), vsync: bool) -> Result<VkRenderer, String> {
        let slots = (0..FRAMES_IN_FLIGHT)
            .map(|_| create_slot(&ctx))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("frame slots: {e}"))?;
        let mut renderer = VkRenderer {
            swapchain: None,
            slots,
            render_finished: Vec::new(),
            slot: 0,
            size,
            vsync,
            msaa: std::env::var("SHOCK2_MSAA")
                .ok()
                .and_then(|v| v.trim().parse::<u32>().ok())
                .unwrap_or(DEFAULT_MSAA)
                .max(1),
            rebuild: true,
            rebuilds: 0,
            capture: false,
            captured: None,
            grade: gamma::Grade::NEUTRAL,
            gamma_target: None,
            ctx,
        };
        crate::gfx::vk_backend::install(&renderer.ctx);
        renderer.rebuild_swapchain();
        // A device that can't make the first swapchain never will; fail now
        // rather than retry every frame (an iGPU that doesn't drive the display).
        if renderer.swapchain.is_none() && size.0 > 0 && size.1 > 0 {
            return Err(format!(
                "{} cannot create a swapchain for this window",
                renderer.ctx.device_name
            ));
        }
        Ok(renderer)
    }

    pub fn device_name(&self) -> &str {
        &self.ctx.device_name
    }

    /// Swapchains built so far, the first included.
    pub fn swapchain_builds(&self) -> u32 {
        self.rebuilds
    }

    /// Read back the next frame presented; [`VkRenderer::take_capture`]
    /// returns it once that frame has been drawn.
    pub fn request_capture(&mut self) {
        self.capture = true;
    }

    pub fn take_capture(&mut self) -> Option<Capture> {
        self.captured.take()
    }

    pub fn swapchain(&self) -> Option<&VkSwapchain> {
        self.swapchain.as_ref()
    }

    /// The window's framebuffer changed size; the swapchain follows before
    /// the next frame.
    /// Display gamma (Brightness): 1.0 draws straight to the swapchain.
    pub fn set_gamma(&mut self, gamma: f32) {
        self.grade.gamma = gamma;
    }

    /// The screen grade on top of gamma (see [`Grade`]); neutral values
    /// (0, 0, 1, 1) cost nothing. `time` (seconds) animates the grain - the
    /// caller's clock, so a fixed-step runtime screenshots deterministically.
    pub fn set_grade(
        &mut self,
        [vignette, grain, contrast, saturation, danger, bloom, streak]: [f32; 7],
        time: f32,
    ) {
        self.grade.bloom = bloom.max(0.0);
        self.grade.streak = streak.max(0.0);
        self.grade.danger = danger.clamp(0.0, 1.0);
        self.grade.time = time;
        self.grade.vignette = vignette;
        self.grade.grain = grain;
        self.grade.contrast = contrast;
        self.grade.saturation = saturation;
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if (width, height) != self.size {
            self.size = (width, height);
            self.rebuild = true;
        }
    }

    /// Request `samples` MSAA samples per pixel (1 = off); takes effect
    /// with the next frame.
    pub fn set_msaa(&mut self, samples: u32) {
        let samples = samples.max(1);
        if samples != self.msaa {
            self.msaa = samples;
            self.rebuild = true;
        }
    }

    /// Samples per pixel the current swapchain renders with.
    pub fn msaa_samples(&self) -> u32 {
        self.swapchain
            .as_ref()
            .map_or(1, |swapchain| swapchain.samples.as_raw())
    }

    pub fn set_vsync(&mut self, vsync: bool) {
        if vsync != self.vsync {
            self.vsync = vsync;
            self.rebuild = true;
        }
    }

    fn rebuild_swapchain(&mut self) {
        unsafe {
            let _ = self.ctx.device.device_wait_idle();
        }
        if let Some(mut target) = self.gamma_target.take() {
            target.destroy(&self.ctx.device);
        }
        if let Some(mut old) = self.swapchain.take() {
            old.destroy(&self.ctx);
        }
        for semaphore in self.render_finished.drain(..) {
            unsafe { self.ctx.device.destroy_semaphore(semaphore, None) };
        }
        // A minimised window has no drawable area; try again once it has one.
        if self.size.0 == 0 || self.size.1 == 0 {
            return;
        }
        let samples = swapchain::supported_samples(&self.ctx, self.msaa);
        match VkSwapchain::new(&self.ctx, self.size, self.vsync, samples) {
            Ok(swapchain) => {
                self.render_finished = swapchain
                    .images
                    .iter()
                    .map(|_| unsafe {
                        self.ctx
                            .device
                            .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                            .expect("vkCreateSemaphore")
                    })
                    .collect();
                if self.rebuilds == 0 {
                    tracing::info!("MSAA: {}x", swapchain.samples.as_raw());
                }
                self.swapchain = Some(swapchain);
                self.rebuild = false;
                self.rebuilds += 1;
            }
            Err(e) => warn!("swapchain rebuild failed: {e}"),
        }
    }

    /// Start a frame cleared to `clear` (RGBA). `None` when there is nothing
    /// to draw into this frame (minimised, or the swapchain was just rebuilt).
    pub fn begin_frame(&mut self, clear: [f32; 4]) -> Option<Frame> {
        if self.rebuild {
            self.rebuild_swapchain();
        }
        let swapchain = self.swapchain.as_ref()?;
        let gamma = (!self.grade.is_neutral()).then_some(self.grade);
        if gamma.is_some() && self.gamma_target.is_none() {
            match gamma::GammaTarget::new(&self.ctx, swapchain) {
                Ok(target) => self.gamma_target = Some(target),
                Err(e) => {
                    warn!("{e}; brightness off");
                    self.grade = gamma::Grade::NEUTRAL;
                }
            }
        }
        let gamma = gamma.filter(|_| self.gamma_target.is_some());
        let device = &self.ctx.device;
        let slot = &self.slots[self.slot];
        let waited = std::time::Instant::now();
        unsafe {
            device
                .wait_for_fences(&[slot.in_flight], true, u64::MAX)
                .expect("vkWaitForFences");
        }
        let fence_wait = waited.elapsed();
        let acquired = std::time::Instant::now();
        let image_index = match unsafe {
            self.ctx.swapchain_loader.acquire_next_image(
                swapchain.handle,
                u64::MAX,
                slot.image_available,
                vk::Fence::null(),
            )
        } {
            Ok((index, suboptimal)) => {
                self.rebuild |= suboptimal;
                index
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.rebuild = true;
                return None;
            }
            Err(e) => panic!("vkAcquireNextImageKHR: {e}"),
        };
        unsafe {
            device
                .reset_fences(&[slot.in_flight])
                .expect("vkResetFences");
            device
                .reset_command_pool(slot.pool, vk::CommandPoolResetFlags::empty())
                .expect("vkResetCommandPool");
            device
                .begin_command_buffer(
                    slot.commands,
                    &vk::CommandBufferBeginInfo::builder()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .expect("vkBeginCommandBuffer");
            // With gamma the scene goes offscreen; end_frame copies it over.
            let (render_pass, framebuffer) = match (&self.gamma_target, gamma) {
                (Some(target), Some(_)) => (target.render_pass, target.framebuffer),
                _ => (
                    swapchain.render_pass,
                    swapchain.framebuffers[image_index as usize],
                ),
            };
            let clears = [
                vk::ClearValue {
                    color: vk::ClearColorValue { float32: clear },
                },
                vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: 0,
                    },
                },
            ];
            device.cmd_begin_render_pass(
                slot.commands,
                &vk::RenderPassBeginInfo::builder()
                    .render_pass(render_pass)
                    .framebuffer(framebuffer)
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D::default(),
                        extent: swapchain.extent,
                    })
                    .clear_values(&clears),
                vk::SubpassContents::INLINE,
            );
        }
        crate::gfx::vk_backend::note_waits(fence_wait, acquired.elapsed());
        let (resume_pass, framebuffer) = match (&self.gamma_target, gamma) {
            (Some(target), Some(_)) => (target.resume_pass, target.framebuffer),
            _ => (
                swapchain.resume_pass,
                swapchain.framebuffers[image_index as usize],
            ),
        };
        crate::gfx::vk_backend::begin_frame(
            slot.commands,
            swapchain.render_pass,
            resume_pass,
            framebuffer,
            (swapchain.format, swapchain.depth_format),
            swapchain.samples,
            swapchain.extent,
            self.slot,
        );
        Some(Frame {
            commands: slot.commands,
            extent: swapchain.extent,
            image_index,
            gamma,
        })
    }

    /// Finish `frame`: close its render pass, submit and present it.
    pub fn end_frame(&mut self, frame: Frame) {
        let Some(swapchain) = self.swapchain.as_ref() else {
            return;
        };
        let device = &self.ctx.device;
        let slot = &self.slots[self.slot];
        let render_finished = self.render_finished[frame.image_index as usize];
        crate::gfx::vk_backend::end_frame();
        if let (Some(gamma), Some(target)) = (frame.gamma, &self.gamma_target) {
            let clears = [
                vk::ClearValue {
                    color: vk::ClearColorValue { float32: [0.0; 4] },
                },
                vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: 0,
                    },
                },
            ];
            unsafe {
                device.cmd_end_render_pass(frame.commands);
                device.cmd_begin_render_pass(
                    frame.commands,
                    &vk::RenderPassBeginInfo::builder()
                        .render_pass(swapchain.render_pass)
                        .framebuffer(swapchain.framebuffers[frame.image_index as usize])
                        .render_area(vk::Rect2D {
                            offset: vk::Offset2D::default(),
                            extent: swapchain.extent,
                        })
                        .clear_values(&clears),
                    vk::SubpassContents::INLINE,
                );
            }
            target.record_copy(device, frame.commands, swapchain.extent, &gamma);
        }
        let readback = if std::mem::take(&mut self.capture) {
            self.record_readback(frame.commands, swapchain, frame.image_index)
        } else {
            None
        };
        unsafe {
            device.cmd_end_render_pass(frame.commands);
            if let Some((buffer, _, _)) = readback {
                let image = swapchain.images[frame.image_index as usize];
                image_barrier(
                    device,
                    frame.commands,
                    image,
                    vk::ImageLayout::PRESENT_SRC_KHR,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                );
                device.cmd_copy_image_to_buffer(
                    frame.commands,
                    image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    buffer,
                    &[vk::BufferImageCopy {
                        image_subresource: vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            mip_level: 0,
                            base_array_layer: 0,
                            layer_count: 1,
                        },
                        image_extent: vk::Extent3D {
                            width: swapchain.extent.width,
                            height: swapchain.extent.height,
                            depth: 1,
                        },
                        ..Default::default()
                    }],
                );
                image_barrier(
                    device,
                    frame.commands,
                    image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    vk::ImageLayout::PRESENT_SRC_KHR,
                );
            }
            device
                .end_command_buffer(frame.commands)
                .expect("vkEndCommandBuffer");
            let wait = [slot.image_available];
            let stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            let commands = [frame.commands];
            let signal = [render_finished];
            device
                .queue_submit(
                    self.ctx.queue,
                    &[vk::SubmitInfo::builder()
                        .wait_semaphores(&wait)
                        .wait_dst_stage_mask(&stages)
                        .command_buffers(&commands)
                        .signal_semaphores(&signal)
                        .build()],
                    slot.in_flight,
                )
                .expect("vkQueueSubmit");
            let swapchains = [swapchain.handle];
            let indices = [frame.image_index];
            match self.ctx.swapchain_loader.queue_present(
                self.ctx.queue,
                &vk::PresentInfoKHR::builder()
                    .wait_semaphores(&signal)
                    .swapchains(&swapchains)
                    .image_indices(&indices),
            ) {
                Ok(suboptimal) => self.rebuild |= suboptimal,
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => self.rebuild = true,
                Err(e) => panic!("vkQueuePresentKHR: {e}"),
            }
            if let Some((buffer, memory, size)) = readback {
                device
                    .wait_for_fences(&[slot.in_flight], true, u64::MAX)
                    .expect("vkWaitForFences");
                let ptr = device
                    .map_memory(memory, 0, size, vk::MemoryMapFlags::empty())
                    .expect("vkMapMemory") as *const u8;
                let mut rgba = std::slice::from_raw_parts(ptr, size as usize).to_vec();
                device.unmap_memory(memory);
                device.destroy_buffer(buffer, None);
                device.free_memory(memory, None);
                if swapchain.format == vk::Format::B8G8R8A8_UNORM {
                    for p in rgba.chunks_exact_mut(4) {
                        p.swap(0, 2);
                    }
                }
                self.captured = Some(Capture {
                    width: swapchain.extent.width,
                    height: swapchain.extent.height,
                    rgba,
                });
            }
        }
        self.slot = (self.slot + 1) % FRAMES_IN_FLIGHT;
    }

    /// A host-visible buffer for one frame's pixels, or `None` when the
    /// swapchain can't be copied from.
    fn record_readback(
        &self,
        _commands: vk::CommandBuffer,
        swapchain: &VkSwapchain,
        _image_index: u32,
    ) -> Option<(vk::Buffer, vk::DeviceMemory, u64)> {
        if !swapchain.transfer_src {
            warn!("this swapchain cannot be read back");
            return None;
        }
        let device = &self.ctx.device;
        let size = swapchain.extent.width as u64 * swapchain.extent.height as u64 * 4;
        unsafe {
            let buffer = device
                .create_buffer(
                    &vk::BufferCreateInfo::builder()
                        .size(size)
                        .usage(vk::BufferUsageFlags::TRANSFER_DST)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE),
                    None,
                )
                .ok()?;
            let req = device.get_buffer_memory_requirements(buffer);
            let memory_type = self.ctx.memory_type(
                req.memory_type_bits,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;
            let memory = device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::builder()
                        .allocation_size(req.size)
                        .memory_type_index(memory_type),
                    None,
                )
                .ok()?;
            device.bind_buffer_memory(buffer, memory, 0).ok()?;
            Some((buffer, memory, size))
        }
    }
}

impl Drop for VkRenderer {
    fn drop(&mut self) {
        let device = &self.ctx.device;
        unsafe {
            let _ = device.device_wait_idle();
            crate::gfx::vk_backend::shutdown();
            if let Some(mut target) = self.gamma_target.take() {
                target.destroy(device);
            }
            for slot in &self.slots {
                device.destroy_command_pool(slot.pool, None);
                device.destroy_semaphore(slot.image_available, None);
                device.destroy_fence(slot.in_flight, None);
            }
            for &semaphore in &self.render_finished {
                device.destroy_semaphore(semaphore, None);
            }
        }
        if let Some(mut swapchain) = self.swapchain.take() {
            swapchain.destroy(&self.ctx);
        }
    }
}

fn create_slot(ctx: &VkContext) -> Result<FrameSlot, vk::Result> {
    let device = &ctx.device;
    unsafe {
        let pool = device.create_command_pool(
            &vk::CommandPoolCreateInfo::builder()
                .flags(vk::CommandPoolCreateFlags::TRANSIENT)
                .queue_family_index(ctx.queue_family),
            None,
        )?;
        let commands = device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::builder()
                .command_pool(pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1),
        )?[0];
        Ok(FrameSlot {
            pool,
            commands,
            image_available: device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?,
            // Signalled, so the first wait on each slot returns at once.
            in_flight: device.create_fence(
                &vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED),
                None,
            )?,
        })
    }
}

unsafe fn image_barrier(
    device: &ash::Device,
    commands: vk::CommandBuffer,
    image: vk::Image,
    from: vk::ImageLayout,
    to: vk::ImageLayout,
) {
    let barrier = vk::ImageMemoryBarrier::builder()
        .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE | vk::AccessFlags::TRANSFER_READ)
        .dst_access_mask(vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::MEMORY_READ)
        .old_layout(from)
        .new_layout(to)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(swapchain::subresource(vk::ImageAspectFlags::COLOR))
        .build();
    unsafe {
        device.cmd_pipeline_barrier(
            commands,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        );
    }
}
