//! The Vulkan 1.0 renderer.
//!
//! [`VkRenderer`] owns the device and the window's swapchain and runs the
//! frame loop: [`VkRenderer::begin_frame`] waits for a free frame slot,
//! acquires a swapchain image and opens its render pass on a fresh command
//! buffer; [`VkRenderer::end_frame`] closes it, submits and presents. Two
//! frames are in flight, so the CPU records one while the GPU draws the other.

mod context;
mod swapchain;

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
    rebuild: bool,
    rebuilds: u32,
    pub ctx: VkContext,
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
            rebuild: true,
            rebuilds: 0,
            ctx,
        };
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

    pub fn swapchain(&self) -> Option<&VkSwapchain> {
        self.swapchain.as_ref()
    }

    /// The window's framebuffer changed size; the swapchain follows before
    /// the next frame.
    pub fn resize(&mut self, width: u32, height: u32) {
        if (width, height) != self.size {
            self.size = (width, height);
            self.rebuild = true;
        }
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
        match VkSwapchain::new(&self.ctx, self.size, self.vsync) {
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
        let device = &self.ctx.device;
        let slot = &self.slots[self.slot];
        unsafe {
            device
                .wait_for_fences(&[slot.in_flight], true, u64::MAX)
                .expect("vkWaitForFences");
        }
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
                    .render_pass(swapchain.render_pass)
                    .framebuffer(swapchain.framebuffers[image_index as usize])
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D::default(),
                        extent: swapchain.extent,
                    })
                    .clear_values(&clears),
                vk::SubpassContents::INLINE,
            );
        }
        Some(Frame {
            commands: slot.commands,
            extent: swapchain.extent,
            image_index,
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
        unsafe {
            device.cmd_end_render_pass(frame.commands);
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
        }
        self.slot = (self.slot + 1) % FRAMES_IN_FLIGHT;
    }
}

impl Drop for VkRenderer {
    fn drop(&mut self) {
        let device = &self.ctx.device;
        unsafe {
            let _ = device.device_wait_idle();
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
