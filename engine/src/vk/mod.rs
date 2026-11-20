//! The Vulkan 1.0 renderer.
//!
//! [`VkRenderer`] owns the device and the window's swapchain and runs the
//! frame loop: [`VkRenderer::begin_frame`] waits for a free frame slot,
//! acquires a swapchain image and opens its render pass on a fresh command
//! buffer; [`VkRenderer::end_frame`] closes it, submits and presents. Two
//! frames are in flight, so the CPU records one while the GPU draws the other.

mod context;
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
    capture: bool,
    captured: Option<Capture>,
    pub ctx: VkContext,
}

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
            rebuild: true,
            rebuilds: 0,
            capture: false,
            captured: None,
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
        crate::gfx::vk_backend::begin_frame(
            slot.commands,
            swapchain.render_pass,
            (swapchain.format, swapchain.depth_format),
            swapchain.extent,
            self.slot,
        );
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
        crate::gfx::vk_backend::end_frame();
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
