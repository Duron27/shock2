//! Vulkan bring-up check: open a window, clear it to a cycling colour, and
//! exit non-zero if the validation layer reported any error.
//!
//!   cargo run -p engine --example vk_clear -- [--frames N] [--hidden] [--no-vsync] [--resize-test]
//!
//! `--resize-test` resizes the window every 30 frames, to exercise rebuilds.
//!
//! `SHOCK2_VK_DEVICE=llvmpipe` runs on the software rasteriser.

fn main() {
    // This example exists to catch validation errors: on unless told not.
    if std::env::var_os("SHOCK2_VK_VALIDATION").is_none() {
        unsafe { std::env::set_var("SHOCK2_VK_VALIDATION", "1") };
    }
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();
    let args: Vec<String> = std::env::args().collect();
    let frames: Option<u64> = args
        .iter()
        .position(|a| a == "--frames")
        .and_then(|i| args.get(i + 1))
        .and_then(|n| n.parse().ok());
    let hidden = args.iter().any(|a| a == "--hidden");
    let vsync = !args.iter().any(|a| a == "--no-vsync");
    let resize_test = args.iter().any(|a| a == "--resize-test");

    let mut glfw = glfw::init(glfw::fail_on_errors).unwrap();
    glfw.window_hint(glfw::WindowHint::ClientApi(glfw::ClientApiHint::NoApi));
    glfw.window_hint(glfw::WindowHint::Visible(!hidden));
    let (mut window, events) = glfw
        .create_window(800, 600, "Vulkan clear", glfw::WindowMode::Windowed)
        .expect("window");
    window.set_framebuffer_size_polling(true);
    window.set_key_polling(true);

    let extensions = glfw
        .get_required_instance_extensions()
        .expect("GLFW reports no Vulkan support");
    let ctx = engine::vk::VkContext::new("shock2 vk_clear", &extensions, |instance| {
        let mut surface = ash::vk::SurfaceKHR::null();
        match window.create_window_surface(instance, std::ptr::null(), &mut surface) {
            ash::vk::Result::SUCCESS => Ok(surface),
            e => Err(e),
        }
    })
    .expect("Vulkan context");
    let (w, h) = window.get_framebuffer_size();
    let mut renderer =
        engine::vk::VkRenderer::new(ctx, (w as u32, h as u32), vsync).expect("renderer");
    println!("device: {}", renderer.device_name());

    let start = std::time::Instant::now();
    let mut drawn = 0u64;
    while !window.should_close() && frames.is_none_or(|n| drawn < n) {
        glfw.poll_events();
        for (_, event) in glfw::flush_messages(&events) {
            match event {
                glfw::WindowEvent::FramebufferSize(w, h) => {
                    renderer.resize(w.max(0) as u32, h.max(0) as u32)
                }
                glfw::WindowEvent::Key(glfw::Key::Escape, _, glfw::Action::Press, _) => {
                    window.set_should_close(true)
                }
                _ => {}
            }
        }
        let t = start.elapsed().as_secs_f32();
        let clear = [
            0.5 + 0.5 * t.sin(),
            0.5 + 0.5 * (t * 0.7 + 2.0).sin(),
            0.5 + 0.5 * (t * 1.3 + 4.0).sin(),
            1.0,
        ];
        if resize_test && drawn % 30 == 29 {
            let sizes = [(640, 480), (1024, 300), (300, 700), (800, 600)];
            let (w, h) = sizes[(drawn / 30) as usize % sizes.len()];
            window.set_size(w, h);
        }
        if let Some(frame) = renderer.begin_frame(clear) {
            renderer.end_frame(frame);
            drawn += 1;
        }
    }
    let seconds = start.elapsed().as_secs_f64();
    println!(
        "{} swapchains built, last {:?}",
        renderer.swapchain_builds(),
        renderer.swapchain().map(|s| s.extent)
    );
    drop(renderer);
    let errors = engine::vk::validation_error_count();
    println!(
        "{drawn} frames in {seconds:.2}s ({:.0} fps), {errors} validation errors",
        drawn as f64 / seconds
    );
    std::process::exit(if errors == 0 { 0 } else { 1 });
}
