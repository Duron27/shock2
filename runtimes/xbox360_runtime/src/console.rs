//! The title itself: xsdk's entry point, the log, the display and the frame
//! loop (the SDL runtime's, with the pad as the only input).

use std::ffi::c_char;
use std::time::{Duration, Instant};

use cgmath::prelude::*;
use cgmath::{Vector2, vec2, vec3};
use dark::SCALE_FACTOR;
use engine::scene::Scene;
use engine::util::compute_view_matrix_from_render_context;
use engine::vk::ash::{self, vk};
use gamepad::{PAD_LOOK_DEGREES_PER_SECOND, PAD_POINTER_SPEED, PadState, Rumble};
use shock2vr::input::InputActionState;
use shock2vr::input_context::{InputContext, Pointer2D};
use shock2vr::time::Time;
use shock2vr::{GameOptions, SpawnLocation};
use tracing::{info, warn};

use crate::{Look, XInputGamepad, pad_state, starting_mission, wants_exit};

/// The precompiled shaders, from `xbox360/tools/console_shaders.sh`.
const SHADER_TABLE: &str = r"game:\shock2_shaders.table";
const MISSION_TXT: &str = r"game:\mission.txt";
/// Experimental feature names (as `--experimental` takes them), comma or line separated, e.g. `profile_cpu`.
const EXPERIMENTAL_TXT: &str = r"game:\experimental.txt";

fn experimental_features(text: &str) -> std::collections::HashSet<String> {
    text.split(|c: char| c == ',' || c.is_whitespace())
        .filter(|name| !name.is_empty())
        .map(str::to_owned)
        .collect()
}

#[repr(C)]
#[derive(Default)]
struct InputState {
    packet_number: u32,
    gamepad: XInputGamepad,
}

#[repr(C)]
struct InputVibration {
    left_motor_speed: u16,
    right_motor_speed: u16,
}

unsafe extern "C" {
    fn xsdk_log_write(format: *const c_char, ...);
    fn XamInputGetState(user: u32, flags: u32, state: *mut InputState) -> u32;
    fn XamInputSetState(user: u32, flags: u32, vibration: *const InputVibration) -> u32;
    fn exit(status: i32) -> !;
    fn MmQueryStatistics(statistics: *mut MemoryStatistics) -> i32;
}

/// `MM_QUERY_STATISTICS_SECTION` (xsdk `kernel.h`), in 4 KB pages.
#[repr(C)]
#[derive(Default)]
struct MemorySection {
    available_pages: u32,
    total_virtual_memory_bytes: u32,
    reserved_virtual_memory_bytes: u32,
    physical_pages: u32,
    pool_pages: u32,
    stack_pages: u32,
    image_pages: u32,
    heap_pages: u32,
    virtual_pages: u32,
    page_table_pages: u32,
    cache_pages: u32,
}

/// `MM_QUERY_STATISTICS_RESULT`.
#[repr(C)]
#[derive(Default)]
struct MemoryStatistics {
    size: u32,
    total_physical_pages: u32,
    kernel_pages: u32,
    title: MemorySection,
    system: MemorySection,
    highest_physical_page: u32,
}

/// Logs the title's and the system's memory every two seconds, so a run
/// that starves the console shows it coming.
fn watch_memory() {
    const MIB_PAGES: u32 = 256;
    let _ = std::thread::Builder::new()
        .name("memwatch".into())
        .spawn(|| {
            loop {
                let mut stats = MemoryStatistics {
                    size: std::mem::size_of::<MemoryStatistics>() as u32,
                    ..Default::default()
                };
                if unsafe { MmQueryStatistics(&mut stats) } >= 0 {
                    let t = &stats.title;
                    log_line(&format!(
                        "mem: title {} MiB free, physical {} heap {} virtual {}; system {} MiB free",
                        t.available_pages / MIB_PAGES,
                        t.physical_pages / MIB_PAGES,
                        t.heap_pages / MIB_PAGES,
                        t.virtual_pages / MIB_PAGES,
                        stats.system.available_pages / MIB_PAGES,
                    ));
                }
                // The GPU's registers, readable by titles at 0x7FC80000 (xsdk
                // docs/HARDWARE.md): RBBM_STATUS and the ring's read and write
                // pointers, to see where a hang stopped.
                let gpu = |offset: u32| unsafe {
                    std::ptr::read_volatile((0x7FC8_0000u32 + offset) as usize as *const u32)
                };
                log_line(&format!(
                    "gpu: RBBM_STATUS {:08x}, ring read {:#x} write {:#x}",
                    gpu(0x1740),
                    gpu(0x710),
                    gpu(0x714)
                ));
                std::thread::sleep(std::time::Duration::from_secs(2));
            }
        });
}

unsafe extern "system" {
    fn vkGetInstanceProcAddr(instance: vk::Instance, name: *const c_char)
    -> vk::PFN_vkVoidFunction;
}

fn log_line(line: &str) {
    unsafe {
        xsdk_log_write(
            c"%.*s".as_ptr(),
            line.len() as i32,
            line.as_ptr() as *const c_char,
        )
    }
}

/// tracing's output, a line at a time, into xsdk's log.
struct XsdkLog;

impl std::io::Write for XsdkLog {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let text = String::from_utf8_lossy(bytes);
        for line in text.lines().filter(|l| !l.is_empty()) {
            log_line(line);
        }
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn pad() -> PadState {
    let mut state = InputState::default();
    // ERROR_SUCCESS; anything else is no pad on port 1.
    if unsafe { XamInputGetState(0, 0, &mut state) } == 0 {
        pad_state(&state.gamepad)
    } else {
        PadState::default()
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn main(_argc: i32, _argv: *const *const u8) -> i32 {
    std::panic::set_hook(Box::new(|info| {
        log_line(&format!("panic: {info}"));
        // newlib's exit writes xsdk's last log lines.
        unsafe { exit(3) }
    }));
    let _ = tracing_subscriber::fmt()
        .with_ansi(false)
        .without_time()
        .with_max_level(crate::log_level(
            std::fs::read_to_string(r"game:\log_level.txt")
                .ok()
                .as_deref(),
        ))
        .with_writer(|| XsdkLog)
        .try_init();
    log_line("shock2: Back + Start together returns to the dashboard");
    if let Ok(text) = std::fs::read_to_string(r"game:\dev_params.txt") {
        for (key, value) in crate::dev_param_lines(&text) {
            match shock2vr::dev_params::find(key) {
                Some(id) => {
                    let set = shock2vr::dev_params::set(id, value);
                    info!("dev param {key} = {set}");
                }
                None => warn!("dev_params.txt: no parameter {key}"),
            }
        }
    }
    watch_memory();
    match run() {
        Ok(()) => 0,
        Err(e) => {
            log_line(&format!("shock2: {e}"));
            1
        }
    }
}

/// VK_KHR_display's first display at its first mode, plane 0 (vulkan4xenon
/// docs/SUPPORT.md, "Display and swapchain"); its size is the swapchain's.
fn display_surface(
    entry: &ash::Entry,
    handle: vk::Instance,
) -> Result<(vk::SurfaceKHR, vk::Extent2D), vk::Result> {
    let instance = unsafe { ash::Instance::load(entry.static_fn(), handle) };
    let physical = *unsafe { instance.enumerate_physical_devices() }?
        .first()
        .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
    let display_fn = ash::extensions::khr::Display::new(entry, &instance);
    let display = *unsafe { display_fn.get_physical_device_display_properties(physical) }?
        .first()
        .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
    let mode = *unsafe { display_fn.get_display_mode_properties(physical, display.display) }?
        .first()
        .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
    let extent = mode.parameters.visible_region;
    info!(
        "display {}x{} at {} mHz",
        extent.width, extent.height, mode.parameters.refresh_rate
    );
    let surface = unsafe {
        display_fn.create_display_plane_surface(
            &vk::DisplaySurfaceCreateInfoKHR::builder()
                .display_mode(mode.display_mode)
                .plane_index(0)
                .transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
                .alpha_mode(vk::DisplayPlaneAlphaFlagsKHR::OPAQUE)
                .image_extent(extent),
            None,
        )
    }?;
    Ok((surface, extent))
}

fn run() -> Result<(), String> {
    let table = std::fs::read(SHADER_TABLE).map_err(|e| format!("{SHADER_TABLE}: {e}"))?;
    engine::vk::shader::set_console_table(Box::leak(table.into_boxed_slice()));

    let entry = unsafe {
        ash::Entry::from_static_fn(vk::StaticFn {
            get_instance_proc_addr: vkGetInstanceProcAddr,
        })
    };
    let extensions = [
        ash::extensions::khr::Surface::name(),
        ash::extensions::khr::Display::name(),
    ]
    .map(|n| n.to_string_lossy().into_owned());
    let mut extent = vk::Extent2D::default();
    let ctx = engine::vk::VkContext::new("Shock Engine", &extensions, |instance| {
        let (surface, size) = display_surface(&entry, instance)?;
        extent = size;
        Ok(surface)
    })?;
    let mut renderer = engine::vk::VkRenderer::new(ctx, (extent.width, extent.height), true)?;
    info!("Vulkan device: {}", renderer.device_name());

    let engine = engine::renderer();
    let mission_txt = std::fs::read_to_string(MISSION_TXT).ok();
    let mission = starting_mission(mission_txt.as_deref());
    info!("starting {mission}");
    let options = GameOptions {
        build_label: env!("SHOCK2QUEST_BUILD_LABEL"),
        save_writer: shock2vr::save_load::SaveWriter::new(
            env!("SHOCK2QUEST_BUILD_VERSION"),
            env!("SHOCK2QUEST_BUILD_COMMIT"),
            env!("SHOCK2QUEST_BUILD_DIRTY"),
        ),
        mission,
        presentation_mode: shock2vr::PresentationMode::Flat,
        spawn_location: SpawnLocation::MapDefault,
        render_particles: true,
        experimental_features: std::fs::read_to_string(EXPERIMENTAL_TXT)
            .map(|text| experimental_features(&text))
            .unwrap_or_default(),
        ..GameOptions::default()
    };
    info!("experimental: {:?}", options.experimental_features);
    info!(
        "main thread on hardware thread {:?}",
        engine::hardware_threads::current()
    );
    let mut game = shock2vr::App::init(options, engine.get_storage());

    let mut look = Look::default();
    let mut action_state = InputActionState::new();
    let mut previous_pad = PadState::default();
    let mut rumble = Rumble::default();
    let (width, height) = (extent.width as f32, extent.height as f32);
    let mut pointer = vec2(width / 2.0, height / 2.0);
    let mut pointer_mode: Option<bool> = None;

    let start = Instant::now();
    let mut last = start;
    let mut frame_count: u64 = 0;
    // Frames to read back into game:\capture<N>.ppm, from `capture.txt`
    // (frame numbers): xeedee's screenshots of this title come back banded.
    let capture_frames: Vec<u64> = std::fs::read_to_string(r"game:\capture.txt")
        .unwrap_or_default()
        .split_whitespace()
        .filter_map(|word| word.parse().ok())
        .collect();
    // Each stage's milliseconds summed over the frames since the last log:
    // update, render, per eye, lights, begin, record, end, the whole loop.
    let mut sums = [0.0f32; 8];
    let mut summed = 0u32;
    let mut slowest = 0.0f32;
    let mut loop_start = Instant::now();
    loop {
        let wants_pointer = game.wants_pointer();
        if pointer_mode != Some(wants_pointer) {
            pointer = vec2(width / 2.0, height / 2.0);
            pointer_mode = Some(wants_pointer);
        }

        let now = Instant::now();
        // A long stall (loading) must not become one huge step.
        let delta = (now - last).min(Duration::from_millis(100));
        last = now;
        let dt = delta.as_secs_f32();
        frame_count += 1;

        let pad_now = pad();
        if wants_exit(&pad_now) {
            info!("Back + Start: leaving");
            break;
        }
        previous_pad.sync_actions(&pad_now, wants_pointer, &mut action_state);
        previous_pad = pad_now.clone();

        let pc = shock2vr::user_settings::get().pc;
        let invert = if pc.invert_mouse_y { -1.0 } else { 1.0 };
        if !wants_pointer {
            let pad_scale = PAD_LOOK_DEGREES_PER_SECOND * pc.mouse_sensitivity * dt;
            look.yaw += pad_now.right_stick.x * pad_scale;
            look.pitch -= pad_now.right_stick.y * pad_scale * invert;
            look.pitch = look.pitch.clamp(-89.0, 89.0);
        }

        let mut input_context = InputContext::default();
        input_context.head.rotation = look.rotation();
        input_context.head.position = vec3(0.0, shock2vr::input_context::DEFAULT_HEAD_HEIGHT, 0.0);
        let (right_position, right_rotation) = look.hand_pose(1.0);
        let (left_position, left_rotation) = look.hand_pose(-1.0);
        input_context.right_hand.position = right_position;
        input_context.right_hand.rotation = right_rotation;
        input_context.left_hand.position = left_position;
        input_context.left_hand.rotation = left_rotation;
        input_context.left_hand.thumbstick = Vector2::zero();
        if wants_pointer {
            // The left stick moves the cursor; A clicks, B right-clicks.
            pointer.x += pad_now.left_stick.x * PAD_POINTER_SPEED * width * dt;
            pointer.y -= pad_now.left_stick.y * PAD_POINTER_SPEED * height * dt;
            pointer.x = pointer.x.clamp(0.0, width);
            pointer.y = pointer.y.clamp(0.0, height);
            input_context.pointer = Some(Pointer2D {
                position: vec2(pointer.x / width, pointer.y / height),
                secondary: pad_now.pointer_secondary(),
                pressed: pad_now.pointer_pressed(),
            });
        } else {
            input_context.right_hand.thumbstick = pad_now.left_stick;
            input_context.right_hand.trigger_value = f32::from(u8::from(pad_now.right_trigger));
            input_context.right_hand.squeeze_value = f32::from(u8::from(pad_now.left_trigger));
            input_context.lean = pad_now.lean();
            input_context.crouch = pad_now.crouch();
            input_context.jump = pad_now.jump();
        }

        let time = Time {
            elapsed: delta,
            total: now - start,
        };
        game.update(&time, &input_context, &mut action_state);
        if pc.gamepad_rumble {
            rumble.push(game.take_haptics(), Instant::now());
        } else {
            let _ = game.take_haptics();
            rumble.stop();
        }
        if let Some((low, high, _)) = rumble.command(Instant::now()) {
            let vibration = InputVibration {
                left_motor_speed: low,
                right_motor_speed: high,
            };
            unsafe { XamInputSetState(0, 0, &vibration) };
        }
        if game.should_quit() {
            break;
        }

        let updated = Instant::now();
        let projection_matrix = cgmath::perspective(
            cgmath::Deg(game.desired_fov_deg()),
            width / height,
            shock2vr::NEAR_PLANE,
            shock2vr::FAR_PLANE,
        );
        let screen_size = vec2(width, height);
        let (mut scene, pawn_offset, pawn_rotation) = game.render();
        let rendered = Instant::now();
        let head_height = game.player_eye_height();
        let render_context = game
            .resolve_camera(
                pawn_offset,
                pawn_rotation,
                vec3(0.0, head_height / SCALE_FACTOR, 0.0),
                look.rotation(),
            )
            .into_render_context(game.render_time(), projection_matrix, screen_size);
        let view = compute_view_matrix_from_render_context(&render_context);
        let per_eye_scene = game.render_per_eye(view, projection_matrix, screen_size);
        game.finish_render(view, projection_matrix, screen_size);
        scene.extend(per_eye_scene);
        let per_eye = Instant::now();

        let mut scene_for_render = Scene::from_objects(scene);
        for spotlight in game.get_hand_lights() {
            scene_for_render.lights_mut().add_light(spotlight);
        }
        (
            scene_for_render.point_shadows,
            scene_for_render.point_shadow_strength,
        ) = game.get_shadow_lights();
        scene_for_render.fog_density = game.fog_density();
        scene_for_render.world_gloss = game.world_gloss();
        scene_for_render.bump_strength = game.bump_strength();
        scene_for_render.lights_mut().environment = game.environment();

        renderer.set_gamma(pc.brightness);
        renderer.set_grade(game.screen_grade(), start.elapsed().as_secs_f32());
        let built = Instant::now();
        let (mut began, mut recorded) = (built, built);
        match renderer.begin_frame([0.0, 0.0, 0.0, 1.0]) {
            Some(frame) => {
                began = Instant::now();
                engine.render(&render_context, &scene_for_render);
                recorded = Instant::now();
                let capture = capture_frames.contains(&frame_count);
                if capture {
                    renderer.request_capture();
                }
                renderer.end_frame(frame);
                if let Some(image) = capture.then(|| renderer.take_capture()).flatten() {
                    write_capture(frame_count, &image);
                }
            }
            None => warn!("no frame to draw into"),
        }
        let ms = |a: Instant, b: Instant| (b - a).as_secs_f32() * 1000.0;
        let ended = Instant::now();
        let whole = ms(loop_start, ended);
        loop_start = ended;
        for (sum, stage) in sums.iter_mut().zip([
            ms(now, updated),
            ms(updated, rendered),
            ms(rendered, per_eye),
            ms(per_eye, built),
            ms(built, began),
            ms(began, recorded),
            ms(recorded, ended),
            whole,
        ]) {
            *sum += stage;
        }
        summed += 1;
        slowest = slowest.max(whole);
        if frame_count <= 30 || frame_count % 60 == 0 {
            let n = summed.max(1) as f32;
            let [update, render, per_eye, lights, begin, record, end, whole] =
                sums.map(|sum| sum / n);
            info!(
                "frame {frame_count}: mean of {summed}: update {update:.1} ms, scene {:.1} (render {render:.1}, per eye {per_eye:.1}, lights {lights:.1}), begin {begin:.1}, record {record:.1}, end {end:.1}; loop {whole:.1}, slowest {slowest:.1}; draws {}",
                render + per_eye + lights,
                engine::gfx::vk_backend::take_draw_counts(),
            );
            (sums, summed, slowest) = ([0.0; 8], 0, 0.0);
            if let Some(report) = shock2vr::perf::take_report() {
                let phases = report["phases"].as_object().map(|phases| {
                    phases
                        .iter()
                        .map(|(name, p)| {
                            format!("{name} {:.1}", p["mean_ms"].as_f64().unwrap_or(0.0))
                        })
                        .collect::<Vec<_>>()
                        .join(", ")
                });
                info!("cpu phases (mean ms): {}", phases.unwrap_or_default());
            }
        }
    }
    let stop = InputVibration {
        left_motor_speed: 0,
        right_motor_speed: 0,
    };
    unsafe { XamInputSetState(0, 0, &stop) };
    Ok(())
}

/// A frame read back, as a binary PPM beside the `.xex`.
fn write_capture(frame: u64, image: &engine::vk::Capture) {
    let mut ppm = format!("P6\n{} {}\n255\n", image.width, image.height).into_bytes();
    ppm.extend(image.rgba.chunks_exact(4).flat_map(|p| [p[0], p[1], p[2]]));
    let path = format!(r"game:\capture{frame}.ppm");
    match std::fs::write(&path, ppm) {
        Ok(()) => info!("captured frame {frame} into {path}"),
        Err(e) => warn!("capture {path}: {e}"),
    }
}
