//! The flatscreen game on SDL2.
//!
//! One loop for two hosts: the `shock2_sdl` desktop binary, and
//! `libshock2.so`, which alpha3's `SDLActivity` loads on Android and enters
//! through `SDL_main` (see `android.rs`). SDL owns the window, the Vulkan
//! surface and input; the game and renderer are the same ones the GLFW
//! desktop runtime drives.
//!
//! Pointer contract with the host: while a scene wants a 2D cursor
//! (`App::wants_pointer`: menus, use mode, MFD panels) [`pointer_shown`] is
//! true and mouse motion moves that cursor; otherwise motion turns the view.
//! alpha3 polls this through `SDLActivity.isMouseShown()` every frame to
//! switch its touch controls between mouse-look and a visible cursor.

use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::time::{Duration, Instant};

use cgmath::prelude::*;
use cgmath::{Decomposed, Quaternion, Vector2, Vector3, point3, vec2, vec3};
use clap::Parser;
use dark::SCALE_FACTOR;
use engine::scene::Scene;
use engine::util::compute_view_matrix_from_render_context;
use engine::vk::ash::vk::{self, Handle};
use gamepad::{Gamepads, PAD_LOOK_DEGREES_PER_SECOND, PAD_POINTER_SPEED};
use sdl2::event::Event;
use sdl2::keyboard::Scancode;
use shock2vr::input::InputActionState;
use shock2vr::input::keyboard::{Key, KeyboardActionMapper};
use shock2vr::input_context::{InputContext, Pointer2D};
use shock2vr::time::Time;
use shock2vr::{GameOptions, SpawnLocation};
use tracing::warn;

#[cfg(target_os = "android")]
mod android;

/// Mouse-look degrees per pixel at sensitivity 1x (the desktop runtime's).
const MOUSE_DEGREES_PER_PIXEL: f32 = 0.15;
/// Hand yaw the simulated hands rest at: aiming straight down the view.
const REST_HAND_YAW: f32 = 90.0;

static POINTER_SHOWN: AtomicBool = AtomicBool::new(false);
static POINTER_X: AtomicI32 = AtomicI32::new(0);
static POINTER_Y: AtomicI32 = AtomicI32::new(0);

/// Whether the game is showing a 2D cursor (menus, use mode, MFD panels).
pub fn pointer_shown() -> bool {
    POINTER_SHOWN.load(Ordering::Relaxed)
}

/// The cursor's position in window coordinates.
pub fn pointer_position() -> (i32, i32) {
    (
        POINTER_X.load(Ordering::Relaxed),
        POINTER_Y.load(Ordering::Relaxed),
    )
}

#[derive(Parser, Debug, Default)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Mission or debug scene to load (`medsci1.mis`, `earth.mis:12`). If
    /// omitted, boots the main menu.
    #[arg(short, long)]
    mission: Option<String>,

    #[arg(short, long)]
    save_file: Option<String>,

    #[arg(short, long)]
    experimental: Option<Vec<String>>,
}

/// Run the game until it quits; the process exit code.
pub fn run(args: Vec<String>) -> i32 {
    let args = Args::try_parse_from(&args).unwrap_or_else(|e| {
        // A host passing arguments this runtime doesn't know (Android hands
        // over whatever the launcher configured) still gets a game.
        eprintln!("shock2: ignoring arguments: {e}");
        Args::default()
    });
    match run_game(args) {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("shock2: {e}");
            1
        }
    }
}

fn run_game(args: Args) -> Result<(), String> {
    engine::logging::init_logging("SHOCK2_LOG");
    #[cfg(target_os = "android")]
    android::prepare_paths();

    let sdl = sdl2::init()?;
    let video = sdl.video()?;
    let mut gamepads = Gamepads::new(&sdl);
    let mut window_builder = video.window("System Shock 2", 1280, 720);
    window_builder.vulkan().allow_highdpi().resizable();
    if cfg!(target_os = "android") {
        window_builder.fullscreen_desktop();
    } else {
        window_builder.position_centered();
    }
    let mut window = window_builder.build().map_err(|e| e.to_string())?;

    let mut renderer = {
        let extensions: Vec<String> = window
            .vulkan_instance_extensions()?
            .into_iter()
            .map(str::to_owned)
            .collect();
        let ctx = engine::vk::VkContext::new("Shock Engine", &extensions, |instance| {
            create_surface(&window, instance)
        })?;
        let renderer = engine::vk::VkRenderer::new(
            ctx,
            window.vulkan_drawable_size(),
            shock2vr::user_settings::get().pc.vsync,
        )?;
        println!("Vulkan device: {}", renderer.device_name());
        renderer
    };

    let engine = engine::renderer();
    let (mission, spawn_location) = parse_mission(args.mission.as_deref().unwrap_or("main_menu"))?;
    let options = GameOptions {
        build_label: env!("SHOCK2QUEST_BUILD_LABEL"),
        save_writer: shock2vr::save_load::SaveWriter::new(
            env!("SHOCK2QUEST_BUILD_VERSION"),
            env!("SHOCK2QUEST_BUILD_COMMIT"),
            env!("SHOCK2QUEST_BUILD_DIRTY"),
        ),
        mission,
        presentation_mode: shock2vr::PresentationMode::Flat,
        spawn_location,
        save_file: args.save_file,
        render_particles: true,
        experimental_features: args.experimental.unwrap_or_default().into_iter().collect(),
        ..GameOptions::default()
    };
    let mut game = shock2vr::App::init(options, engine.get_storage());

    let mut event_pump = sdl.event_pump()?;
    let mut look = Look::default();
    let mut keyboard = KeyboardActionMapper::new();
    let mut action_state = InputActionState::new();
    let mut pointer_mode: Option<bool> = None;
    let mut pointer = vec2(0.0f32, 0.0);
    // A mouse button still held when a cursor UI closes must not fire the
    // weapon: swallow it until every button is up (as the desktop runtime).
    let mut suppress_mouse_until_release = false;
    let mut in_background = false;
    let mut applied_fullscreen: Option<bool> = None;

    let start = Instant::now();
    let mut last = start;
    let mut frame_number = 0u64;
    'frames: loop {
        let wants_pointer = game.wants_pointer();
        if pointer_mode != Some(wants_pointer) {
            // The host draws its own cursor on Android and feeds absolute or
            // relative motion itself; grabbing the mouse there would ask
            // Android for pointer capture and fight the launcher's controls.
            if !cfg!(target_os = "android") {
                sdl.mouse().set_relative_mouse_mode(!wants_pointer);
            }
            if wants_pointer {
                let (w, h) = window.size();
                pointer = vec2(w as f32 / 2.0, h as f32 / 2.0);
                sdl.mouse()
                    .warp_mouse_in_window(&window, pointer.x as i32, pointer.y as i32);
            }
            pointer_mode = Some(wants_pointer);
        }
        POINTER_SHOWN.store(wants_pointer, Ordering::Relaxed);

        let mut look_pixels = vec2(0.0f32, 0.0);
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. } => break 'frames,
                // Android destroys the native window in the background; the
                // swapchain and surface go with it and return with the app.
                Event::AppWillEnterBackground { .. } | Event::AppDidEnterBackground { .. } => {
                    if !in_background {
                        renderer.release_surface();
                        in_background = true;
                    }
                }
                Event::AppDidEnterForeground { .. } => {
                    if in_background {
                        match renderer.recreate_surface(|i| create_surface(&window, i)) {
                            Ok(()) => in_background = false,
                            Err(e) => warn!("no surface on returning to the foreground: {e}"),
                        }
                        last = Instant::now();
                    }
                }
                Event::MouseMotion {
                    x, y, xrel, yrel, ..
                } => {
                    pointer = vec2(x as f32, y as f32);
                    if !wants_pointer {
                        look_pixels += vec2(xrel as f32, yrel as f32);
                    }
                }
                Event::ControllerDeviceAdded { .. } | Event::ControllerDeviceRemoved { .. } => {
                    gamepads.handle_event(&event);
                }
                _ => {}
            }
        }
        if in_background {
            std::thread::sleep(Duration::from_millis(50));
            continue;
        }
        if renderer.surface_lost() {
            if let Err(e) = renderer.recreate_surface(|i| create_surface(&window, i)) {
                warn!("window surface lost and not recreated: {e}");
                std::thread::sleep(Duration::from_millis(50));
                continue;
            }
        }
        if !cfg!(target_os = "android") {
            apply_fullscreen(&mut window, &mut applied_fullscreen);
        }

        let now = Instant::now();
        // A long stall (loading, a debugger) must not become one huge step.
        let delta = (now - last).min(Duration::from_millis(100));
        last = now;
        let dt = delta.as_secs_f32();

        let keys = event_pump.keyboard_state();
        let down = |key: Key| keys.is_scancode_pressed(scancode(key));
        let mouse = event_pump.mouse_state();
        let pad_now = gamepads.poll(wants_pointer, &mut action_state);

        // Look: mouse pixels plus the right stick, scaled by the settings.
        let pc = shock2vr::user_settings::get().pc;
        let invert = if pc.invert_mouse_y { -1.0 } else { 1.0 };
        if !wants_pointer {
            let mouse_scale = MOUSE_DEGREES_PER_PIXEL * pc.mouse_sensitivity;
            let pad_scale = PAD_LOOK_DEGREES_PER_SECOND * pc.mouse_sensitivity * dt;
            look.yaw += look_pixels.x * mouse_scale + pad_now.right_stick.x * pad_scale;
            look.pitch +=
                (look_pixels.y * mouse_scale - pad_now.right_stick.y * pad_scale) * invert;
            look.pitch = look.pitch.clamp(-89.0, 89.0);
        }

        let alt = down(Key::LeftAlt) || down(Key::RightAlt);
        let shift = down(Key::LeftShift) || down(Key::RightShift);
        let key_axis =
            |neg: Key, pos: Key| f32::from(u8::from(down(pos))) - f32::from(u8::from(down(neg)));
        let mut movement = Vector2::zero();
        let mut turn = Vector2::zero();
        if !alt {
            movement = vec2(key_axis(Key::A, Key::D), key_axis(Key::S, Key::W));
        }
        turn += vec2(
            key_axis(Key::Left, Key::Right),
            key_axis(Key::Down, Key::Up),
        );
        if shift {
            movement *= 2.0;
            turn *= 2.0;
        }
        if !wants_pointer {
            movement += pad_now.left_stick;
        }

        let mut input_context = InputContext::default();
        input_context.head.rotation = look.rotation();
        input_context.head.position = vec3(0.0, shock2vr::input_context::DEFAULT_HEAD_HEIGHT, 0.0);
        let (right_position, right_rotation) = look.hand_pose(1.0);
        let (left_position, left_rotation) = look.hand_pose(-1.0);
        input_context.right_hand.position = right_position;
        input_context.right_hand.rotation = right_rotation;
        input_context.right_hand.thumbstick = movement;
        input_context.left_hand.position = left_position;
        input_context.left_hand.rotation = left_rotation;
        input_context.left_hand.thumbstick = turn;

        if wants_pointer {
            // The stick nudges the cursor too, so a pad alone can work menus.
            if pad_now.left_stick != Vector2::zero() {
                let (w, h) = window.size();
                pointer.x += pad_now.left_stick.x * PAD_POINTER_SPEED * w as f32 * dt;
                pointer.y -= pad_now.left_stick.y * PAD_POINTER_SPEED * h as f32 * dt;
                pointer.x = pointer.x.clamp(0.0, w as f32);
                pointer.y = pointer.y.clamp(0.0, h as f32);
            }
            let (w, h) = window.size();
            input_context.pointer = Some(Pointer2D {
                position: vec2(pointer.x / w.max(1) as f32, pointer.y / h.max(1) as f32),
                secondary: mouse.right() || pad_now.pointer_secondary(),
                pressed: mouse.left() || pad_now.pointer_pressed(),
            });
            suppress_mouse_until_release = true;
        } else {
            if !mouse.left() && !mouse.right() {
                suppress_mouse_until_release = false;
            }
            let mouse_live = !suppress_mouse_until_release;
            let lmb = mouse_live && mouse.left();
            let fire = (lmb && !shift) || pad_now.right_trigger;
            let use_item = (mouse_live && mouse.right()) || (lmb && shift) || pad_now.left_trigger;
            input_context.right_hand.trigger_value = f32::from(u8::from(fire));
            input_context.right_hand.squeeze_value = f32::from(u8::from(use_item));
            if !alt {
                input_context.lean = (f32::from(u8::from(down(Key::E)))
                    - f32::from(u8::from(down(Key::Q)))
                    + pad_now.lean())
                .clamp(-1.0, 1.0);
            }
            input_context.crouch = down(Key::LeftControl) || pad_now.crouch();
            input_context.jump = down(Key::Space) || pad_now.jump();
        }
        POINTER_X.store(pointer.x as i32, Ordering::Relaxed);
        POINTER_Y.store(pointer.y as i32, Ordering::Relaxed);

        keyboard.poll(down, &mut action_state);

        let time = Time {
            elapsed: delta,
            total: now - start,
        };
        game.update(&time, &input_context, &mut action_state);
        gamepads.rumble(
            game.take_haptics(),
            shock2vr::user_settings::get().pc.gamepad_rumble,
        );
        if game.should_quit() {
            break;
        }

        let (fb_width, fb_height) = window.vulkan_drawable_size();
        let (fb_width, fb_height) = (fb_width.max(1) as f32, fb_height.max(1) as f32);
        let projection_matrix = cgmath::perspective(
            cgmath::Deg(game.desired_fov_deg()),
            fb_width / fb_height,
            shock2vr::NEAR_PLANE,
            shock2vr::FAR_PLANE,
        );
        let screen_size = vec2(fb_width, fb_height);

        let (mut scene, pawn_offset, pawn_rotation) = game.render();
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

        renderer.resize(fb_width as u32, fb_height as u32);
        renderer.set_vsync(pc.vsync);
        renderer.set_gamma(pc.brightness);
        renderer.set_grade(game.screen_grade(), start.elapsed().as_secs_f32());
        frame_number += 1;
        if let Some(frame) = renderer.begin_frame([0.0, 0.0, 0.0, 1.0]) {
            engine.render(&render_context, &scene_for_render);
            // SHOCK2_VK_CAPTURE=path[@frame]: save one presented frame (as
            // the desktop runtime does).
            let capture = std::env::var("SHOCK2_VK_CAPTURE").ok();
            if let Some(spec) = &capture {
                let at = spec.split_once('@').map_or("300", |(_, at)| at);
                if frame_number == at.parse::<u64>().unwrap_or(300) {
                    renderer.request_capture();
                }
            }
            renderer.end_frame(frame);
            if let (Some(c), Some(spec)) = (renderer.take_capture(), capture) {
                let path = spec.split('@').next().unwrap_or_default();
                match image::save_buffer(path, &c.rgba, c.width, c.height, image::ColorType::Rgba8)
                {
                    Ok(()) => println!("SHOCK2_VK_CAPTURED {path}"),
                    Err(e) => warn!("capture {path}: {e}"),
                }
            }
        }
    }
    POINTER_SHOWN.store(false, Ordering::Relaxed);
    Ok(())
}

fn create_surface(
    window: &sdl2::video::Window,
    instance: vk::Instance,
) -> Result<vk::SurfaceKHR, vk::Result> {
    window
        .vulkan_create_surface(instance.as_raw() as usize)
        .map(vk::SurfaceKHR::from_raw)
        .map_err(|e| {
            warn!("SDL_Vulkan_CreateSurface: {e}");
            vk::Result::ERROR_INITIALIZATION_FAILED
        })
}

/// Follow the Options menu's fullscreen switch (desktop only).
fn apply_fullscreen(window: &mut sdl2::video::Window, applied: &mut Option<bool>) {
    let wanted = shock2vr::user_settings::get().pc.fullscreen;
    if *applied == Some(wanted) {
        return;
    }
    let mode = if wanted {
        sdl2::video::FullscreenType::Desktop
    } else {
        sdl2::video::FullscreenType::Off
    };
    if let Err(e) = window.set_fullscreen(mode) {
        warn!("fullscreen: {e}");
    }
    *applied = Some(wanted);
}

/// The view's yaw and pitch, in degrees.
#[derive(Default)]
struct Look {
    yaw: f32,
    pitch: f32,
}

impl Look {
    fn forward(yaw: f32, pitch: f32) -> Vector3<f32> {
        vec3(
            yaw.to_radians().cos() * pitch.to_radians().cos(),
            pitch.to_radians().sin(),
            yaw.to_radians().sin() * pitch.to_radians().cos(),
        )
        .normalize()
    }

    fn rotation_of(yaw: f32, pitch: f32) -> Quaternion<f32> {
        let f = Self::forward(yaw, pitch);
        let mat: Decomposed<Vector3<f32>, Quaternion<f32>> = cgmath::Transform::look_at_rh(
            point3(f.x, f.y, f.z),
            point3(0.0, 0.0, 0.0),
            vec3(0.0, 1.0, 0.0),
        );
        mat.rot.invert()
    }

    /// The head's rotation (the desktop runtime's `camera_rotation`).
    fn rotation(&self) -> Quaternion<f32> {
        Self::rotation_of(self.yaw, self.pitch)
    }

    /// A resting simulated hand in front of the view, aiming down it (the
    /// desktop runtime's `hand_pose` with no hand-aim key held); `side` is
    /// +1 for the right hand, -1 for the left.
    fn hand_pose(&self, side: f32) -> (Vector3<f32>, Quaternion<f32>) {
        let forward = -Self::forward(self.yaw, self.pitch);
        let right = forward.cross(vec3(0.0, 1.0, 0.0)).normalize();
        let position = vec3(0.0, 2.0 / SCALE_FACTOR, 0.0)
            + right * side * 2.0 / SCALE_FACTOR
            + forward * 4.0 / SCALE_FACTOR;
        (
            position,
            self.rotation() * Self::rotation_of(REST_HAND_YAW, 0.0),
        )
    }
}

fn parse_mission(mission: &str) -> Result<(String, SpawnLocation), String> {
    let Some((name, spawn)) = mission.split_once(':') else {
        return Ok((mission.to_owned(), SpawnLocation::MapDefault));
    };
    let spawn = if spawn.contains(',') {
        let v: Vec<f32> = spawn
            .split(',')
            .map(|p| p.trim().parse::<f32>())
            .collect::<Result<_, _>>()
            .map_err(|e| format!("mission position {spawn}: {e}"))?;
        let [x, y, z] = v[..] else {
            return Err(format!("mission position {spawn}: want x,y,z"));
        };
        SpawnLocation::PositionRotation(vec3(x, y, z), Quaternion::new(1.0, 0.0, 0.0, 0.0))
    } else {
        spawn
            .parse::<i32>()
            .map_or(SpawnLocation::MapDefault, SpawnLocation::Marker)
    };
    Ok((name.to_owned(), spawn))
}

fn scancode(key: Key) -> Scancode {
    use Scancode as S;
    match key {
        Key::A => S::A,
        Key::B => S::B,
        Key::C => S::C,
        Key::D => S::D,
        Key::E => S::E,
        Key::F => S::F,
        Key::G => S::G,
        Key::H => S::H,
        Key::I => S::I,
        Key::J => S::J,
        Key::K => S::K,
        Key::L => S::L,
        Key::M => S::M,
        Key::N => S::N,
        Key::O => S::O,
        Key::P => S::P,
        Key::Q => S::Q,
        Key::R => S::R,
        Key::S => S::S,
        Key::T => S::T,
        Key::U => S::U,
        Key::V => S::V,
        Key::W => S::W,
        Key::X => S::X,
        Key::Y => S::Y,
        Key::Z => S::Z,
        Key::Num0 => S::Num0,
        Key::Num1 => S::Num1,
        Key::Num2 => S::Num2,
        Key::Num3 => S::Num3,
        Key::Num4 => S::Num4,
        Key::Num5 => S::Num5,
        Key::Num6 => S::Num6,
        Key::Num7 => S::Num7,
        Key::Num8 => S::Num8,
        Key::Num9 => S::Num9,
        Key::Minus => S::Minus,
        Key::Equal => S::Equals,
        Key::Backslash => S::Backslash,
        Key::GraveAccent => S::Grave,
        Key::Tab => S::Tab,
        Key::Escape => S::Escape,
        Key::Space => S::Space,
        Key::Enter => S::Return,
        Key::Up => S::Up,
        Key::Down => S::Down,
        Key::Left => S::Left,
        Key::Right => S::Right,
        Key::LeftShift => S::LShift,
        Key::RightShift => S::RShift,
        Key::LeftControl => S::LCtrl,
        Key::RightControl => S::RCtrl,
        Key::LeftAlt => S::LAlt,
        Key::RightAlt => S::RAlt,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mission_argument_takes_a_marker_or_a_position() {
        assert!(matches!(
            parse_mission("earth.mis").unwrap().1,
            SpawnLocation::MapDefault
        ));
        assert!(matches!(
            parse_mission("earth.mis:12").unwrap().1,
            SpawnLocation::Marker(12)
        ));
        assert!(matches!(
            parse_mission("earth.mis:1,2,3").unwrap().1,
            SpawnLocation::PositionRotation(..)
        ));
        assert!(parse_mission("earth.mis:1,2").is_err());
    }

    #[test]
    fn hand_rest_pose_aims_down_the_view() {
        // The same math as the desktop runtime: local -Z is the aim axis.
        let look = Look {
            yaw: 30.0,
            pitch: -10.0,
        };
        let (_, rotation) = look.hand_pose(1.0);
        let aim = rotation * vec3(0.0, 0.0, -1.0);
        let view = look.rotation() * vec3(0.0, 0.0, -1.0);
        assert!((aim - view).magnitude() < 1e-4, "{aim:?} vs {view:?}");
    }
}
