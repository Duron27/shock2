extern crate glfw;
use self::glfw::{Action, Key};
use cgmath::Decomposed;
use cgmath::point3;

use clap::Parser;

use dark::SCALE_FACTOR;
use engine::profile;
use engine::scene::Scene;

use engine::util::compute_view_matrix_from_render_context;
use glfw::GlfwReceiver;
use glfw::Modifiers;

use shock2vr::GameOptions;
use shock2vr::SpawnLocation;
use tracing::trace;

use cgmath::prelude::*;
use cgmath::vec2;
use cgmath::{Quaternion, Vector3, vec3};

use glfw::MouseButton;
use shock2vr::input::InputActionState;
use shock2vr::input_context::InputContext;

mod input_mapper;
use input_mapper::DesktopInputMapper;
use shock2vr::time::Time;
use std::collections::HashSet;
use std::time::Duration;

// settings
const SCR_WIDTH: u32 = 800;
const SCR_HEIGHT: u32 = 600;
/// Mouse-look degrees per pixel at sensitivity 1x (what the old frame-time
/// scaling gave at 60 fps).
const MOUSE_DEGREES_PER_PIXEL: f32 = 0.15;

/// Switch fullscreen and vsync when the Options menu changes them.
fn apply_display_settings(
    glfw: &mut glfw::Glfw,
    window: &mut glfw::PWindow,
    applied: &mut Option<(bool, bool)>,
) {
    let pc = shock2vr::user_settings::get().pc;
    let wanted = (pc.fullscreen, pc.vsync);
    if *applied == Some(wanted) {
        return;
    }
    let was_fullscreen = applied.map_or(false, |(fullscreen, _)| fullscreen);
    if pc.fullscreen != was_fullscreen {
        if pc.fullscreen {
            glfw.with_primary_monitor(|_, monitor| {
                let Some(monitor) = monitor else { return };
                let Some(mode) = monitor.get_video_mode() else {
                    return;
                };
                window.set_monitor(
                    glfw::WindowMode::FullScreen(monitor),
                    0,
                    0,
                    mode.width,
                    mode.height,
                    Some(mode.refresh_rate),
                );
            });
        } else {
            window.set_monitor(
                glfw::WindowMode::Windowed,
                100,
                100,
                SCR_WIDTH,
                SCR_HEIGHT,
                None,
            );
        }
    }
    *applied = Some(wanted);
}

struct MousePosition {
    x: f32,
    y: f32,
}

struct CameraContext {
    pitch: f32,
    yaw: f32,
    mouse_position: Option<MousePosition>,
}

impl CameraContext {
    pub fn new() -> CameraContext {
        CameraContext {
            pitch: 0.0,
            yaw: 0.0,
            mouse_position: None,
        }
    }
}
struct HandContext {
    left_hand_context: CameraContext,
    right_hand_context: CameraContext,

    right_trigger_pressed: bool,
    right_squeeze_pressed: bool,
    right_a_pressed: bool,

    left_trigger_pressed: bool,
    left_squeeze_pressed: bool,
    left_a_pressed: bool,

    /// Which hand owns each physically-held mouse button (trigger, squeeze, A).
    button_owner: [Option<MouseLookTarget>; 3],
}

impl HandContext {
    pub fn new() -> HandContext {
        HandContext {
            left_hand_context: CameraContext::new(),
            right_hand_context: CameraContext::new(),

            right_trigger_pressed: false,
            right_squeeze_pressed: false,
            right_a_pressed: false,

            left_squeeze_pressed: false,
            left_trigger_pressed: false,
            left_a_pressed: false,

            button_owner: [None; 3],
        }
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Mission or debug scene to load. If omitted, boots the main menu.
    #[arg(short, long)]
    mission: Option<String>,

    #[arg(long = "debug-physics")]
    debug_physics: bool,

    #[arg(long = "debug-portals")]
    debug_portals: bool,

    #[arg(long = "debug-draw")]
    debug_draw: bool,

    #[arg(long = "debug-show-ids")]
    debug_show_ids: bool,

    #[arg(long = "debug-skeletons")]
    debug_skeletons: bool,

    #[arg(long = "debug-ai")]
    debug_ai: bool,

    #[arg(long = "debug-pathfinding")]
    debug_pathfinding: bool,

    #[arg(short, long, default_value = None)]
    save_file: Option<String>,
    // Number of times to greet
    // #[arg(short, long, default_value_t = 1)]
    // count: u8,
    #[arg(short, long, default_value = None)]
    experimental: Option<Vec<String>>,
}
struct MouseUpdateResult {
    delta_x: f32,
    delta_y: f32,
}

fn camera_update_mouse(camera: &mut CameraContext, x_pos: f32, y_pos: f32) -> MouseUpdateResult {
    match camera.mouse_position {
        None => {
            camera.mouse_position = Some(MousePosition { x: x_pos, y: y_pos });
            MouseUpdateResult {
                delta_x: 0.0,
                delta_y: 0.0,
            }
        }
        Some(MousePosition { x, y }) => {
            let delta_x = x_pos - x;
            let delta_y = y_pos - y;
            camera.mouse_position = Some(MousePosition { x: x_pos, y: y_pos });
            MouseUpdateResult { delta_x, delta_y }
        }
    }
}

fn camera_forward(camera: &CameraContext) -> Vector3<f32> {
    let yaw = camera.yaw;
    let pitch = camera.pitch;
    let front = Vector3 {
        x: yaw.to_radians().cos() * pitch.to_radians().cos(),
        y: pitch.to_radians().sin(),
        z: yaw.to_radians().sin() * pitch.to_radians().cos(),
    };

    front.normalize()
}

fn camera_rotation(camera: &CameraContext) -> Quaternion<f32> {
    let forward = camera_forward(camera);
    let forward_p = cgmath::Point3::new(forward.x, forward.y, forward.z);

    let up: Vector3<f32> = vec3(0.0, 1.0, 0.0);
    let mat: Decomposed<Vector3<f32>, Quaternion<f32>> =
        cgmath::Transform::look_at_rh(forward_p, point3(0.0, 0.0, 0.0), up);
    mat.rot.invert()
}

/// Hand yaw the rig snaps back to when neither hand-aim key (E/Q) is held: the
/// hand then aims straight down the head's facing.
const REST_HAND_YAW: f32 = 90.0;

/// World pose of one simulated VR hand: `side` is `+1.0` for the right hand and
/// `-1.0` for the left. The hand sits out in front of the head (so it is
/// visible) and aims along the head's facing turned by the hand's own
/// yaw/pitch, which mouse-look drives while E (right) or Q (left) is held.
///
/// Its aim axis is **-Z**, the OpenXR aim convention every VR consumer in
/// `shock2vr` assumes (`main_menu::vr_pointer`, the hand raycasts):
/// `camera_rotation` maps local +Z to `camera_forward`, and the rig's own
/// forward is `-camera_forward`, so the two negations cancel.
fn hand_pose(
    camera_context: &CameraContext,
    hand_camera_context: &CameraContext,
    side: f32,
) -> (Vector3<f32>, Quaternion<f32>) {
    let forward = -1.0 * camera_forward(camera_context);
    let right = Vector3::cross(forward, vec3(0.0, 1.0, 0.0)).normalize();
    let position = vec3(0.0, 2.0 / SCALE_FACTOR, 0.0)
        + right * side * 2.0 / SCALE_FACTOR
        + (forward * 4.0 / SCALE_FACTOR);
    let rotation = camera_rotation(camera_context) * camera_rotation(hand_camera_context);
    (position, rotation)
}

/// What this frame's mouse motion turns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MouseLookTarget {
    Head,
    RightHand,
    LeftHand,
    /// Nothing: a flatscreen 2D cursor UI is up and owns the mouse.
    Cursor,
}

/// Route this frame's mouse motion: a scene asking for a pointer
/// (`Game::wants_pointer`) takes the mouse - the cursor *is* the menu, so
/// mouse-look stops.
fn mouse_look_target(
    _right_hand_key: bool,
    _left_hand_key: bool,
    scene_wants_pointer: bool,
    _presentation: shock2vr::PresentationMode,
) -> MouseLookTarget {
    if !scene_wants_pointer {
        MouseLookTarget::Head
    } else {
        MouseLookTarget::Cursor
    }
}

/// Which hand a held mouse button belongs to.
///
/// Ownership is claimed on the press edge by the hand the mouse was aiming
/// then, and dropped only when the button physically comes up - so releasing
/// Q/E while still holding the mouse keeps the grab alive and you can walk
/// around with the item. Ownership is never transferred mid-press: a stuck
/// button on an unaimed hand would swallow the next rising edge.
fn update_button_owner(
    owner: Option<MouseLookTarget>,
    pressed: bool,
    look_target: MouseLookTarget,
) -> Option<MouseLookTarget> {
    if !pressed {
        return None;
    }
    match owner {
        Some(existing) => Some(existing),
        None => match look_target {
            MouseLookTarget::RightHand | MouseLookTarget::LeftHand => Some(look_target),
            MouseLookTarget::Head | MouseLookTarget::Cursor => None,
        },
    }
}

fn f32_from_bool(v: bool) -> f32 {
    if v { 1.0 } else { 0.0 }
}

pub enum Mode {
    Gameplay,
    Editor,
}

pub enum Effect {
    SwitchToEditorMode,
    SwitchToGameplayMode,
}

pub fn main() {
    // glfw: initialize and configure
    // ------------------------------

    let args = Args::parse();
    // Seed before any HTTP server or game starts: later live writes must win.
    if args.debug_physics {
        shock2vr::dev_params::set(shock2vr::dev_params::DEBUG_PHYSICS, 1.0);
    }
    //panic!("args: {:?}", args);
    let mut glfw = glfw::init(glfw::fail_on_errors).unwrap();
    glfw.window_hint(glfw::WindowHint::ClientApi(glfw::ClientApiHint::NoApi));

    // glfw window creation
    // --------------------
    let (mut window, events) = glfw
        .create_window(
            SCR_WIDTH,
            SCR_HEIGHT,
            "Shock Engine - Game Mode",
            glfw::WindowMode::Windowed,
        )
        .expect("Failed to create GLFW window");

    let mut renderer = {
        let extensions = glfw
            .get_required_instance_extensions()
            .expect("GLFW reports no Vulkan support");
        let ctx = engine::vk::VkContext::new("Shock Engine", &extensions, |instance| {
            let mut surface = engine::vk::ash::vk::SurfaceKHR::null();
            match window.create_window_surface(instance, std::ptr::null(), &mut surface) {
                engine::vk::ash::vk::Result::SUCCESS => Ok(surface),
                e => Err(e),
            }
        })
        .expect("Vulkan device");
        let (w, h) = window.get_framebuffer_size();
        let renderer = engine::vk::VkRenderer::new(
            ctx,
            (w as u32, h as u32),
            shock2vr::user_settings::get().pc.vsync,
        )
        .expect("Vulkan renderer");
        println!("Vulkan device: {}", renderer.device_name());
        renderer
    };
    window.set_key_polling(true);
    window.set_cursor_pos_polling(true);
    window.set_framebuffer_size_polling(true);
    window.set_cursor_mode(glfw::CursorMode::Disabled);

    println!(
        "camera eye height (standing): {} SS2 units = {} world units (above pawn)",
        shock2vr::PLAYER_EYE_HEIGHT,
        shock2vr::PLAYER_EYE_HEIGHT / SCALE_FACTOR
    );

    let engine = engine::renderer();
    let bundle_storage = engine.get_storage();
    let experimental_features: HashSet<String> =
        args.experimental.unwrap_or(vec![]).into_iter().collect();

    let presentation_mode = shock2vr::PresentationMode::Flat;

    // With no explicit mission, both presentations boot the main menu. VR used
    // to skip it because the rig had no pointer; it now drives the menu with a
    // controller ray, so there is nothing to special-case.
    let mission_arg = args
        .mission
        .clone()
        .unwrap_or_else(|| "main_menu".to_owned());
    let (mission, spawn_location) = parse_mission(&mission_arg);

    let options = GameOptions {
        build_label: env!("SHOCK2QUEST_BUILD_LABEL"),
        save_writer: shock2vr::save_load::SaveWriter::new(
            env!("SHOCK2QUEST_BUILD_VERSION"),
            env!("SHOCK2QUEST_BUILD_COMMIT"),
            env!("SHOCK2QUEST_BUILD_DIRTY"),
        ),
        mission,
        presentation_mode,
        spawn_location,
        save_file: args.save_file,
        debug_draw: args.debug_draw,
        debug_portals: args.debug_portals,
        debug_show_ids: args.debug_show_ids,
        debug_skeletons: args.debug_skeletons,
        debug_ai: args.debug_ai,
        debug_pathfinding: args.debug_pathfinding,
        render_particles: true,
        experimental_features,
        ..GameOptions::default()
    };
    let mut game = shock2vr::App::init(options, bundle_storage);
    // FOR SCREENSHOT
    // let mut camera_context = CameraContext {
    //     camera_offset: cgmath::Vector3::new(1.25, -14.0, -24.0),
    //     pitch: 4.81,
    //     yaw: -213.0,
    //     mouse_position: None,
    // };

    let mut camera_context = CameraContext::new();

    let mut hand_context = HandContext::new();

    let mut last_time = glfw.get_time() as f32;
    let start_time = last_time;

    let mut _frame = 0;
    let mut fps_frames = 0u32;
    let mut fps_since = std::time::Instant::now();
    let mut input_mapper = DesktopInputMapper::new();

    // Tracks whether the OS cursor is currently visible (Normal) vs captured
    // for mouse-look (Disabled). The window starts with the cursor disabled.
    let mut cursor_visible = false;
    // A mouse button still held when a cursor UI closes (e.g. LMB-closing an
    // MFD panel) must not fire the weapon on the next frame - swallow the
    // held press until every button is released.
    let mut suppress_mouse_until_release = false;
    let mut action_state = InputActionState::new();
    // The Options menu's fullscreen / vsync, as last applied to the window.
    let mut applied_display: Option<(bool, bool)> = None;

    let _mode = Mode::Gameplay;
    // render loop
    // -----------
    while !window.should_close() {
        // events
        // -----
        apply_display_settings(&mut glfw, &mut window, &mut applied_display);
        let time = glfw.get_time() as f32;
        let delta_time = time - last_time;
        last_time = time;

        // A scene (e.g. the main menu) may want a visible 2D cursor instead of
        // captured mouse-look. Toggle the OS cursor when that changes.
        let wants_pointer = game.wants_pointer();
        if wants_pointer != cursor_visible {
            window.set_cursor_mode(if wants_pointer {
                glfw::CursorMode::Normal
            } else {
                glfw::CursorMode::Disabled
            });
            if wants_pointer {
                // The captured (Disabled) cursor position is a virtual,
                // unbounded accumulator - garbage as a screen pointer. Start
                // the visible cursor centered and sync the tracked position
                // so the first CursorPos event doesn't produce a huge delta.
                let (w, h) = window.get_size();
                let (cx, cy) = (w as f64 / 2.0, h as f64 / 2.0);
                window.set_cursor_pos(cx, cy);
                camera_context.mouse_position = Some(MousePosition {
                    x: cx as f32,
                    y: cy as f32,
                });
            } else {
                // Re-capturing: forget the pointer position so the next
                // mouse-look delta starts fresh instead of jumping.
                camera_context.mouse_position = None;
            }
            cursor_visible = wants_pointer;
        }

        let mut input_context = process_events(
            &mut window,
            &mut camera_context,
            &mut hand_context,
            &events,
            delta_time,
            &mut input_mapper,
            &mut action_state,
            wants_pointer,
            presentation_mode,
        );

        // Flat first-person: left mouse button fires the wielded weapon; right
        // mouse button - or shift+left-click, which is easier on a trackpad -
        // uses/frobs/picks up. (The flat controller reads the right-hand trigger
        // + squeeze.)
        // While a 2D cursor UI is up (e.g. an MFD panel), mouse buttons drive
        // the cursor (`input_context.pointer`), not fire/frob - the original
        // surrenders the mouse to the metagame the same way. A button still
        // held when the UI closes stays swallowed until released.
        {
            let pressed = |b| window.get_mouse_button(b) == Action::Press;
            let any_mouse = pressed(MouseButton::Button1) || pressed(MouseButton::Button2);
            if wants_pointer {
                suppress_mouse_until_release = true;
            } else if !any_mouse {
                suppress_mouse_until_release = false;
            }
            if !wants_pointer && !suppress_mouse_until_release {
                let shift = window.get_key(Key::LeftShift) == Action::Press
                    || window.get_key(Key::RightShift) == Action::Press;
                let lmb = pressed(MouseButton::Button1);
                input_context.right_hand.trigger_value = if lmb && !shift { 1.0 } else { 0.0 };
                input_context.right_hand.squeeze_value =
                    if pressed(MouseButton::Button2) || (lmb && shift) {
                        1.0
                    } else {
                        0.0
                    };
            }
        }

        let time = Time {
            elapsed: Duration::from_secs_f32(delta_time),
            total: Duration::from_secs_f32(time - start_time),
        };

        profile!(
            "game.update",
            game.update(&time, &input_context, &mut action_state)
        );

        // A scene may request quitting (e.g. the main menu's Quit item).
        if game.should_quit() {
            window.set_should_close(true);
        }

        // Built after `game.update` so this frame's `desired_fov_deg()` is
        // reflected immediately rather than lagging a frame.
        let (fb_width, fb_height) = window.get_framebuffer_size();
        let (fb_width, fb_height) = (fb_width.max(1) as f32, fb_height.max(1) as f32);
        let ratio = fb_width / fb_height;
        let projection_matrix: cgmath::Matrix4<f32> = cgmath::perspective(
            cgmath::Deg(game.desired_fov_deg()),
            ratio,
            shock2vr::NEAR_PLANE,
            shock2vr::FAR_PLANE,
        );

        let screen_size = vec2(fb_width, fb_height);

        let (mut scene, pawn_offset, pawn_rotation) = profile!("game.render", game.render());

        // Crouch-aware, and reflects the ACTUAL collider state (standing up
        // is refused without headroom), so the camera can't rise through a
        // ceiling the body still crouches under.
        let head_height = game.player_eye_height();
        // Routed through `resolve_camera` rather than used directly: while the
        // player is dying the game blends this tracked pose toward the fallen
        // death pose, once, for every runtime (see `shock2vr::death_camera`).
        // Flat gameplay supplies the shared collision-resolved leaning eye.
        let render_context = game
            .resolve_camera(
                pawn_offset,
                pawn_rotation,
                vec3(0.0, head_height / SCALE_FACTOR, 0.0),
                camera_rotation(&camera_context),
            )
            .into_render_context(game.render_time(), projection_matrix, screen_size);

        let view = compute_view_matrix_from_render_context(&render_context);
        let per_eye_scene = profile!(
            "game.render_per_eye",
            game.render_per_eye(view, projection_matrix, screen_size)
        );

        game.finish_render(view, projection_matrix, screen_size);

        _frame += 1;
        // SHOCK2_FPS=1: average frame time once a second.
        if std::env::var_os("SHOCK2_FPS").is_some() {
            fps_frames += 1;
            let elapsed = fps_since.elapsed().as_secs_f32();
            if elapsed >= 1.0 {
                println!(
                    "SHOCK2_FPS {:.1} fps {:.2} ms",
                    fps_frames as f32 / elapsed,
                    elapsed * 1000.0 / fps_frames as f32
                );
                fps_frames = 0;
                fps_since = std::time::Instant::now();
            }
        }

        scene.extend(per_eye_scene);

        let mut scene_for_render = Scene::from_objects(scene);

        // Add hand spotlights (`hand_spotlights` dev param)
        let hand_spotlights = game.get_hand_lights();
        for spotlight in hand_spotlights {
            scene_for_render.lights_mut().add_light(spotlight);
        }
        (
            scene_for_render.point_shadows,
            scene_for_render.point_shadow_strength,
        ) = game.get_shadow_lights();
        scene_for_render.fog_density = game.fog_density();
        scene_for_render.world_gloss = game.world_gloss();
        scene_for_render.bump_strength = game.bump_strength();
        // Reflection strength 0: only the world shader mirrors it from here;
        // objects carry their own capture and strength.
        scene_for_render.lights_mut().environment = game.environment();

        {
            renderer.resize(fb_width as u32, fb_height as u32);
            renderer.set_vsync(shock2vr::user_settings::get().pc.vsync);
            renderer.set_gamma(shock2vr::user_settings::get().pc.brightness);
            renderer.set_grade(game.screen_grade(), glfw.get_time() as f32);
            if let Some(frame) = renderer.begin_frame([0.0, 0.0, 0.0, 1.0]) {
                profile!(
                    "engine.render",
                    engine.render(&render_context, &scene_for_render)
                );
                // SHOCK2_VK_CAPTURE=path[@frame]: save one presented frame.
                if let Ok(spec) = std::env::var("SHOCK2_VK_CAPTURE") {
                    let at = spec.split_once('@').map_or("300", |(_, at)| at);
                    if _frame == at.parse::<u64>().unwrap_or(300) {
                        renderer.request_capture();
                    }
                }
                renderer.end_frame(frame);
                if let Some(c) = renderer.take_capture() {
                    let path = std::env::var("SHOCK2_VK_CAPTURE").unwrap_or_default();
                    let path = path.split('@').next().unwrap_or_default().to_owned();
                    image::save_buffer(&path, &c.rgba, c.width, c.height, image::ColorType::Rgba8)
                        .expect("save capture");
                    println!("SHOCK2_VK_CAPTURED {path}");
                }
            }
        }
        glfw.poll_events();
    }
}

fn parse_mission(mission: &str) -> (String, SpawnLocation) {
    if !mission.contains(':') {
        return (mission.to_owned(), SpawnLocation::MapDefault);
    }

    let parts: Vec<&str> = mission.split(':').collect();

    if parts.len() > 2 {
        panic!("Unable to parse mission argument: {}", mission);
    }

    let mission = parts[0];

    let spawn_location = if parts[1].contains(",") {
        let vec_parts: Vec<&str> = parts[1].split(',').collect();
        if vec_parts.len() != 3 {
            panic!("Unable to parse position: {}", parts[1]);
        }

        let x = vec_parts[0].parse::<f32>().unwrap();
        let y = vec_parts[1].parse::<f32>().unwrap();
        let z = vec_parts[2].parse::<f32>().unwrap();
        SpawnLocation::PositionRotation(vec3(x, y, z), Quaternion::<f32>::new(1.0, 0.0, 0.0, 0.0))
    } else {
        match parts[1].parse::<i32>() {
            Ok(num) => SpawnLocation::Marker(num),
            Err(_) => SpawnLocation::MapDefault,
        }
    };

    return (mission.to_owned(), spawn_location);
}

// NOTE: not the same version as in common.rs!
fn process_events(
    //audio: &mut AudioContext,
    window: &mut glfw::Window,
    camera_context: &mut CameraContext,
    hand_context: &mut HandContext,
    events: &GlfwReceiver<(f64, glfw::WindowEvent)>,
    delta_time: f32,
    input_mapper: &mut DesktopInputMapper,
    action_state: &mut InputActionState,
    wants_pointer: bool,
    presentation: shock2vr::PresentationMode,
) -> InputContext {
    let _speed = 20.0;

    let _movement = cgmath::Vector3::new(0.0, 0.0, 0.0);

    trace!("delta time: {delta_time}");
    let mut rot_yaw = 0.0;
    let mut rot_pitch = 0.0;
    // if window.get_key(Key::Left) == Action::Press {
    //     println!("left key pressed");
    //     rot_yaw += 1.0;
    // }

    // if window.get_key(Key::Right) == Action::Press {
    //     rot_yaw += -1.0;
    // }

    for (_, event) in glfw::flush_messages(events) {
        match event {
            // The renderer follows the framebuffer size each frame.
            glfw::WindowEvent::FramebufferSize(..) => {}
            glfw::WindowEvent::Key(Key::E, _, Action::Press, Modifiers::Alt) => {
                window.set_cursor_mode(glfw::CursorMode::Normal);
                //effects.push(Effect::SwitchToEditorMode)
            }
            glfw::WindowEvent::Key(Key::G, _, Action::Press, Modifiers::Alt) => {
                window.set_cursor_mode(glfw::CursorMode::Disabled);
                //effects.push(Effect::SwitchToGameplayMode)
            }
            // glfw::WindowEvent::Key(Key::Space, _, Action::Press, _) => {
            //     engine::audio::play_audio(audio)
            // }
            glfw::WindowEvent::CursorPos(x, y) => {
                // Sum every event this frame: keeping only the last one
                // dropped motion whenever the OS delivered several.
                let mouse_update = camera_update_mouse(camera_context, x as f32, y as f32);
                rot_yaw += mouse_update.delta_x;
                rot_pitch += mouse_update.delta_y;
            }
            _ => {}
        }
    }

    // Mouse look is degrees per pixel - like any PC shooter, independent of
    // frame rate (it used to scale by frame time, so it sped up as FPS fell).
    let pc = shock2vr::user_settings::get().pc;
    let look_scale = MOUSE_DEGREES_PER_PIXEL * pc.mouse_sensitivity;
    rot_yaw *= look_scale;
    rot_pitch *= look_scale * if pc.invert_mouse_y { -1.0 } else { 1.0 };

    let mouse_pressed = |button| window.get_mouse_button(button) == Action::Press;
    let look_target = mouse_look_target(
        window.get_key(Key::E) == Action::Press,
        window.get_key(Key::Q) == Action::Press,
        wants_pointer,
        presentation,
    );
    match look_target {
        MouseLookTarget::RightHand => {
            hand_context.right_hand_context.yaw += rot_yaw;
            hand_context.right_hand_context.pitch += rot_pitch;
        }
        MouseLookTarget::LeftHand => {
            hand_context.left_hand_context.yaw += rot_yaw;
            hand_context.left_hand_context.pitch += rot_pitch;
        }
        target @ (MouseLookTarget::Head | MouseLookTarget::Cursor) => {
            if target == MouseLookTarget::Head {
                camera_context.yaw += rot_yaw;
                camera_context.pitch += rot_pitch;
            }

            hand_context.left_hand_context.yaw = REST_HAND_YAW;
            hand_context.left_hand_context.pitch = 0.0;

            hand_context.right_hand_context.yaw = REST_HAND_YAW;
            hand_context.right_hand_context.pitch = 0.0;
        }
    }

    // A hand's buttons are the mouse buttons *while the mouse is driving that
    // hand*, but ownership latches for as long as the button is physically
    // down: letting go of E/Q must not drop a held item.
    let buttons = [
        MouseButton::Button1,
        MouseButton::Button2,
        MouseButton::Button3,
    ];
    for (i, button) in buttons.into_iter().enumerate() {
        hand_context.button_owner[i] = update_button_owner(
            hand_context.button_owner[i],
            mouse_pressed(button),
            look_target,
        );
    }
    let owner = hand_context.button_owner;
    let owned_by = |i: usize, hand: MouseLookTarget| owner[i] == Some(hand);
    let (right, left) = (MouseLookTarget::RightHand, MouseLookTarget::LeftHand);
    hand_context.right_trigger_pressed = owned_by(0, right);
    hand_context.right_squeeze_pressed = owned_by(1, right);
    hand_context.right_a_pressed = owned_by(2, right);
    hand_context.left_trigger_pressed = owned_by(0, left);
    hand_context.left_squeeze_pressed = owned_by(1, left);
    hand_context.left_a_pressed = owned_by(2, left);

    if camera_context.pitch < -89.0 {
        camera_context.pitch = -89.0
    }

    if camera_context.pitch > 89.0 {
        camera_context.pitch = 89.0
    }

    let mut right_thumbstick_value = vec2(0.0, 0.0);
    let mut left_thumbstick_value = vec2(0.0, 0.0);
    //let mut trigger_value = 0.0;

    let is_alt_pressed = window.get_key(Key::LeftAlt) == Action::Press
        || window.get_key(Key::RightAlt) == Action::Press;

    if window.get_key(Key::W) == Action::Press && !is_alt_pressed {
        right_thumbstick_value += vec2(0.0, 1.0);
    }

    if window.get_key(Key::S) == Action::Press && !is_alt_pressed {
        right_thumbstick_value += vec2(0.0, -1.0);
    }

    if window.get_key(Key::A) == Action::Press && !is_alt_pressed {
        right_thumbstick_value += vec2(-1.0, 0.0);
    }

    if window.get_key(Key::D) == Action::Press && !is_alt_pressed {
        right_thumbstick_value += vec2(1.0, 0.0);
    }

    if window.get_key(Key::Up) == Action::Press {
        left_thumbstick_value += vec2(0.0, 1.0);
    }

    if window.get_key(Key::Down) == Action::Press {
        left_thumbstick_value += vec2(0.0, -1.0);
    }

    if window.get_key(Key::Left) == Action::Press {
        left_thumbstick_value += vec2(-1.0, 0.0);
    }

    if window.get_key(Key::Right) == Action::Press {
        left_thumbstick_value += vec2(1.0, 0.0);
    }

    if window.get_key(Key::LeftShift) == Action::Press
        || window.get_key(Key::RightShift) == Action::Press
    {
        left_thumbstick_value *= 2.0;
        right_thumbstick_value *= 2.0;
    }

    let mut input_context = InputContext::default();
    if presentation == shock2vr::PresentationMode::Flat && !wants_pointer && !is_alt_pressed {
        input_context.lean = f32_from_bool(window.get_key(Key::E) == Action::Press)
            - f32_from_bool(window.get_key(Key::Q) == Action::Press);
    }
    let head_rotation = camera_rotation(camera_context);
    input_context.head.rotation = head_rotation;
    // The flat rig has no tracked head: the eye sits directly above the pawn
    // origin at the fixed camera height, which is exactly the default.
    input_context.head.position = vec3(0.0, shock2vr::input_context::DEFAULT_HEAD_HEIGHT, 0.0);
    let (right_hand_position, right_hand_rotation) =
        hand_pose(camera_context, &hand_context.right_hand_context, 1.0);
    input_context.right_hand.position = right_hand_position;
    input_context.right_hand.rotation = right_hand_rotation;
    input_context.right_hand.thumbstick = right_thumbstick_value;
    input_context.right_hand.trigger_value = f32_from_bool(hand_context.right_trigger_pressed);
    input_context.right_hand.squeeze_value = f32_from_bool(hand_context.right_squeeze_pressed);
    input_context.right_hand.a_value = f32_from_bool(hand_context.right_a_pressed);

    let (left_hand_position, left_hand_rotation) =
        hand_pose(camera_context, &hand_context.left_hand_context, -1.0);
    input_context.left_hand.position = left_hand_position;
    input_context.left_hand.rotation = left_hand_rotation;
    input_context.left_hand.thumbstick = left_thumbstick_value;
    input_context.left_hand.trigger_value = f32_from_bool(hand_context.left_trigger_pressed);
    input_context.left_hand.squeeze_value = f32_from_bool(hand_context.left_squeeze_pressed);
    input_context.left_hand.a_value = f32_from_bool(hand_context.left_a_pressed);
    // input_context.left_hand.trigger_value = trigger_value;
    // input_context.left_hand.squeeze_value = squeeze_value;

    // Flatscreen UI (menus) reads an absolute 2D pointer rather than the VR
    // hands. The cursor is in Normal mode here, so positions are window coords.
    if wants_pointer {
        if let Some(MousePosition { x, y }) = &camera_context.mouse_position {
            input_context.pointer = Some(shock2vr::input_context::Pointer2D {
                position: {
                    let (w, h) = window.get_size();
                    vec2(x / w.max(1) as f32, y / h.max(1) as f32)
                },
                secondary: window.get_mouse_button(MouseButton::Button2) == Action::Press,
                pressed: window.get_mouse_button(MouseButton::Button1) == Action::Press,
            });
        }
    }

    input_context.crouch = window.get_key(Key::LeftControl) == Action::Press;
    input_context.jump = window.get_key(Key::Space) == Action::Press;

    // Discrete actions (spawn, save/load, inventory, pathfinding test) are
    // handled by the mapper, which edge-detects key presses.
    input_mapper.poll(window, action_state);

    input_context
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx(yaw: f32, pitch: f32) -> CameraContext {
        CameraContext {
            yaw,
            pitch,
            mouse_position: None,
        }
    }

    #[test]
    fn leaning_keys_keep_mouse_look_in_flat_mode() {
        use shock2vr::PresentationMode::Flat;
        for (q, e) in [(true, false), (false, true), (true, true)] {
            assert_eq!(mouse_look_target(e, q, false, Flat), MouseLookTarget::Head);
            assert_eq!(mouse_look_target(e, q, true, Flat), MouseLookTarget::Cursor);
        }
    }

    /// `camera_rotation` maps local **+Z** (not -Z) onto `camera_forward`, and
    /// the rig's own forward is `-camera_forward` - so a pose built from it
    /// aims along -Z, the convention every VR consumer assumes. Pinned here
    /// because the two negations cancelling is easy to "fix" into a backwards
    /// ray.
    #[test]
    fn hand_aim_axis_is_negative_z_and_matches_the_render_direction() {
        let camera = ctx(0.0, 0.0);
        let render_direction = camera_rotation(&camera).rotate_vector(vec3(0.0, 0.0, -1.0));
        let (_, rotation) = hand_pose(&camera, &ctx(REST_HAND_YAW, 0.0), 1.0);
        let aim = rotation.rotate_vector(vec3(0.0, 0.0, -1.0));

        // Yaw 0 looks along -X in this rig, and the resting hand aims with it.
        assert!((render_direction - vec3(-1.0, 0.0, 0.0)).magnitude() < 1e-4);
        assert!((aim - render_direction).magnitude() < 1e-4);
    }

    /// The fix: surrendering mouse-look to a 2D cursor is a flatscreen bargain.
    /// Doing it under `--vr` froze the head and hands, so the menu ray stuck at
    /// the resting pose and hover/click never responded. Under `--vr` a
    /// pointer-wanting scene routes the mouse to the right hand (the only thing
    /// that moves the ray across the head-anchored panel); VR E/Q always win.
    #[test]
    fn pointer_scenes_take_the_mouse_as_a_cursor() {
        use shock2vr::PresentationMode::Flat;

        assert_eq!(
            mouse_look_target(false, false, true, Flat),
            MouseLookTarget::Cursor
        );
        assert_eq!(
            mouse_look_target(false, false, false, Flat),
            MouseLookTarget::Head
        );
    }

    /// Releasing the aim key must not drop a grab: the hand that owned the
    /// button on the press edge keeps it until the button physically releases.
    #[test]
    fn a_held_mouse_button_stays_with_the_hand_that_claimed_it() {
        use MouseLookTarget::{Head, LeftHand, RightHand};

        // Squeeze while aiming the right hand, then let go of E.
        let owner = update_button_owner(None, true, RightHand);
        assert_eq!(owner, Some(RightHand));
        let owner = update_button_owner(owner, true, Head);
        assert_eq!(owner, Some(RightHand));
        // ...and re-aiming the other hand mid-press does not steal it.
        let owner = update_button_owner(owner, true, LeftHand);
        assert_eq!(owner, Some(RightHand));
        // Releasing the button ends the grab.
        assert_eq!(update_button_owner(owner, false, Head), None);

        // A press with no hand aimed belongs to no hand.
        assert_eq!(update_button_owner(None, true, Head), None);
    }
}
