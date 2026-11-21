//! System Shock (1994) on the shock2quest engine.
//!
//! ```text
//! cargo s1 --level 1                       # walk Medical
//! cargo s1 --level 1 --screenshot out.png  # render one frame and exit
//! ```
//!
//! Mouse looks; WASD moves; Shift runs; F toggles walking and flying (Space
//! and C rise and sink); L toggles the lamp; Tab toggles the interface
//! frame; PageUp/PageDown change level;
//! Escape quits.

mod hud;
mod world;

use std::path::PathBuf;

use anyhow::{Result, anyhow};
use cgmath::{Deg, InnerSpace, Quaternion, Rad, Rotation, Rotation3, Vector3, vec2, vec3};
use clap::Parser;
use engine::scene::{Scene, SpotLight};
use glfw::{Action, Context, Key};
use world::{TILE, World};

#[derive(Parser)]
#[command(about = "System Shock (1994) on the shock2quest engine")]
struct Cli {
    /// Level number (0-15; 1 is Medical).
    #[arg(long, default_value_t = 1)]
    level: u16,
    /// The game's data folder (default: $SS1_DATA_PATH, or an install beside the repo).
    #[arg(long)]
    data: Option<PathBuf>,
    /// Start at this tile, `x,y`.
    #[arg(long, value_parser = parse_pair)]
    tile: Option<(f32, f32)>,
    /// Starting view direction in degrees (0 looks north, 90 east).
    #[arg(long, default_value_t = 0.0)]
    yaw: f32,
    #[arg(long, default_value_t = 0.0)]
    pitch: f32,
    /// Render one frame to this PNG and exit.
    #[arg(long)]
    screenshot: Option<PathBuf>,
    #[arg(long, value_parser = parse_pair, default_value = "1280,800")]
    window_size: (f32, f32),
    /// Window position on the desktop, `x,y`.
    #[arg(long, value_parser = parse_pair)]
    window_pos: Option<(f32, f32)>,
    /// Start flying instead of walking.
    #[arg(long)]
    fly: bool,
    /// Start in full-screen view, without the interface frame.
    #[arg(long)]
    fullscreen_view: bool,
}

fn parse_pair(text: &str) -> Result<(f32, f32), String> {
    let (a, b) = text
        .split_once([',', 'x'])
        .ok_or_else(|| format!("expected two numbers like 10,20, got {text:?}"))?;
    let parse = |s: &str| s.trim().parse::<f32>().map_err(|e| e.to_string());
    Ok((parse(a)?, parse(b)?))
}

/// Eye height above the floor, and the tallest step a walk climbs.
const EYE: f32 = 0.55 * TILE;
const STEP: f32 = 0.3 * TILE;
const WALK_SPEED: f32 = 1.6 * TILE;

struct Player {
    feet: Vector3<f32>,
    yaw: f32,
    pitch: f32,
    flying: bool,
    lamp: bool,
}

impl Player {
    fn rotation(&self) -> Quaternion<f32> {
        Quaternion::from_angle_y(Rad(-self.yaw.to_radians()))
            * Quaternion::from_angle_x(Rad(-self.pitch.to_radians()))
    }

    fn eye(&self) -> Vector3<f32> {
        self.feet + vec3(0.0, EYE, 0.0)
    }

    /// Move by `delta` (world units), walking over the tile map.
    fn travel(&mut self, world: &World, delta: Vector3<f32>) {
        if self.flying {
            self.feet += delta;
            return;
        }
        // Axis by axis, so a wall slides the player along it.
        for axis in [vec3(delta.x, 0.0, 0.0), vec3(0.0, 0.0, delta.z)] {
            let next = self.feet + axis;
            let probe = next
                + axis
                    .normalize_to(0.25 * TILE)
                    .map(|v| if v.is_nan() { 0.0 } else { v });
            let (Some(floor), Some(probe_floor)) = (
                world.floor_at(next.x, next.z),
                world.floor_at(probe.x, probe.z),
            ) else {
                continue;
            };
            let ceiling = world.ceiling_at(probe.x, probe.z).unwrap_or(f32::MAX);
            if probe_floor - self.feet.y > STEP || ceiling - probe_floor < EYE + 0.1 * TILE {
                continue;
            }
            self.feet = vec3(next.x, floor, next.z);
        }
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    tracing_subscriber::fmt::init();
    let data = cli
        .data
        .clone()
        .or_else(ss1::paths::data_root)
        .ok_or_else(|| anyhow!("no System Shock data found; pass --data or set SS1_DATA_PATH"))?;

    let mut glfw = glfw::init(glfw::fail_on_errors)?;
    glfw.window_hint(glfw::WindowHint::ClientApi(glfw::ClientApiHint::NoApi));
    let (width, height) = (cli.window_size.0 as u32, cli.window_size.1 as u32);
    let (mut window, events) = glfw
        .create_window(width, height, "System Shock", glfw::WindowMode::Windowed)
        .ok_or_else(|| anyhow!("cannot open a window"))?;
    if let Some((x, y)) = cli.window_pos {
        window.set_pos(x as i32, y as i32);
    }
    let mut renderer = {
        let extensions = glfw
            .get_required_instance_extensions()
            .ok_or_else(|| anyhow!("GLFW reports no Vulkan support"))?;
        let ctx = engine::vk::VkContext::new("System Shock", &extensions, |instance| {
            let mut surface = engine::vk::ash::vk::SurfaceKHR::null();
            match window.create_window_surface(instance, std::ptr::null(), &mut surface) {
                engine::vk::ash::vk::Result::SUCCESS => Ok(surface),
                e => Err(e),
            }
        })
        .map_err(|e| anyhow!("Vulkan device: {e:?}"))?;
        let (w, h) = window.get_framebuffer_size();
        engine::vk::VkRenderer::new(ctx, (w as u32, h as u32), true)
            .map_err(|e| anyhow!("Vulkan renderer: {e:?}"))?
    };
    window.set_key_polling(true);
    window.set_cursor_pos_polling(true);
    window.set_framebuffer_size_polling(true);
    if cli.screenshot.is_none() {
        window.set_cursor_mode(glfw::CursorMode::Disabled);
    }
    let engine = engine::renderer();

    let hud = hud::Hud::load(&data)?;
    let mut show_frame = !cli.fullscreen_view;
    let mut level_number = cli.level;
    let mut world = World::load(&data, level_number)?;
    println!(
        "level {level_number}: {} faces in {} draws",
        world.face_count,
        world.objects.len()
    );
    let start = |world: &World| match cli.tile {
        Some((x, y)) => {
            let (x, z) = (x * TILE, -y * TILE);
            vec3(x, world.floor_at(x, z).unwrap_or(0.0), z)
        }
        None => world.start_position(),
    };
    let mut player = Player {
        feet: start(&world),
        yaw: cli.yaw,
        pitch: cli.pitch,
        flying: cli.fly,
        lamp: true,
    };

    let mut keys = std::collections::HashSet::new();
    let mut last_mouse: Option<(f64, f64)> = None;
    let mut last_time = glfw.get_time();
    while !window.should_close() {
        let now = glfw.get_time();
        let dt = ((now - last_time) as f32).min(0.1);
        last_time = now;

        glfw.poll_events();
        let mut change_level: Option<u16> = None;
        for (_, event) in glfw::flush_messages(&events) {
            match event {
                glfw::WindowEvent::Key(key, _, Action::Press, _) => {
                    keys.insert(key);
                    match key {
                        Key::Escape => window.set_should_close(true),
                        Key::F => player.flying = !player.flying,
                        Key::L => player.lamp = !player.lamp,
                        Key::Tab => show_frame = !show_frame,
                        Key::PageUp => change_level = Some((level_number + 1) % 16),
                        Key::PageDown => change_level = Some((level_number + 15) % 16),
                        _ => {}
                    }
                }
                glfw::WindowEvent::Key(key, _, Action::Release, _) => {
                    keys.remove(&key);
                }
                glfw::WindowEvent::CursorPos(x, y) => {
                    if let Some((lx, ly)) = last_mouse {
                        player.yaw += (x - lx) as f32 * 0.12;
                        player.pitch = (player.pitch - (y - ly) as f32 * 0.12).clamp(-85.0, 85.0);
                    }
                    last_mouse = Some((x, y));
                }
                _ => {}
            }
        }
        if let Some(number) = change_level {
            match World::load(&data, number) {
                Ok(next) => {
                    level_number = number;
                    world = next;
                    player.feet = start(&world);
                    println!("level {level_number}: {} faces", world.face_count);
                }
                Err(err) => eprintln!("level {number}: {err:#}"),
            }
        }

        // Movement.
        let yaw = player.yaw.to_radians();
        let forward = vec3(yaw.sin(), 0.0, -yaw.cos());
        let right = vec3(yaw.cos(), 0.0, yaw.sin());
        let mut wish = vec3(0.0, 0.0, 0.0);
        let held = |k: Key| keys.contains(&k);
        if held(Key::W) {
            wish += forward;
        }
        if held(Key::S) {
            wish -= forward;
        }
        if held(Key::D) {
            wish += right;
        }
        if held(Key::A) {
            wish -= right;
        }
        if player.flying {
            if held(Key::Space) {
                wish.y += 1.0;
            }
            if held(Key::C) {
                wish.y -= 1.0;
            }
        }
        if wish.magnitude2() > 0.0 {
            let speed = WALK_SPEED * if held(Key::LeftShift) { 2.2 } else { 1.0 };
            player.travel(&world, wish.normalize() * speed * dt);
        }

        // Render.
        let (fb_w, fb_h) = window.get_framebuffer_size();
        let aspect = fb_w.max(1) as f32 / fb_h.max(1) as f32;
        let rotation = player.rotation();
        let screen = (fb_w.max(1) as f32, fb_h.max(1) as f32);
        let projection_matrix = if show_frame {
            hud.window_projection(
                screen,
                cgmath::perspective(Deg(60.0), hud.window_aspect(), 0.1, 4000.0),
            )
        } else {
            cgmath::perspective(Deg(70.0), aspect, 0.1, 4000.0)
        };
        let render_context = engine::EngineRenderContext {
            terrain_wetness: 0.0,
            self_depth: engine::scene::self_depth::Phase::None,
            world_wave: Default::default(),
            ambient_light_intensity: 1.0,
            level_light_intensity: 1.0,
            time: now as f32,
            camera_offset: player.eye(),
            camera_rotation: Quaternion::new(1.0, 0.0, 0.0, 0.0),
            head_offset: vec3(0.0, 0.0, 0.0),
            head_rotation: rotation,
            projection_matrix,
            screen_size: vec2(fb_w as f32, fb_h as f32),
        };

        let mut objects = world.objects.clone();
        if show_frame {
            objects.extend(hud.objects(screen));
        }
        let mut scene = Scene::from_objects(objects);
        scene.fog_density = 0.025;
        scene.world_gloss = 0.5;
        scene.bump_strength = 0.8;
        if player.lamp {
            // The player's lamp: a wide, soft headlight.
            let look = rotation.rotate_vector(vec3(0.0, 0.0, -1.0));
            let mut lamp = SpotLight::new(
                player.eye() + vec3(0.0, -0.08 * TILE, 0.0),
                look,
                vec3(1.0, 0.95, 0.85),
                0.9,
            );
            lamp.inner_cone_angle = 20f32.to_radians();
            lamp.outer_cone_angle = 50f32.to_radians();
            lamp.range = 5.0 * TILE;
            scene.lights_mut().add_light(lamp);
        }

        renderer.resize(fb_w.max(0) as u32, fb_h.max(0) as u32);
        if let Some(frame) = renderer.begin_frame([0.0, 0.0, 0.0, 1.0]) {
            engine.render(&render_context, &scene);
            if cli.screenshot.is_some() {
                renderer.request_capture();
            }
            renderer.end_frame(frame);
        }
        if let Some(path) = &cli.screenshot
            && let Some(capture) = renderer.take_capture()
        {
            let rgb: Vec<u8> = capture
                .rgba
                .chunks_exact(4)
                .flat_map(|p| [p[0], p[1], p[2]])
                .collect();
            image::save_buffer(
                path,
                &rgb,
                capture.width,
                capture.height,
                image::ColorType::Rgb8,
            )?;
            println!("Screenshot saved to {}", path.display());
            break;
        }
        window.swap_buffers();
    }
    Ok(())
}
