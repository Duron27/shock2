//! Game controller input and rumble for the flatscreen runtimes.
//!
//! Both the GLFW desktop runtime and the SDL runtime read pads through SDL2's
//! GameController API (the GLFW host initializes SDL without video just for
//! this), so a pad maps to the same controls - and rumbles the same way - in
//! either. The host owns the SDL event pump and forwards controller events to
//! [`Gamepads::handle_event`]; everything else lives here.
//!
//! Layout (Xbox names; SDL maps other pads onto them):
//! left stick move, right stick look, RT fire, LT use/frob, A jump, B crouch,
//! d-pad left/right lean, d-pad up read the last unread log, X reload,
//! Y use mode (inventory cursor), LB cycle ammo, RB gun setting, Back map,
//! Start pause. While a cursor UI is up the left stick moves the cursor,
//! A clicks and B right-clicks.
//!
//! [`PadState`], its actions and [`Rumble`] don't need SDL: the Xbox 360
//! runtime fills a `PadState` from XInput (`default-features = false`, no
//! `sdl` feature) so its pad plays the same.

use std::time::{Duration, Instant};

use cgmath::prelude::*;
use cgmath::{Vector2, vec2};
#[cfg(feature = "sdl")]
use sdl2::GameControllerSubsystem;
#[cfg(feature = "sdl")]
use sdl2::controller::{Axis, GameController};
#[cfg(feature = "sdl")]
use sdl2::event::Event;
use shock2vr::haptics::HapticPulse;
use shock2vr::input::{InputAction, InputActionState};
#[cfg(feature = "sdl")]
use tracing::{info, warn};

/// A controller's buttons, by their Xbox names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Button {
    A,
    B,
    X,
    Y,
    Back,
    Start,
    LeftShoulder,
    RightShoulder,
    DPadUp,
    DPadDown,
    DPadLeft,
    DPadRight,
}

/// Right-stick look speed at full deflection and sensitivity 1x.
pub const PAD_LOOK_DEGREES_PER_SECOND: f32 = 180.0;
/// Fraction of the window per second the left stick moves a UI cursor.
pub const PAD_POINTER_SPEED: f32 = 0.8;
const STICK_DEADZONE: f32 = 0.2;
const TRIGGER_THRESHOLD: f32 = 0.3;

/// Controller buttons that fire discrete actions during play.
const PAD_ACTIONS: [(Button, InputAction); 7] = [
    (Button::Start, InputAction::TogglePauseMenu),
    (Button::Back, InputAction::ToggleMap),
    (Button::Y, InputAction::ToggleUseMode),
    (Button::X, InputAction::Reload),
    (Button::LeftShoulder, InputAction::CycleAmmo),
    (Button::RightShoulder, InputAction::CycleGunSetting),
    (Button::DPadUp, InputAction::ReadLastUnreadLog),
];

#[cfg(feature = "sdl")]
const READ_BUTTONS: [(sdl2::controller::Button, Button); 12] = {
    use sdl2::controller::Button as Sdl;
    [
        (Sdl::A, Button::A),
        (Sdl::B, Button::B),
        (Sdl::X, Button::X),
        (Sdl::Y, Button::Y),
        (Sdl::Back, Button::Back),
        (Sdl::Start, Button::Start),
        (Sdl::LeftShoulder, Button::LeftShoulder),
        (Sdl::RightShoulder, Button::RightShoulder),
        (Sdl::DPadUp, Button::DPadUp),
        (Sdl::DPadDown, Button::DPadDown),
        (Sdl::DPadLeft, Button::DPadLeft),
        (Sdl::DPadRight, Button::DPadRight),
    ]
};

/// A stick's axes, -1..1 with +y forward, through the deadzone.
pub fn stick(x: f32, y: f32) -> Vector2<f32> {
    let v = vec2(x, y);
    if v.magnitude() < STICK_DEADZONE {
        Vector2::zero()
    } else {
        v
    }
}

/// Whether a trigger, 0..1, counts as pulled.
pub fn trigger(value: f32) -> bool {
    value > TRIGGER_THRESHOLD
}

/// One frame of a game controller.
#[derive(Clone, Debug)]
pub struct PadState {
    /// +x right, +y forward (pushed away), deadzoned.
    pub left_stick: Vector2<f32>,
    pub right_stick: Vector2<f32>,
    pub left_trigger: bool,
    pub right_trigger: bool,
    pub buttons: Vec<Button>,
}

impl Default for PadState {
    fn default() -> Self {
        PadState {
            left_stick: Vector2::zero(),
            right_stick: Vector2::zero(),
            left_trigger: false,
            right_trigger: false,
            buttons: Vec::new(),
        }
    }
}

impl PadState {
    #[cfg(feature = "sdl")]
    pub fn read(controller: Option<&GameController>) -> PadState {
        let Some(c) = controller else {
            return PadState::default();
        };
        let axis = |a: Axis| f32::from(c.axis(a)) / 32767.0;
        // SDL's +y is down; the game's sticks read +y as forward.
        let sdl_stick = |x: Axis, y: Axis| stick(axis(x), -axis(y));
        PadState {
            left_stick: sdl_stick(Axis::LeftX, Axis::LeftY),
            right_stick: sdl_stick(Axis::RightX, Axis::RightY),
            left_trigger: trigger(axis(Axis::TriggerLeft)),
            right_trigger: trigger(axis(Axis::TriggerRight)),
            buttons: READ_BUTTONS
                .into_iter()
                .filter(|&(sdl, _)| c.button(sdl))
                .map(|(_, b)| b)
                .collect(),
        }
    }

    pub fn button(&self, button: Button) -> bool {
        self.buttons.contains(&button)
    }

    /// Lean from the d-pad: +1 right, -1 left.
    pub fn lean(&self) -> f32 {
        f32::from(u8::from(self.button(Button::DPadRight)))
            - f32::from(u8::from(self.button(Button::DPadLeft)))
    }

    pub fn jump(&self) -> bool {
        self.button(Button::A)
    }

    pub fn crouch(&self) -> bool {
        self.button(Button::B)
    }

    /// The cursor's primary / secondary click while a cursor UI is up.
    pub fn pointer_pressed(&self) -> bool {
        self.button(Button::A)
    }

    pub fn pointer_secondary(&self) -> bool {
        self.button(Button::B)
    }

    /// Edge-trigger [`PAD_ACTIONS`] from last frame (`self`) to `now`. Only
    /// Start (the pause menu) works while a cursor UI is up: A and B are the
    /// cursor's buttons there.
    pub fn sync_actions(&self, now: &PadState, wants_pointer: bool, state: &mut InputActionState) {
        for (button, action) in PAD_ACTIONS {
            let was = self.button(button);
            let is = now.button(button) && (!wants_pointer || button == Button::Start);
            state.sync_discrete_button(action, true, was != is, is);
        }
    }
}

/// One rumble motor: its current strength and when it stops.
#[derive(Clone, Copy, Debug, Default)]
struct Motor {
    amplitude: f32,
    until: Option<Instant>,
}

impl Motor {
    fn live(&self, now: Instant) -> f32 {
        match self.until {
            Some(until) if until > now => self.amplitude,
            _ => 0.0,
        }
    }

    /// A stronger pulse takes over, an equal one keeps whichever lasts
    /// longer, and a weaker one is felt through the stronger anyway: a tap
    /// during a blast's jolt does not cut it short.
    fn push(&mut self, pulse: HapticPulse, now: Instant) {
        let until = now + Duration::from_millis(u64::from(pulse.duration_ms));
        let current = self.live(now);
        if pulse.amplitude > current {
            self.amplitude = pulse.amplitude;
            self.until = Some(until);
        } else if pulse.amplitude == current {
            self.until = self.until.max(Some(until));
        }
    }
}

/// Mixes the game's per-hand pulses onto a pad's two motors: the left hand
/// drives the low-frequency (heavy, left grip) motor and the right hand the
/// high-frequency one, so a gunshot is a sharp buzz and a hit a body thump.
#[derive(Debug, Default)]
pub struct Rumble {
    motors: [Motor; 2],
    sent: (u16, u16),
}

impl Rumble {
    pub fn push(&mut self, pulses: [Option<HapticPulse>; 2], now: Instant) {
        for (motor, pulse) in self.motors.iter_mut().zip(pulses) {
            if let Some(pulse) = pulse {
                motor.push(pulse, now);
            }
        }
    }

    /// The (low, high) motor strengths now and how long the longer lasts.
    pub fn output(&self, now: Instant) -> (u16, u16, u32) {
        let strength = |m: &Motor| (m.live(now).clamp(0.0, 1.0) * f32::from(u16::MAX)) as u16;
        let remaining = self
            .motors
            .iter()
            .filter(|m| m.live(now) > 0.0)
            .filter_map(|m| m.until)
            .map(|until| (until - now).as_millis() as u32)
            .max()
            .unwrap_or(0);
        (
            strength(&self.motors[0]),
            strength(&self.motors[1]),
            remaining,
        )
    }

    /// The command to send the pad this frame, if its motors changed.
    pub fn command(&mut self, now: Instant) -> Option<(u16, u16, u32)> {
        let (low, high, remaining) = self.output(now);
        if (low, high) == self.sent {
            return None;
        }
        self.sent = (low, high);
        Some((low, high, remaining.max(1)))
    }

    /// Silence the motors; the next [`Rumble::command`] sends the stop.
    pub fn stop(&mut self) {
        self.motors = Default::default();
    }
}

/// The connected pads. The first one connected plays; the rest stay open so
/// unplugging it hands control to the next.
#[cfg(feature = "sdl")]
pub struct Gamepads {
    subsystem: Option<GameControllerSubsystem>,
    pads: Vec<GameController>,
    previous: PadState,
    rumble: Rumble,
}

#[cfg(feature = "sdl")]
impl Gamepads {
    pub fn new(sdl: &sdl2::Sdl) -> Gamepads {
        let subsystem = match sdl.game_controller() {
            Ok(subsystem) => Some(subsystem),
            Err(e) => {
                warn!("no game controller support: {e}");
                None
            }
        };
        Gamepads {
            subsystem,
            pads: Vec::new(),
            previous: PadState::default(),
            rumble: Rumble::default(),
        }
    }

    /// Open and close pads as they come and go. SDL also reports the pads
    /// already plugged in at startup this way.
    pub fn handle_event(&mut self, event: &Event) {
        match *event {
            Event::ControllerDeviceAdded { which, .. } => {
                let Some(subsystem) = &self.subsystem else {
                    return;
                };
                match subsystem.open(which) {
                    Ok(controller) => {
                        info!(
                            "controller connected: {} (rumble: {})",
                            controller.name(),
                            controller.has_rumble()
                        );
                        println!("controller: {}", controller.name());
                        self.pads.push(controller);
                    }
                    Err(e) => warn!("controller {which}: {e}"),
                }
            }
            Event::ControllerDeviceRemoved { which, .. } => {
                let was_first = self.pads.first().is_some_and(|c| c.instance_id() == which);
                self.pads.retain(|c| c.instance_id() != which);
                if was_first {
                    self.rumble.stop();
                }
            }
            _ => {}
        }
    }

    /// Read this frame's state and edge-trigger the pad's discrete actions.
    pub fn poll(&mut self, wants_pointer: bool, actions: &mut InputActionState) -> PadState {
        let now = PadState::read(self.pads.first());
        self.previous.sync_actions(&now, wants_pointer, actions);
        self.previous = now.clone();
        now
    }

    /// Feed this frame's [`shock2vr::App::take_haptics`] output to the motors.
    pub fn rumble(&mut self, pulses: [Option<HapticPulse>; 2], enabled: bool) {
        let now = Instant::now();
        if enabled {
            self.rumble.push(pulses, now);
        } else {
            self.rumble.stop();
        }
        let Some((low, high, ms)) = self.rumble.command(now) else {
            return;
        };
        if let Some(pad) = self.pads.first_mut() {
            // Pads without motors report an error; that is not worth a log
            // line every shot.
            let _ = pad.set_rumble(low, high, ms);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TAP: HapticPulse = HapticPulse {
        amplitude: 0.3,
        duration_ms: 50,
    };
    const JOLT: HapticPulse = HapticPulse {
        amplitude: 1.0,
        duration_ms: 400,
    };

    #[test]
    fn controller_actions_fire_on_press_edges_only() {
        let idle = PadState::default();
        let pressed = PadState {
            buttons: vec![Button::Y],
            ..PadState::default()
        };
        let mut state = InputActionState::new();
        idle.sync_actions(&pressed, false, &mut state);
        assert!(state.just_triggered(InputAction::ToggleUseMode));
        state.clear_triggered();
        pressed.sync_actions(&pressed, false, &mut state);
        assert!(!state.just_triggered(InputAction::ToggleUseMode));
    }

    #[test]
    fn a_cursor_ui_keeps_only_start_from_the_controller() {
        let pressed = PadState {
            buttons: vec![Button::Y, Button::Start],
            ..PadState::default()
        };
        let mut state = InputActionState::new();
        PadState::default().sync_actions(&pressed, true, &mut state);
        assert!(!state.just_triggered(InputAction::ToggleUseMode));
        assert!(state.just_triggered(InputAction::TogglePauseMenu));
    }

    #[test]
    fn hands_drive_separate_motors_and_expire() {
        let t = Instant::now();
        let mut rumble = Rumble::default();
        rumble.push([None, Some(TAP)], t);
        let (low, high, ms) = rumble.output(t);
        assert_eq!(low, 0);
        assert!(high > 0);
        assert_eq!(ms, 50);
        assert_eq!(rumble.output(t + Duration::from_millis(60)), (0, 0, 0));
    }

    #[test]
    fn a_tap_does_not_cut_a_jolt_short() {
        let t = Instant::now();
        let mut rumble = Rumble::default();
        rumble.push([Some(JOLT), None], t);
        rumble.push([Some(TAP), None], t + Duration::from_millis(10));
        let (low, _, _) = rumble.output(t + Duration::from_millis(200));
        assert_eq!(low, u16::MAX);
    }

    #[test]
    fn commands_are_sent_only_on_change_and_stop_at_the_end() {
        let t = Instant::now();
        let mut rumble = Rumble::default();
        assert_eq!(rumble.command(t), None);
        rumble.push([Some(JOLT), Some(TAP)], t);
        assert!(rumble.command(t).is_some());
        assert_eq!(rumble.command(t + Duration::from_millis(5)), None);
        // The right motor stops first: resend with only the left running.
        let (low, high, _) = rumble.command(t + Duration::from_millis(60)).unwrap();
        assert_eq!((low, high), (u16::MAX, 0));
        assert_eq!(
            rumble.command(t + Duration::from_millis(500)),
            Some((0, 0, 1))
        );
    }
}
