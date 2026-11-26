//! Game controllers for the GLFW runtime, read through SDL2.
//!
//! GLFW reads pads but cannot rumble them, so this initializes SDL with only
//! its game controller subsystem (no video, no window) and drives the shared
//! [`gamepad`] crate - the same controls and rumble as the SDL runtime.

use gamepad::{Gamepads, PadState};
use shock2vr::haptics::HapticPulse;
use shock2vr::input::InputActionState;
use tracing::warn;

/// `None` inside when SDL could not start: the game still runs on mouse and
/// keyboard, and every call is a no-op.
pub struct Controllers(Option<Sdl>);

struct Sdl {
    // Field order matters: the pads close before SDL shuts down.
    gamepads: Gamepads,
    events: sdl2::EventPump,
    _context: sdl2::Sdl,
}

impl Controllers {
    pub fn new() -> Controllers {
        // SDL has no window of its own here, so it must not wait for one to
        // have focus before reporting the pad.
        sdl2::hint::set("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1");
        let start = || -> Result<Sdl, String> {
            let context = sdl2::init()?;
            Ok(Sdl {
                gamepads: Gamepads::new(&context),
                events: context.event_pump()?,
                _context: context,
            })
        };
        Controllers(
            start()
                .inspect_err(|e| warn!("game controllers unavailable: {e}"))
                .ok(),
        )
    }

    /// Pump SDL's events (pads arriving and leaving) and read this frame.
    pub fn poll(&mut self, wants_pointer: bool, actions: &mut InputActionState) -> PadState {
        let Some(sdl) = &mut self.0 else {
            return PadState::default();
        };
        for event in sdl.events.poll_iter() {
            sdl.gamepads.handle_event(&event);
        }
        sdl.gamepads.poll(wants_pointer, actions)
    }

    pub fn rumble(&mut self, pulses: [Option<HapticPulse>; 2]) {
        if let Some(sdl) = &mut self.0 {
            let enabled = shock2vr::user_settings::get().pc.gamepad_rumble;
            sdl.gamepads.rumble(pulses, enabled);
        }
    }
}
