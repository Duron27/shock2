//! The flatscreen game on the Xbox 360 (xbox360/README.md).
//!
//! The title is `xbox360/build.sh game`: this crate as a staticlib, linked
//! with vulkan4xenon, xsdk and the shader bundle `console_shaders.sh` makes.
//! The console has no window system and no GLSL compiler, so:
//! - the surface is VK_KHR_display's one plane at the display's mode;
//! - shaders come from `game:\shock2_shaders.table` (GLSL -> SPIR-V), and
//!   vulkan4xenon finds each one's microcode in the bundle linked in;
//! - the pad is XInput, read into [`gamepad::PadState`] so it plays as the
//!   SDL runtime's does;
//! - the log goes to xsdk's (`<title>.log` beside the `.xex`).
//!
//! What is here outside [`console`] is plain Rust, tested on the host.

use cgmath::prelude::*;
use cgmath::{Decomposed, Quaternion, Vector3, point3, vec3};
use dark::SCALE_FACTOR;
use gamepad::{Button, PadState};

#[cfg(target_os = "xbox360")]
mod console;

/// Hand yaw the simulated hands rest at: aiming straight down the view.
const REST_HAND_YAW: f32 = 90.0;

/// `INPUT_GAMEPAD` (xsdk `include/xsdk/kernel.h`, XInput's layout). The
/// console is big-endian like the code reading it, so it is read as is.
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct XInputGamepad {
    pub buttons: u16,
    pub left_trigger: u8,
    pub right_trigger: u8,
    pub thumb_lx: i16,
    pub thumb_ly: i16,
    pub thumb_rx: i16,
    pub thumb_ry: i16,
}

/// XInput's button bits.
const XINPUT_BUTTONS: [(u16, Button); 12] = [
    (0x0001, Button::DPadUp),
    (0x0002, Button::DPadDown),
    (0x0004, Button::DPadLeft),
    (0x0008, Button::DPadRight),
    (0x0010, Button::Start),
    (0x0020, Button::Back),
    (0x0100, Button::LeftShoulder),
    (0x0200, Button::RightShoulder),
    (0x1000, Button::A),
    (0x2000, Button::B),
    (0x4000, Button::X),
    (0x8000, Button::Y),
];

/// One frame of the pad in the shared controller layout. XInput's sticks
/// already read +y as up, which the game reads as forward.
pub fn pad_state(raw: &XInputGamepad) -> PadState {
    let axis = |v: i16| (f32::from(v) / 32767.0).max(-1.0);
    let trigger = |v: u8| f32::from(v) / 255.0;
    PadState {
        left_stick: gamepad::stick(axis(raw.thumb_lx), axis(raw.thumb_ly)),
        right_stick: gamepad::stick(axis(raw.thumb_rx), axis(raw.thumb_ry)),
        left_trigger: gamepad::trigger(trigger(raw.left_trigger)),
        right_trigger: gamepad::trigger(trigger(raw.right_trigger)),
        buttons: XINPUT_BUTTONS
            .into_iter()
            .filter(|&(bit, _)| raw.buttons & bit != 0)
            .map(|(_, b)| b)
            .collect(),
    }
}

/// Back and Start together leave the title, back to the dashboard.
pub fn wants_exit(pad: &PadState) -> bool {
    pad.button(Button::Back) && pad.button(Button::Start)
}

/// The mission the title starts: the first line of `mission.txt` beside the
/// `.xex` when there is one (to go straight to a level), else the menu.
pub fn starting_mission(mission_txt: Option<&str>) -> String {
    mission_txt
        .and_then(|text| text.lines().next())
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .unwrap_or("main_menu")
        .to_owned()
}

/// Where the view looks, in degrees: the SDL runtime's `Look`, which turns
/// the head and rests the simulated hands in front of it.
#[derive(Default)]
pub struct Look {
    pub yaw: f32,
    pub pitch: f32,
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

    /// The head's rotation.
    pub fn rotation(&self) -> Quaternion<f32> {
        Self::rotation_of(self.yaw, self.pitch)
    }

    /// A resting simulated hand in front of the view, aiming down it; `side`
    /// is +1 for the right hand, -1 for the left.
    pub fn hand_pose(&self, side: f32) -> (Vector3<f32>, Quaternion<f32>) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use cgmath::vec2;

    #[test]
    fn xinput_buttons_sticks_and_triggers_map_to_the_shared_layout() {
        let raw = XInputGamepad {
            buttons: 0x1000 | 0x0008 | 0x0200,
            left_trigger: 255,
            right_trigger: 40,
            thumb_lx: 0,
            thumb_ly: 32767,
            thumb_rx: -32768,
            thumb_ry: 1000,
        };
        let pad = pad_state(&raw);
        assert_eq!(
            pad.buttons,
            [Button::DPadRight, Button::RightShoulder, Button::A]
        );
        assert_eq!(pad.left_stick, vec2(0.0, 1.0));
        // Past the deadzone on x only; y alone would be inside it, but the
        // deadzone is on the stick's length.
        assert_eq!(pad.right_stick.x, -1.0);
        assert!(pad.left_trigger);
        assert!(!pad.right_trigger, "40/255 is under the trigger threshold");
        assert!(pad.jump() && pad.lean() == 1.0);
    }

    #[test]
    fn a_resting_stick_is_still() {
        let raw = XInputGamepad {
            thumb_lx: 3000,
            thumb_ly: -3000,
            ..Default::default()
        };
        assert_eq!(pad_state(&raw).left_stick, vec2(0.0, 0.0));
    }

    #[test]
    fn back_and_start_together_exit() {
        let both = pad_state(&XInputGamepad {
            buttons: 0x0010 | 0x0020,
            ..Default::default()
        });
        let back = pad_state(&XInputGamepad {
            buttons: 0x0020,
            ..Default::default()
        });
        assert!(wants_exit(&both));
        assert!(!wants_exit(&back));
    }

    #[test]
    fn mission_txt_picks_the_start() {
        assert_eq!(starting_mission(None), "main_menu");
        assert_eq!(starting_mission(Some("  \n")), "main_menu");
        assert_eq!(starting_mission(Some("earth.mis\r\nignored")), "earth.mis");
    }
}
