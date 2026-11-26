use glfw::{Action, Window};
use shock2vr::input::InputActionState;
use shock2vr::input::keyboard::{Key, KeyboardActionMapper};

/// The shared keyboard bindings ([`KeyboardActionMapper`]), fed from GLFW's
/// polled key state.
pub struct DesktopInputMapper {
    mapper: KeyboardActionMapper,
}

impl DesktopInputMapper {
    pub fn new() -> Self {
        Self {
            mapper: KeyboardActionMapper::new(),
        }
    }

    /// Call once per frame. Polls bound keys and triggers actions on
    /// rising edges; releases them when the keys come back up.
    pub fn poll(&mut self, window: &Window, state: &mut InputActionState) {
        self.mapper
            .poll(|key| window.get_key(glfw_key(key)) == Action::Press, state);
    }
}

fn glfw_key(key: Key) -> glfw::Key {
    use glfw::Key as G;
    match key {
        Key::A => G::A,
        Key::B => G::B,
        Key::C => G::C,
        Key::D => G::D,
        Key::E => G::E,
        Key::F => G::F,
        Key::G => G::G,
        Key::H => G::H,
        Key::I => G::I,
        Key::J => G::J,
        Key::K => G::K,
        Key::L => G::L,
        Key::M => G::M,
        Key::N => G::N,
        Key::O => G::O,
        Key::P => G::P,
        Key::Q => G::Q,
        Key::R => G::R,
        Key::S => G::S,
        Key::T => G::T,
        Key::U => G::U,
        Key::V => G::V,
        Key::W => G::W,
        Key::X => G::X,
        Key::Y => G::Y,
        Key::Z => G::Z,
        Key::Num0 => G::Num0,
        Key::Num1 => G::Num1,
        Key::Num2 => G::Num2,
        Key::Num3 => G::Num3,
        Key::Num4 => G::Num4,
        Key::Num5 => G::Num5,
        Key::Num6 => G::Num6,
        Key::Num7 => G::Num7,
        Key::Num8 => G::Num8,
        Key::Num9 => G::Num9,
        Key::Minus => G::Minus,
        Key::Equal => G::Equal,
        Key::Backslash => G::Backslash,
        Key::GraveAccent => G::GraveAccent,
        Key::Tab => G::Tab,
        Key::Escape => G::Escape,
        Key::Space => G::Space,
        Key::Enter => G::Enter,
        Key::Up => G::Up,
        Key::Down => G::Down,
        Key::Left => G::Left,
        Key::Right => G::Right,
        Key::LeftShift => G::LeftShift,
        Key::RightShift => G::RightShift,
        Key::LeftControl => G::LeftControl,
        Key::RightControl => G::RightControl,
        Key::LeftAlt => G::LeftAlt,
        Key::RightAlt => G::RightAlt,
    }
}
