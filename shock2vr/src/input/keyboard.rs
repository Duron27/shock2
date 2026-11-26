//! Keyboard bindings for discrete actions, shared by every keyboard runtime
//! (the GLFW desktop runtime and the SDL runtime alpha3 hosts on Android).
//!
//! Runtimes translate their own key codes into [`Key`] and call
//! [`KeyboardActionMapper::poll`] once per frame; the table and the edge
//! logic live here once so the two runtimes cannot drift apart.

use super::{InputAction, InputActionState};
use std::collections::HashSet;

/// The keys bindings can name, independent of any windowing library.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Key {
    A,
    B,
    C,
    D,
    E,
    F,
    G,
    H,
    I,
    J,
    K,
    L,
    M,
    N,
    O,
    P,
    Q,
    R,
    S,
    T,
    U,
    V,
    W,
    X,
    Y,
    Z,
    Num0,
    Num1,
    Num2,
    Num3,
    Num4,
    Num5,
    Num6,
    Num7,
    Num8,
    Num9,
    Minus,
    Equal,
    Backslash,
    GraveAccent,
    Tab,
    Escape,
    Space,
    Enter,
    Up,
    Down,
    Left,
    Right,
    LeftShift,
    RightShift,
    LeftControl,
    RightControl,
    LeftAlt,
    RightAlt,
}

/// Modifier requirement for a binding. `None` requires Alt NOT held (so a
/// plain-key binding doesn't also fire during a system chord like Alt+Tab);
/// `Alt` requires it held.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Modifier {
    None,
    Alt,
}

struct Binding {
    key: Key,
    modifier: Modifier,
    action: InputAction,
}

/// Maps polled keyboard state to discrete input actions.
///
/// Runtimes report which keys are down every frame (rather than key
/// events), so the mapper tracks which actions were down last frame and
/// triggers actions only on rising edges.
pub struct KeyboardActionMapper {
    bindings: Vec<Binding>,
    prev_down: HashSet<InputAction>,
}

impl Default for KeyboardActionMapper {
    fn default() -> Self {
        Self::new()
    }
}

impl KeyboardActionMapper {
    pub fn new() -> Self {
        let bindings = vec![
            Binding {
                key: Key::P,
                modifier: Modifier::None,
                action: InputAction::PathfindingTestCycle,
            },
            Binding {
                key: Key::B,
                modifier: Modifier::Alt,
                action: InputAction::DebugCycleWeapon,
            },
            // Original System Shock 2 direct weapon bindings. These select a
            // matching carried item; unlike DebugCycleWeapon they never spawn one.
            Binding {
                key: Key::Num1,
                modifier: Modifier::None,
                action: InputAction::EquipWrench,
            },
            Binding {
                key: Key::Num2,
                modifier: Modifier::None,
                action: InputAction::EquipPistol,
            },
            Binding {
                key: Key::Num3,
                modifier: Modifier::None,
                action: InputAction::EquipShotgun,
            },
            Binding {
                key: Key::Num4,
                modifier: Modifier::None,
                action: InputAction::EquipAssaultRifle,
            },
            Binding {
                key: Key::Num5,
                modifier: Modifier::None,
                action: InputAction::EquipLaserPistol,
            },
            Binding {
                key: Key::Num6,
                modifier: Modifier::None,
                action: InputAction::EquipEmpRifle,
            },
            Binding {
                key: Key::Num7,
                modifier: Modifier::None,
                action: InputAction::EquipElectroShock,
            },
            Binding {
                key: Key::Num8,
                modifier: Modifier::None,
                action: InputAction::EquipGrenadeLauncher,
            },
            Binding {
                key: Key::Num9,
                modifier: Modifier::None,
                action: InputAction::EquipStasisFieldGenerator,
            },
            Binding {
                key: Key::Num0,
                modifier: Modifier::None,
                action: InputAction::EquipFusionCannon,
            },
            Binding {
                key: Key::Minus,
                modifier: Modifier::None,
                action: InputAction::EquipCrystalShard,
            },
            Binding {
                key: Key::Equal,
                modifier: Modifier::None,
                action: InputAction::EquipViralProliferator,
            },
            Binding {
                key: Key::Backslash,
                modifier: Modifier::None,
                action: InputAction::EquipWormLauncher,
            },
            Binding {
                key: Key::GraveAccent,
                modifier: Modifier::None,
                action: InputAction::EquipPsiAmp,
            },
            Binding {
                key: Key::R,
                modifier: Modifier::None,
                action: InputAction::Reload,
            },
            Binding {
                key: Key::B,
                modifier: Modifier::None,
                action: InputAction::CycleAmmo,
            },
            Binding {
                key: Key::T,
                modifier: Modifier::Alt,
                action: InputAction::ClearPsiTeleport,
            },
            Binding {
                key: Key::T,
                modifier: Modifier::None,
                action: InputAction::CycleAmmo,
            },
            // Alt-modified so a stray press cannot dump a magazine mid-fight;
            // in VR this is the gun hand's lower face button instead. NOT
            // `Alt+R`: `poll` keys its edges on the action, so with `R` held
            // down a mere Alt tap would flip `Reload` off and fire the eject.
            Binding {
                key: Key::X,
                modifier: Modifier::Alt,
                action: InputAction::EjectClip,
            },
            Binding {
                key: Key::F,
                modifier: Modifier::None,
                action: InputAction::CycleGunSetting,
            },
            Binding {
                key: Key::Y,
                modifier: Modifier::None,
                action: InputAction::CyclePsiPower,
            },
            Binding {
                key: Key::S,
                modifier: Modifier::Alt,
                action: InputAction::QuickSave,
            },
            Binding {
                key: Key::L,
                modifier: Modifier::Alt,
                action: InputAction::QuickLoad,
            },
            Binding {
                key: Key::R,
                modifier: Modifier::Alt,
                action: InputAction::ToggleInputRecording,
            },
            Binding {
                key: Key::Tab,
                modifier: Modifier::None,
                action: InputAction::ToggleUseMode,
            },
            // Escape opens/closes the in-game pause menu, as it does in the
            // original. It used to close the window outright (a raw glfw
            // handler in `main.rs`); the pause menu's "Quit to Main Menu" plus
            // the main menu's "Quit" is now the way out.
            Binding {
                key: Key::Escape,
                modifier: Modifier::None,
                action: InputAction::TogglePauseMenu,
            },
            // AI debug: make everything hunt the player (pinned) / calm down
            Binding {
                key: Key::G,
                modifier: Modifier::Alt,
                action: InputAction::DebugForceChase,
            },
            Binding {
                key: Key::C,
                modifier: Modifier::Alt,
                action: InputAction::DebugCalmAll,
            },
            Binding {
                key: Key::U,
                modifier: Modifier::None,
                action: InputAction::ReadLastUnreadLog,
            },
            Binding {
                key: Key::M,
                modifier: Modifier::None,
                action: InputAction::ToggleMap,
            },
            // Detach/re-attach the debug camera. Alt-modified like the other
            // debug bindings above, and inert unless the `free_camera` dev
            // param is on (the gate lives in `Game`, so every runtime and
            // HTTP injection share one rule). Alt is Option on macOS.
            Binding {
                key: Key::V,
                modifier: Modifier::Alt,
                action: InputAction::ToggleFreeCamera,
            },
        ];

        Self {
            bindings,
            prev_down: HashSet::new(),
        }
    }

    pub fn poll(&mut self, is_key_down: impl Fn(Key) -> bool, state: &mut InputActionState) {
        let is_alt_pressed = is_key_down(Key::LeftAlt) || is_key_down(Key::RightAlt);

        let mut down_now: HashSet<InputAction> = HashSet::new();
        for binding in &self.bindings {
            let modifier_satisfied = match binding.modifier {
                Modifier::None => !is_alt_pressed,
                Modifier::Alt => is_alt_pressed,
            };
            if modifier_satisfied && is_key_down(binding.key) {
                down_now.insert(binding.action);
            }
        }

        for action in down_now.difference(&self.prev_down) {
            state.trigger(*action);
        }
        for action in self.prev_down.difference(&down_now) {
            state.release(*action);
        }
        self.prev_down = down_now;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One simulated frame: `held` are the keys physically down.
    fn frame(mapper: &mut KeyboardActionMapper, state: &mut InputActionState, held: &[Key]) {
        state.clear_triggered();
        mapper.poll(|key| held.contains(&key), state);
    }

    #[test]
    fn alt_t_clears_the_marker_without_also_cycling_ammo() {
        let mut mapper = KeyboardActionMapper::new();
        let mut state = InputActionState::new();
        frame(&mut mapper, &mut state, &[Key::LeftAlt, Key::T]);
        assert!(state.just_triggered(InputAction::ClearPsiTeleport));
        assert!(!state.just_triggered(InputAction::CycleAmmo));
        frame(&mut mapper, &mut state, &[Key::LeftAlt, Key::T]);
        assert!(!state.just_triggered(InputAction::ClearPsiTeleport));
        frame(&mut mapper, &mut state, &[]);
        frame(&mut mapper, &mut state, &[Key::T]);
        assert!(state.just_triggered(InputAction::CycleAmmo));
        assert!(!state.just_triggered(InputAction::ClearPsiTeleport));
    }

    #[test]
    fn b_cycles_ammo_and_alt_b_is_debug_only() {
        let mut mapper = KeyboardActionMapper::new();
        let mut state = InputActionState::new();
        frame(&mut mapper, &mut state, &[Key::B]);
        assert!(state.just_triggered(InputAction::CycleAmmo));
        assert!(!state.just_triggered(InputAction::DebugCycleWeapon));
        frame(&mut mapper, &mut state, &[]);
        frame(&mut mapper, &mut state, &[Key::LeftAlt, Key::B]);
        assert!(state.just_triggered(InputAction::DebugCycleWeapon));
        assert!(!state.just_triggered(InputAction::CycleAmmo));
    }

    /// The baseline every binding relies on: hold a key, get one edge. This
    /// holds under the old per-binding logic too - the flicker only needed a
    /// second binding, which `twin_bindings_for_one_action_do_not_fight`
    /// builds.
    #[test]
    fn a_held_key_triggers_its_action_once_not_every_frame() {
        let mut mapper = KeyboardActionMapper::new();
        let mut state = InputActionState::new();

        frame(&mut mapper, &mut state, &[Key::Tab]);
        assert!(state.just_triggered(InputAction::ToggleUseMode));

        for _ in 0..10 {
            frame(&mut mapper, &mut state, &[Key::Tab]);
            assert!(
                !state.just_triggered(InputAction::ToggleUseMode),
                "a held Tab must not re-trigger use mode"
            );
            assert!(state.is_held(InputAction::ToggleUseMode));
        }
    }

    /// And releasing re-arms it, so tap-tap is two toggles.
    #[test]
    fn releasing_rearms_a_shared_action() {
        let mut mapper = KeyboardActionMapper::new();
        let mut state = InputActionState::new();

        frame(&mut mapper, &mut state, &[Key::Tab]);
        frame(&mut mapper, &mut state, &[]);
        assert!(!state.is_held(InputAction::ToggleUseMode));

        frame(&mut mapper, &mut state, &[Key::Tab]);
        assert!(state.just_triggered(InputAction::ToggleUseMode));
    }

    /// The flicker regression test. Two keys on one action: either alone
    /// works, and handing off from one to the other while the action stays
    /// down mints no second edge. Keying edges per *binding* fails here - the
    /// unpressed twin releases the action every frame while the pressed one
    /// re-triggers it, which is exactly what `Tab` did while `I` was bound to
    /// `ToggleUseMode` as well. This also protects the B/T ammo-cycle aliases.
    #[test]
    fn twin_bindings_for_one_action_do_not_fight() {
        let mut mapper = KeyboardActionMapper::new();
        mapper.bindings.push(Binding {
            key: Key::I,
            modifier: Modifier::None,
            action: InputAction::ToggleUseMode,
        });
        let mut state = InputActionState::new();

        frame(&mut mapper, &mut state, &[Key::I]);
        assert!(state.just_triggered(InputAction::ToggleUseMode));

        // Tab pressed while I is still down: the action never came up.
        frame(&mut mapper, &mut state, &[Key::I, Key::Tab]);
        assert!(!state.just_triggered(InputAction::ToggleUseMode));
        frame(&mut mapper, &mut state, &[Key::Tab]);
        assert!(!state.just_triggered(InputAction::ToggleUseMode));
        assert!(state.is_held(InputAction::ToggleUseMode));

        frame(&mut mapper, &mut state, &[]);
        assert!(!state.is_held(InputAction::ToggleUseMode));
    }

    /// And `I` really is gone: it must not toggle use mode any more.
    #[test]
    fn i_no_longer_toggles_use_mode() {
        let mut mapper = KeyboardActionMapper::new();
        let mut state = InputActionState::new();

        frame(&mut mapper, &mut state, &[Key::I]);
        assert!(!state.just_triggered(InputAction::ToggleUseMode));
        assert!(!state.is_held(InputAction::ToggleUseMode));
    }

    /// Alt gating still holds: a plain binding must not fire during Alt+key,
    /// and an Alt binding only fires with Alt down.
    #[test]
    fn alt_gates_plain_and_alt_bindings() {
        let mut mapper = KeyboardActionMapper::new();
        let mut state = InputActionState::new();

        frame(&mut mapper, &mut state, &[Key::LeftAlt, Key::Tab]);
        assert!(!state.just_triggered(InputAction::ToggleUseMode));

        frame(&mut mapper, &mut state, &[Key::S]);
        assert!(!state.just_triggered(InputAction::QuickSave));
        frame(&mut mapper, &mut state, &[Key::LeftAlt, Key::S]);
        assert!(state.just_triggered(InputAction::QuickSave));
    }

    #[test]
    fn original_weapon_keys_map_to_direct_equip_actions() {
        let mapper = KeyboardActionMapper::new();
        let expected = [
            (Key::Num1, InputAction::EquipWrench),
            (Key::Num2, InputAction::EquipPistol),
            (Key::Num3, InputAction::EquipShotgun),
            (Key::Num4, InputAction::EquipAssaultRifle),
            (Key::Num5, InputAction::EquipLaserPistol),
            (Key::Num6, InputAction::EquipEmpRifle),
            (Key::Num7, InputAction::EquipElectroShock),
            (Key::Num8, InputAction::EquipGrenadeLauncher),
            (Key::Num9, InputAction::EquipStasisFieldGenerator),
            (Key::Num0, InputAction::EquipFusionCannon),
            (Key::Minus, InputAction::EquipCrystalShard),
            (Key::Equal, InputAction::EquipViralProliferator),
            (Key::Backslash, InputAction::EquipWormLauncher),
            (Key::GraveAccent, InputAction::EquipPsiAmp),
        ];

        for (key, action) in expected {
            let binding = mapper
                .bindings
                .iter()
                .find(|binding| binding.key == key && binding.modifier == Modifier::None);
            assert_eq!(binding.map(|binding| binding.action), Some(action));
        }
    }
}
