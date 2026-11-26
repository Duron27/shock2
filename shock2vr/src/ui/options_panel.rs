//! Retail Options canvas, shared by the main menu and the paused mission.
//!
//! PC settings only: mouse, display and audio. (The VR comfort/turning
//! preferences in `user_settings::VrSettings` keep their defaults.)
use super::{HAlign, Rect, UiCanvas, VAlign, resolve_menu_label, resolve_menu_rects};
use crate::user_settings::{
    self, BRIGHTNESSES, FIELDS_OF_VIEW, MOUSE_SENSITIVITIES, PcSettings, VOLUMES,
};
use cgmath::{Vector2, vec2};
use dark::importers::{STRINGS_IMPORTER, UI_LAYOUT_IMPORTER};
use engine::assets::asset_cache::AssetCache;

const TABS: [Rect; 4] = [
    Rect::new(4.0, 8.0, 152.0, 56.0),
    Rect::new(164.0, 8.0, 152.0, 56.0),
    Rect::new(324.0, 8.0, 152.0, 56.0),
    Rect::new(484.0, 8.0, 151.0, 56.0),
];
const FRAME: [Rect; 2] = [
    Rect::new(244.0, 408.0, 152.0, 56.0),
    Rect::new(20.0, 67.0, 601.0, 335.0),
];
// Keep retail audio's five rows; extend its pitch for additional controls.
const ROWS: [Rect; 10] = [
    Rect::new(93.0, 73.0, 481.0, 23.0),
    Rect::new(93.0, 102.0, 481.0, 23.0),
    Rect::new(93.0, 130.0, 481.0, 23.0),
    Rect::new(93.0, 158.0, 481.0, 23.0),
    Rect::new(93.0, 187.0, 481.0, 23.0),
    Rect::new(93.0, 216.0, 481.0, 23.0),
    Rect::new(93.0, 245.0, 481.0, 23.0),
    Rect::new(93.0, 274.0, 481.0, 23.0),
    Rect::new(93.0, 303.0, 481.0, 23.0),
    Rect::new(93.0, 332.0, 481.0, 23.0),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptionsEvent {
    Tab(usize),
    Row(usize),
    Done,
}

pub struct OptionsLayout {
    pub tabs: Vec<Rect>,
    pub frame: Vec<Rect>,
    pub rows: Vec<Rect>,
}

impl Default for OptionsLayout {
    fn default() -> Self {
        Self {
            tabs: TABS.to_vec(),
            frame: FRAME.to_vec(),
            rows: ROWS.to_vec(),
        }
    }
}

impl OptionsLayout {
    pub fn load(assets: &mut AssetCache) -> Self {
        let mut read = |name, fallback: &[Rect]| {
            let layout = assets.get_opt(&UI_LAYOUT_IMPORTER, name);
            resolve_menu_rects(layout.as_deref().map(|r| r.as_slice()), fallback)
        };
        // Only the first five OPTIONAR rects are full-width option rows;
        // later retail widgets have different geometry. Extend the final row
        // using the authored pitch, rather than borrowing an unrelated rect.
        let mut rows = read("OPTIONAR.BIN", &ROWS[..5]);
        let last = rows[4];
        let pitch = last.y - rows[3].y;
        for step in 1..=(ROWS.len() - 5) {
            rows.push(Rect::new(
                last.x,
                last.y + pitch * step as f32,
                last.w,
                last.h,
            ));
        }
        Self {
            tabs: read("OPTIONTR.BIN", &TABS),
            frame: read("OPTIONSR.BIN", &FRAME),
            rows,
        }
    }
}

#[derive(Default)]
pub struct OptionsPanel {
    tab: usize,
    error: bool,
}

const TAB_LABELS: [&str; 3] = ["Controls", "Display", "Audio"];

impl OptionsPanel {
    pub fn hit(&self, layout: &OptionsLayout, point: Vector2<f32>) -> Option<OptionsEvent> {
        if layout.frame[0].contains(point) {
            return Some(OptionsEvent::Done);
        }
        // The fourth retail tab slot has no page and stays inert.
        for (i, rect) in layout.tabs.iter().take(TAB_LABELS.len()).enumerate() {
            if rect.contains(point) {
                return Some(OptionsEvent::Tab(i));
            }
        }
        let count = row_labels(&user_settings::get().pc, self.tab).len();
        for (i, rect) in layout.rows.iter().take(count).enumerate() {
            if rect.contains(point) {
                return Some(OptionsEvent::Row(i));
            }
        }
        None
    }

    pub fn activate(&mut self, event: OptionsEvent) -> bool {
        match event {
            OptionsEvent::Done => return true,
            OptionsEvent::Tab(tab) if tab < TAB_LABELS.len() => {
                self.tab = tab;
                self.error = false;
            }
            OptionsEvent::Row(row) => {
                let tab = self.tab;
                self.error =
                    user_settings::update(|settings| change(&mut settings.pc, tab, row)).is_err();
            }
            _ => {}
        }
        false
    }

    pub fn canvas(&self, assets: &mut AssetCache, pointer: Option<Vector2<f32>>) -> UiCanvas {
        let layout = OptionsLayout::load(assets);
        let settings = user_settings::get().pc;
        let mut canvas = UiCanvas::new(vec2(640.0, 480.0));
        canvas.image(Rect::new(0.0, 0.0, 640.0, 480.0), "OPTIONS.PCX");
        let hover = pointer.and_then(|p| self.hit(&layout, p));
        for (i, label) in TAB_LABELS.iter().enumerate() {
            canvas
                .text_native_fit(
                    layout.tabs[i],
                    label,
                    "metafont.fon",
                    HAlign::Center,
                    VAlign::Middle,
                )
                .opacity(if i == self.tab || hover == Some(OptionsEvent::Tab(i)) {
                    1.0
                } else {
                    0.55
                });
        }
        for (i, label) in row_labels(&settings, self.tab).iter().enumerate() {
            canvas
                .text_native_fit(
                    layout.rows[i],
                    label,
                    "metafont.fon",
                    HAlign::Center,
                    VAlign::Middle,
                )
                .opacity(if hover == Some(OptionsEvent::Row(i)) {
                    1.0
                } else {
                    0.7
                });
        }
        if self.error {
            let body = layout.frame[1];
            canvas
                .text(
                    Rect::new(body.x + 16.0, body.y + body.h - 38.0, body.w - 32.0, 24.0),
                    "Could not save settings. Previous values kept.",
                    "mainfont.fon",
                    14.0,
                    HAlign::Center,
                    VAlign::Middle,
                )
                .opacity(0.55);
        }
        let strings = assets.get_opt(&STRINGS_IMPORTER, "options.str");
        let done = resolve_menu_label(strings.as_deref(), "done", "Done");
        canvas
            .text_native_fit(
                layout.frame[0],
                &done,
                "metafont.fon",
                HAlign::Center,
                VAlign::Middle,
            )
            .opacity(if hover == Some(OptionsEvent::Done) {
                1.0
            } else {
                0.7
            });
        canvas
    }
}

fn on_off(on: bool) -> &'static str {
    if on { "On" } else { "Off" }
}

fn percent(value: f32) -> String {
    format!("{:.0}%", value * 100.0)
}

fn row_labels(pc: &PcSettings, tab: usize) -> Vec<String> {
    match tab {
        0 => vec![
            format!("Mouse sensitivity: {}x", pc.mouse_sensitivity),
            format!("Invert mouse: {}", on_off(pc.invert_mouse_y)),
            format!("Gamepad rumble: {}", on_off(pc.gamepad_rumble)),
            "Reset control defaults".to_owned(),
        ],
        1 => vec![
            format!("Field of view: {} degrees", pc.field_of_view),
            format!("Brightness: {:.1}", pc.brightness),
            format!("Fullscreen: {}", on_off(pc.fullscreen)),
            format!("VSync: {}", on_off(pc.vsync)),
            "Reset display defaults".to_owned(),
        ],
        _ => vec![
            format!("Master volume: {}", percent(pc.master_volume)),
            format!("Music volume: {}", percent(pc.music_volume)),
            format!("Effects volume: {}", percent(pc.effects_volume)),
            "Reset audio defaults".to_owned(),
        ],
    }
}

fn change(pc: &mut PcSettings, tab: usize, row: usize) {
    let defaults = PcSettings::default();
    match (tab, row) {
        (0, 0) => pc.mouse_sensitivity = cycle(pc.mouse_sensitivity, &MOUSE_SENSITIVITIES),
        (0, 1) => pc.invert_mouse_y = !pc.invert_mouse_y,
        (0, 2) => pc.gamepad_rumble = !pc.gamepad_rumble,
        (0, 3) => {
            pc.mouse_sensitivity = defaults.mouse_sensitivity;
            pc.invert_mouse_y = defaults.invert_mouse_y;
            pc.gamepad_rumble = defaults.gamepad_rumble;
        }
        (1, 0) => pc.field_of_view = cycle(pc.field_of_view, &FIELDS_OF_VIEW),
        (1, 1) => pc.brightness = cycle(pc.brightness, &BRIGHTNESSES),
        (1, 2) => pc.fullscreen = !pc.fullscreen,
        (1, 3) => pc.vsync = !pc.vsync,
        (1, 4) => {
            pc.field_of_view = defaults.field_of_view;
            pc.brightness = defaults.brightness;
            pc.fullscreen = defaults.fullscreen;
            pc.vsync = defaults.vsync;
        }
        (2, 0) => pc.master_volume = cycle(pc.master_volume, &VOLUMES),
        (2, 1) => pc.music_volume = cycle(pc.music_volume, &VOLUMES),
        (2, 2) => pc.effects_volume = cycle(pc.effects_volume, &VOLUMES),
        (2, 3) => {
            pc.master_volume = defaults.master_volume;
            pc.music_volume = defaults.music_volume;
            pc.effects_volume = defaults.effects_volume;
        }
        _ => {}
    }
}

/// The step after `value`, wrapping; an off-list value restarts at the first.
fn cycle(value: f32, values: &[f32]) -> f32 {
    let current = values.iter().position(|v| (*v - value).abs() < 1e-4);
    current.map_or(values[0], |i| values[(i + 1) % values.len()])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout() -> OptionsLayout {
        OptionsLayout {
            tabs: TABS.to_vec(),
            frame: FRAME.to_vec(),
            rows: ROWS.to_vec(),
        }
    }

    #[test]
    fn three_pc_tabs_are_live_and_the_fourth_slot_is_inert() {
        let panel = OptionsPanel::default();
        let layout = layout();
        assert_eq!(
            panel.hit(&layout, vec2(320.0, 440.0)),
            Some(OptionsEvent::Done)
        );
        assert_eq!(
            panel.hit(&layout, vec2(400.0, 30.0)),
            Some(OptionsEvent::Tab(2))
        );
        assert_eq!(panel.hit(&layout, vec2(560.0, 30.0)), None);
        assert_eq!(
            panel.hit(&layout, vec2(320.0, 80.0)),
            Some(OptionsEvent::Row(0))
        );
    }

    #[test]
    fn rows_past_a_tabs_last_setting_do_not_hit() {
        let panel = OptionsPanel::default();
        let layout = layout();
        let beyond = layout.rows[4];
        assert_eq!(
            panel.hit(&layout, vec2(beyond.x + 1.0, beyond.y + 1.0)),
            None
        );
    }

    #[test]
    fn controls_cycle_sensitivity_and_reset() {
        let mut pc = PcSettings::default();
        change(&mut pc, 0, 0);
        assert_eq!(pc.mouse_sensitivity, 1.25);
        change(&mut pc, 0, 1);
        assert!(pc.invert_mouse_y);
        change(&mut pc, 0, 2);
        assert!(!pc.gamepad_rumble);
        change(&mut pc, 0, 3);
        assert_eq!(pc, PcSettings::default());
    }

    #[test]
    fn display_and_audio_rows_change_their_own_settings() {
        let mut pc = PcSettings::default();
        change(&mut pc, 1, 0);
        assert_eq!(pc.field_of_view, 50.0);
        change(&mut pc, 1, 1);
        assert_eq!(pc.brightness, 1.6);
        change(&mut pc, 1, 2);
        assert!(pc.fullscreen);
        change(&mut pc, 2, 0);
        assert_eq!(pc.master_volume, 0.0, "100% wraps to mute");
        change(&mut pc, 2, 3);
        assert_eq!(pc.master_volume, 1.0);
        assert!(pc.fullscreen, "audio reset leaves display alone");
    }
}
