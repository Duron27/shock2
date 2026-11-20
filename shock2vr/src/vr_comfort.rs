//! Artificial turn input and movement vignette for both presentations. Physical tracking is
//! never filtered, slowed, or used as a reason to restrict the view.
use crate::ui::entry_ramp::{EntryExitRamp, RampParams};
use crate::user_settings::{TurnMode, VignetteStrength, VrSettings};
use cgmath::{InnerSpace, Matrix4, SquareMatrix, Vector2, Vector3, vec3};
use engine::scene::SceneObject;

/// Sustained travel of the body across the physics step (platform carry,
/// falling, water currents), as opposed to contact jitter or one correction.
/// Above 0.08 wu/s for 50 ms - three 60Hz frames.
#[derive(Default)]
pub struct SustainedTravel {
    moving_secs: f32,
}

impl SustainedTravel {
    pub fn update(&mut self, displacement: Vector3<f32>, dt: f32) -> bool {
        let speed = displacement.magnitude() / dt;
        self.moving_secs = if speed.is_finite() && speed > 0.08 {
            self.moving_secs + dt
        } else {
            0.0
        };
        self.moving_secs >= 0.05
    }
}

#[derive(Default)]
pub struct VrComfort {
    turn_armed: bool,
    last_turn_settings: Option<(TurnMode, f32, f32)>,
    vignette: EntryExitRamp,
    travel: SustainedTravel,
}

impl VrComfort {
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    pub fn disarm_turn(&mut self) {
        self.turn_armed = false;
    }

    pub fn turn_radians(&mut self, axis: f32, dt: f32, settings: VrSettings) -> f32 {
        if !axis.is_finite() || !dt.is_finite() || dt <= 0.0 {
            return 0.0;
        }
        let signature = (settings.turning, settings.snap_angle, settings.smooth_speed);
        if self.last_turn_settings != Some(signature) {
            self.turn_armed = false;
            self.last_turn_settings = Some(signature);
        }
        let axis = axis.clamp(-1.0, 1.0);
        if axis.abs() < 0.25 {
            self.turn_armed = true;
        }
        match settings.turning {
            TurnMode::Snap if self.turn_armed && axis.abs() >= 0.7 => {
                self.turn_armed = false;
                axis.signum() * settings.snap_angle.to_radians()
            }
            TurnMode::Smooth if self.turn_armed && axis.abs() > 0.15 => {
                axis * settings.smooth_speed.to_radians() * dt
            }
            _ => 0.0,
        }
    }

    /// `displacement` is the body's travel across this frame's physics step,
    /// so a passive ride tunnels like stick movement does.
    pub fn update_vignette(
        &mut self,
        movement: Vector2<f32>,
        displacement: Vector3<f32>,
        turn_radians: f32,
        dt: f32,
        settings: VrSettings,
    ) {
        if settings.vignette == VignetteStrength::Off {
            self.vignette.snap_closed();
            self.travel = SustainedTravel::default();
            return;
        }
        if !dt.is_finite() || dt <= 0.0 {
            return;
        }
        let riding = self.travel.update(displacement, dt);
        let moving = settings.vignette_movement
            && (movement.x.abs() > 0.15 || movement.y.abs() > 0.15 || riding);
        let turning = settings.vignette_turning
            && settings.turning == TurnMode::Smooth
            && turn_radians.abs() > 0.0001;
        // A short eased entry and a gentler release avoid a flashing border as
        // the stick crosses its dead zone. Snap turns do not trigger the mask.
        if moving || turning {
            self.vignette.open(RampParams {
                attack_secs: 0.2,
                release_secs: 0.4,
            });
        } else {
            self.vignette.close();
        }
        self.vignette.update(dt);
    }

    pub fn render(
        &self,
        view: Matrix4<f32>,
        projection: Matrix4<f32>,
        settings: VrSettings,
    ) -> Option<SceneObject> {
        if settings.vignette == VignetteStrength::Off || self.vignette.eased() <= 0.0 {
            return None;
        }
        let eye = view.invert()?;
        Some(crate::hit_feedback::vignette_layer_at_pose(
            crate::hit_feedback::view_extents_from_projection(projection),
            eye,
            vec3(0.0, 0.0, 0.0),
            self.vignette.eased(),
            settings.vignette.radii(),
            "vr_comfort_vignette",
        ))
    }
}
