//! Cosmetic walking bob for the flat viewmodel: whatever the player holds
//! drifts in a figure-eight and dips on each footstep while they move, the way
//! the original's held weapon swayed. Driven by the player's measured ground
//! speed, so it scales with walking/running and fades out on stopping. A
//! camera-space layer like `flat_turn_sway`: it never bends the aim ray.
use cgmath::{Deg, InnerSpace, Quaternion, Rotation3, Vector3, vec2, vec3};

/// Ground speed (world units/s) at which the bob reaches full amplitude.
const FULL_SPEED: f32 = 5.0;
/// Ground covered by one full bob cycle (two footsteps), in world units.
const CYCLE_LENGTH: f32 = 3.2;
/// Side-to-side and footstep-dip amplitudes at full speed, in world units of
/// camera space, and the roll that accompanies the sideways drift.
const SIDE_AMPLITUDE: f32 = 0.018;
const DIP_AMPLITUDE: f32 = 0.014;
const ROLL_DEG: f32 = 1.2;
/// How quickly the bob fades in/out as speed changes.
const FADE_SECONDS: f32 = 0.12;
/// Faster than any locomotion: a teleport, load or level change, not a step.
const DISCONTINUITY_SPEED: f32 = 30.0;

#[derive(Default)]
pub(crate) struct FlatWalkBob {
    previous: Option<Vector3<f32>>,
    phase: f32,
    intensity: f32,
}

/// The bob for one frame: a camera-space offset (+x right, +y up) and a roll
/// about the view axis.
pub(crate) struct Bob {
    pub offset: Vector3<f32>,
    pub roll: Quaternion<f32>,
}

impl FlatWalkBob {
    pub fn step(&mut self, player_pos: Vector3<f32>, dt: f32, scale: f32) -> Bob {
        // Paused redraws: hold the current pose rather than re-sampling.
        if dt > 0.0 && dt.is_finite() {
            let previous = self.previous.replace(player_pos);
            let speed = previous
                .map(|old| {
                    let d = player_pos - old;
                    vec2(d.x, d.z).magnitude() / dt
                })
                .unwrap_or(0.0);
            if speed > DISCONTINUITY_SPEED || dt > 0.25 {
                self.intensity = 0.0;
            } else {
                let target = (speed / FULL_SPEED).min(1.0);
                self.intensity += (target - self.intensity) * (1.0 - (-dt / FADE_SECONDS).exp());
                self.phase = (self.phase + speed * dt * std::f32::consts::TAU / CYCLE_LENGTH)
                    .rem_euclid(std::f32::consts::TAU);
            }
        }
        self.bob(scale)
    }

    fn bob(&self, scale: f32) -> Bob {
        let amount = self.intensity * scale;
        let side = self.phase.sin();
        // Twice per cycle: lowest as each foot lands, back to rest between.
        let dip = (2.0 * self.phase).cos() * 0.5 - 0.5;
        Bob {
            offset: vec3(side * SIDE_AMPLITUDE, dip * DIP_AMPLITUDE, 0.0) * amount,
            roll: Quaternion::from_angle_z(Deg(-side * ROLL_DEG * amount)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn walk(bob: &mut FlatWalkBob, speed: f32, frames: usize) -> Bob {
        let dt = 1.0 / 60.0;
        let mut pos = bob.previous.unwrap_or(vec3(0.0, 0.0, 0.0));
        let mut last = bob.step(pos, dt, 1.0);
        for _ in 0..frames {
            pos.z += speed * dt;
            last = bob.step(pos, dt, 1.0);
        }
        last
    }

    #[test]
    fn standing_still_does_not_bob() {
        let mut bob = FlatWalkBob::default();
        let pose = walk(&mut bob, 0.0, 120);
        assert!(pose.offset.magnitude() < 1e-6);
    }

    #[test]
    fn walking_moves_the_held_item_and_stopping_settles_it() {
        let mut bob = FlatWalkBob::default();
        let mut peak: f32 = 0.0;
        for _ in 0..60 {
            peak = peak.max(walk(&mut bob, FULL_SPEED, 1).offset.magnitude());
        }
        assert!(peak > SIDE_AMPLITUDE * 0.5, "walking bobs: {peak}");
        let settled = walk(&mut bob, 0.0, 60);
        assert!(settled.offset.magnitude() < peak * 0.05, "stopping settles");
    }

    #[test]
    fn a_teleport_is_not_a_stride() {
        let mut bob = FlatWalkBob::default();
        walk(&mut bob, 0.0, 2);
        let pose = bob.step(vec3(100.0, 0.0, 0.0), 1.0 / 60.0, 1.0);
        assert!(pose.offset.magnitude() < 1e-6);
    }
}
