//! Deterministic pickup fitting against visible triangles. All outputs are
//! hand-local; tracked motion never re-runs the search or changes item scale.
use cgmath::{
    Matrix4, Vector3,
};

use crate::vr_config::Handedness;

pub const CURL_STEPS: usize = 24;
/// The auto-fit's finger thickness; Tighten uses the glove's own
/// (`FingerChain` spheres).
pub const FINGER_RADIUS: f32 = 0.007 / crate::METERS_PER_WORLD_UNIT;

/// Sampled from the very same glove skeleton and pose endpoints used to draw it.
pub struct GripKinematics {
    pub fingers: [Vec<Vec<Vector3<f32>>>; 5],
    pub palm: Vector3<f32>,
    pub normal: Vector3<f32>,
}






/// A posed finger as `GripSurface::tighten` tests it: per phalanx, from its
/// base to its tip, `PHALANX_SAMPLES` spheres it fills, swept between
/// neighbours (glove frame, authored size). Phalanx `k` of finger `f` turns
/// with `FINGER_BONES[f]` up to `PHALANX_BONES[f] + k`, as the glove's do:
/// the thumb's metacarpal first, the other fingers' from the knuckle.
#[derive(Clone, Debug)]
pub struct FingerChain {
    pub phalanges: Vec<[Sphere; PHALANX_SAMPLES]>,
}

pub const PHALANX_SAMPLES: usize = 5;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sphere {
    pub center: Vector3<f32>,
    pub radius: f32,
}

impl FingerChain {
    /// Phalanges between neighbouring `joints`, as thick as `radius`
    /// everywhere.
    pub fn uniform(joints: &[Vector3<f32>], radius: f32) -> Self {
        let phalanges = joints
            .windows(2)
            .map(|ends| {
                std::array::from_fn(|i| {
                    let t = i as f32 / (PHALANX_SAMPLES - 1) as f32;
                    Sphere {
                        center: ends[0] * (1.0 - t) + ends[1] * t,
                        radius,
                    }
                })
            })
            .collect();
        FingerChain { phalanges }
    }
}



/// Shared boundary conversion for both the rendered glove and fitting samples.
pub fn glove_to_hand(hand: Handedness) -> Matrix4<f32> {
    hand.mirror()
        * Matrix4::from_angle_y(cgmath::Deg(180.0))
        * Matrix4::from_scale(crate::hand_glove::GLOVE_SCALE)
}






/// Stable hashes invalidate a bake when either the visible mesh or glove's
/// sampled pose arcs change. Quantize below visible precision (0.076 mm) so
/// host/Quest floating-point differences and signed zero do not invalidate a bake.
fn fingerprint(values: impl IntoIterator<Item = f32>) -> String {
    let mut hash = 0xcbf29ce484222325_u64;
    for value in values {
        for byte in ((value * 10_000.0).round() as i64).to_le_bytes() {
            hash = (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3);
        }
    }
    format!("{hash:016x}")
}

impl GripKinematics {
    pub fn fingerprint(&self) -> String {
        fingerprint(
            self.fingers
                .iter()
                .flatten()
                .flatten()
                .copied()
                .chain([self.palm, self.normal])
                .flat_map(|p| [p.x, p.y, p.z]),
        )
    }
}
