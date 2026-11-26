//! Visual-only preparation of an empty glove for its available interaction.
//! Targets come from gameplay arbitration; no input, ownership or wrist motion changes.
use crate::hand_pose::{FingerAmounts, Pose};

#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub struct HandAnticipation {
    curls: [f32; 5],
    point: f32,
}

impl HandAnticipation {
    pub(crate) fn pose(
        &self,
        open: &Pose,
        point: &Pose,
        fist: &Pose,
        input: FingerAmounts,
    ) -> Pose {
        let [thumb, index, middle, ring, pinky] = self.curls;
        // Explicit analog input always wins: a full squeeze remains a full fist,
        // and a trigger pull still curls the extended index finger.
        open.blend(point, self.point).blend_per_finger(
            fist,
            &FingerAmounts {
                thumb: input.thumb.max(thumb),
                index: input.index.max(index),
                middle: input.middle.max(middle),
                ring: input.ring.max(ring),
                pinky: input.pinky.max(pinky),
            },
        )
    }
}
