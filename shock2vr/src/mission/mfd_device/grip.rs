//! Live tricorder placement, independent of the ordinary access-card fit.

#[derive(Clone, Copy, Debug)]
pub(super) enum Edge {
    Bottom,
    Left,
    Top,
    Right,
}

pub(super) struct Tuning {
    pub edge: Edge,
}

pub(super) fn tuning(hand: usize) -> Tuning {
    use crate::dev_params::*;
    let keys = if hand == 0 {
        [
            VR_MFD_LEFT_GRIP_EDGE,
            VR_MFD_LEFT_GRIP_X,
            VR_MFD_LEFT_GRIP_Y,
            VR_MFD_LEFT_GRIP_Z,
            VR_MFD_LEFT_GRIP_PITCH,
            VR_MFD_LEFT_GRIP_YAW,
            VR_MFD_LEFT_GRIP_ROLL,
        ]
    } else {
        [
            VR_MFD_RIGHT_GRIP_EDGE,
            VR_MFD_RIGHT_GRIP_X,
            VR_MFD_RIGHT_GRIP_Y,
            VR_MFD_RIGHT_GRIP_Z,
            VR_MFD_RIGHT_GRIP_PITCH,
            VR_MFD_RIGHT_GRIP_YAW,
            VR_MFD_RIGHT_GRIP_ROLL,
        ]
    };
    let edge = get(keys[0]);
    Tuning {
        edge: match edge.round() as u8 {
            1 => Edge::Left,
            2 => Edge::Top,
            3 => Edge::Right,
            _ => Edge::Bottom,
        },
    }
}

/// The authored left pinch reaches farther across alternate edges. Keep this
/// fit correction separate from position so the frame still meets the fingers.
pub(super) fn margin(hand: usize) -> f32 {
    use crate::dev_params::*;
    super::grip_margin()
        + if matches!(tuning(hand).edge, Edge::Bottom) {
            0.0
        } else {
            get(if hand == 0 {
                VR_MFD_LEFT_GRIP_CLEARANCE
            } else {
                VR_MFD_RIGHT_GRIP_CLEARANCE
            }) / crate::world_scale::meters_per_world_unit()
        }
}

#[cfg(test)]
mod tests {
    
    
}
