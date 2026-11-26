//! Transient controller output, separate from input and persistent game state.
use shipyard::{Unique, UniqueViewMut, World};

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub struct HapticPulse {
    pub amplitude: f32,
    pub duration_ms: u32,
}

/// One psi selector step: a light detent, like a notched dial.
pub const PSI_NAV_TICK: HapticPulse = HapticPulse {
    amplitude: 0.25,
    duration_ms: 12,
};

/// A released item was consumed by its target: firmer than the ready tick.
pub const TOOL_APPLIED: HapticPulse = HapticPulse {
    amplitude: 0.7,
    duration_ms: 90,
};

/// Smasher feedback follows charge transitions, never the per-frame readout.
pub(crate) fn melee_charge_pulse(previous: Option<f32>, next: Option<f32>) -> Option<HapticPulse> {
    match (previous, next) {
        (previous, Some(0.0)) if previous != Some(0.0) => Some(HapticPulse {
            amplitude: 0.2,
            duration_ms: 30,
        }),
        (Some(previous), Some(next)) if previous < 1.0 && next >= 1.0 => Some(HapticPulse {
            amplitude: 0.6,
            duration_ms: 70,
        }),
        _ => None,
    }
}

/// Psi amp charge, sent every frame of the hold (each pulse just outlasts a
/// frame): a rumble swelling toward the overload zone, a detent on entering
/// it, then a steady buzz that says "release now".
pub(crate) fn psi_charge_pulse(previous: f32, fraction: f32) -> HapticPulse {
    let zone = crate::psi::OVERLOAD_ZONE_START;
    if previous < zone && fraction >= zone {
        PSI_OVERLOAD_READY
    } else if fraction >= zone {
        HapticPulse {
            amplitude: 0.45,
            duration_ms: 20,
        }
    } else {
        HapticPulse {
            amplitude: 0.1 + 0.25 * (fraction / zone).clamp(0.0, 1.0),
            duration_ms: 20,
        }
    }
}

pub const PSI_OVERLOAD_READY: HapticPulse = HapticPulse {
    amplitude: 0.8,
    duration_ms: 40,
};
pub const PSI_CAST: HapticPulse = HapticPulse {
    amplitude: 0.7,
    duration_ms: 50,
};
pub const PSI_OVERLOAD_CAST: HapticPulse = HapticPulse {
    amplitude: 1.0,
    duration_ms: 120,
};
pub const PSI_BURNOUT: HapticPulse = HapticPulse {
    amplitude: 1.0,
    duration_ms: 350,
};
/// A cast that went nowhere: a long dull buzz, unlike a cast's sharp kick.
/// Still louder than the charge, which the mixer would otherwise let mask it.
pub const PSI_FIZZLE: HapticPulse = HapticPulse {
    amplitude: 0.5,
    duration_ms: 150,
};

/// A shot leaving the player's gun: a sharp kick on the trigger hand.
pub const WEAPON_FIRE: HapticPulse = HapticPulse {
    amplitude: 0.55,
    duration_ms: 70,
};

/// The player losing `damage` hit points, sent to both hands: a scratch is a
/// tap, a blast a long jolt. Reads the loss actually applied, so armour that
/// soaked a hit makes it gentler.
pub fn player_hit_pulse(damage: f32) -> HapticPulse {
    let damage = damage.max(0.0);
    HapticPulse {
        amplitude: (0.35 + damage / 20.0).min(1.0),
        duration_ms: (120.0 + damage * 12.0).min(450.0) as u32,
    }
}

#[derive(Default, Unique, serde::Serialize)]
pub struct HapticFeedback {
    pub pending: [Option<HapticPulse>; 2],
    /// Monotonic request counters for headless/device diagnostics, left/right.
    pub sequence: [u64; 2],
    /// The latest winning request per hand, never consumed (read with
    /// `sequence` to tell new from stale): the debug runtime's idle zero-dt
    /// frames `take` a pulse before a headless client can read it.
    pub last: [Option<HapticPulse>; 2],
}

impl HapticFeedback {
    pub fn request(&mut self, hand: crate::Handedness, pulse: HapticPulse) {
        let hand = crate::vr_config::hand_slot(hand);
        if !pulse.amplitude.is_finite() || pulse.amplitude <= 0.0 || pulse.duration_ms == 0 {
            return;
        }
        let pulse = HapticPulse {
            amplitude: pulse.amplitude.min(1.0),
            duration_ms: pulse.duration_ms,
        };
        self.sequence[hand] += 1;
        // Same-frame effects resolve deterministically: the stronger pulse
        // wins, then the longer one. A ready tick cannot overwrite an impact.
        if self.pending[hand].is_none_or(|current| {
            pulse.amplitude > current.amplitude
                || (pulse.amplitude == current.amplitude && pulse.duration_ms > current.duration_ms)
        }) {
            self.pending[hand] = Some(pulse);
        }
        self.last[hand] = self.pending[hand];
    }
}

/// Consume once. A menu/transition world without gameplay output is silent.
pub fn take(world: &World) -> [Option<HapticPulse>; 2] {
    world
        .borrow::<UniqueViewMut<HapticFeedback>>()
        .map(|mut feedback| std::mem::take(&mut feedback.pending))
        .unwrap_or([None; 2])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn player_hits_scale_with_damage_and_stay_in_range() {
        let scratch = player_hit_pulse(1.0);
        let blast = player_hit_pulse(40.0);
        assert!(scratch.amplitude < blast.amplitude);
        assert!(scratch.duration_ms < blast.duration_ms);
        assert_eq!(blast.amplitude, 1.0);
        assert_eq!(player_hit_pulse(1000.0).duration_ms, 450);
        assert!(player_hit_pulse(-5.0).amplitude > 0.0);
    }
}
