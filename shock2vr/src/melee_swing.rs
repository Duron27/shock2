//! A support hand must be attached when a VR wrench swing starts and stay
//! attached through contact. Uses the current interaction attachment, never
//! controller proximity or the weapon body's collision-recovery velocity.
use shipyard::{EntityId, Unique, UniqueView};

/// One hand's swing state. `weapon` is what the hand is holding, so a swing
/// cannot survive putting the weapon down and picking another one up.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct SwingLatch {
    weapon: Option<EntityId>,
    hot: bool,
    two_handed: bool,
    sound: SwingSound,
}

impl SwingLatch {
}

/// A whoosh marks crossing the damage-speed gate, once until the hand slows.
/// The lower rearm point keeps tracking noise around the gate from chattering.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct SwingSound {
    armed: bool,
    cooldown: f32,
}

impl SwingSound {}

/// Both flat attacks and physical VR swings use the weapon's motion schema.
/// The shipped psi sword has none, so it falls back to the shard's whoosh.
pub(crate) fn sound_effect(world: &shipyard::World, entity: EntityId) -> crate::scripts::Effect {
    use crate::scripts::{Effect, script_util::play_environmental_sound};
    use dark::{EnvSoundQuery, properties::PropLimbModel};
    use engine::audio::AudioHandle;
    use shipyard::{Get, View};
    if !world
        .borrow::<View<PropLimbModel>>()
        .is_ok_and(|models| models.get(entity).is_ok())
    {
        return Effect::NoEffect;
    }
    // Smasher windup deliberately cannot damage until the trigger is released.
    // Keep the speed cue honest while that charge is being held.
    if world
        .borrow::<View<crate::runtime_props::RuntimePropMeleeCharge>>()
        .is_ok_and(|charges| {
            charges
                .get(entity)
                .is_ok_and(|charge| charge.held_seconds.is_some())
        })
    {
        return Effect::NoEffect;
    }
    let effect = play_environmental_sound(
        world,
        entity,
        "motion",
        vec![("plyrmelee", "2"), ("plyrmeleeswing", "1")],
        AudioHandle::new(),
    );
    match effect {
        Effect::PlayEnvironmentalSound {
            audio_handle,
            query,
            position,
        } => Effect::PlayEnvironmentalSoundWithFallback {
            audio_handle,
            query,
            position,
            fallback: EnvSoundQuery::from_tag_values(vec![
                ("event", "motion"),
                ("weapontype", "crystalshard"),
                ("plyrmelee", "2"),
                ("plyrmeleeswing", "1"),
            ]),
        },
        other => other,
    }
}

#[derive(Clone, Copy, Debug, Default, Unique)]
pub(crate) struct MeleeSwings(pub [SwingLatch; 2]);

impl MeleeSwings {
    /// Current-frame releases can downgrade the physics step's latch; a new
    /// grab cannot upgrade that already-started swing.
    pub fn release_unsupported(&mut self, supported: impl Fn(EntityId) -> bool) {
        for latch in &mut self.0 {
            latch.two_handed &= latch.weapon.is_some_and(&supported);
        }
    }
}

pub(crate) fn damage_scale(world: &shipyard::World, weapon: EntityId) -> f32 {
    use shipyard::{Get, View};
    let eligible = world
        .borrow::<View<dark::properties::PropModelName>>()
        .ok()
        .and_then(|models| {
            models.get(weapon).ok().map(|model| {
                // Two-handed melee weapons, once they have a support socket.
                // The psi sword is one-handed by design.
                matches!(model.0.as_str(), "wrench_h" | "rapier_h" | "shard_h")
                    && crate::vr_support::supports_model(&model.0)
            })
        })
        .unwrap_or(false);
    if !eligible {
        return 1.0;
    }
    let supported = world
        .borrow::<UniqueView<MeleeSwings>>()
        .is_ok_and(|swings| {
            swings
                .0
                .iter()
                .any(|latch| latch.weapon == Some(weapon) && latch.two_handed)
        });
    if supported {
        1.0
    } else {
        crate::dev_params::get(crate::dev_params::MELEE_ONE_HAND_SCALE)
    }
}

#[cfg(test)]
mod sound_tests {

    #[test]
    fn charging_smasher_is_silent_until_released() {
        use crate::runtime_props::{RuntimePropMeleeCharge, RuntimePropTransform};
        use cgmath::{Matrix4, SquareMatrix};
        use dark::properties::{PropClassTag, PropLimbModel};
        let mut world = shipyard::World::new();
        let weapon = world.add_entity((
            PropLimbModel("wrench_h".into()),
            PropClassTag::from_string("WeaponType Wrench"),
            RuntimePropTransform(Matrix4::identity()),
            RuntimePropMeleeCharge {
                fraction: 1.0,
                held_seconds: Some(1.0),
            },
        ));
        assert!(matches!(
            super::sound_effect(&world, weapon),
            crate::scripts::Effect::NoEffect
        ));
        world.add_component(
            weapon,
            RuntimePropMeleeCharge {
                fraction: 1.0,
                held_seconds: None,
            },
        );
        assert!(matches!(
            super::sound_effect(&world, weapon),
            crate::scripts::Effect::PlayEnvironmentalSoundWithFallback { .. }
        ));
    }
}
