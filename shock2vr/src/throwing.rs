//! Tracked release motion and bounded, one-impact damage for player-thrown props.
use std::collections::HashMap;

use cgmath::{InnerSpace, Vector3, Zero};
use dark::properties::{PropHitPoints, PropMaterial};
use shipyard::{EntityId, Get, IntoIter, IntoWithId, UniqueView, View, World};

use crate::{
    physics::{CollisionContact, PhysicsWorld},
    scripts::{DamageImpact, Effect, Message, MessagePayload, impact_sound::ImpactSoundGuard},
};

/// Read once at release/impact so one calculation uses a consistent set of knobs.
struct ThrowTuning {
    impact_min_speed: f32,
    damage_speed: f32,
    organic_cap: f32,
    inorganic_cap: f32,
}
impl ThrowTuning {
    fn read(read_value: impl Fn(crate::dev_params::DevParamId) -> f32) -> Self {
        use crate::dev_params::*;
        Self {
            impact_min_speed: read_value(THROW_IMPACT_MIN_SPEED),
            damage_speed: read_value(THROW_DAMAGE_SPEED),
            organic_cap: read_value(THROW_ORGANIC_CAP),
            inorganic_cap: read_value(THROW_INORGANIC_CAP),
        }
    }
    fn current() -> Self {
        Self::read(crate::dev_params::get)
    }
    #[cfg(test)]
    fn defaults() -> Self {
        Self::read(|id| crate::dev_params::spec(id).default)
    }
}

#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub(crate) struct ReleaseMotion {
    pub linear: Vector3<f32>,
    pub angular: Vector3<f32>,
}

impl Default for ReleaseMotion {
    fn default() -> Self {
        Self {
            linear: Vector3::zero(),
            angular: Vector3::zero(),
        }
    }
}

#[derive(shipyard::Unique, Default)]
struct TrackingEpoch(u64);
pub(crate) fn cancel_tracking(world: &World) {
    if let Ok(mut epoch) = world.borrow::<shipyard::UniqueViewMut<TrackingEpoch>>() {
        epoch.0 = epoch.0.wrapping_add(1);
    } else {
        world.add_unique(TrackingEpoch(1));
    }
}

#[derive(Clone, Copy)]
pub(crate) struct BodyMotion {
    pub center: Vector3<f32>,
    pub linear: Vector3<f32>,
    pub angular: Vector3<f32>,
}
impl BodyMotion {
    fn at(self, point: Vector3<f32>) -> Vector3<f32> {
        self.linear + self.angular.cross(point - self.center)
    }
}

/// In-flight state persisted with the prop, including spent throws' remaining motion.
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize, shipyard::Component)]
pub struct SavedThrow {
    mass: f32,
    remaining: f32,
    armed: bool,
    motion: ReleaseMotion,
}
#[derive(shipyard::Unique, Default)]
pub(crate) struct SavedThrows(pub HashMap<u64, SavedThrow>);
#[derive(Default)]
pub(crate) struct ThrownItems {
    active: HashMap<EntityId, SavedThrow>,
    before_step: HashMap<EntityId, BodyMotion>,
    sound_guards: HashMap<EntityId, ImpactSoundGuard>,
}

fn impact_damage(mass: f32, closing_speed: f32, organic: bool, tuning: &ThrowTuning) -> f32 {
    if !closing_speed.is_finite() || closing_speed < tuning.impact_min_speed {
        return 0.0;
    }
    let cap = if organic {
        tuning.organic_cap
    } else {
        tuning.inorganic_cap
    };
    // Mug reaches its cap at 6 world units/s; light/heavy objects differ in
    // how quickly they reach the same intentionally small ceiling.
    (cap * mass.clamp(0.1, 4.0) * (closing_speed / tuning.damage_speed).powi(2))
        .min(cap)
        .floor()
}

impl ThrownItems {
    /// Track a release whose motion is already set (a flat inventory toss, a
    /// fumbled VR weapon): impact audio and provenance, never armed to damage.
    pub fn track_existing_motion(
        &mut self,
        world: &World,
        physics: &PhysicsWorld,
        entity: EntityId,
    ) {
        let Some(body) = physics.snapshot_body_motion().get(&entity).copied() else {
            return;
        };
        self.cancel(entity);
        self.active.insert(
            entity,
            SavedThrow {
                mass: 1.0,
                remaining: 5.0,
                armed: false,
                motion: ReleaseMotion {
                    linear: body.linear,
                    angular: body.angular,
                },
            },
        );
        self.publish(world, physics);
    }

    pub fn cancel(&mut self, entity: EntityId) {
        self.active.remove(&entity);
        self.sound_guards.remove(&entity);
    }

    pub fn restore(&mut self, world: &World, physics: &mut PhysicsWorld) {
        for (entity, saved) in world.borrow::<View<SavedThrow>>().unwrap().iter().with_id() {
            if physics.release_motion(entity, saved.motion) {
                self.active.insert(entity, *saved);
            }
        }
        self.publish(world, physics);
    }

    pub fn publish(&mut self, world: &World, physics: &PhysicsWorld) {
        let motion = if self.active.is_empty() {
            HashMap::new()
        } else {
            physics.snapshot_body_motion()
        };
        self.active.retain(|id, saved| {
            let Some(body) = motion.get(id) else {
                return false;
            };
            saved.motion = ReleaseMotion {
                linear: body.linear,
                angular: body.angular,
            };
            true
        });
        let saved = SavedThrows(
            self.active
                .iter()
                .map(|(id, value)| (id.inner(), *value))
                .collect(),
        );
        if let Ok(mut current) = world.borrow::<shipyard::UniqueViewMut<SavedThrows>>() {
            *current = saved;
        } else {
            world.add_unique(saved);
        }
    }

    pub fn prepare(&mut self, physics: &PhysicsWorld, dt: f32) {
        self.active.retain(|entity, throw| {
            throw.remaining = (throw.remaining - dt).max(0.0);
            throw.armed &= throw.remaining > 0.0;
            !physics.is_held_item(*entity)
                && physics.get_velocity(*entity).is_some_and(|v| {
                    throw.remaining > 0.0
                        || v.magnitude() > 0.05
                        || throw.motion.angular.magnitude() > 0.05
                })
        });
        self.sound_guards.retain(|entity, guard| {
            guard.tick(dt);
            self.active.contains_key(entity)
        });
        self.before_step = if self.active.is_empty() {
            HashMap::new()
        } else {
            physics.snapshot_body_motion()
        };
    }

    /// Audible collisions for player-released props, independent of their
    /// one-shot damage budget. Melee and terminal projectiles already own
    /// their sound path; never duplicate their playback here.
    pub fn impact_sound(
        &mut self,
        world: &World,
        physics: &PhysicsWorld,
        item: EntityId,
        target: EntityId,
        contact: Option<CollisionContact>,
    ) -> Effect {
        use dark::properties::{CollisionType, PropCollisionType, PropLimbModel};
        if !self.active.contains_key(&item)
            || world
                .borrow::<View<PropLimbModel>>()
                .is_ok_and(|v| v.get(item).is_ok())
            || world.borrow::<View<PropCollisionType>>().is_ok_and(|v| {
                v.get(item).is_ok_and(|p| {
                    p.collision_type.intersects(
                        CollisionType::SLAY_ON_IMPACT | CollisionType::DESTROY_ON_IMPACT,
                    )
                })
            })
        {
            return Effect::NoEffect;
        }
        let Some(contact) = contact else {
            return Effect::NoEffect;
        };
        let target = crate::util::resolve_proxy_entity(world, target);
        let speed = self.contact_speed(item, target, contact).unwrap_or(0.0);
        if !self
            .sound_guards
            .entry(item)
            .or_default()
            .should_play_speed(target, speed)
        {
            return Effect::NoEffect;
        }
        crate::scripts::impact_sound::impact_sound_effect(
            item,
            target,
            world,
            physics,
            Some(contact),
        )
    }

    fn contact_speed(
        &self,
        item: EntityId,
        target: EntityId,
        contact: CollisionContact,
    ) -> Option<f32> {
        let incoming = self.before_step.get(&item)?.at(contact.point);
        let other = self
            .before_step
            .get(&target)
            .map(|m| m.at(contact.point))
            .unwrap_or_else(Vector3::zero);
        Some((incoming - other).dot(contact.normal).max(0.0))
    }

    pub fn impact(
        &mut self,
        world: &World,
        item: EntityId,
        target: EntityId,
        contact: Option<CollisionContact>,
    ) -> Option<Message> {
        // First solid contact spends the throw, even against scenery. Keep
        // its motion for saves, but never bill bounces or duplicate colliders.
        let throw = self.active.get_mut(&item)?;
        if !std::mem::replace(&mut throw.armed, false) {
            return None;
        }
        let contact = contact?;
        let mass = throw.mass;
        let target = crate::util::resolve_proxy_entity(world, target);
        if world
            .borrow::<UniqueView<crate::mission::PlayerInfo>>()
            .is_ok_and(|p| p.entity_id == target)
        {
            return None;
        }
        let hp = world.borrow::<View<PropHitPoints>>().ok()?;
        if hp.get(target).ok()?.hit_points <= 0 {
            return None;
        }
        let speed = self.contact_speed(item, target, contact)?;
        let organic = world
            .borrow::<View<PropMaterial>>()
            .ok()
            .and_then(|v| {
                v.get(target)
                    .ok()
                    .map(|p| p.0.to_ascii_lowercase().contains("flesh"))
            })
            .unwrap_or(false);
        let amount = impact_damage(mass, speed, organic, &ThrowTuning::current());
        (amount > 0.0).then_some(Message {
            to: target,
            payload: MessagePayload::Damage {
                amount,
                impact: Some(DamageImpact {
                    direction: contact.normal,
                    point: contact.point,
                    bone: None,
                }),
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cgmath::vec3;

    fn impact_fixture(
        material: &str,
    ) -> (World, ThrownItems, EntityId, EntityId, CollisionContact) {
        let mut world = World::new();
        let item = world.add_entity(());
        let target = world.add_entity((
            PropHitPoints { hit_points: 20 },
            PropMaterial(material.into()),
        ));
        let motion = ReleaseMotion {
            linear: vec3(6.0, 0.0, 0.0),
            angular: Vector3::zero(),
        };
        let mut throws = ThrownItems::default();
        throws.active.insert(
            item,
            SavedThrow {
                mass: 1.0,
                remaining: 5.0,
                armed: true,
                motion,
            },
        );
        throws.before_step.insert(
            item,
            BodyMotion {
                center: Vector3::zero(),
                linear: motion.linear,
                angular: motion.angular,
            },
        );
        let contact = CollisionContact {
            point: Vector3::zero(),
            normal: Vector3::unit_x(),
            closing_speed: None,
            surface_material: None,
        };
        (world, throws, item, target, contact)
    }

    #[test]
    fn thrown_prop_sound_is_thresholded_rate_limited_and_independent_of_damage() {
        use dark::properties::{CollisionType, PropCollisionType};
        let (mut world, mut throws, item, target, contact) = impact_fixture("Material Metal");
        world.add_component(item, PropMaterial("Material Glass".into()));
        world.add_unique(SavedThrows(
            throws
                .active
                .iter()
                .map(|(id, t)| (id.inner(), *t))
                .collect(),
        ));
        // A stationary release or a spent damage budget still clatters; sound
        // uses the pre-solve contact speed, not the damage latch or remaining HP.
        throws.active.get_mut(&item).unwrap().armed = false;
        throws.before_step.get_mut(&item).unwrap().linear = vec3(0.01, 0.0, 0.0);
        assert!(matches!(
            throws.impact_sound(&world, &PhysicsWorld::new(), item, target, Some(contact)),
            Effect::NoEffect
        ));
        throws.before_step.get_mut(&item).unwrap().linear = vec3(6.0, 0.0, 0.0);
        let effects = Effect::flatten(vec![throws.impact_sound(
            &world,
            &PhysicsWorld::new(),
            item,
            target,
            Some(contact),
        )]);
        assert_eq!(
            effects
                .iter()
                .filter(|e| matches!(e, Effect::PlayImpactSound { .. }))
                .count(),
            1
        );
        let Effect::PlayImpactSound { query, .. } = &effects[0] else {
            panic!("missing sound");
        };
        assert!(
            query
                .tag_values()
                .contains(&("material".into(), "glass".into()))
        );
        assert!(
            query
                .tag_values()
                .contains(&("material2".into(), "metal".into()))
        );

        assert_eq!(
            effects
                .iter()
                .filter(|e| matches!(e, Effect::PlayImpactSound { source, .. } if *source == item))
                .count(),
            1
        );
        assert!(matches!(
            throws.impact_sound(&world, &PhysicsWorld::new(), item, target, Some(contact)),
            Effect::NoEffect
        ));
        throws.sound_guards.get_mut(&item).unwrap().tick(0.2);
        world.add_component(
            item,
            PropCollisionType {
                collision_type: CollisionType::NO_COLLISION_SOUND,
            },
        );
        assert!(matches!(
            throws.impact_sound(&world, &PhysicsWorld::new(), item, target, Some(contact)),
            Effect::NoEffect
        ));
    }

    #[test]
    fn only_player_owned_impacts_generate_investigation_cues() {
        use crate::runtime_props::{
            RuntimePropLaunchedProjectile, RuntimePropPlayerFiredProjectile,
        };
        use dark::properties::PropClassTag;
        let mut world = World::new();
        let projectile = world.add_entity((
            PropClassTag::from_string("AmmoType Standard"),
            RuntimePropLaunchedProjectile,
        ));
        let wall = world.add_entity(());
        let effects = Effect::flatten(vec![crate::scripts::script_util::play_impact_sound(
            &world,
            projectile,
            wall,
            Vector3::zero(),
            None,
        )]);
        assert_eq!(effects.len(), 1, "enemy impact plays audio only");
        assert!(matches!(effects[0], Effect::PlayEnvironmentalSound { .. }));
        world.add_component(projectile, RuntimePropPlayerFiredProjectile);
        let effects = Effect::flatten(vec![crate::scripts::script_util::play_impact_sound(
            &world,
            projectile,
            wall,
            Vector3::zero(),
            None,
        )]);
        assert!(
            effects
                .iter()
                .any(|e| matches!(e, Effect::PlayImpactSound { .. }))
        );
    }

    #[test]
    fn incoming_speed_caps_material_damage_and_spends_the_throw() {
        for (material, expected) in [("Material FleshTarget", 2.0), ("Material Metal", 1.0)] {
            let (world, mut throws, item, target, contact) = impact_fixture(material);
            let hit = throws.impact(&world, item, target, Some(contact)).unwrap();
            assert!(
                matches!(hit.payload, MessagePayload::Damage { amount, .. } if amount == expected)
            );
            assert!(throws.impact(&world, item, target, Some(contact)).is_none());
        }
    }

    #[test]
    fn scenery_and_same_speed_targets_do_not_deal_damage() {
        let (mut world, mut throws, item, target, contact) = impact_fixture("Material FleshTarget");
        throws
            .before_step
            .insert(target, *throws.before_step.get(&item).unwrap());
        assert!(throws.impact(&world, item, target, Some(contact)).is_none());
        throws.active.get_mut(&item).unwrap().armed = true;
        let wall = world.add_entity(());
        assert!(throws.impact(&world, item, wall, Some(contact)).is_none());
        assert!(throws.impact(&world, item, target, Some(contact)).is_none());
    }

    #[test]
    fn saved_throw_remaps_identity_preserves_spin_and_does_not_rearm() {
        let (_, mut throws, item, _, _) = impact_fixture("Material FleshTarget");
        let saved = throws.active.get_mut(&item).unwrap();
        saved.armed = false;
        saved.motion.angular = vec3(0.0, 4.0, 0.0);
        let mut data = crate::save_load::EntitySaveData::empty();
        data.all_entities.push(item.inner());
        data.thrown_props.insert(item.inner(), *saved);
        let decoded: crate::save_load::EntitySaveData =
            serde_json::from_str(&serde_json::to_string(&data).unwrap()).unwrap();
        let mut restored = World::new();
        restored.add_entity(()); // Ensure identity actually changes.
        let (_, remap) = decoded.instantiate(&mut restored);
        let values = restored.borrow::<View<SavedThrow>>().unwrap();
        let value = values.get(remap[&item]).unwrap();
        assert!(!value.armed);
        assert_eq!(value.motion.angular, vec3(0.0, 4.0, 0.0));
        assert_eq!(value.motion.linear, vec3(6.0, 0.0, 0.0));
        assert_eq!(value.remaining, 5.0);
    }

    #[test]
    fn cup_damage_caps_and_grazes() {
        assert_eq!(impact_damage(1.0, 6.0, true, &ThrowTuning::defaults()), 2.0);
        assert_eq!(
            impact_damage(1.0, 6.0, false, &ThrowTuning::defaults()),
            1.0
        );
        assert_eq!(
            impact_damage(100.0, 100.0, true, &ThrowTuning::defaults()),
            2.0
        );
        assert_eq!(
            impact_damage(100.0, 100.0, false, &ThrowTuning::defaults()),
            1.0
        );
        assert_eq!(impact_damage(1.0, 1.9, true, &ThrowTuning::defaults()), 0.0);
        assert_eq!(
            impact_damage(1.0, f32::NAN, true, &ThrowTuning::defaults()),
            0.0
        );
        assert_eq!(impact_damage(0.2, 6.0, true, &ThrowTuning::defaults()), 0.0);
    }
}
