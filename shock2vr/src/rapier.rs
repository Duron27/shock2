//! Transient VR rapier activation. Inventory/hand ownership remains authoritative;
//! this only follows it with blade animation, an attached hum, and a draw pulse.
use std::collections::HashMap;

use dark::{importers::VrHeldWeaponBounds, properties::PropLimbModel};
use engine::audio::AudioHandle;
use shipyard::{Component, EntityId, Get, View, World};

#[derive(Component, Clone, Copy)]
pub(crate) struct Blade(pub f32);

/// Endpoint envelopes measured from the authored hilt and complete weapon.
/// Interpolation keeps the whole hilt inside the collider at every extension.
#[derive(Component, Clone, Copy)]
pub(crate) struct Collider {
    pub hilt: VrHeldWeaponBounds,
    pub full: VrHeldWeaponBounds,
}

impl Collider {
    pub fn bounds(self, extension: f32) -> VrHeldWeaponBounds {
        let t = extension.clamp(0.0, 1.0);
        VrHeldWeaponBounds {
            size: self.hilt.size + (self.full.size - self.hilt.size) * t,
            center: self.hilt.center + (self.full.center - self.hilt.center) * t,
        }
    }
}

pub(crate) fn collider_bounds(world: &World, entity: EntityId) -> Option<VrHeldWeaponBounds> {
    let collider = world
        .borrow::<View<Collider>>()
        .ok()?
        .get(entity)
        .ok()
        .copied()?;
    let scale = world
        .borrow::<View<crate::runtime_props::RuntimePropGloveWeapon>>()
        .ok()
        .and_then(|v| v.get(entity).ok().map(|weapon| weapon.item_scale))
        .unwrap_or(1.0);
    let mut bounds = collider.bounds(amount(world, entity));
    bounds.size *= scale;
    bounds.center *= scale;
    Some(bounds)
}

#[derive(Default)]
struct Activation {
    hum: Option<AudioHandle>,
}

#[derive(Default)]
pub(crate) struct RapierFeedback(HashMap<EntityId, Activation>);

pub(crate) fn is_rapier(world: &World, entity: EntityId) -> bool {
    world.borrow::<View<PropLimbModel>>().is_ok_and(|v| {
        v.get(entity)
            .is_ok_and(|m| m.0.eq_ignore_ascii_case("rapier_h"))
    })
}

pub(crate) fn amount(world: &World, entity: EntityId) -> f32 {
    world
        .borrow::<View<Blade>>()
        .ok()
        .and_then(|v| v.get(entity).ok().map(|b| b.0))
        .unwrap_or(0.0)
}

/// Until the visible blade has extended, its full-length contact body must not
/// deal damage or advertise a damaging swing. Other melee weapons are unchanged.
pub(crate) fn ready(world: &World, entity: EntityId) -> bool {
    !is_rapier(world, entity) || amount(world, entity) >= 1.0
}

impl RapierFeedback {
    pub fn take_handles(&mut self) -> Vec<AudioHandle> {
        self.0.drain().filter_map(|(_, s)| s.hum).collect()
    }
}

#[cfg(test)]
mod tests {}
