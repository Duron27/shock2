//! Permanent personal credential/wallet tool. Ownership is transient presentation;
//! balances and permissions remain in QuestInfo. Reuses the body inventory frame.
use shipyard::{EntityId, Get, View, World};

pub(crate) fn is_reader(world: &World, entity: EntityId) -> bool {
    world
        .borrow::<View<dark::properties::PropKeyDst>>()
        .is_ok_and(|keys| keys.get(entity).is_ok())
        || super::mfd_device::machine_uses_mfd(world, entity)
        || [
            "Computer",
            "SecurityComputer",
            "EnergyStation",
            "ResurrectMachine",
        ]
        .iter()
        .any(|script| crate::scripts::script_util::entity_has_script(world, entity, script))
}

/// Key targets, MFD machines and chargers accept frob as well as scanning.
pub(crate) fn requires_scan(world: &World, entity: EntityId) -> bool {
    is_reader(world, entity)
        && !super::mfd_device::machine_uses_mfd(world, entity)
        && !crate::scripts::script_util::entity_has_script(world, entity, "EnergyStation")
        && !world
            .borrow::<View<dark::properties::PropKeyDst>>()
            .is_ok_and(|keys| keys.get(entity).is_ok())
}
