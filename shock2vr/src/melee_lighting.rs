//! Short-range light emitted by the two luminous melee weapons while held.
use cgmath::{EuclideanSpace, Point3, Transform, Vector4};
use dark::properties::PropLimbModel;
use engine::scene::light::PointLight;
use shipyard::{EntityId, Get, View, World};

use crate::runtime_props::{RuntimePropTransform, RuntimePropViewmodelToWorld};

pub(crate) fn held_lights(world: &World, hands: [Option<EntityId>; 2]) -> Vec<PointLight> {
    let models = world.borrow::<View<PropLimbModel>>().unwrap();
    let transforms = world.borrow::<View<RuntimePropTransform>>().unwrap();
    hands
        .into_iter()
        .enumerate()
        .filter_map(|(hand, entity)| {
            let entity = entity?;
            if hand == 1 && hands[0] == Some(entity) {
                return None;
            }
            let model = &models.get(entity).ok()?.0;
            let (color_intensity, extension) = if model.eq_ignore_ascii_case("rapier_h") {
                (Vector4::new(0.12, 0.65, 1.0, 0.35), 1.0)
            } else if model.eq_ignore_ascii_case("shard_h") {
                (Vector4::new(0.4, 0.9, 0.7, 0.15), 1.0)
            } else {
                return None;
            };
            if extension <= 0.0 {
                return None;
            }
            // This is the rendered weapon's frame, including physical hand following.
            // Flat viewmodels use a separate depth layer; map that frame back into
            // the mission so their light falls on nearby world surfaces too.
            let mut position = transforms
                .get(entity)
                .ok()?
                .0
                .transform_point(Point3::new(0.0, 0.0, 0.0));
            if let Some(mapping) = world
                .borrow::<View<RuntimePropViewmodelToWorld>>()
                .ok()
                .and_then(|v| v.get(entity).ok().copied())
            {
                position = mapping.0.transform_point(position);
            }
            Some(PointLight {
                position: position.to_vec(),
                color_intensity: Vector4::new(
                    color_intensity.x,
                    color_intensity.y,
                    color_intensity.z,
                    color_intensity.w * extension,
                ),
                range: 2.0,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use cgmath::{Matrix4, vec3};

    #[test]
    fn only_held_luminous_weapons_emit() {
        let mut world = World::new();
        let rapier = world.add_entity((
            PropLimbModel("rapier_h".into()),
            RuntimePropTransform(Matrix4::from_translation(vec3(1.0, 2.0, 3.0))),
        ));
        let shard = world.add_entity((
            PropLimbModel("shard_h".into()),
            RuntimePropTransform(Matrix4::from_translation(vec3(4.0, 5.0, 6.0))),
        ));
        assert!(held_lights(&world, [None, None]).is_empty());
        let lights = held_lights(&world, [Some(rapier), Some(shard)]);
        assert_eq!(lights.len(), 2);
        assert_eq!(lights[0].position, vec3(1.0, 2.0, 3.0));
        assert_eq!(lights[1].position, vec3(4.0, 5.0, 6.0));
        assert_eq!(held_lights(&world, [Some(shard), Some(shard)]).len(), 1);
        world.add_component(
            rapier,
            RuntimePropViewmodelToWorld(Matrix4::from_translation(vec3(10.0, 0.0, 0.0))),
        );
        let flat = held_lights(&world, [Some(rapier), None]);
        assert_eq!(flat[0].position, vec3(11.0, 2.0, 3.0));
        assert_eq!(flat[0].color_intensity.w, 0.35);
    }
}
