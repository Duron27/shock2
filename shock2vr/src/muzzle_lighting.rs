//! A gunshot lights its surroundings for as long as its muzzle flash lives,
//! and an explosion for as long as its effect does.
//!
//! Not in the original (Dark lit nothing dynamically from a flash); added so
//! gunfire in a dark corridor briefly shows what is there - player and AI
//! shots alike. A flash is any live entity bolted to its weapon
//! (`RuntimePropAttachment`) whose template descends from "Muzzle Flashes".
use std::{cell::RefCell, collections::HashMap};

use cgmath::{EuclideanSpace, Point3, Transform, Vector4};
use dark::properties::PropTemplateId;
use engine::scene::light::PointLight;
use shipyard::{Get, IntoIter, IntoWithId, View, World};

use crate::runtime_props::{
    RuntimePropAttachment, RuntimePropTransform, RuntimePropViewmodelToWorld,
};

/// Template -370, "Muzzle Flashes": the assault and slug flashes.
pub(crate) const MUZZLE_FLASHES: i32 = -370;

const COLOR_INTENSITY: Vector4<f32> = Vector4::new(1.0, 0.72, 0.38, 1.4);
const RANGE: f32 = 12.0;

pub(crate) fn flash_lights(world: &World, hierarchy: &HashMap<i32, Vec<i32>>) -> Vec<PointLight> {
    let (Ok(attachments), Ok(templates), Ok(transforms), Ok(mappings)) = (
        world.borrow::<View<RuntimePropAttachment>>(),
        world.borrow::<View<PropTemplateId>>(),
        world.borrow::<View<RuntimePropTransform>>(),
        world.borrow::<View<RuntimePropViewmodelToWorld>>(),
    ) else {
        return Vec::new();
    };
    (&attachments, &templates, &transforms)
        .iter()
        .with_id()
        .filter(|(_, (_, template, _))| is_muzzle_flash(hierarchy, template.template_id))
        .map(|(_, (attachment, _, transform))| {
            let mut position = transform.0.transform_point(Point3::new(0.0, 0.0, 0.0));
            // A flat viewmodel's flash lives in the viewmodel's space.
            if let Ok(mapping) = mappings.get(attachment.parent) {
                position = mapping.0.transform_point(position);
            }
            PointLight {
                position: position.to_vec(),
                color_intensity: COLOR_INTENSITY,
                range: RANGE,
            }
        })
        .collect()
}

/// Template -382, "Explosions": grenades, fusion, cryo, viral, EMP...
pub(crate) const EXPLOSIONS: i32 = -382;
const CRYO_EXPLOSIONS: i32 = -2211;
const VIRAL_EXPLOSIONS: i32 = -2696;
const EMP_EXPLOSION: i32 = -4858;
const STASIS_EXPLOSION: i32 = -3869;

/// Explosion tint per template, memoized: classifying walks the template
/// ancestry, which is too slow to repeat for every entity every frame.
pub(crate) type ExplosionClasses = RefCell<HashMap<i32, Option<Vector4<f32>>>>;

fn explosion_color(hierarchy: &HashMap<i32, Vec<i32>>, id: i32) -> Option<Vector4<f32>> {
    let ancestors = dark::ss2_entity_info::get_ancestors(hierarchy, &id);
    let is = |family: i32| id == family || ancestors.contains(&family);
    if !is(EXPLOSIONS) {
        return None;
    }
    Some(
        if is(CRYO_EXPLOSIONS) || is(EMP_EXPLOSION) || is(STASIS_EXPLOSION) {
            Vector4::new(0.45, 0.7, 1.0, 2.2)
        } else if is(VIRAL_EXPLOSIONS) {
            Vector4::new(0.5, 1.0, 0.35, 1.8)
        } else {
            Vector4::new(1.0, 0.6, 0.25, 2.0)
        },
    )
}

/// Live explosion effects, tinted by family. Brighter and wider than a
/// muzzle flash.
pub(crate) fn explosion_lights(
    world: &World,
    hierarchy: &HashMap<i32, Vec<i32>>,
    classes: &ExplosionClasses,
) -> Vec<PointLight> {
    let (Ok(templates), Ok(transforms)) = (
        world.borrow::<View<PropTemplateId>>(),
        world.borrow::<View<RuntimePropTransform>>(),
    ) else {
        return Vec::new();
    };
    let mut classes = classes.borrow_mut();
    (&templates, &transforms)
        .iter()
        // Explosions are spawned from (negative-id) archetypes at runtime.
        .filter(|(template, _)| template.template_id < 0)
        .filter_map(|(template, transform)| {
            let id = template.template_id;
            let color = *classes
                .entry(id)
                .or_insert_with(|| explosion_color(hierarchy, id));
            Some(PointLight {
                position: transform
                    .0
                    .transform_point(Point3::new(0.0, 0.0, 0.0))
                    .to_vec(),
                color_intensity: color?,
                range: 18.0,
            })
        })
        .collect()
}

fn is_muzzle_flash(hierarchy: &HashMap<i32, Vec<i32>>, template_id: i32) -> bool {
    template_id == MUZZLE_FLASHES
        || dark::ss2_entity_info::get_ancestors(hierarchy, &template_id).contains(&MUZZLE_FLASHES)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cgmath::{Matrix4, SquareMatrix, vec3};

    #[test]
    fn only_attached_muzzle_flash_descendants_light() {
        let hierarchy = HashMap::from([(-2653, vec![MUZZLE_FLASHES]), (-9, vec![-5])]);
        let mut world = World::new();
        let gun = world.add_entity(());
        let attach = RuntimePropAttachment {
            parent: gun,
            local_transform: Matrix4::identity(),
        };
        let at = |x| RuntimePropTransform(Matrix4::from_translation(vec3(x, 0.0, 0.0)));
        world.add_entity((attach, PropTemplateId { template_id: -2653 }, at(1.0)));
        // Attached, but not a flash (e.g. stun sparks).
        world.add_entity((attach, PropTemplateId { template_id: -9 }, at(2.0)));
        // A flash template that is not attached.
        world.add_entity((PropTemplateId { template_id: -2653 }, at(3.0)));
        let lights = flash_lights(&world, &hierarchy);
        assert_eq!(lights.len(), 1);
        assert_eq!(lights[0].position, vec3(1.0, 0.0, 0.0));
    }

    #[test]
    fn explosions_light_tinted_by_family() {
        let hierarchy = HashMap::from([
            (-2720, vec![EXPLOSIONS]),
            (-9000, vec![CRYO_EXPLOSIONS]),
            (CRYO_EXPLOSIONS, vec![EXPLOSIONS]),
        ]);
        let mut world = World::new();
        let at = |x| RuntimePropTransform(Matrix4::from_translation(vec3(x, 0.0, 0.0)));
        world.add_entity((PropTemplateId { template_id: -2720 }, at(1.0)));
        world.add_entity((PropTemplateId { template_id: -9000 }, at(2.0)));
        world.add_entity((PropTemplateId { template_id: -2653 }, at(3.0)));
        let mut lights = explosion_lights(&world, &hierarchy, &Default::default());
        lights.sort_by(|a, b| a.position.x.total_cmp(&b.position.x));
        assert_eq!(lights.len(), 2);
        assert!(
            lights[0].color_intensity.x > lights[0].color_intensity.z,
            "fire is warm"
        );
        assert!(
            lights[1].color_intensity.z > lights[1].color_intensity.x,
            "cryo is cold"
        );
    }
}
