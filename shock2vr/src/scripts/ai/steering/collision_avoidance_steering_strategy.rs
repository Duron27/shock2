use cgmath::{Deg, InnerSpace, Quaternion, Rotation, Rotation3, vec3, vec4};
use dark::SCALE_FACTOR;

use shipyard::{EntityId, World};

use crate::{
    physics::{InternalCollisionGroups, PhysicsWorld},
    scripts::{
        Effect,
        ai::ai_util::{self},
    },
    time::Time,
    util::{get_position_from_transform, get_rotation_from_transform},
};

use super::{Steering, SteeringOutput, SteeringStrategy};

pub struct CollisionAvoidanceSteeringStrategy {
    extra_whisker_distance: f32,
    main_whisker_distance: f32,
    last_mitigation_direction: Deg<f32>,
}

impl CollisionAvoidanceSteeringStrategy {
    pub fn conservative() -> CollisionAvoidanceSteeringStrategy {
        CollisionAvoidanceSteeringStrategy {
            last_mitigation_direction: Deg(30.0),
            extra_whisker_distance: 3.0 / SCALE_FACTOR,
            main_whisker_distance: 3.0 / SCALE_FACTOR,
        }
    }

    pub fn comprehensive() -> CollisionAvoidanceSteeringStrategy {
        CollisionAvoidanceSteeringStrategy {
            last_mitigation_direction: Deg(30.0),
            extra_whisker_distance: 8.0 / SCALE_FACTOR,
            main_whisker_distance: 6.0 / SCALE_FACTOR,
        }
    }
}

impl SteeringStrategy for CollisionAvoidanceSteeringStrategy {
    fn steer(
        &mut self,
        current_heading: Deg<f32>,
        world: &World,
        physics: &PhysicsWorld,
        entity_id: EntityId,
        _time: &Time,
    ) -> Option<(SteeringOutput, Effect)> {
        let whisker_angle = Deg(30.0);
        let target_offset = 8.0 / SCALE_FACTOR;

        // Cast from the body center, NOT below it: a knee-height whisker
        // looking down a descending staircase hits the lower steps' risers
        // (vertical faces, so the mostly-vertical-normal filter doesn't
        // reject them) and treats the descent the route chose as a wall.
        // Real walls extend above body center and are still seen.
        let position = get_position_from_transform(
            world,
            entity_id,
            crate::creature::sense_offset(world, entity_id),
        );

        let mut debug_lines = vec![];

        let left_whisker_rotation = Quaternion::from_angle_y(-whisker_angle);
        let right_whisker_rotation = Quaternion::from_angle_y(whisker_angle);
        let rotation = get_rotation_from_transform(world, entity_id);

        let extra_whisker_distance = self.extra_whisker_distance;
        let main_whisker_distance = self.main_whisker_distance;

        let whiskers_to_check = vec![
            (
                rotation * left_whisker_rotation,
                extra_whisker_distance,
                Some(Deg(30.0)),
            ),
            (
                rotation * right_whisker_rotation,
                extra_whisker_distance,
                Some(Deg(-30.0)),
            ),
            // The main (straight-ahead) whisker deflects along the wall via
            // the hit normal (below) instead of carrying a fixed mitigation:
            // a fixed 180 reversal fights the path steering command straight
            // back at the obstacle, flip-flopping the heading forever while
            // the AI runs in place (issue #481's freeze).
            (rotation, main_whisker_distance, None),
            //(rotation, main_whisker_distance, None),
        ];

        let mut maybe_steering_output = None;
        let _dist = f32::MAX;
        for (rotation, distance, mitigation) in whiskers_to_check {
            let forward = rotation.rotate_vector(vec3(0.0, 0.0, 1.0)).normalize();
            let end_point = position + forward * distance;

            let main_whisker_hit_result = physics.ray_cast2_as_actor(
                position,
                forward,
                distance,
                InternalCollisionGroups::ALL_COLLIDABLE,
                Some(entity_id),
                true,
            );

            if let Some(main_whisker_hit_result) = main_whisker_hit_result {
                debug_lines.push((
                    position,
                    main_whisker_hit_result.hit_point,
                    vec4(0.0, 0.0, 1.0, 1.0),
                ));
                let target = main_whisker_hit_result.hit_point
                    + main_whisker_hit_result.hit_normal * target_offset;

                debug_lines.push((
                    main_whisker_hit_result.hit_point,
                    target,
                    vec4(1.0, 1.0, 0.0, 1.0),
                ));

                // If the normal is valid, we can use it to calculate a new heading
                // Ignore the normal if it's pointing up or down

                let is_normal_mostly_vertical = main_whisker_hit_result.hit_normal.y.abs() > 0.5;
                let entity_is_door = main_whisker_hit_result
                    .maybe_entity_id
                    .map(|m| ai_util::is_entity_door(world, m))
                    .unwrap_or(false);
                // The creature's own target is where it is going, not
                // something to steer around: a whisker that lands on the
                // player a stride ahead aims along the hit normal, which is
                // straight back past the creature, and the 180 degree error
                // stops the body and plays a turn-around clip in the
                // player's face (the medsci1 blast-door hybrid stood ~2 s).
                let entity_is_target = main_whisker_hit_result
                    .maybe_entity_id
                    .is_some_and(|hit| is_combat_target(world, physics, entity_id, hit));

                // let distance_to_hit = (main_whisker_hit_result.hit_point - position).magnitude();
                // if distance_to_hit < dist {
                // dist = distance_to_hit;

                if !is_normal_mostly_vertical && !entity_is_door && !entity_is_target {
                    match mitigation {
                        // Main whisker: steer toward the point deflected off
                        // the obstacle along its surface normal - glancing
                        // away instead of reversing into a heading war with
                        // the path steering
                        None => {
                            maybe_steering_output = Some(Steering::turn_to_point(position, target));
                        }
                        Some(mitigation) => {
                            self.last_mitigation_direction = mitigation;
                            maybe_steering_output =
                                Some(Steering::from_current(current_heading + mitigation));
                        }
                    }
                }
                // }
                // }
            } else {
                debug_lines.push((position, end_point, vec4(0.0, 1.0, 0.0, 1.0)));
            };
        }

        if maybe_steering_output.is_some() {
            Some((
                maybe_steering_output.unwrap(),
                Effect::DrawDebugLines { lines: debug_lines },
            ))
        } else {
            None
        }
    }
}

/// Whether `hit` is the entity `observer` is fighting - the player for a
/// monster, the nearest visible hostile for a friendly AI.
fn is_combat_target(
    world: &World,
    physics: &PhysicsWorld,
    observer: EntityId,
    hit: EntityId,
) -> bool {
    ai_util::combat_target(world, physics, observer).is_some_and(|(target, _)| target == hit)
}
