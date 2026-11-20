//! Permanent personal credential/wallet tool. Ownership is transient presentation;
//! balances and permissions remain in QuestInfo. Reuses the body inventory frame.
use super::body_inventory::BodyPose;
use crate::{input_context::InputContext, vr_support::GripPose};
use cgmath::{InnerSpace, Matrix4, Quaternion, Vector3, vec3};
use shipyard::{EntityId, Get, View, World};

const DOWNLOAD_SECONDS: f32 = 1.15;

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

pub(super) struct PersonalCard {
    pub center: Option<Vector3<f32>>,
    pub device_mode: bool,
    pub hand: Option<usize>,
    belt_yaw: f32,
    rest_pose: crate::vr_belt::BeltCardPose,
    mfd_rest_pose: crate::vr_belt::BeltCardPose,
    pose_loaded: bool,
    pub blocked: [bool; 2],
    pressed: [bool; 2],
    tracked: [bool; 2],
    reader: Option<EntityId>,
    withdrawn: f32,
    pub scans: u64,
    pub last_scan: Option<EntityId>,
    pub grip: Option<crate::vr_grip::ResolvedGrip>,
    downloads: Vec<(Vector3<f32>, f32)>,
    pub held_pose: Option<(Vector3<f32>, Quaternion<f32>)>,
}

impl Default for PersonalCard {
    fn default() -> Self {
        Self {
            center: None,
            device_mode: false,
            hand: None,
            belt_yaw: 0.0,
            rest_pose: Default::default(),
            mfd_rest_pose: crate::vr_belt::BeltCardPose::mfd_default(),
            pose_loaded: false,
            blocked: [false; 2],
            pressed: [true; 2],
            tracked: [false; 2],
            reader: None,
            withdrawn: 0.0,
            scans: 0,
            last_scan: None,
            held_pose: None,
            downloads: Vec::new(),
            grip: None,
        }
    }
}

impl PersonalCard {
    pub fn load_pose(&mut self, assets: &mut engine::assets::asset_cache::AssetCache) {
        if self.pose_loaded {
            return;
        }
        self.pose_loaded = true;
        for (name, target) in [
            ("vr-belt-card.json", &mut self.rest_pose),
            ("vr-belt-mfd.json", &mut self.mfd_rest_pose),
        ] {
            if let Some(text) = assets.get_opt(&engine::assets::text_importer::TEXT_IMPORTER, name)
            {
                match crate::vr_belt::BeltCardPose::parse(&text) {
                    Ok(pose) => *target = pose,
                    Err(error) => eprintln!("Invalid {name}; using default: {error}"),
                }
            }
        }
    }

    pub fn advance_downloads(&mut self, dt: f32) {
        for (_, age) in &mut self.downloads {
            *age += dt.max(0.0);
        }
        self.downloads.retain(|(_, age)| *age < DOWNLOAD_SECONDS);
    }

    pub fn download(&mut self, origin: Vector3<f32>) {
        self.downloads.push((origin, 0.0));
    }

    /// Returns true only for a deliberate, tracked release back to the belt.
    pub fn update(
        &mut self,
        input: &InputContext,
        body: Option<BodyPose>,
        available: [bool; 2],
        enabled: bool,
    ) -> bool {
        self.center = body.map(|body| {
            self.belt_yaw = body.yaw;
            let pose = if self.device_mode {
                &self.mfd_rest_pose
            } else {
                &self.rest_pose
            };
            body.front(
                crate::dev_params::get(crate::dev_params::VR_BELT_DROP),
                crate::dev_params::get(crate::dev_params::VR_BELT_DISTANCE) - 0.30,
            ) + (Matrix4::from_angle_y(cgmath::Rad(-body.yaw))
                * ((Vector3::from(pose.position_m)
                    + if self.device_mode {
                        vec3(
                            0.0,
                            crate::dev_params::get(crate::dev_params::VR_MFD_BELT_Y),
                            0.0,
                        )
                    } else {
                        vec3(0.0, 0.0, 0.0)
                    })
                    / crate::world_scale::meters_per_world_unit())
                .extend(0.0))
            .truncate()
        });
        self.blocked = [false; 2];
        if !enabled {
            self.reader = None;
            self.withdrawn = 0.0;
            self.hand = None;
            self.pressed = [true; 2];
            self.tracked = [false; 2];
            self.held_pose = None;
            return false;
        }
        let mut returned = false;
        for (i, hand) in [&input.left_hand, &input.right_hand]
            .into_iter()
            .enumerate()
        {
            let tracked = body.is_some()
                && input.pose_tracking.is_none_or(|p| p.head && p.hands[i])
                && GripPose {
                    position: hand.position,
                    rotation: hand.rotation,
                }
                .is_tracked()
                && hand.squeeze_value.is_finite();
            let pressed = hand.squeeze_value >= 0.5;
            if self.hand == Some(i) {
                self.blocked[i] = true;
                // Tracking loss never invents a release. Recovery requires a
                // new squeeze before an open hand can return the card.
                if tracked && self.tracked[i] && !pressed && self.pressed[i] {
                    returned = true;
                    self.hand = None;
                    self.held_pose = None;
                    self.reader = None;
                    self.withdrawn = 0.0;
                }
            } else if self.hand.is_none()
                && available[i]
                && tracked
                && self.tracked[i]
                && pressed
                && !self.pressed[i]
                && self.center.is_some_and(|center| {
                    (hand.position - center).magnitude()
                        <= 0.05 / crate::world_scale::meters_per_world_unit()
                            + super::body_inventory::hand_radius()
                        && body.is_none_or(|body| {
                            (hand.position - center).magnitude2()
                                < (hand.position - super::ammo_pouch::center_for(body)).magnitude2()
                        })
                })
            {
                self.hand = Some(i);
                self.blocked[i] = true;
            }
            self.pressed[i] = !tracked || pressed;
            self.tracked[i] = tracked;
        }
        returned
    }

    pub fn sample_hand(
        &mut self,
        input: &InputContext,
        pawn: Vector3<f32>,
        rotation: Quaternion<f32>,
    ) {
        if let Some(i) = self.hand.filter(|i| self.tracked[*i]) {
            let hand = [&input.left_hand, &input.right_hand][i];
            self.held_pose = Some((
                crate::virtual_hand::hand_world_position(pawn, rotation, hand.position),
                rotation * hand.rotation,
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (PersonalCard, InputContext, BodyPose) {
        let mut input = InputContext::default();
        input.head.rotation = Quaternion::new(1.0, 0.0, 0.0, 0.0);
        input.left_hand.rotation = input.head.rotation;
        input.right_hand.rotation = input.head.rotation;
        let body = BodyPose {
            head: vec3(0.0, 1.6, 0.0),
            yaw: 0.0,
        };
        (PersonalCard::default(), input, body)
    }

    #[test]
    fn one_permanent_card_draws_on_fresh_grip_and_returns_anywhere() {
        let (mut card, mut input, body) = fixture();
        card.update(&input, Some(body), [true; 2], true);
        input.left_hand.position = card.center.unwrap();
        input.right_hand.position = card.center.unwrap();
        card.update(&input, Some(body), [true; 2], true);
        input.left_hand.squeeze_value = 1.0;
        input.right_hand.squeeze_value = 1.0;
        card.update(&input, Some(body), [true; 2], true);
        assert_eq!(card.hand, Some(0));
        input.left_hand.position = vec3(5.0, 2.0, 5.0);
        input.left_hand.squeeze_value = 0.0;
        assert!(card.update(&input, Some(body), [true; 2], true));
        assert_eq!(card.hand, None);
        assert!(card.blocked[0], "return edge is consumed");
        assert!(!card.update(&input, Some(body), [true; 2], true));
        assert_eq!(
            card.hand, None,
            "other hand's held squeeze cannot steal returned card"
        );
    }
    #[test]
    fn disabling_or_losing_tracking_does_not_announce_a_belt_return() {
        let (mut card, mut input, body) = fixture();
        card.hand = Some(0);
        card.pressed[0] = true;
        card.tracked[0] = true;
        input.left_hand.squeeze_value = 0.0;
        assert!(!card.update(&input, None, [true; 2], true));
        assert_eq!(card.hand, Some(0));
        assert!(!card.update(&input, Some(body), [true; 2], false));
        assert_eq!(card.hand, None);
    }
    #[test]
    fn recovery_does_not_draw_from_a_stale_squeeze() {
        let (mut card, mut input, body) = fixture();
        card.update(&input, Some(body), [true; 2], true);
        input.left_hand.position = card.center.unwrap();
        input.left_hand.squeeze_value = 1.0;
        card.update(&input, None, [true; 2], true);
        card.update(&input, Some(body), [true; 2], true);
        assert_eq!(card.hand, None);
        input.left_hand.squeeze_value = 0.0;
        card.update(&input, Some(body), [true; 2], true);
        input.left_hand.squeeze_value = 1.0;
        card.update(&input, Some(body), [true; 2], true);
        assert_eq!(card.hand, Some(0));
    }
}
