//! The free-hand pose library: 25AE authored hands, snapped between.
//!
//! The remaster's first-person weapon models each carry a hand posed onto that
//! weapon. Those hands have no finger joints (see
//! `~/notes/projects/shock2quest/vr-hand-rigging.md`), so we cannot blend
//! between poses - but the poses themselves are good, and there are enough of
//! them to cover the states a free hand needs. We pick one per state and snap.
//!
//! Held weapons do not come through here: for those the whole `_h` model is
//! drawn, hand included, which is already the artist's grip.
//!
//! # Aligning a pose
//!
//! Each hand is anchored by its geometric frame ([`dark`'s `HandFrame`]): the
//! wrist goes to the hand origin and the fingers point along the hand's forward
//! axis. That fixes position and two of three rotational degrees of freedom
//! from the geometry itself, so the only thing left to author per pose is the
//! **roll** about the finger axis - one number, tuned visually in
//! `debug_hand_poses`.
//!
//! # Known limitation: the wrist is estimated, not authored
//!
//! `HandFrame` derives the wrist by stepping back a hand's length from the
//! far end of the point cloud. That endpoint is an extended fingertip on one
//! hand and a knuckle on a curled one, so the estimate lands a few centimetres
//! differently per pose - visibly, the axis marker sits mid-palm on `ar15_h`
//! rather than at the wrist. It also means the authored `forward` is the
//! *forearm* axis on the models that include a forearm, not the finger axis.
//!
//! Consequences: the poses do not present the same amount of sleeve, and a
//! common-cuff trim built on this landmark cut into the wrist on the worst
//! case. A per-pose wrist offset alongside `roll`, tuned in the harness, is
//! the intended fix.


