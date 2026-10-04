//! Cached separations and swept bounds that let a query skip regions with no contact.

#[cfg(test)]
use super::TerrainContactScene;
use super::geometry::{Collider, MachineCollisionGeometry};
use crate::{BodyPose, PhysicsError};
use bevy_math::{DQuat, DVec3};
use mechanic_core::{
    ContactCylinder, ContactPolytope, ConvexSeparation, SeparatingFace, SeparationOutcome,
};
use mechanic_world::TerrainNodeId;
use rustc_hash::FxHashMap;
use std::sync::OnceLock;

// Cached construction geometry is relative to the simulation origin, independent
// of terrain publications. Exact body-pose changes invalidate only that body's
// shapes. Topology owns the cache and cannot be changed in place.

/// Collider-pair separations at the current poses, and for each pair the face
/// that last proved it separated beyond the query's reach.
///
/// Nearly every candidate pair of a construction is separated, and from one
/// substep to the next the same face keeps proving it. A witness remembers
/// that face, the gap it measured and how far the two bodies had drifted
/// apart by then. While the drift since stays well inside the gap's excess
/// over the reach, the face provably still separates and the pair is skipped;
/// otherwise the face is tested first. Either way the answer is exactly the
/// one the full separating-axis test gives, so witnesses outlive pose
/// changes; separations do not.
#[derive(Default)]
pub(super) struct PairCache {
    // Bumped whenever any pose changes. A record from an older epoch keeps
    // only its witness.
    epoch: u64,
    records: Records,
    // Separations within reach at the current poses; few pairs have one.
    within: FxHashMap<[usize; 2], ConvexSeparation>,
    drifts: Drifts,
}

// Pair records by the pair's first collider, sorted by the second. Queries
// visit pairs in ascending order, so a cursor finds most records without
// searching.
#[derive(Default)]
struct Records {
    lists: Vec<Vec<PairRecord>>,
    cursor: [usize; 2],
}

#[derive(Default)]
struct Drifts {
    epoch: u64,
    // Never dropped: a witness measures from the path length it saw, and there
    // are at most as many as body pairs.
    paths: FxHashMap<[usize; 2], Drift>,
    // Drifts already brought up to this epoch, direct mapped by body pair.
    recent: Vec<Option<([usize; 2], [f64; 2])>>,
}

const RECENT_DRIFTS: usize = 256;

#[derive(Clone, Copy)]
struct PairRecord {
    second: usize,
    // The epoch `reach` and `within` describe, and the last one the pair was
    // looked up in.
    epoch: u64,
    // The largest reach this pair was found separated beyond, if any.
    reach: f64,
    // Whether `PairCache::within` holds this pair's separation.
    within: bool,
    witness: Option<Witness>,
}

#[derive(Clone, Copy)]
struct Witness {
    face: SeparatingFace,
    // The face's gap, and the drift of the opposing body in the face body's
    // frame, when the gap was measured.
    gap: f64,
    drift: [f64; 2],
}

// One body's pose in another's frame, and the path length both have covered
// since the pair was first seen: translation in metres, and twice the chord
// between unit quaternions, which bounds how far a point at unit distance
// from the body origin can turn.
#[derive(Clone, Copy)]
struct Drift {
    epoch: u64,
    rotation: DQuat,
    translation: DVec3,
    travelled: [f64; 2],
}

// Pairs not looked up for this many pose changes are dropped, checked as often.
const PAIR_RECORD_EPOCHS: u64 = 128;

impl PairCache {
    fn advance(&mut self) {
        self.epoch += 1;
        self.within.clear();
        self.drifts.epoch = self.epoch;
        self.drifts.recent.fill(None);
        if self.epoch.is_multiple_of(PAIR_RECORD_EPOCHS) {
            let oldest = self.epoch - PAIR_RECORD_EPOCHS;
            for records in &mut self.records.lists {
                records.retain(|record| record.epoch >= oldest);
            }
        }
    }
}

impl Records {
    // The pair's record, its reach and separation reset in a new epoch.
    fn get(&mut self, [first, second]: [usize; 2], epoch: u64) -> &mut PairRecord {
        if self.lists.len() <= first {
            self.lists.resize_with(first + 1, Vec::new);
        }
        let list = &mut self.lists[first];
        let [last_first, after] = self.cursor;
        let mut index = if last_first == first
            && after <= list.len()
            && (after == 0 || list[after - 1].second < second)
        {
            after
        } else {
            list.partition_point(|record| record.second < second)
        };
        while list.get(index).is_some_and(|record| record.second < second) {
            index += 1;
        }
        if list.get(index).is_none_or(|record| record.second != second) {
            list.insert(
                index,
                PairRecord {
                    second,
                    epoch,
                    reach: f64::NEG_INFINITY,
                    within: false,
                    witness: None,
                },
            );
        }
        self.cursor = [first, index + 1];
        let record = &mut list[index];
        if record.epoch != epoch {
            record.epoch = epoch;
            record.reach = f64::NEG_INFINITY;
            record.within = false;
        }
        record
    }
}

impl Drifts {
    // Path length covered by `moving`'s frame as seen from `face`'s, up to now.
    fn drift(&mut self, [face, moving]: [usize; 2], poses: &[BodyPose]) -> [f64; 2] {
        let key = [face, moving];
        let slot = (face.wrapping_mul(31) ^ moving) % RECENT_DRIFTS;
        if self.recent.is_empty() {
            self.recent.resize(RECENT_DRIFTS, None);
        }
        if let Some((_, travelled)) = self.recent[slot].filter(|(k, _)| *k == key) {
            return travelled;
        }
        let inverse = poses[face].rotation.normalize().conjugate();
        let rotation = inverse * poses[moving].rotation.normalize();
        let translation = inverse * (poses[moving].position - poses[face].position);
        let epoch = self.epoch;
        let drift = self.paths.entry(key).or_insert(Drift {
            epoch,
            rotation,
            translation,
            travelled: [0.0; 2],
        });
        if drift.epoch != epoch {
            let turn = (rotation - drift.rotation)
                .length()
                .min((rotation + drift.rotation).length());
            drift.travelled[0] += (translation - drift.translation).length();
            drift.travelled[1] += 2.0 * turn;
            drift.rotation = rotation;
            drift.translation = translation;
            drift.epoch = epoch;
        }
        let travelled = drift.travelled;
        self.recent[slot] = Some((key, travelled));
        travelled
    }
}

impl PairCache {
    /// The pair's separation when it lies within `reach`, as
    /// [`ContactPolytope::convex_separation_within`] reports it at these poses
    /// for the colliders `shapes` transforms. `bodies` and `radii` are each
    /// collider's body and conservative radius about that body's origin.
    ///
    /// # Errors
    /// Whatever `shapes` returns, and invalid geometry.
    pub(super) fn separation<'a>(
        &mut self,
        pair: [usize; 2],
        [first_body, second_body]: [usize; 2],
        radii: [f64; 2],
        poses: &[BodyPose],
        reach: f64,
        shapes: impl Fn() -> Result<[&'a ContactPolytope; 2], PhysicsError>,
    ) -> Result<Option<ConvexSeparation>, PhysicsError> {
        let Self {
            epoch,
            records,
            within,
            drifts,
        } = self;
        let record = records.get(pair, *epoch);
        if record.within {
            return Ok(within.get(&pair).copied());
        }
        if record.reach >= reach {
            return Ok(None);
        }
        // The face's body, the opposing body, and the opposing collider's radius.
        let frame = |face: SeparatingFace| match face {
            SeparatingFace::Own(_) => ([first_body, second_body], radii[1]),
            SeparatingFace::Other(_) => ([second_body, first_body], radii[0]),
        };
        let mut witness = record.witness;
        let mut beyond = false;
        if let Some(found) = witness {
            let (bodies, radius) = frame(found.face);
            let drift = drifts.drift(bodies, poses);
            // Every opposing vertex has moved at most this far relative to the
            // face, so the face's gap has shrunk by at most this much.
            let moved = (drift[0] - found.drift[0]) + (drift[1] - found.drift[1]) * radius;
            // Far more than the rounding of either measured gap.
            let rounding = 1e-9
                + 1e-10
                    * (poses[first_body].position.length()
                        + poses[second_body].position.length()
                        + radii[0]
                        + radii[1]);
            // Transforming a shape would refuse a rotation this far from unit.
            let transformable = [first_body, second_body]
                .iter()
                .all(|&body| (poses[body].rotation.length_squared() - 1.0).abs() < 1e-6);
            if transformable && found.gap - moved - rounding > reach {
                beyond = true;
            } else {
                let [first, second] = shapes()?;
                let gap = first.face_gap(second, found.face);
                if gap > reach {
                    witness = Some(Witness {
                        gap,
                        drift,
                        ..found
                    });
                    beyond = true;
                }
            }
        }
        let separation = if beyond {
            None
        } else {
            let [first, second] = shapes()?;
            match first
                .convex_separation_witnessed(second, reach)
                .map_err(|_| PhysicsError::InvalidCollision)?
            {
                SeparationOutcome::Within(separation) => Some(separation),
                SeparationOutcome::Beyond(face) => {
                    let (bodies, _) = frame(face);
                    witness = Some(Witness {
                        face,
                        gap: first.face_gap(second, face),
                        drift: drifts.drift(bodies, poses),
                    });
                    None
                }
            }
        };
        record.witness = witness;
        match separation {
            Some(separation) => {
                within.insert(pair, separation);
                record.within = true;
            }
            None => record.reach = reach,
        }
        Ok(separation)
    }
}

#[derive(Default)]
pub(super) struct PoseCache {
    pub(super) poses: Vec<BodyPose>,
    pub(super) bounds: Vec<[DVec3; 2]>,
    pub(super) shapes: Vec<OnceLock<ContactPolytope>>,
    pub(super) rounds: Vec<Option<ContactCylinder>>,
    pub(super) pairs: std::cell::RefCell<PairCache>,
    // Each collider's last transformed shape, kept after a pose change so its
    // buffers, already the right size, take the next transform.
    pub(super) stale_shapes: std::cell::RefCell<Vec<Option<ContactPolytope>>>,
    pub(super) clipping: std::cell::RefCell<mechanic_core::TriangleClipScratch>,
    pub(super) terrain_candidates: Vec<Vec<TerrainNodeId>>,
    pub(super) terrain_bounds: Vec<[DVec3; 2]>,
}

impl PoseCache {
    pub(super) fn update(
        &mut self,
        machine: &MachineCollisionGeometry,
        poses: &[BodyPose],
    ) -> Result<(), PhysicsError> {
        if poses.iter().any(|p| {
            !p.position.is_finite()
                || !p.rotation.is_finite()
                || (p.rotation.length_squared() - 1.0).abs() > 1e-6
        }) {
            return Err(PhysicsError::InvalidCollision);
        }
        self.bounds
            .resize(machine.colliders.len(), [DVec3::ZERO; 2]);
        self.shapes
            .resize_with(machine.colliders.len(), OnceLock::new);
        self.rounds.resize(machine.colliders.len(), None);
        self.stale_shapes
            .get_mut()
            .resize_with(machine.colliders.len(), || None);
        if self.poses != poses {
            self.pairs.get_mut().advance();
        }
        for (body, &pose) in poses.iter().enumerate() {
            if self.poses.get(body) == Some(&pose) {
                continue;
            }
            for &row in &machine.body_colliders[body] {
                if let Some(shape) = self.shapes[row].take() {
                    self.stale_shapes.get_mut()[row] = Some(shape);
                }
                self.bounds[row] = machine.colliders[row]
                    .local
                    .transformed_bounds(pose.position, pose.rotation)
                    .map_err(|_| PhysicsError::InvalidCollision)?;
                self.rounds[row] = machine.colliders[row]
                    .round
                    .map(|round| round.transformed(pose.position, pose.rotation))
                    .transpose()
                    .map_err(|_| PhysicsError::InvalidCollision)?;
            }
        }
        self.poses.clear();
        self.poses.extend_from_slice(poses);
        Ok(())
    }
    pub(super) fn shape(
        &self,
        machine: &MachineCollisionGeometry,
        poses: &[BodyPose],
        row: usize,
    ) -> Result<&ContactPolytope, PhysicsError> {
        let slot = &self.shapes[row];
        if slot.get().is_none() {
            let pose = poses[machine.colliders[row].body];
            let local = &machine.colliders[row].local;
            let mut shape = self.stale_shapes.borrow_mut()[row]
                .take()
                .unwrap_or_else(|| local.clone());
            local
                .transformed_into(pose.position, pose.rotation, &mut shape)
                .map_err(|_| PhysicsError::InvalidCollision)?;
            let _ = slot.set(shape);
        }
        slot.get().ok_or(PhysicsError::InvalidCollision)
    }
}

pub(super) fn overlaps(a: [DVec3; 2], b: [DVec3; 2]) -> bool {
    !a[0].cmpgt(b[1]).any() && !b[0].cmpgt(a[1]).any()
}

pub(super) struct CandidatePairs<'a> {
    pub(super) scratch: std::sync::MutexGuard<'a, PairScratch>,
}

impl std::ops::Deref for CandidatePairs<'_> {
    type Target = [[usize; 2]];
    fn deref(&self) -> &Self::Target {
        &self.scratch.pairs
    }
}

#[derive(Default)]
pub(super) struct PairScratch {
    pub(super) body_bounds: Vec<[DVec3; 2]>,
    pub(super) body_nodes: Vec<[DVec3; 2]>,
    pub(super) body_candidates: Vec<usize>,
    pub(super) trees: Vec<Vec<[DVec3; 2]>>,
    pub(super) refitted: Vec<bool>,
    pub(super) stack: Vec<usize>,
    pub(super) candidates: Vec<usize>,
    pub(super) node_pair_tests: usize,
    pub(super) pairs: Vec<[usize; 2]>,
    pub(super) unsorted: Vec<[usize; 2]>,
    pub(super) pair_stack: Vec<[usize; 2]>,
}

// Every vertex moves at most its integrated point-speed bound. Intersect that
// initial-shape envelope with the origin/radius envelope: both contain the
// entire unwrapped trajectory, including full rotations.
pub(super) fn swept_bounds(
    collider: &Collider,
    pose: BodyPose,
    motion: crate::MotionBound,
    tolerance: f64,
    [minimum, maximum]: [DVec3; 2],
) -> [DVec3; 2] {
    let travel = DVec3::splat((motion.point_speed(collider.radius) + tolerance).next_up());
    let reach = DVec3::splat((motion.origin_speed + collider.radius + tolerance).next_up());
    [
        (minimum - travel)
            .max(pose.position - reach)
            .map(f64::next_down),
        (maximum + travel)
            .min(pose.position + reach)
            .map(f64::next_up),
    ]
}

// Tighten the general path envelope with a directional translation, an
// acceleration-bounded endpoint chord, and (for long fixed-axis paths) a full
// circle. Each independently encloses the whole motion, including many turns.
pub(super) fn path_bounds(
    collider: &Collider,
    path: &crate::MachineMotion<'_>,
    tolerance: f64,
) -> Result<[DVec3; 2], PhysicsError> {
    path_bounds_from(collider, path, tolerance, None)
}

// The optional bound must belong to the exact initial pose and this collider.
pub(super) fn path_bounds_from(
    collider: &Collider,
    path: &crate::MachineMotion<'_>,
    tolerance: f64,
    initial_bounds: Option<[DVec3; 2]>,
) -> Result<[DVec3; 2], PhysicsError> {
    let pose = path.initial_poses()[collider.body];
    let bound = path.bounds()[collider.body];
    let start = match initial_bounds {
        Some(bounds) => bounds,
        None => collider
            .local
            .transformed_bounds(pose.position, pose.rotation)
            .map_err(|_| PhysicsError::InvalidCollision)?,
    };
    let fallback = swept_bounds(collider, pose, bound, tolerance, start);
    // If travel is already below the query padding, the general speed bound
    // is tight enough. Retain that conservative envelope instead of calculating
    // endpoint rotations for every slowly settling piece elsewhere in a scene.
    if bound.point_speed(collider.radius) <= tolerance {
        return Ok(fallback);
    }
    // Every material point follows its endpoint chord to within A / 8 over a
    // unit interval when its acceleration magnitude is bounded by A. This
    // encloses the whole trajectory, including articulated ancestors, rather
    // than assuming matching endpoint orientations imply no intervening turn.
    // Unlike a speed-radius expansion, the error shrinks quadratically with
    // substep duration. That matters for finely decomposed pipes near bodies.
    let acceleration = bound.point_acceleration(collider.radius);
    let chord = if bound.angular_speed != 0.0 && acceleration.is_finite() {
        let end_pose = path.final_poses()[collider.body];
        let end = collider
            .local
            .transformed_bounds(end_pose.position, end_pose.rotation)
            .map_err(|_| PhysicsError::InvalidCollision)?;
        let deviation = DVec3::splat((acceleration * 0.125).next_up());
        Some([
            start[0].min(end[0]) - deviation,
            start[1].max(end[1]) + deviation,
        ])
    } else {
        None
    };
    let mut result = [DVec3::INFINITY, DVec3::NEG_INFINITY];
    if bound.angular_speed == 0.0 {
        let delta = path.final_poses()[collider.body].position - pose.position;
        result = [
            start[0] + delta.min(DVec3::ZERO),
            start[1] + delta.max(DVec3::ZERO),
        ];
    } else if let Some(chord) = chord.filter(|_| bound.angular_speed <= 1.0) {
        // For short arcs the chord certificate is already tight; avoid building
        // an additional full-circle envelope for every small pipe sector.
        result = chord;
    } else if let Some((pivot, axis, delta)) = path.fixed_rotation(collider.body) {
        let [lo, hi] = collider.bounds;
        for x in [lo.x, hi.x] {
            for y in [lo.y, hi.y] {
                for z in [lo.z, hi.z] {
                    let offset = pose.position + pose.rotation * DVec3::new(x, y, z) - pivot;
                    let parallel = axis * axis.dot(offset);
                    let radial = offset - parallel;
                    let tangent = axis.cross(radial);
                    let extent = (radial * radial + tangent * tangent).map(f64::sqrt);
                    let center = pivot + parallel;
                    result[0] = result[0].min(center - extent + delta.min(DVec3::ZERO));
                    result[1] = result[1].max(center + extent + delta.max(DVec3::ZERO));
                }
            }
        }
    } else if let Some(chord) = chord {
        result = chord;
    } else {
        return Ok(fallback);
    }
    if let Some(chord) = chord {
        result = [result[0].max(chord[0]), result[1].min(chord[1])];
    }
    // Cover roundoff in transforms, projection, and cancellation around a pivot.
    let scale = pose
        .position
        .abs()
        .max(result[0].abs())
        .max(result[1].abs())
        .max_element()
        + collider.radius
        + bound.origin_speed
        + 1.0;
    let padding = DVec3::splat(tolerance + 128.0 * f64::EPSILON * scale);
    Ok([
        (result[0] - padding).map(f64::next_down).max(fallback[0]),
        (result[1] + padding).map(f64::next_up).min(fallback[1]),
    ])
}

// A broadphase-only proof of empty space. Scoped to a single tick by the caller;
// generations and origin are still checked so a changed scene cannot reuse it.
#[cfg(test)]
pub(crate) struct EmptyContactRegion {
    pub(super) bounds: Vec<[DVec3; 2]>,
    pub(super) terrain_generation: u64,
    pub(super) topology_generation: u64,
    pub(super) origin: DVec3,
}

#[cfg(test)]
impl EmptyContactRegion {
    pub(crate) fn contains(
        &self,
        scene: &TerrainContactScene,
        geometry: &MachineCollisionGeometry,
        poses: &[BodyPose],
        origin: DVec3,
        margins: &[f64],
    ) -> bool {
        if self.terrain_generation != scene.generation
            || self.topology_generation != geometry.generation
            || self.origin != origin
            || self.bounds.len() != geometry.colliders.len()
            || margins.len() != self.bounds.len()
            || poses.len() != geometry.bodies
        {
            return false;
        }
        geometry
            .colliders
            .iter()
            .zip(&self.bounds)
            .zip(margins)
            .all(|((collider, region), &margin)| {
                let pose = poses[collider.body];
                collider
                    .local
                    .transformed_bounds(pose.position, pose.rotation)
                    .is_ok_and(|bounds| {
                        let padding = DVec3::splat(margin);
                        (bounds[0] - padding)
                            .map(f64::next_down)
                            .cmpge(region[0])
                            .all()
                            && (bounds[1] + padding)
                                .map(f64::next_up)
                                .cmple(region[1])
                                .all()
                    })
            })
    }
}
