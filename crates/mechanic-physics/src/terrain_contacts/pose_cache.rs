//! Cached separations and swept bounds that let a query skip regions with no contact.

#[cfg(test)]
use super::TerrainContactScene;
use super::geometry::{Collider, MachineCollisionGeometry};
use crate::{BodyPose, PhysicsError};
use bevy_math::DVec3;
use mechanic_core::{ContactCylinder, ContactPolytope, ConvexSeparation};
use mechanic_world::TerrainNodeId;
use std::collections::HashMap;
use std::sync::OnceLock;

// Cached construction geometry is relative to the simulation origin, independent
// of terrain publications. Exact body-pose changes invalidate only that body's
// shapes. Topology owns the cache and cannot be changed in place.
pub(super) type CachedSeparations = HashMap<[usize; 2], (f64, Option<ConvexSeparation>)>;

#[derive(Default)]
pub(super) struct PoseCache {
    pub(super) poses: Vec<BodyPose>,
    pub(super) bounds: Vec<[DVec3; 2]>,
    pub(super) shapes: Vec<OnceLock<ContactPolytope>>,
    pub(super) rounds: Vec<Option<ContactCylinder>>,
    pub(super) separations: std::cell::RefCell<CachedSeparations>,
    pub(super) spare_shapes: std::cell::RefCell<Vec<ContactPolytope>>,
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
        if self.poses != poses {
            self.separations.get_mut().clear();
        }
        for (body, &pose) in poses.iter().enumerate() {
            if self.poses.get(body) == Some(&pose) {
                continue;
            }
            for &row in &machine.body_colliders[body] {
                if let Some(shape) = self.shapes[row].take() {
                    self.spare_shapes.get_mut().push(shape);
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
            let mut shape = self
                .spare_shapes
                .borrow_mut()
                .pop()
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
