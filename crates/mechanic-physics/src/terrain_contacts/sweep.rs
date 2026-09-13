//! Conservative advancement along a reconstructed articulated trajectory.

use super::{
    CONTACT_ACTIVATION_DISTANCE, ContactTarget, MachineCollisionGeometry, PAIR_ACTIVATION_DISTANCE,
    TerrainContactScene, valid_bounds,
};
use crate::{BodyPose, MachineMotion, PhysicsError};
use bevy_math::{DQuat, DVec3, Vec3};
use mechanic_core::ContactPolytope;
use mechanic_world::{WorldBounds, WorldPosition};

/// First terrain triangle or collider approached by a certified candidate path.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TerrainSweepHit {
    /// Collider row within the query's topology generation.
    pub collider: usize,
    /// Approached terrain triangle or collider on another body.
    pub target: ContactTarget,
    /// Candidate motion fraction, in [0, 1].
    pub fraction: f64,
    /// Signed SAT gap at this fraction, in metres.
    pub separation: f64,
}

/// Continuous query result. Unconverged work is never interpreted as clear space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TerrainSweepOutcome {
    /// Every candidate was certified separated for the entire path.
    Clear,
    /// Earliest approach within the requested geometric tolerance.
    Impact(TerrainSweepHit),
    /// At least one candidate exhausted its work/progress bound. Its path
    /// remainder is uncertified; retry or retain the last valid snapshot.
    Unconverged(TerrainSweepHit),
}

/// Tagged continuous query and actual work, including failed advancement.
#[derive(Clone, Debug)]
pub struct TerrainSweepQuery {
    /// Continuous result, which must be checked before accepting the path.
    pub outcome: TerrainSweepOutcome,
    /// Construction generation used by the query.
    pub topology_generation: u64,
    /// Complete terrain publication used by the query.
    pub terrain_generation: u64,
    /// Chunk candidates across the swept collider bounds.
    pub chunk_candidates: usize,
    /// Exact finite triangles tested.
    pub triangle_candidates: usize,
    /// Whole-tree pose reconstructions; none assemble or factor inertia.
    pub pose_evaluations: usize,
    /// Separating-axis evaluations, including failed advancement work.
    pub separation_evaluations: usize,
    /// Finite initial contact queries identifying existing support candidates.
    pub initial_contact_evaluations: usize,
    /// Initial touching pairs excluded only from the new-impact search.
    pub supported_pairs: usize,
    /// Slow colliders deferred to endpoint contact; not a penetration certificate.
    pub slow_contact_deferrals: usize,
    /// Exact finite translation-interval queries; they need no advancement loop.
    pub linear_interval_evaluations: usize,
    /// Fixed-axis, per-vertex quadratic clearance certificates.
    pub quadratic_interval_evaluations: usize,
    /// Whole-tree spatial derivative traversals used by quadratic certificates.
    pub velocity_evaluations: usize,
    /// Maximum point displacement over all moving colliders for the full path.
    /// This is computed before traversal, including on an inconclusive query.
    pub maximum_point_displacement: f64,
    /// Collider pairs on different bodies whose swept bounds overlap.
    pub collider_pair_candidates: usize,
}

#[derive(Clone, Copy)]
pub(crate) struct SlowContactMotion<'a> {
    pub initial: &'a [crate::MotionBound],
    pub final_bounds: &'a [crate::MotionBound],
    pub maximum_displacement: f64,
}

impl TerrainContactScene {
    /// Sweeps every moving compiled collider along the exact generalized path.
    /// Bounds include ancestor rotation, root motion, and suspension extension.
    /// Existing touching geometry returns an impact at fraction zero; the tick
    /// solver must handle its support constraints before continuing the path.
    ///
    /// # Errors
    /// Rejects incompatible topology, invalid bounds/settings, or invalid geometry.
    pub fn sweep(
        &self,
        machine: &MachineCollisionGeometry,
        motion: &MachineMotion<'_>,
        origin: DVec3,
        tolerance: f64,
        maximum_evaluations_per_triangle: usize,
    ) -> Result<TerrainSweepQuery, PhysicsError> {
        self.sweep_internal(
            machine,
            motion,
            origin,
            tolerance,
            maximum_evaluations_per_triangle,
            false,
            None,
        )
    }

    // Gathers new-impact candidates while excluding initial numerical-zero
    // finite contacts. Clear here is NOT whole-path clearance: every accepted
    // trajectory must independently pass supported penetration validation.
    pub(crate) fn sweep_new_contacts(
        &self,
        machine: &MachineCollisionGeometry,
        motion: &MachineMotion<'_>,
        origin: DVec3,
        tolerance: f64,
        maximum_evaluations_per_triangle: usize,
    ) -> Result<TerrainSweepQuery, PhysicsError> {
        self.sweep_internal(
            machine,
            motion,
            origin,
            tolerance,
            maximum_evaluations_per_triangle,
            true,
            None,
        )
    }

    pub(crate) fn sweep_substep_contacts(
        &self,
        machine: &MachineCollisionGeometry,
        motion: &MachineMotion<'_>,
        origin: DVec3,
        tolerance: f64,
        maximum_evaluations_per_triangle: usize,
        slow: Option<SlowContactMotion<'_>>,
    ) -> Result<TerrainSweepQuery, PhysicsError> {
        if slow.is_none() {
            return self.sweep_new_contacts(
                machine,
                motion,
                origin,
                tolerance,
                maximum_evaluations_per_triangle,
            );
        }
        self.sweep_internal(
            machine,
            motion,
            origin,
            tolerance,
            maximum_evaluations_per_triangle,
            true,
            slow,
        )
    }

    #[allow(clippy::too_many_arguments, clippy::too_many_lines, clippy::float_cmp)] // Ordered bounded traversal; excluded supports require separate path validation.
    fn sweep_internal(
        &self,
        machine: &MachineCollisionGeometry,
        motion: &MachineMotion<'_>,
        origin: DVec3,
        tolerance: f64,
        maximum_evaluations_per_triangle: usize,
        exclude_initial_supports: bool,
        slow: Option<SlowContactMotion<'_>>,
    ) -> Result<TerrainSweepQuery, PhysicsError> {
        if machine.generation != motion.generation()
            || machine.bodies != motion.initial_poses().len()
            || !origin.is_finite()
            || !tolerance.is_finite()
            || tolerance <= 0.0
            || maximum_evaluations_per_triangle == 0
            || slow.as_ref().is_some_and(|slow| {
                slow.initial.len() != machine.bodies
                    || slow.final_bounds.len() != machine.bodies
                    || !slow.maximum_displacement.is_finite()
                    || slow.maximum_displacement < 0.0
            })
        {
            return Err(PhysicsError::InvalidCollision);
        }
        let mut query = TerrainSweepQuery {
            outcome: TerrainSweepOutcome::Clear,
            topology_generation: machine.generation,
            terrain_generation: self.generation,
            chunk_candidates: 0,
            triangle_candidates: 0,
            pose_evaluations: 0,
            separation_evaluations: 0,
            initial_contact_evaluations: 0,
            supported_pairs: 0,
            slow_contact_deferrals: 0,
            linear_interval_evaluations: 0,
            quadratic_interval_evaluations: 0,
            velocity_evaluations: 0,
            maximum_point_displacement: 0.0,
            collider_pair_candidates: 0,
        };
        for collider in &machine.colliders {
            if collider.moving {
                let displacement = motion.bounds()[collider.body].point_speed(collider.radius);
                if !displacement.is_finite() {
                    return Err(PhysicsError::InvalidCollision);
                }
                query.maximum_point_displacement =
                    query.maximum_point_displacement.max(displacement);
            }
        }
        let mut earliest: Option<TerrainSweepHit> = None;
        for (collider_row, collider) in machine.colliders.iter().enumerate() {
            if !collider.moving {
                continue;
            }
            if let Some(slow) = &slow
                && [motion.bounds(), slow.initial, slow.final_bounds]
                    .iter()
                    .all(|bounds| {
                        bounds[collider.body].point_speed(collider.radius)
                            <= slow.maximum_displacement
                    })
            {
                query.slow_contact_deferrals += 1;
                continue;
            }
            let bound = motion.bounds()[collider.body];
            // Zero accumulated angular speed proves all ancestors are fixed in
            // rotation. Matching endpoint quaternions would miss complete turns.
            let translation = (bound.angular_speed == 0.0).then(|| {
                motion.final_poses()[collider.body].position
                    - motion.initial_poses()[collider.body].position
            });
            let reach = bound.origin_speed + collider.radius + tolerance;
            let center = origin + motion.initial_poses()[collider.body].position;
            let bounds = WorldBounds {
                minimum: WorldPosition(center - DVec3::splat(reach)),
                maximum: WorldPosition(center + DVec3::splat(reach)),
            };
            if !valid_bounds(bounds) {
                return Err(PhysicsError::InvalidCollision);
            }
            let speed = bound.point_speed(collider.radius);
            if !speed.is_finite() {
                return Err(PhysicsError::InvalidCollision);
            }
            let chunks = self.index.bounds_candidates(bounds);
            query.chunk_candidates += chunks.len();
            for node in chunks {
                let chunk = &self.chunks[&node].geometry;
                for triangle_row in chunk.bounds_candidates(bounds) {
                    let triangle =
                        chunk.triangle_bvh.triangles[triangle_row]
                            .indices
                            .map(|index| {
                                chunk.origin.0 - origin
                                    + Vec3::from_array(chunk.vertices[index as usize]).as_dvec3()
                            });
                    if (triangle[1] - triangle[0])
                        .cross(triangle[2] - triangle[0])
                        .try_normalize()
                        .is_none()
                    {
                        continue;
                    }
                    query.triangle_candidates += 1;
                    let end = earliest.map_or(1.0, |hit| hit.fraction);
                    if let Some(translation) = translation {
                        let pose = motion.initial_poses()[collider.body];
                        let shape = collider
                            .local
                            .transformed(pose.position, pose.rotation)
                            .map_err(|_| PhysicsError::InvalidCollision)?;
                        if initial_support(&shape, triangle, exclude_initial_supports, &mut query)?
                        {
                            continue;
                        }
                        query.linear_interval_evaluations += 1;
                        if let Some([fraction, _]) = shape
                            .translation_interval(triangle, translation)
                            .map_err(|_| PhysicsError::InvalidCollision)?
                            && fraction <= end
                        {
                            let at_impact = shape
                                .transformed(translation * fraction, DQuat::IDENTITY)
                                .map_err(|_| PhysicsError::InvalidCollision)?;
                            let separation = at_impact
                                .triangle_separation(triangle)
                                .map_err(|_| PhysicsError::InvalidCollision)?;
                            query.separation_evaluations += 1;
                            if earliest.is_none_or(|previous| fraction < previous.fraction) {
                                earliest = Some(TerrainSweepHit {
                                    collider: collider_row,
                                    target: ContactTarget::Terrain {
                                        node,
                                        triangle: triangle_row,
                                    },
                                    fraction,
                                    separation,
                                });
                            }
                        }
                        continue;
                    }
                    let mut fraction = 0.0;
                    for evaluation in 1..=maximum_evaluations_per_triangle {
                        let poses = motion.poses_at(fraction)?;
                        query.pose_evaluations += 1;
                        let pose = poses[collider.body];
                        let shape = collider
                            .local
                            .transformed(pose.position, pose.rotation)
                            .map_err(|_| PhysicsError::InvalidCollision)?;
                        if evaluation == 1
                            && initial_support(
                                &shape,
                                triangle,
                                exclude_initial_supports,
                                &mut query,
                            )?
                        {
                            break;
                        }
                        let separation = shape
                            .triangle_separation(triangle)
                            .map_err(|_| PhysicsError::InvalidCollision)?;
                        query.separation_evaluations += 1;
                        let hit = TerrainSweepHit {
                            collider: collider_row,
                            target: ContactTarget::Terrain {
                                node,
                                triangle: triangle_row,
                            },
                            fraction,
                            separation,
                        };
                        if separation <= tolerance {
                            if earliest.is_none_or(|previous| fraction < previous.fraction) {
                                earliest = Some(hit);
                            }
                            break;
                        }
                        if speed == 0.0 || separation > speed * (end - fraction) {
                            break;
                        }
                        let velocities = motion.velocities_at_poses(&poses);
                        query.velocity_evaluations += 1;
                        query.quadratic_interval_evaluations += 1;
                        let prefix = shape
                            .triangle_motion_prefix(
                                triangle,
                                pose.position,
                                velocities[collider.body],
                                bound.point_acceleration(collider.radius),
                                end - fraction,
                            )
                            .map_err(|_| PhysicsError::InvalidCollision)?;
                        if prefix == end - fraction {
                            break;
                        }
                        let next = fraction + prefix.max(separation / speed);
                        if evaluation == maximum_evaluations_per_triangle || next <= fraction {
                            query.outcome = TerrainSweepOutcome::Unconverged(hit);
                            return Ok(query);
                        }
                        fraction = next.min(end);
                    }
                }
            }
        }
        // Colliders of different bodies approaching each other. The separating-
        // axis gap is a lower bound on distance, and distance closes no faster
        // than both colliders' point-speed bounds together, so advancing by that
        // sum stays conservative with both sides moving.
        let mut bounds = Vec::with_capacity(machine.colliders.len());
        for collider in &machine.colliders {
            let reach = motion.bounds()[collider.body].origin_speed + collider.radius + tolerance;
            let center = motion.initial_poses()[collider.body].position;
            let corners = [center - DVec3::splat(reach), center + DVec3::splat(reach)];
            if !corners[0].is_finite() || !corners[1].is_finite() {
                return Err(PhysicsError::InvalidCollision);
            }
            bounds.push(corners);
        }
        for [first, second] in machine.candidate_pairs(&bounds) {
            query.collider_pair_candidates += 1;
            let colliders = [&machine.colliders[first], &machine.colliders[second]];
            if let Some(slow) = &slow
                && colliders.iter().all(|collider| {
                    [motion.bounds(), slow.initial, slow.final_bounds]
                        .iter()
                        .all(|bounds| {
                            bounds[collider.body].point_speed(collider.radius)
                                <= slow.maximum_displacement
                        })
                })
            {
                query.slow_contact_deferrals += 1;
                continue;
            }
            let speed = colliders
                .iter()
                .map(|collider| motion.bounds()[collider.body].point_speed(collider.radius))
                .sum::<f64>();
            if !speed.is_finite() {
                return Err(PhysicsError::InvalidCollision);
            }
            let end = earliest.map_or(1.0, |hit| hit.fraction);
            let mut fraction = 0.0;
            for evaluation in 1..=maximum_evaluations_per_triangle {
                let sampled;
                let poses: &[BodyPose] = if evaluation == 1 {
                    motion.initial_poses()
                } else {
                    sampled = motion.poses_at(fraction)?;
                    query.pose_evaluations += 1;
                    &sampled
                };
                let [own, other] = colliders.map(|collider| {
                    let pose = poses[collider.body];
                    collider
                        .local
                        .transformed(pose.position, pose.rotation)
                        .map_err(|_| PhysicsError::InvalidCollision)
                });
                let (own, other) = (own?, other?);
                let separation = own
                    .convex_separation(&other)
                    .map_err(|_| PhysicsError::InvalidCollision)?
                    .separation;
                query.separation_evaluations += 1;
                if evaluation == 1 && exclude_initial_supports {
                    query.initial_contact_evaluations += 1;
                    if separation <= PAIR_ACTIVATION_DISTANCE {
                        query.supported_pairs += 1;
                        break;
                    }
                }
                let hit = TerrainSweepHit {
                    collider: first,
                    target: ContactTarget::Collider(second),
                    fraction,
                    separation,
                };
                // Scale the requested terrain tolerance to the pair window.
                if separation
                    <= tolerance.max(
                        PAIR_ACTIVATION_DISTANCE
                            * (tolerance / CONTACT_ACTIVATION_DISTANCE).min(1.0),
                    )
                {
                    if earliest.is_none_or(|previous| fraction < previous.fraction) {
                        earliest = Some(hit);
                    }
                    break;
                }
                if speed == 0.0 || separation > speed * (end - fraction) {
                    break;
                }
                // Bodies moving together keep their gap even when both move fast,
                // so certify the prefix from relative motion before falling back
                // to dividing the gap by both absolute speeds.
                let velocities = motion.velocities_at_poses(poses);
                query.velocity_evaluations += 1;
                query.quadratic_interval_evaluations += 1;
                let [own_body, other_body] = colliders.map(|collider| collider.body);
                let acceleration = colliders
                    .iter()
                    .map(|collider| {
                        motion.bounds()[collider.body].point_acceleration(collider.radius)
                    })
                    .sum::<f64>();
                let prefix = own
                    .convex_motion_prefix(
                        poses[own_body].position,
                        velocities[own_body],
                        &other,
                        poses[other_body].position,
                        velocities[other_body],
                        acceleration,
                        end - fraction,
                    )
                    .map_err(|_| PhysicsError::InvalidCollision)?;
                if prefix == end - fraction {
                    break;
                }
                let next = fraction + prefix.max(separation / speed);
                if next <= fraction
                    || (evaluation == maximum_evaluations_per_triangle && fraction == 0.0)
                {
                    query.outcome = TerrainSweepOutcome::Unconverged(hit);
                    return Ok(query);
                }
                if evaluation == maximum_evaluations_per_triangle {
                    // A body rolling over another's edge a fraction of a micron
                    // away closes its gap far slower than any bound admits, so
                    // advancement crawls. Everything before `fraction` is still
                    // certified clear: report it as the earliest candidate so the
                    // event search commits that prefix and continues from it.
                    if earliest.is_none_or(|previous| fraction < previous.fraction) {
                        earliest = Some(hit);
                    }
                    break;
                }
                fraction = next.min(end);
            }
        }
        query.outcome = earliest.map_or(TerrainSweepOutcome::Clear, TerrainSweepOutcome::Impact);
        Ok(query)
    }
}

fn initial_support(
    shape: &ContactPolytope,
    triangle: [DVec3; 3],
    exclude: bool,
    query: &mut TerrainSweepQuery,
) -> Result<bool, PhysicsError> {
    if !exclude {
        return Ok(false);
    }
    query.initial_contact_evaluations += 1;
    let supported =
        !super::activation_points(shape, triangle, CONTACT_ACTIVATION_DISTANCE)?.is_empty();
    query.supported_pairs += usize::from(supported);
    Ok(supported)
}

#[cfg(test)]
mod tests;
