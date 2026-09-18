//! Bounded interval validation for supports that already overlap at path start.

use super::{ContactTarget, MachineCollisionGeometry, TerrainContactScene, valid_bounds};
use crate::{BodyPose, MachineMotion, PhysicsError};
use bevy_math::{DVec3, Vec3};
use mechanic_world::{WorldBounds, WorldPosition};

/// Finite geometry and unresolved interval, suitable for retry diagnostics.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TerrainPathFailure {
    /// Collider row within the query's topology generation.
    pub collider: usize,
    /// Terrain triangle or collider on another body.
    pub target: ContactTarget,
    /// Normalized interval that could not be certified.
    pub interval: [f64; 2],
    /// Conservative envelope depth; this may overestimate physical overlap.
    /// None means the budget expired before evaluating this interval.
    pub upper_bound: Option<f64>,
    /// Largest retained physical contact depth at the interval midpoint.
    /// None means this interval was not evaluated; retained points need not
    /// include the deepest feature, so this value cannot certify the path.
    pub observed_depth: Option<f64>,
}

/// A supported path needs a depth certificate, even when its initial SAT gap is zero.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TerrainPathOutcome {
    /// Every candidate interval meets the requested penetration limit.
    Bounded,
    /// A sampled physical contact exceeds the limit. No state was published.
    ExcessPenetration(TerrainPathFailure),
    /// The interval bound was inconclusive when the work/progress bound expired.
    Unconverged(TerrainPathFailure),
}

/// Tagged depth validation and all work, including intervals rejected before subdivision.
#[derive(Clone, Debug)]
pub struct TerrainPathQuery {
    /// Must be checked before accepting a supported candidate path.
    pub outcome: TerrainPathOutcome,
    /// Topology used throughout the query.
    pub topology_generation: u64,
    /// Terrain publication used throughout the query.
    pub terrain_generation: u64,
    /// Chunk candidates across swept bounds.
    pub chunk_candidates: usize,
    /// Exact finite triangle candidates.
    pub triangle_candidates: usize,
    /// Whole-tree midpoint reconstructions; none assemble inertia.
    pub pose_evaluations: usize,
    /// Reuses of the exact full-interval midpoint reconstruction in this query.
    pub pose_cache_hits: usize,
    /// Interval envelope evaluations, including failed bounds.
    pub envelope_evaluations: usize,
    /// Intervals whose entire trajectory was bounded, rather than merely sampled.
    pub certified_intervals: usize,
    /// Collider pairs on different bodies whose swept bounds overlap.
    pub collider_pair_candidates: usize,
}

impl TerrainContactScene {
    /// Bounds finite normal penetration along a complete articulated candidate path.
    /// Uses deterministic midpoint interval envelopes and the path's point-speed
    /// bounds. Existing supports are never skipped. This validates geometry only;
    /// it does not activate impacts, supply missing terrain, or publish a tick.
    ///
    /// # Errors
    /// Rejects incompatible topology, invalid settings, or uncertifiable geometry.
    #[expect(
        clippy::too_many_lines,
        reason = "ordered candidate traversal with explicit failed-work accounting"
    )]
    pub fn validate_penetration(
        &self,
        machine: &MachineCollisionGeometry,
        motion: &MachineMotion<'_>,
        origin: DVec3,
        maximum_depth: f64,
        maximum_evaluations_per_triangle: usize,
    ) -> Result<TerrainPathQuery, PhysicsError> {
        if machine.generation != motion.generation()
            || machine.bodies != motion.initial_poses().len()
            || !origin.is_finite()
            || !maximum_depth.is_finite()
            || maximum_depth < 0.0
            || maximum_evaluations_per_triangle == 0
        {
            return Err(PhysicsError::InvalidCollision);
        }
        let mut query = TerrainPathQuery {
            outcome: TerrainPathOutcome::Bounded,
            topology_generation: machine.generation,
            terrain_generation: self.generation,
            chunk_candidates: 0,
            triangle_candidates: 0,
            pose_evaluations: 0,
            pose_cache_hits: 0,
            envelope_evaluations: 0,
            certified_intervals: 0,
            collider_pair_candidates: 0,
        };
        // Every triangle first asks for the same dyadic midpoint. Its immutable
        // trajectory and generation are fixed for this query. Retain just this
        // one whole-tree sample so storage remains linear in the body count.
        let mut midpoint_poses = None;
        for (collider_row, collider) in machine.colliders.iter().enumerate() {
            if !collider.moving {
                continue;
            }
            let motion_bound = motion.bounds()[collider.body];
            let [minimum, maximum] = super::path_bounds(collider, motion, 0.0)?;
            let bounds = WorldBounds {
                minimum: WorldPosition((origin + minimum).map(f64::next_down)),
                maximum: WorldPosition((origin + maximum).map(f64::next_up)),
            };
            if !valid_bounds(bounds) {
                return Err(PhysicsError::InvalidCollision);
            }
            let speed = motion_bound.point_speed(collider.radius);
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
                    let mut pending = vec![[0.0_f64, 1.0_f64]];
                    let mut evaluations = 0;
                    while let Some(interval) = pending.pop() {
                        if evaluations >= maximum_evaluations_per_triangle {
                            query.outcome = TerrainPathOutcome::Unconverged(TerrainPathFailure {
                                collider: collider_row,
                                target: ContactTarget::Terrain {
                                    node,
                                    triangle: triangle_row,
                                },
                                interval,
                                upper_bound: None,
                                observed_depth: None,
                            });
                            return Ok(query);
                        }
                        let midpoint = interval[0] + (interval[1] - interval[0]) * 0.5;
                        let pose = if midpoint.to_bits() == 0.5_f64.to_bits() {
                            if midpoint_poses.is_none() {
                                midpoint_poses = Some(motion.poses_at(midpoint)?);
                                query.pose_evaluations += 1;
                            } else {
                                query.pose_cache_hits += 1;
                            }
                            midpoint_poses
                                .as_ref()
                                .ok_or(PhysicsError::InvalidCollision)?[collider.body]
                        } else {
                            query.pose_evaluations += 1;
                            motion.poses_at(midpoint)?[collider.body]
                        };
                        let shape = collider
                            .local
                            .transformed(pose.position, pose.rotation)
                            .map_err(|_| PhysicsError::InvalidCollision)?;
                        let half_width = (midpoint - interval[0])
                            .max(interval[1] - midpoint)
                            .next_up();
                        let displacement = (speed * half_width).next_up();
                        let bound = shape
                            .triangle_penetration_bound(triangle, displacement)
                            .map_err(|_| PhysicsError::InvalidCollision)?;
                        evaluations += 1;
                        query.envelope_evaluations += 1;
                        let Some(upper_bound) = bound else {
                            query.certified_intervals += 1;
                            continue;
                        };
                        if upper_bound <= maximum_depth {
                            query.certified_intervals += 1;
                            continue;
                        }
                        let observed_depth = shape
                            .triangle_contacts(triangle)
                            .map_err(|_| PhysicsError::InvalidCollision)?
                            .iter()
                            .map(|point| point.depth)
                            .fold(0.0_f64, f64::max);
                        let failure = TerrainPathFailure {
                            collider: collider_row,
                            target: ContactTarget::Terrain {
                                node,
                                triangle: triangle_row,
                            },
                            interval,
                            upper_bound: Some(upper_bound),
                            observed_depth: Some(observed_depth),
                        };
                        if observed_depth > maximum_depth {
                            query.outcome = TerrainPathOutcome::ExcessPenetration(failure);
                            return Ok(query);
                        }
                        if evaluations == maximum_evaluations_per_triangle
                            || midpoint <= interval[0]
                            || midpoint >= interval[1]
                        {
                            query.outcome = TerrainPathOutcome::Unconverged(failure);
                            return Ok(query);
                        }
                        // Push right first; visit the left interval first at every depth.
                        pending.push([midpoint, interval[1]]);
                        pending.push([interval[0], midpoint]);
                    }
                }
            }
        }
        // Colliders of different bodies. Over an interval, their relative point
        // displacement is within both colliders' bounds together, so the
        // midpoint's separating-axis overlap plus that sum bounds the whole path.
        let mut bounds = Vec::with_capacity(machine.colliders.len());
        for collider in &machine.colliders {
            let corners = super::path_bounds(collider, motion, 0.0)?;
            if !corners[0].is_finite() || !corners[1].is_finite() {
                return Err(PhysicsError::InvalidCollision);
            }
            bounds.push(corners);
        }
        for &[first, second] in machine.candidate_pairs(&bounds).iter() {
            query.collider_pair_candidates += 1;
            let colliders = [&machine.colliders[first], &machine.colliders[second]];
            let speed = colliders.iter().fold(0.0_f64, |sum, collider| {
                (sum + motion.bounds()[collider.body].point_speed(collider.radius)).next_up()
            });
            if !speed.is_finite() {
                return Err(PhysicsError::InvalidCollision);
            }
            let failure = |interval, upper_bound, observed_depth| TerrainPathFailure {
                collider: first,
                target: ContactTarget::Collider(second),
                interval,
                upper_bound,
                observed_depth,
            };
            let mut pending = vec![[0.0_f64, 1.0_f64]];
            let mut evaluations = 0;
            while let Some(interval) = pending.pop() {
                if evaluations >= maximum_evaluations_per_triangle {
                    query.outcome = TerrainPathOutcome::Unconverged(failure(interval, None, None));
                    return Ok(query);
                }
                let midpoint = interval[0] + (interval[1] - interval[0]) * 0.5;
                let sampled;
                let poses: &[BodyPose] = if midpoint.to_bits() == 0.5_f64.to_bits() {
                    if midpoint_poses.is_none() {
                        midpoint_poses = Some(motion.poses_at(midpoint)?);
                        query.pose_evaluations += 1;
                    } else {
                        query.pose_cache_hits += 1;
                    }
                    midpoint_poses
                        .as_deref()
                        .ok_or(PhysicsError::InvalidCollision)?
                } else {
                    query.pose_evaluations += 1;
                    sampled = motion.poses_at(midpoint)?;
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
                let half_width = (midpoint - interval[0])
                    .max(interval[1] - midpoint)
                    .next_up();
                let displacement = (speed * half_width).next_up();
                let bound = own
                    .convex_penetration_bound(&other, displacement)
                    .map_err(|_| PhysicsError::InvalidCollision)?;
                evaluations += 1;
                query.envelope_evaluations += 1;
                let Some(upper_bound) = bound else {
                    query.certified_intervals += 1;
                    continue;
                };
                if upper_bound <= maximum_depth {
                    query.certified_intervals += 1;
                    continue;
                }
                let observed_depth = (-own
                    .convex_separation(&other)
                    .map_err(|_| PhysicsError::InvalidCollision)?
                    .separation)
                    .max(0.0);
                let failure = failure(interval, Some(upper_bound), Some(observed_depth));
                if observed_depth > maximum_depth {
                    query.outcome = TerrainPathOutcome::ExcessPenetration(failure);
                    return Ok(query);
                }
                if evaluations == maximum_evaluations_per_triangle
                    || midpoint <= interval[0]
                    || midpoint >= interval[1]
                {
                    query.outcome = TerrainPathOutcome::Unconverged(failure);
                    return Ok(query);
                }
                pending.push([midpoint, interval[1]]);
                pending.push([interval[0], midpoint]);
            }
        }
        Ok(query)
    }
}

#[cfg(test)]
mod tests;
