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
    /// Shapes transformed during continuous collision checks.
    pub shape_transformations: usize,
    /// Starting shapes reused during continuous collision checks.
    pub shape_cache_hits: usize,
    /// Collider hierarchy node pairs tested during continuous collision checks.
    pub hierarchy_node_pair_tests: usize,
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
    /// Collider paths prepared after conservative body rejection.
    pub detailed_preparations: usize,
    /// Finite initial supports reused only at their exact queried poses.
    pub cached_supports: usize,
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
            None,
            &[],
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
            None,
            &[],
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
            None,
            &[],
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn sweep_contact_groups(
        &self,
        machine: &MachineCollisionGeometry,
        motion: &MachineMotion<'_>,
        origin: DVec3,
        tolerance: f64,
        maximum_evaluations_per_triangle: usize,
        groups: &super::ContactGroups,
        supported: &[super::TerrainContactFeature],
    ) -> Result<TerrainSweepQuery, PhysicsError> {
        self.sweep_internal(
            machine,
            motion,
            origin,
            tolerance,
            maximum_evaluations_per_triangle,
            true,
            None,
            Some(groups),
            supported,
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
        groups: Option<&super::ContactGroups>,
        supported: &[super::TerrainContactFeature],
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
            shape_transformations: 0,
            shape_cache_hits: 0,
            hierarchy_node_pair_tests: 0,
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
            detailed_preparations: 0,
            cached_supports: 0,
        };
        for (body, &radius) in machine.body_radii.iter().enumerate() {
            if machine.body_colliders[body]
                .first()
                .is_some_and(|&row| machine.colliders[row].moving)
            {
                let displacement = motion.bounds()[body].point_speed(radius);
                if !displacement.is_finite() {
                    return Err(PhysicsError::InvalidCollision);
                }
                query.maximum_point_displacement =
                    query.maximum_point_displacement.max(displacement);
            }
        }
        // Lock order is shared with contact queries: pose cache, sweep scratch,
        // then pair scratch. Velocities belong to this path, never the pose cache.
        let mut cache = machine
            .cache
            .lock()
            .map_err(|_| PhysicsError::InvalidCollision)?;

        let mut scratch = machine
            .sweep_scratch
            .lock()
            .map_err(|_| PhysicsError::InvalidCollision)?;
        let SweepScratch {
            shapes: [first_shape, second_shape],
            clipping,
        } = &mut *scratch;
        let body_bounds = machine.body_path_bounds(motion, tolerance)?;
        let mut prepare = vec![false; machine.bodies];
        let mut body_terrain = vec![Vec::new(); machine.bodies];
        for (body, bounds) in body_bounds.iter().enumerate() {
            if machine.body_colliders[body].is_empty() {
                continue;
            }
            let world = WorldBounds {
                minimum: WorldPosition((origin + bounds[0]).map(f64::next_down)),
                maximum: WorldPosition((origin + bounds[1]).map(f64::next_up)),
            };
            if !valid_bounds(world) {
                return Err(PhysicsError::InvalidCollision);
            }
            if groups.is_none_or(|g| g.includes(machine, body, None))
                && machine.colliders[machine.body_colliders[body][0]].moving
            {
                body_terrain[body] = self.index.bounds_candidates(world);
                prepare[body] = !body_terrain[body].is_empty();
            }
        }
        for [a, b] in machine.body_candidates(&body_bounds) {
            if groups.is_some_and(|g| !g.includes(machine, a, Some(b))) {
                continue;
            }
            prepare[a] = true;
            prepare[b] = true;
        }
        if reuse_start() && !prepare.iter().any(|&body| body) {
            return Ok(query);
        }
        // The body hierarchy rejects distant assemblies. Within surviving
        // bodies, first use the cheap speed envelope to reject individual
        // colliders before computing endpoint/curvature bounds.
        let mut starts = vec![None; machine.colliders.len()];
        let coarse = machine
            .colliders
            .iter()
            .enumerate()
            .map(|(row, collider)| {
                if !prepare[collider.body] {
                    return Ok(body_bounds[collider.body]);
                }
                let pose = motion.initial_poses()[collider.body];
                let start = if cache.poses.get(collider.body) == Some(&pose) {
                    cache.bounds[row]
                } else {
                    collider
                        .local
                        .transformed_bounds(pose.position, pose.rotation)
                        .map_err(|_| PhysicsError::InvalidCollision)?
                };
                starts[row] = Some(start);
                Ok(super::swept_bounds(
                    collider,
                    pose,
                    motion.bounds()[collider.body],
                    tolerance,
                    start,
                ))
            })
            .collect::<Result<Vec<_>, PhysicsError>>()?;
        let mut detailed = vec![false; machine.colliders.len()];
        for (row, collider) in machine.colliders.iter().enumerate() {
            if collider.moving && groups.is_none_or(|g| g.includes(machine, collider.body, None)) {
                let bounds = WorldBounds {
                    minimum: WorldPosition((origin + coarse[row][0]).map(f64::next_down)),
                    maximum: WorldPosition((origin + coarse[row][1]).map(f64::next_up)),
                };
                if !valid_bounds(bounds) {
                    return Err(PhysicsError::InvalidCollision);
                }
                detailed[row] = body_terrain[collider.body].iter().any(|node| {
                    let chunk = &self.chunks[node].geometry;
                    let chunk_bounds = chunk
                        .triangle_bvh
                        .nodes
                        .first()
                        .map_or(chunk.bounds, |node| node.bounds);
                    super::overlaps(
                        [bounds.minimum.0, bounds.maximum.0],
                        [chunk_bounds.minimum.0, chunk_bounds.maximum.0],
                    )
                });
            }
        }
        {
            let pairs = machine.candidate_pairs_groups(&coarse, groups);
            query.hierarchy_node_pair_tests += pairs.scratch.node_pair_tests;
            for &[a, b] in pairs.iter() {
                detailed[a] = true;
                detailed[b] = true;
            }
        }
        let bounds = machine
            .colliders
            .iter()
            .enumerate()
            .map(|(row, collider)| {
                if !detailed[row] && reuse_start() {
                    return Ok(coarse[row]);
                }
                query.detailed_preparations += 1;
                super::path_bounds_from(collider, motion, tolerance, starts[row])
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut terrain_candidates = Vec::with_capacity(bounds.len());
        for (collider, &[minimum, maximum]) in machine.colliders.iter().zip(&bounds) {
            let world = WorldBounds {
                minimum: WorldPosition((origin + minimum).map(f64::next_down)),
                maximum: WorldPosition((origin + maximum).map(f64::next_up)),
            };
            if !valid_bounds(world) {
                return Err(PhysicsError::InvalidCollision);
            }
            terrain_candidates.push(
                if collider.moving
                    && groups.is_none_or(|g| g.includes(machine, collider.body, None))
                {
                    if reuse_start() {
                        body_terrain[collider.body]
                            .iter()
                            .copied()
                            .filter(|node| {
                                let chunk = &self.chunks[node].geometry;
                                let chunk_bounds = chunk
                                    .triangle_bvh
                                    .nodes
                                    .first()
                                    .map_or(chunk.bounds, |node| node.bounds);
                                super::overlaps(
                                    [world.minimum.0, world.maximum.0],
                                    [chunk_bounds.minimum.0, chunk_bounds.maximum.0],
                                )
                            })
                            .collect()
                    } else {
                        self.index.bounds_candidates(world)
                    }
                } else {
                    Vec::new()
                },
            );
        }
        let candidates = machine.candidate_pairs_groups(&bounds, groups);
        query.hierarchy_node_pair_tests += candidates.scratch.node_pair_tests;
        if candidates.iter().next().is_none() && terrain_candidates.iter().all(Vec::is_empty) {
            return Ok(query);
        }
        cache.update(machine, motion.initial_poses())?;
        let initial_velocities = std::sync::OnceLock::new();
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
            if terrain_candidates[collider_row].is_empty() {
                continue;
            }
            let bound = motion.bounds()[collider.body];
            // Zero accumulated angular speed proves all ancestors are fixed in
            // rotation. Matching endpoint quaternions would miss complete turns.
            let translation = (bound.angular_speed == 0.0).then(|| {
                motion.final_poses()[collider.body].position
                    - motion.initial_poses()[collider.body].position
            });
            let [minimum, maximum] = bounds[collider_row];
            let bounds = WorldBounds {
                minimum: WorldPosition((origin + minimum).map(f64::next_down)),
                maximum: WorldPosition((origin + maximum).map(f64::next_up)),
            };
            let speed = bound.point_speed(collider.radius);
            let chunks = &terrain_candidates[collider_row];
            query.chunk_candidates += chunks.len();
            for &node in chunks {
                let chunk = &self.chunks[&node].geometry;
                for triangle_row in chunk.bounds_candidates(bounds) {
                    let triangle =
                        chunk.triangle_bvh.triangles[triangle_row]
                            .indices
                            .map(|index| {
                                chunk.origin.0 - origin
                                    + Vec3::from_array(chunk.vertices[index as usize]).as_dvec3()
                            });
                    // BVH leaves batch triangles; their shared node bounds do
                    // not prove that this individual triangle is reachable.
                    let triangle_bounds = triangle.iter().fold(
                        [DVec3::INFINITY, DVec3::NEG_INFINITY],
                        |[lo, hi], &point| [lo.min(point), hi.max(point)],
                    );
                    if !super::overlaps([minimum, maximum], triangle_bounds) {
                        continue;
                    }
                    if (triangle[1] - triangle[0])
                        .cross(triangle[2] - triangle[0])
                        .try_normalize()
                        .is_none()
                    {
                        continue;
                    }
                    query.triangle_candidates += 1;
                    if exclude_initial_supports
                        && reuse_start()
                        && supported.iter().any(|feature| {
                            feature.touches(
                                collider_row,
                                ContactTarget::Terrain {
                                    node,
                                    triangle: triangle_row,
                                },
                            )
                        })
                    {
                        query.supported_pairs += 1;
                        query.cached_supports += 1;
                        continue;
                    }
                    let end = earliest.map_or(1.0, |hit| hit.fraction);
                    if let Some(translation) = translation {
                        let shape = if reuse_start() {
                            starting_shape(
                                &cache,
                                machine,
                                motion.initial_poses(),
                                collider_row,
                                &mut query,
                            )?
                        } else {
                            query.shape_transformations += 1;
                            let pose = motion.initial_poses()[collider.body];
                            transform(&collider.local, pose.position, pose.rotation, second_shape)?
                        };
                        if initial_support(
                            shape,
                            triangle,
                            exclude_initial_supports,
                            &mut query,
                            clipping,
                        )? {
                            continue;
                        }
                        query.linear_interval_evaluations += 1;
                        if let Some([fraction, _]) = shape
                            .translation_interval(triangle, translation)
                            .map_err(|_| PhysicsError::InvalidCollision)?
                            && fraction <= end
                        {
                            query.shape_transformations += 1;
                            let at_impact = transform(
                                shape,
                                translation * fraction,
                                DQuat::IDENTITY,
                                first_shape,
                            )?;
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
                        let sampled;
                        let poses = if fraction == 0.0 && reuse_start() {
                            motion.initial_poses()
                        } else {
                            sampled = motion.poses_at(fraction)?;
                            query.pose_evaluations += 1;
                            &sampled
                        };
                        let pose = poses[collider.body];
                        let shape = if fraction == 0.0 && reuse_start() {
                            starting_shape(&cache, machine, poses, collider_row, &mut query)?
                        } else {
                            query.shape_transformations += 1;
                            transform(&collider.local, pose.position, pose.rotation, first_shape)?
                        };
                        if evaluation == 1
                            && initial_support(
                                shape,
                                triangle,
                                exclude_initial_supports,
                                &mut query,
                                clipping,
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
                        let sampled_velocities;
                        let velocities = if fraction == 0.0 && reuse_start() {
                            initial_velocities.get_or_init(|| {
                                query.velocity_evaluations += 1;
                                motion.velocities_at_poses(motion.initial_poses())
                            })
                        } else {
                            query.velocity_evaluations += 1;
                            sampled_velocities = motion.velocities_at_poses(poses);
                            &sampled_velocities
                        };
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
        for &[first, second] in candidates.iter() {
            query.collider_pair_candidates += 1;
            if exclude_initial_supports
                && reuse_start()
                && supported
                    .iter()
                    .any(|feature| feature.touches(first, ContactTarget::Collider(second)))
            {
                query.supported_pairs += 1;
                query.cached_supports += 1;
                continue;
            }
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
                let poses: &[BodyPose] = if evaluation == 1 && reuse_start() {
                    motion.initial_poses()
                } else {
                    sampled = motion.poses_at(fraction)?;
                    query.pose_evaluations += 1;
                    &sampled
                };
                let (own, other) = if fraction == 0.0 && reuse_start() {
                    (
                        starting_shape(&cache, machine, poses, first, &mut query)?,
                        starting_shape(&cache, machine, poses, second, &mut query)?,
                    )
                } else {
                    query.shape_transformations += 2;
                    let [own_pose, other_pose] = colliders.map(|collider| poses[collider.body]);
                    (
                        transform(
                            &colliders[0].local,
                            own_pose.position,
                            own_pose.rotation,
                            first_shape,
                        )?,
                        transform(
                            &colliders[1].local,
                            other_pose.position,
                            other_pose.rotation,
                            second_shape,
                        )?,
                    )
                };
                let separation = own
                    .convex_separation(other)
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
                let sampled_velocities;
                let velocities = if fraction == 0.0 && reuse_start() {
                    initial_velocities.get_or_init(|| {
                        query.velocity_evaluations += 1;
                        motion.velocities_at_poses(motion.initial_poses())
                    })
                } else {
                    query.velocity_evaluations += 1;
                    sampled_velocities = motion.velocities_at_poses(poses);
                    &sampled_velocities
                };
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
                        other,
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
    clipping: &mut mechanic_core::TriangleClipScratch,
) -> Result<bool, PhysicsError> {
    if !exclude {
        return Ok(false);
    }
    query.initial_contact_evaluations += 1;
    let supported = !super::activation_points_with_scratch(
        shape,
        triangle,
        CONTACT_ACTIVATION_DISTANCE,
        clipping,
    )?
    .is_empty();
    query.supported_pairs += usize::from(supported);
    Ok(supported)
}

#[derive(Default)]
pub(super) struct SweepScratch {
    shapes: [Option<ContactPolytope>; 2],
    clipping: mechanic_core::TriangleClipScratch,
}

fn transform<'a>(
    local: &ContactPolytope,
    position: DVec3,
    rotation: DQuat,
    buffer: &'a mut Option<ContactPolytope>,
) -> Result<&'a ContactPolytope, PhysicsError> {
    let shape = buffer.get_or_insert_with(|| local.clone());
    local
        .transformed_into(position, rotation, shape)
        .map_err(|_| PhysicsError::InvalidCollision)?;
    Ok(shape)
}

fn starting_shape<'a>(
    cache: &'a super::PoseCache,
    machine: &MachineCollisionGeometry,
    poses: &[BodyPose],
    row: usize,
    query: &mut TerrainSweepQuery,
) -> Result<&'a ContactPolytope, PhysicsError> {
    if cache.shapes[row].get().is_some() {
        query.shape_cache_hits += 1;
    } else {
        query.shape_transformations += 1;
    }
    cache.shape(machine, poses, row)
}

// The uncached test path reconstructs and transforms each evaluation, as the
// pre-reuse sweep did. This switch is absent from production builds.
#[cfg(test)]
std::thread_local! {
    static REUSE_START: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
}
fn reuse_start() -> bool {
    #[cfg(test)]
    {
        REUSE_START.get()
    }
    #[cfg(not(test))]
    {
        true
    }
}

#[cfg(test)]
mod tests;
