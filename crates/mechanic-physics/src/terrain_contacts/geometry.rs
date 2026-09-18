//! A machine's collision geometry: its colliders and how they move over a tick.

use super::groups::ContactGroups;
use super::pose_cache::{CandidatePairs, PairScratch, PoseCache, overlaps};
use super::{broadphase, sweep};
use crate::PhysicsError;
use bevy_math::DVec3;
use mechanic_core::{CompiledCreation, ContactCylinder, ContactPolytope, MaterialProperties};
use std::sync::Mutex;

pub(super) struct Collider {
    pub(super) body: usize,
    pub(super) center: DVec3,
    pub(super) material: MaterialProperties,
    // Circumscribing prism for a cylinder: every bound, sweep and body pair uses
    // it, since it contains the cylinder.
    pub(super) local: ContactPolytope,
    // The exact cylinder, for rolling terrain contact.
    pub(super) round: Option<ContactCylinder>,
    pub(super) bounds: [DVec3; 2],
    pub(super) moving: bool,
    pub(super) radius: f64,
}

// Frequently traversed motion data stays separate from the larger shape records.
pub(super) struct MotionCollider {
    pub(super) row: usize,
    pub(super) body: usize,
    pub(super) assembly: usize,
    pub(super) radius: f64,
    pub(super) round: Option<ContactCylinder>,
}

// Bounds on how fast a collider's surface moves against terrain along a path,
// and how fast it turns. A cylinder meets terrain as a circle, so its turn
// about its own axis counts as neither.
pub(super) fn terrain_motion(
    round: Option<&ContactCylinder>,
    radius: f64,
    bound: crate::MotionBound,
    spin: crate::motion_path::Spin,
) -> [f64; 2] {
    let point = [bound.point_speed(radius), bound.angular_speed];
    round.map_or(point, |cylinder| {
        let reach = cylinder.radius().hypot(cylinder.half_length()).next_up();
        let [speed, turn] = spin.symmetric(cylinder.center(), cylinder.axis(), reach);
        [speed.min(point[0]), turn.min(point[1])]
    })
}

impl MachineCollisionGeometry {
    /// Body and conservative radius about the body origin of every collider row,
    /// in row order.
    pub(crate) fn collider_reach(&self) -> impl ExactSizeIterator<Item = (usize, f64)> + '_ {
        self.reach.iter().copied()
    }

    /// The exact cylinder of a collider row in body coordinates, if it is one.
    pub(crate) fn rolling_shape(&self, row: usize) -> Option<&ContactCylinder> {
        self.colliders.get(row)?.round.as_ref()
    }
}

/// Immutable local collider data compiled once for a construction generation.
pub struct MachineCollisionGeometry {
    pub(super) generation: u64,
    pub(super) bodies: usize,
    pub(super) colliders: Vec<Collider>,
    pub(super) reach: Vec<(usize, f64)>,
    pub(super) motion_colliders: Vec<MotionCollider>,
    // Sorted body pairs joined by a bearing, which never collide with each other.
    pub(super) suppressed: Vec<[usize; 2]>,
    pub(super) body_colliders: Vec<Vec<usize>>,
    pub(super) body_bounds: Vec<[DVec3; 2]>,
    pub(super) body_radii: Vec<f64>,
    // Bodies whose colliders are all cylinders.
    pub(super) round_bodies: Vec<bool>,
    pub(crate) assemblies: Vec<usize>,
    pub(crate) assembly_count: usize,
    pub(crate) internal_collisions: Vec<bool>,
    // Assemblies with at most one moving tree root, whose bodies' root-frame
    // motion bounds every distance between them.
    pub(crate) tree_frames: Vec<bool>,
    pub(crate) moving_assemblies: Vec<usize>,
    pub(super) collider_trees: Vec<broadphase::Tree>,
    pub(super) body_tree: broadphase::Tree,
    pub(super) pair_scratch: Mutex<PairScratch>,
    pub(super) sweep_scratch: Mutex<sweep::SweepScratch>,
    pub(super) cache: Mutex<PoseCache>,
}

impl MachineCollisionGeometry {
    /// Retains the exact box/convex decomposition and material of every collider.
    /// A solid full cylinder is taken from its own analytic description instead:
    /// its sixteen tangent boxes describe the same prism, but each shared corner
    /// twice, rounded once per box. Two copies of one contact edge about 1e-8 m
    /// apart are two separate arrivals to the event search, and it then spends
    /// every trial localizing contacts the solve already carries. The exact
    /// cylinder is kept beside that prism for rolling terrain contact.
    ///
    /// # Errors
    /// Rejects invalid compiled geometry or body references.
    #[expect(
        clippy::too_many_lines,
        reason = "compile immutable geometry and both hierarchy levels together"
    )]
    pub fn new(
        creation: &CompiledCreation,
        topology_generation: u64,
    ) -> Result<Self, PhysicsError> {
        let mut colliders = Vec::with_capacity(creation.colliders.len());
        let mut row = 0;
        while row < creation.colliders.len() {
            let source = &creation.colliders[row];
            let cylinder = creation
                .cylinders
                .iter()
                .find(|cylinder| cylinder.first_collider as usize == row);
            let local = match cylinder {
                Some(cylinder) => ContactPolytope::from_convex(&cylinder.hull()),
                None => ContactPolytope::from_collider(source),
            }
            .map_err(|_| PhysicsError::InvalidCollision)?;
            let body = source.compound_index as usize;
            let compound = creation
                .compounds
                .get(body)
                .ok_or(PhysicsError::InvalidCollision)?;
            let radius = local
                .conservative_radius()
                .map_err(|_| PhysicsError::InvalidCollision)?;
            colliders.push(Collider {
                body,
                center: cylinder
                    .map_or(source.local_center, |cylinder| cylinder.local_center)
                    .as_dvec3(),
                material: source.material_properties,
                bounds: local.bounds(),
                local,
                round: cylinder
                    .map(ContactCylinder::from_compiled)
                    .transpose()
                    .map_err(|_| PhysicsError::InvalidCollision)?,
                moving: !compound.is_static,
                radius,
            });
            row += cylinder.map_or(1, |_| mechanic_core::CYLINDER_COLLIDER_COUNT);
        }
        let mut suppressed = creation
            .collision_suppression
            .iter()
            .map(|pair| {
                let [a, b] = pair.map(|body| body as usize);
                if a.max(b) >= creation.compounds.len() {
                    return Err(PhysicsError::InvalidCollision);
                }
                Ok([a.min(b), a.max(b)])
            })
            .collect::<Result<Vec<_>, _>>()?;
        suppressed.sort_unstable();
        let mut body_colliders = vec![Vec::new(); creation.compounds.len()];
        for (row, collider) in colliders.iter().enumerate() {
            body_colliders[collider.body].push(row);
        }
        let local_bounds = colliders
            .iter()
            .map(|collider| collider.bounds)
            .collect::<Vec<_>>();
        let collider_trees = body_colliders
            .iter()
            .map(|rows| broadphase::Tree::new(rows.clone(), &local_bounds))
            .collect();
        let body_positions = creation
            .compounds
            .iter()
            .map(|body| [body.root_translation.as_dvec3(); 2])
            .collect::<Vec<_>>();
        let body_tree = broadphase::Tree::new(
            (0..creation.compounds.len())
                .filter(|&body| !body_colliders[body].is_empty())
                .collect(),
            &body_positions,
        );
        let body_bounds = body_colliders
            .iter()
            .map(|rows| {
                rows.iter()
                    .fold([DVec3::INFINITY, DVec3::NEG_INFINITY], |[lo, hi], &row| {
                        [lo.min(local_bounds[row][0]), hi.max(local_bounds[row][1])]
                    })
            })
            .collect();
        let body_radii = body_colliders
            .iter()
            .map(|rows| {
                rows.iter()
                    .map(|&row| colliders[row].radius)
                    .fold(0.0, f64::max)
            })
            .collect();
        let mut assemblies = vec![usize::MAX; creation.compounds.len()];
        let mut assembly_count = 0;
        for component in &creation.loop_topology.mechanism_components {
            for &body in component {
                *assemblies
                    .get_mut(body as usize)
                    .ok_or(PhysicsError::InvalidCollision)? = assembly_count;
            }
            assembly_count += 1;
        }
        for assembly in &mut assemblies {
            if *assembly == usize::MAX {
                *assembly = assembly_count;
                assembly_count += 1;
            }
        }
        let reach = colliders.iter().map(|c| (c.body, c.radius)).collect();
        let motion_colliders = colliders
            .iter()
            .enumerate()
            .filter(|(_, c)| c.moving)
            .map(|(row, c)| MotionCollider {
                row,
                body: c.body,
                assembly: assemblies[c.body],
                radius: c.radius,
                round: c.round,
            })
            .collect();
        let round_bodies = body_colliders
            .iter()
            .map(|rows| !rows.is_empty() && rows.iter().all(|&row| colliders[row].round.is_some()))
            .collect();
        let mut geometry = Self {
            generation: topology_generation,
            bodies: creation.compounds.len(),
            colliders,
            reach,
            motion_colliders,
            suppressed,
            body_colliders,
            body_bounds,
            body_radii,
            round_bodies,
            assemblies,
            assembly_count,
            internal_collisions: vec![false; assembly_count],
            tree_frames: vec![true; assembly_count],
            moving_assemblies: Vec::new(),
            collider_trees,
            body_tree,
            pair_scratch: Mutex::new(PairScratch::default()),
            sweep_scratch: Mutex::new(sweep::SweepScratch::default()),
            cache: Mutex::new(PoseCache::default()),
        };
        let flush = geometry.built_flush(creation)?;
        geometry.suppressed.extend(flush);
        geometry.suppressed.sort_unstable();
        geometry.suppressed.dedup();
        let mut roots = vec![0_usize; assembly_count];
        for (body, parents) in creation.loop_topology.body_parents.iter().enumerate() {
            if parents.is_root && !creation.compounds[body].is_static {
                roots[geometry.assemblies[body]] += 1;
            }
        }
        for (frame, roots) in geometry.tree_frames.iter_mut().zip(roots) {
            *frame = roots <= 1;
        }
        let mut members = vec![Vec::new(); assembly_count];
        for (body, &assembly) in geometry.assemblies.iter().enumerate() {
            if !geometry.body_colliders[body].is_empty() {
                members[assembly].push(body);
            }
        }
        for (assembly, bodies) in members.iter().enumerate() {
            if bodies
                .iter()
                .any(|&body| !creation.compounds[body].is_static)
            {
                geometry.moving_assemblies.push(assembly);
            }
            geometry.internal_collisions[assembly] = bodies.iter().enumerate().any(|(row, &a)| {
                bodies[row + 1..].iter().any(|&b| {
                    geometry
                        .suppressed
                        .binary_search(&[a.min(b), a.max(b)])
                        .is_err()
                        && (!creation.compounds[a].is_static || !creation.compounds[b].is_static)
                })
            });
        }
        Ok(geometry)
    }

    // Bodies of one mechanism built touching each other, such as a wheel face
    // flush against the mount two joints away. They slide on that shared face
    // as one assembly: it carries no load, and a spinning face never clears a
    // conservative sweep bounded by its full point speed. Compilation already
    // suppresses each bearing's own pair; this adds every other pair of one
    // mechanism that touches as built. Separate mechanisms always collide, and
    // bodies of one mechanism built apart still collide when they meet.
    pub(super) fn built_flush(
        &self,
        creation: &CompiledCreation,
    ) -> Result<Vec<[usize; 2]>, PhysicsError> {
        // Authored f32 positions put flush faces within rounding of each other.
        const BUILT_TOUCHING: f64 = 1e-6;
        let mut mechanism = vec![usize::MAX; self.bodies];
        for (index, component) in creation
            .loop_topology
            .mechanism_components
            .iter()
            .enumerate()
        {
            for &body in component {
                *mechanism
                    .get_mut(body as usize)
                    .ok_or(PhysicsError::InvalidCollision)? = index;
            }
        }
        let built = crate::MachineState::at_rest(creation);
        let shapes = self
            .colliders
            .iter()
            .map(|collider| {
                let pose = built.poses[collider.body];
                collider
                    .local
                    .transformed(pose.position, pose.rotation)
                    .map_err(|_| PhysicsError::InvalidCollision)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let bounds = shapes
            .iter()
            .map(|shape| {
                let [minimum, maximum] = shape.bounds();
                [
                    minimum - DVec3::splat(BUILT_TOUCHING),
                    maximum + DVec3::splat(BUILT_TOUCHING),
                ]
            })
            .collect::<Vec<_>>();
        let mut flush = Vec::new();
        for &[first, second] in self.candidate_pairs(&bounds).iter() {
            let bodies = [self.colliders[first].body, self.colliders[second].body];
            let pair = [bodies[0].min(bodies[1]), bodies[0].max(bodies[1])];
            if mechanism[pair[0]] != mechanism[pair[1]]
                || mechanism[pair[0]] == usize::MAX
                || flush.contains(&pair)
            {
                continue;
            }
            let separation = shapes[first]
                .convex_separation(&shapes[second])
                .map_err(|_| PhysicsError::InvalidCollision)?;
            if separation.separation <= BUILT_TOUCHING {
                flush.push(pair);
            }
        }
        Ok(flush)
    }

    // Transform only eight aggregate corners per body. The speed envelope is
    // over the actual generalized path, including ancestor and suspension motion.
    pub(super) fn body_path_bounds(
        &self,
        motion: &crate::MachineMotion<'_>,
        padding: f64,
    ) -> Result<Vec<[DVec3; 2]>, PhysicsError> {
        self.body_bounds
            .iter()
            .enumerate()
            .map(|(body, local)| {
                if self.body_colliders[body].is_empty() {
                    return Ok(*local);
                }
                let pose = motion.initial_poses()[body];
                let mut bounds = [DVec3::INFINITY, DVec3::NEG_INFINITY];
                for x in [local[0].x, local[1].x] {
                    for y in [local[0].y, local[1].y] {
                        for z in [local[0].z, local[1].z] {
                            let point = pose.position + pose.rotation * DVec3::new(x, y, z);
                            bounds[0] = bounds[0].min(point);
                            bounds[1] = bounds[1].max(point);
                        }
                    }
                }
                let bound = motion.bounds()[body];
                let radius = self.body_radii[body];
                let travel = DVec3::splat((bound.point_speed(radius) + padding).next_up());
                let reach = DVec3::splat((bound.origin_speed + radius + padding).next_up());
                let result = [
                    (bounds[0] - travel)
                        .max(pose.position - reach)
                        .map(f64::next_down),
                    (bounds[1] + travel)
                        .min(pose.position + reach)
                        .map(f64::next_up),
                ];
                if result.iter().all(|b| b.is_finite()) {
                    Ok(result)
                } else {
                    Err(PhysicsError::InvalidCollision)
                }
            })
            .collect()
    }

    pub(super) fn body_candidates(&self, bounds: &[[DVec3; 2]]) -> Vec<[usize; 2]> {
        let mut nodes = Vec::new();
        let mut stack = Vec::new();
        let mut found = Vec::new();
        let mut pairs = Vec::new();
        self.body_tree.refit(bounds, &mut nodes);
        for (a, &body_bounds) in bounds.iter().enumerate() {
            if self.body_colliders[a].is_empty() {
                continue;
            }
            found.clear();
            self.body_tree
                .query(&nodes, body_bounds, &mut stack, &mut found);
            for &b in &found {
                if b > a
                    && self.suppressed.binary_search(&[a, b]).is_err()
                    && (self.colliders[self.body_colliders[a][0]].moving
                        || self.colliders[self.body_colliders[b][0]].moving)
                {
                    pairs.push([a, b]);
                }
            }
        }
        pairs
    }

    // Collider pairs that may touch within `bounds`, in sorted order: on
    // different bodies, at least one moving, and not joined by a bearing.
    // Body traversal rejects suppressed pairs before immutable collider trees.
    pub(super) fn candidate_pairs(&self, bounds: &[[DVec3; 2]]) -> CandidatePairs<'_> {
        self.candidate_pairs_groups(bounds, None)
    }

    pub(super) fn candidate_pairs_groups(
        &self,
        bounds: &[[DVec3; 2]],
        groups: Option<&ContactGroups>,
    ) -> CandidatePairs<'_> {
        let mut scratch = self
            .pair_scratch
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let PairScratch {
            body_bounds,
            body_nodes,
            body_candidates,
            trees,
            refitted,
            stack,
            pairs,
            pair_stack,
            node_pair_tests,
            ..
        } = &mut *scratch;
        body_bounds.clear();
        body_bounds.extend(self.body_colliders.iter().map(|rows| {
            rows.iter()
                .fold([DVec3::INFINITY, DVec3::NEG_INFINITY], |[lo, hi], &row| {
                    [lo.min(bounds[row][0]), hi.max(bounds[row][1])]
                })
        }));
        self.body_tree.refit(body_bounds, body_nodes);
        pairs.clear();
        *node_pair_tests = 0;
        trees.resize_with(self.bodies, Vec::new);
        refitted.resize(self.bodies, false);
        refitted.fill(false);
        for a in 0..self.bodies {
            if self.body_colliders[a].is_empty() {
                continue;
            }
            body_candidates.clear();
            self.body_tree
                .query(body_nodes, body_bounds[a], stack, body_candidates);
            for &b in body_candidates.iter().filter(|&&b| b > a) {
                let body_pair = [a.min(b), a.max(b)];
                if groups.is_some_and(|g| !g.includes(self, a, Some(b)))
                    || self.suppressed.binary_search(&body_pair).is_ok()
                    || !(self.colliders[self.body_colliders[a][0]].moving
                        || self.colliders[self.body_colliders[b][0]].moving)
                    || !overlaps(body_bounds[a], body_bounds[b])
                {
                    continue;
                }
                // Whole-body rejection comes first: unrelated distant bodies
                // do not need their detailed collider hierarchy rebuilt.
                for body in [a, b] {
                    if !refitted[body] {
                        self.collider_trees[body].refit(bounds, &mut trees[body]);
                        refitted[body] = true;
                    }
                }
                *node_pair_tests += self.collider_trees[a].pairs(
                    &trees[a],
                    &self.collider_trees[b],
                    &trees[b],
                    pair_stack,
                    pairs,
                );
            }
        }
        pairs.sort_unstable();
        CandidatePairs { scratch }
    }
}
