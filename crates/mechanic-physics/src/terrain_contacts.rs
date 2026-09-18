//! Finite terrain manifolds from immutable colliders and the world's shared BVHs.

use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    sync::{Arc, Mutex, OnceLock},
};

use bevy_math::{DVec3, Vec3};
use mechanic_core::{
    CompiledCreation, ContactCylinder, ContactPolytope, ConvexFeature, ConvexSeparation,
    MaterialProperties, TriangleContactPoint, TriangleSupport,
};
use mechanic_world::{
    TerrainCollisionChunk, TerrainNodeId, TerrainSpatialIndex, WorldBounds, WorldPosition,
};

use crate::{BodyPose, PhysicsError};

mod broadphase;
mod groups;
pub(crate) use groups::{ContactGroups, Measured};
mod penetration;
pub use penetration::{TerrainPathFailure, TerrainPathOutcome, TerrainPathQuery};
mod sweep;
pub(crate) use sweep::SlowContactMotion;
pub use sweep::{TerrainSweepHit, TerrainSweepOutcome, TerrainSweepQuery};
mod constraints;
pub use constraints::TerrainImpactConstraints;

/// Geometry opposing a collider at one contact point.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ContactObstacle {
    /// A finite triangle of published terrain.
    Terrain {
        /// Owning terrain node.
        node: TerrainNodeId,
        /// Chunk geometry generation supplied by the world.
        geometry_generation: u64,
        /// Publication that selected this chunk's active groups/materials.
        publication_generation: u64,
        /// Stable triangle row within the immutable BVH.
        triangle: usize,
    },
    /// Another collider row of the same construction, on a different body.
    Collider(usize),
}

/// Geometry a continuous query found approaching, independent of generations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContactTarget {
    /// A terrain triangle.
    Terrain {
        /// Owning terrain chunk.
        node: TerrainNodeId,
        /// Triangle row in the chunk's immutable BVH.
        triangle: usize,
    },
    /// Another collider row of the same construction.
    Collider(usize),
}

impl ContactObstacle {
    /// The approached geometry, without the generations that published it.
    pub fn target(self) -> ContactTarget {
        match self {
            Self::Terrain { node, triangle, .. } => ContactTarget::Terrain { node, triangle },
            Self::Collider(collider) => ContactTarget::Collider(collider),
        }
    }
}

/// Stable source identity; replacing a chunk or topology invalidates its points.
/// A retained corner is refreshed against geometry before any future reuse.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct TerrainContactFeature {
    /// Compiled construction generation.
    pub topology_generation: u64,
    /// Collider row receiving the contact normal.
    pub collider: usize,
    /// Opposing terrain triangle or collider.
    pub obstacle: ContactObstacle,
    /// Retained finite polygon corner, requiring geometry refresh before reuse.
    pub corner: usize,
}

impl TerrainContactFeature {
    /// Whether this point touches the geometry a continuous query reported for
    /// `collider`. A collider pair matches in either order: which of the two
    /// receives the normal depends on the pose, not on the approach.
    pub fn touches(&self, collider: usize, target: ContactTarget) -> bool {
        let own = self.obstacle.target();
        (self.collider == collider && own == target)
            || matches!(
                (own, target),
                (ContactTarget::Collider(other), ContactTarget::Collider(approached))
                    if other == collider && approached == self.collider
            )
    }
}

// Numerical zero for actual finite opposing points, never a speculative margin.
pub(crate) const CONTACT_ACTIVATION_DISTANCE: f64 = 1e-12;

// Numerical zero between two bodies' solids. Their separating-axis gap comes from
// two independently rounded, moving polytopes, and a body rolling over another's
// edge closes its last fraction of a micron far slower than any conservative
// bound, so a terrain-precision window exhausts the event search. The GPU emits
// body contacts from 1e-5 m apart; this stays ten times tighter.
pub(crate) const PAIR_ACTIVATION_DISTANCE: f64 = 1e-6;

impl ContactTarget {
    /// Largest gap at which a contact with this target counts as touching.
    pub(crate) const fn activation_distance(self) -> f64 {
        match self {
            Self::Terrain { .. } => CONTACT_ACTIVATION_DISTANCE,
            Self::Collider(_) => PAIR_ACTIVATION_DISTANCE,
        }
    }
}

/// Actual finite support against terrain or another body, with mixed materials.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TerrainContact {
    /// Generations and geometric source of this point.
    pub feature: TerrainContactFeature,
    /// Compound body receiving the contact impulse along `normal`.
    pub body: usize,
    /// Body carrying the opposing surface, which receives the opposite impulse.
    /// None for terrain.
    pub other_body: Option<usize>,
    /// Query-local manifold number within this collider/obstacle, for coupled block solving.
    pub manifold: usize,
    /// Opposing surface point, relative to the query's floating origin.
    pub terrain_point: DVec3,
    /// Receiving convex point, relative to the same origin.
    pub body_point: DVec3,
    /// Opposing surface's outward unit normal, toward the receiving body.
    pub normal: DVec3,
    /// Current penetration along the normal, in metres.
    pub depth: f64,
    /// Signed normal gap between the true opposing points; negative in overlap.
    pub separation: f64,
    /// Static/kinetic friction, restitution, rolling resistance, in that order.
    pub response: [f64; 4],
}

/// Retained manifolds and actual collision query work. No solve is implied.
#[derive(Clone, Debug, Default)]
pub struct TerrainContactQuery {
    /// Contacts sorted by stable source identity.
    pub contacts: Vec<TerrainContact>,
    /// One numerical-zero contact witness per source triangle, before manifold
    /// reduction. Event arrival must not depend on which support corners survive.
    pub activation_features: Vec<TerrainContactFeature>,
    /// Chunk candidates visited across all moving colliders.
    pub chunk_candidates: usize,
    /// Triangle candidates passed to finite narrowphase.
    pub triangle_candidates: usize,
    /// Point count before cross-triangle manifold reduction.
    pub unreduced_points: usize,
    /// Collider pairs on different bodies whose bounds overlap.
    pub collider_pair_candidates: usize,
}

struct Collider {
    body: usize,
    center: DVec3,
    material: MaterialProperties,
    // Circumscribing prism for a cylinder: every bound, sweep and body pair uses
    // it, since it contains the cylinder.
    local: ContactPolytope,
    // The exact cylinder, for rolling terrain contact.
    round: Option<ContactCylinder>,
    bounds: [DVec3; 2],
    moving: bool,
    radius: f64,
}

// Frequently traversed motion data stays separate from the larger shape records.
struct MotionCollider {
    row: usize,
    body: usize,
    assembly: usize,
    radius: f64,
    round: Option<ContactCylinder>,
}

// Bounds on how fast a collider's surface moves against terrain along a path,
// and how fast it turns. A cylinder meets terrain as a circle, so its turn
// about its own axis counts as neither.
fn terrain_motion(
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
    generation: u64,
    bodies: usize,
    colliders: Vec<Collider>,
    reach: Vec<(usize, f64)>,
    motion_colliders: Vec<MotionCollider>,
    // Sorted body pairs joined by a bearing, which never collide with each other.
    suppressed: Vec<[usize; 2]>,
    body_colliders: Vec<Vec<usize>>,
    body_bounds: Vec<[DVec3; 2]>,
    body_radii: Vec<f64>,
    // Bodies whose colliders are all cylinders.
    round_bodies: Vec<bool>,
    pub(crate) assemblies: Vec<usize>,
    pub(crate) assembly_count: usize,
    pub(crate) internal_collisions: Vec<bool>,
    // Assemblies with at most one moving tree root, whose bodies' root-frame
    // motion bounds every distance between them.
    pub(crate) tree_frames: Vec<bool>,
    pub(crate) moving_assemblies: Vec<usize>,
    collider_trees: Vec<broadphase::Tree>,
    body_tree: broadphase::Tree,
    pair_scratch: Mutex<PairScratch>,
    sweep_scratch: Mutex<sweep::SweepScratch>,
    cache: Mutex<PoseCache>,
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
    #[allow(clippy::too_many_lines)] // Compile immutable geometry and both hierarchy levels together.
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
    fn built_flush(&self, creation: &CompiledCreation) -> Result<Vec<[usize; 2]>, PhysicsError> {
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
    fn body_path_bounds(
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

    fn body_candidates(&self, bounds: &[[DVec3; 2]]) -> Vec<[usize; 2]> {
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
    fn candidate_pairs(&self, bounds: &[[DVec3; 2]]) -> CandidatePairs<'_> {
        self.candidate_pairs_groups(bounds, None)
    }

    fn candidate_pairs_groups(
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

// Collider/collider friction mixing, matching the GPU route.
fn mixed_response(first: MaterialProperties, second: MaterialProperties) -> [f64; 4] {
    [
        (f64::from(first.static_friction) * f64::from(second.static_friction)).sqrt(),
        (f64::from(first.dynamic_friction) * f64::from(second.dynamic_friction)).sqrt(),
        f64::from(first.restitution.max(second.restitution)),
        (f64::from(first.rolling_resistance) * f64::from(second.rolling_resistance)).sqrt(),
    ]
}

struct Chunk {
    publication: u64,
    geometry: Arc<TerrainCollisionChunk>,
}

/// CPU collision view sharing immutable world-owned chunks. Updates change only
/// the affected ancestor paths; unchanged triangle BVHs and allocations survive.
/// This view does not establish residency readiness or perform a physics tick.
#[derive(Default)]
pub struct TerrainContactScene {
    generation: u64,
    chunks: BTreeMap<TerrainNodeId, Chunk>,
    index: TerrainSpatialIndex,
}

#[derive(Clone, Copy)]
enum QueryKind {
    Surface,
    Activation,
    Recovery,
    BuriedVertices,
}

// How a solid cylinder meets terrain. The soft-step solver rolls it on the exact
// circle. The exact reference solver keeps the prism, because its event search
// and sweep certificates are built on the same polytope.
#[derive(Clone, Copy, PartialEq, Eq)]
enum CylinderContact {
    Analytic,
    Prism,
    // Left out of a buried-vertex query: an exact cylinder's surface points
    // already carry their true depth.
    Omitted,
}

impl TerrainContactScene {
    /// Atomically publishes changed chunks/removals at an increasing generation.
    /// Geometry and hierarchy checks run before changing the visible scene.
    ///
    /// # Errors
    /// Rejects stale generations, duplicate nodes, malformed BVHs or materials.
    pub fn publish(
        &mut self,
        generation: u64,
        upserts: &[Arc<TerrainCollisionChunk>],
        removed: &[TerrainNodeId],
    ) -> Result<(), PhysicsError> {
        if generation <= self.generation {
            return Err(PhysicsError::InvalidCollision);
        }
        let mut seen = BTreeSet::new();
        for &node in removed {
            if !seen.insert(node) {
                return Err(PhysicsError::InvalidCollision);
            }
        }
        for chunk in upserts {
            if !seen.insert(chunk.node) {
                return Err(PhysicsError::InvalidCollision);
            }
            if self
                .chunks
                .get(&chunk.node)
                .is_some_and(|old| chunk.generation < old.geometry.generation)
            {
                return Err(PhysicsError::InvalidCollision);
            }
            validate_chunk(chunk)?;
        }
        for node in removed {
            self.chunks.remove(node);
            self.index.remove(*node);
        }
        for chunk in upserts {
            self.index.insert_bounds(
                chunk.node,
                chunk
                    .triangle_bvh
                    .nodes
                    .first()
                    .map_or(chunk.bounds, |node| node.bounds),
            );
            self.chunks.insert(
                chunk.node,
                Chunk {
                    publication: generation,
                    geometry: Arc::clone(chunk),
                },
            );
        }
        self.generation = generation;
        Ok(())
    }

    /// Latest complete terrain publication.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Computes finite manifolds at reconstructed body poses. A zero contact
    /// result establishes only separation at this pose, not continuous safety.
    ///
    /// # Errors
    /// Rejects invalid poses, origin, or triangle query geometry.
    pub fn contacts(
        &self,
        machine: &MachineCollisionGeometry,
        poses: &[BodyPose],
        origin: DVec3,
    ) -> Result<TerrainContactQuery, PhysicsError> {
        self.proximity(machine, poses, origin, 0.0)
    }

    /// Refreshes finite surface points within a fixed normal separation margin.
    /// Geometry is extruded only along each triangle's normal; tangential holes
    /// remain open. Positive gaps are explicit and must not imply an impact.
    ///
    /// # Errors
    /// Rejects invalid poses, origins, triangles, or negative/non-finite margins.
    pub fn proximity(
        &self,
        machine: &MachineCollisionGeometry,
        poses: &[BodyPose],
        origin: DVec3,
        margin: f64,
    ) -> Result<TerrainContactQuery, PhysicsError> {
        self.query(
            machine,
            poses,
            origin,
            &vec![margin; machine.colliders.len()],
            QueryKind::Surface,
        )
    }

    pub(crate) fn proximity_groups(
        &self,
        machine: &MachineCollisionGeometry,
        poses: &[BodyPose],
        origin: DVec3,
        margins: &[f64],
        groups: Option<&ContactGroups>,
    ) -> Result<TerrainContactQuery, PhysicsError> {
        self.query_groups(
            machine,
            poses,
            origin,
            margins,
            (QueryKind::Surface, CylinderContact::Analytic),
            groups,
        )
    }

    // Buried points a clipped manifold misses, for every collider but an exact
    // cylinder, whose surface points already carry their true depth.
    pub(crate) fn buried_groups(
        &self,
        machine: &MachineCollisionGeometry,
        poses: &[BodyPose],
        origin: DVec3,
        groups: Option<&ContactGroups>,
    ) -> Result<TerrainContactQuery, PhysicsError> {
        self.query_groups(
            machine,
            poses,
            origin,
            &vec![CONTACT_ACTIVATION_DISTANCE; machine.colliders.len()],
            (QueryKind::BuriedVertices, CylinderContact::Omitted),
            groups,
        )
    }

    pub(crate) fn recovery_groups(
        &self,
        machine: &MachineCollisionGeometry,
        poses: &[BodyPose],
        origin: DVec3,
        groups: Option<&ContactGroups>,
    ) -> Result<TerrainContactQuery, PhysicsError> {
        self.query_groups(
            machine,
            poses,
            origin,
            &vec![CONTACT_ACTIVATION_DISTANCE; machine.colliders.len()],
            (QueryKind::Recovery, CylinderContact::Analytic),
            groups,
        )
    }

    // Preserve actual intersection manifolds. Only separated pairs need the
    // numerical-zero proximity query for event activation.
    pub(crate) fn activation_contacts(
        &self,
        machine: &MachineCollisionGeometry,
        poses: &[BodyPose],
        origin: DVec3,
    ) -> Result<TerrainContactQuery, PhysicsError> {
        self.query(
            machine,
            poses,
            origin,
            &vec![CONTACT_ACTIVATION_DISTANCE; machine.colliders.len()],
            QueryKind::Activation,
        )
    }

    pub(crate) fn recovery_contacts(
        &self,
        machine: &MachineCollisionGeometry,
        poses: &[BodyPose],
        origin: DVec3,
    ) -> Result<TerrainContactQuery, PhysicsError> {
        self.query(
            machine,
            poses,
            origin,
            &vec![CONTACT_ACTIVATION_DISTANCE; machine.colliders.len()],
            QueryKind::Recovery,
        )
    }

    // Resolve terrain nodes once per body, then descend its collider hierarchy.
    // Bounds are in world coordinates with the same arithmetic as the old
    // per-collider lookup, so node membership and stable ordering are unchanged.
    fn prepare_terrain_candidates(
        &self,
        machine: &MachineCollisionGeometry,
        cache: &mut PoseCache,
        origin: DVec3,
        margins: &[f64],
        groups: Option<&ContactGroups>,
    ) -> Result<(), PhysicsError> {
        cache
            .terrain_candidates
            .resize_with(machine.colliders.len(), Vec::new);
        for nodes in &mut cache.terrain_candidates {
            nodes.clear();
        }
        cache.terrain_bounds.clear();
        cache
            .terrain_bounds
            .extend(
                cache
                    .bounds
                    .iter()
                    .zip(margins)
                    .map(|(&[minimum, maximum], &margin)| {
                        [
                            origin + minimum - DVec3::splat(margin),
                            origin + maximum + DVec3::splat(margin),
                        ]
                    }),
            );
        let mut scratch = machine
            .pair_scratch
            .lock()
            .map_err(|_| PhysicsError::InvalidCollision)?;
        let PairScratch {
            trees,
            stack,
            candidates,
            ..
        } = &mut *scratch;
        trees.resize_with(machine.bodies, Vec::new);
        let reaches = terrain_reaches(machine, &cache.terrain_bounds, groups)?;
        let Some(reach) = reaches
            .iter()
            .map(|&(_, bounds)| bounds)
            .reduce(|a, b| WorldBounds {
                minimum: WorldPosition(a.minimum.0.min(b.minimum.0)),
                maximum: WorldPosition(a.maximum.0.max(b.maximum.0)),
            })
        else {
            return Ok(());
        };
        // One index traversal serves every body; each keeps the nodes whose
        // indexed mesh bounds it overlaps, in the index's order.
        let nodes = self
            .index
            .bounds_candidates(reach)
            .into_iter()
            .map(|node| {
                let chunk = &self.chunks[&node].geometry;
                let bounds = chunk
                    .triangle_bvh
                    .nodes
                    .first()
                    .map_or(chunk.bounds, |node| node.bounds);
                (node, bounds)
            })
            .collect::<Vec<_>>();
        for (body, reach) in reaches {
            let mut refitted = false;
            for &(node, bounds) in &nodes {
                if !reach.intersects(bounds) {
                    continue;
                }
                if !refitted {
                    machine.collider_trees[body].refit(&cache.terrain_bounds, &mut trees[body]);
                    refitted = true;
                }
                candidates.clear();
                machine.collider_trees[body].query(
                    &trees[body],
                    [bounds.minimum.0, bounds.maximum.0],
                    stack,
                    candidates,
                );
                for &row in candidates.iter() {
                    cache.terrain_candidates[row].push(node);
                }
            }
        }
        Ok(())
    }

    fn query(
        &self,
        machine: &MachineCollisionGeometry,
        poses: &[BodyPose],
        origin: DVec3,
        margins: &[f64],
        kind: QueryKind,
    ) -> Result<TerrainContactQuery, PhysicsError> {
        self.query_groups(
            machine,
            poses,
            origin,
            margins,
            (kind, CylinderContact::Prism),
            None,
        )
    }

    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    fn query_groups(
        &self,
        machine: &MachineCollisionGeometry,
        poses: &[BodyPose],
        origin: DVec3,
        margins: &[f64],
        (kind, cylinders): (QueryKind, CylinderContact),
        groups: Option<&ContactGroups>,
    ) -> Result<TerrainContactQuery, PhysicsError> {
        if poses.len() != machine.bodies
            || !origin.is_finite()
            || margins.len() != machine.colliders.len()
            || margins
                .iter()
                .any(|margin| !margin.is_finite() || *margin < 0.0)
        {
            return Err(PhysicsError::InvalidCollision);
        }
        let mut result = TerrainContactQuery::default();
        let mut cache = machine
            .cache
            .lock()
            .map_err(|_| PhysicsError::InvalidCollision)?;
        cache.update(machine, poses)?;
        self.prepare_terrain_candidates(machine, &mut cache, origin, margins, groups)?;
        let shapes = &cache;
        for (collider_row, collider) in machine.colliders.iter().enumerate() {
            if !collider.moving
                || groups.is_some_and(|g| !g.includes(machine, collider.body, None))
                || (cylinders == CylinderContact::Omitted && collider.round.is_some())
            {
                continue;
            }
            let margin = margins[collider_row];
            let pose = poses[collider.body];

            let center = pose.position + pose.rotation * collider.center;
            let [minimum, maximum] = shapes.bounds[collider_row];
            let bounds = WorldBounds {
                minimum: WorldPosition(origin + minimum - DVec3::splat(margin)),
                maximum: WorldPosition(origin + maximum + DVec3::splat(margin)),
            };
            if !valid_bounds(bounds) {
                return Err(PhysicsError::InvalidCollision);
            }
            let mut groups = Vec::<SupportGroup>::new();
            let mut rolling = Vec::new();
            let round = shapes.rounds[collider_row]
                .as_ref()
                .filter(|_| cylinders == CylinderContact::Analytic);
            // A triangle beside a cylinder's lowest line waits until every point
            // on that line is known: a deeper one of its group discards all of
            // its flank points (see `rolling_supports`).
            let flanks =
                round.is_some() && !matches!(kind, QueryKind::Recovery | QueryKind::BuriedVertices);
            let mut supports = Vec::new();
            let candidates = &shapes.terrain_candidates[collider_row];
            result.chunk_candidates += candidates.len();
            for &node in candidates {
                let published = &self.chunks[&node];
                let chunk = &published.geometry;
                for row in chunk.bounds_candidates(bounds) {
                    let indices = chunk.triangle_bvh.triangles[row].indices;
                    let triangle = indices.map(|index| {
                        chunk.origin.0 - origin
                            + Vec3::from_array(chunk.vertices[index as usize]).as_dvec3()
                    });
                    // BVH leaves contain batches of triangles. Reject their
                    // individual bounds before transforming or clipping a convex.
                    let triangle_bounds = triangle.iter().fold(
                        [DVec3::INFINITY, DVec3::NEG_INFINITY],
                        |[lo, hi], &point| [lo.min(point), hi.max(point)],
                    );
                    let expanded = [
                        (minimum - DVec3::splat(margin)).map(f64::next_down),
                        (maximum + DVec3::splat(margin)).map(f64::next_up),
                    ];
                    if !overlaps(expanded, triangle_bounds) {
                        continue;
                    }
                    result.triangle_candidates += 1;
                    // Zero-area triangles carry no surface. Small but finite
                    // triangles still count; an area cutoff could open a hole.
                    if (triangle[1] - triangle[0])
                        .cross(triangle[2] - triangle[0])
                        .try_normalize()
                        .is_none()
                    {
                        continue;
                    }
                    let support = if let Some(cylinder) = round {
                        // Every query kind reports true depths here, so a
                        // recovery query needs no separate buried vertices.
                        let support = if flanks {
                            cylinder.triangle_support(triangle, margin)
                        } else {
                            cylinder
                                .triangle_contacts(triangle, margin)
                                .map(TriangleSupport::Points)
                        };
                        support.map_err(|_| PhysicsError::InvalidCollision)?
                    } else {
                        let shape = shapes.shape(machine, poses, collider_row)?;
                        TriangleSupport::Points(surface_points_with_scratch(
                            shape,
                            triangle,
                            kind,
                            margin,
                            CONTACT_ACTIVATION_DISTANCE,
                            &mut shapes.clipping.borrow_mut(),
                        )?)
                    };
                    if matches!(&support, TriangleSupport::Points(points) if points.is_empty()) {
                        continue;
                    }
                    let surface = chunk
                        .triangle_surface_response(indices)
                        .map_err(|_| PhysicsError::InvalidCollision)?
                        .to_array()
                        .map(f64::from);
                    let material = collider.material;
                    let response = [
                        (f64::from(material.static_friction) * surface[0]).sqrt(),
                        (f64::from(material.dynamic_friction) * surface[1]).sqrt(),
                        f64::from(material.restitution).max(surface[2]),
                        (f64::from(material.rolling_resistance) * surface[3]).sqrt(),
                    ];
                    supports.push((node, published, row, triangle, response, support));
                }
            }
            let lines = round.filter(|_| flanks).map_or_else(Vec::new, |cylinder| {
                supports
                    .iter()
                    .filter_map(|(.., response, support)| match support {
                        TriangleSupport::Points(points) => Some((response, points)),
                        TriangleSupport::Flank(_) => None,
                    })
                    .flat_map(|(&response, points)| {
                        points.iter().map(move |point| (response, point))
                    })
                    .filter(|(_, point)| on_lowest_line(cylinder, point.normal, point.body_point))
                    .map(|(response, point)| {
                        let surface = Surface {
                            normal: point.normal,
                            distance: point.normal.dot(point.triangle_point),
                            response,
                        };
                        (
                            surface,
                            (point.body_point - point.triangle_point).dot(point.normal),
                        )
                    })
                    .collect::<Vec<_>>()
            });
            for (node, published, row, triangle, response, support) in supports {
                let points = match (support, round) {
                    (TriangleSupport::Points(points), _) => points,
                    (TriangleSupport::Flank(bound), Some(cylinder)) => {
                        let normal = (triangle[1] - triangle[0])
                            .cross(triangle[2] - triangle[0])
                            .normalize();
                        let flank = Surface {
                            normal,
                            distance: normal.dot(triangle[0]),
                            response,
                        };
                        if bound > CONTACT_ACTIVATION_DISTANCE
                            && lines.iter().any(|&(line, separation)| {
                                separation <= bound && same_support(line, flank, center).is_some()
                            })
                        {
                            continue;
                        }
                        cylinder
                            .triangle_contacts(triangle, margin)
                            .map_err(|_| PhysicsError::InvalidCollision)?
                    }
                    (TriangleSupport::Flank(_), None) => continue,
                };
                let chunk = &published.geometry;
                let opposing = match round {
                    Some(cylinder) => Opposing::Cylinder(cylinder),
                    None => Opposing::Polytope(shapes.shape(machine, poses, collider_row)?),
                };
                let mut activation_recorded = false;
                for (corner, point) in points.into_iter().enumerate() {
                    result.unreduced_points += 1;
                    let contact = TerrainContact {
                        feature: TerrainContactFeature {
                            topology_generation: machine.generation,
                            collider: collider_row,
                            obstacle: ContactObstacle::Terrain {
                                node,
                                geometry_generation: chunk.generation,
                                publication_generation: published.publication,
                                triangle: row,
                            },
                            corner,
                        },
                        body: collider.body,
                        other_body: None,
                        manifold: 0,
                        terrain_point: point.triangle_point,
                        body_point: point.body_point,
                        normal: point.normal,
                        depth: point.depth,
                        separation: (point.body_point - point.triangle_point).dot(point.normal),
                        response,
                    };
                    if !activation_recorded && contact.separation <= CONTACT_ACTIVATION_DISTANCE {
                        result.activation_features.push(contact.feature);
                        activation_recorded = true;
                    }
                    if matches!(kind, QueryKind::Recovery | QueryKind::BuriedVertices) {
                        result.contacts.push(contact);
                    } else if matches!(opposing, Opposing::Cylinder(_)) {
                        rolling.push(contact);
                    } else {
                        reduce_support(
                            &mut groups,
                            &mut result.contacts,
                            contact,
                            opposing,
                            center,
                        );
                    }
                }
            }
            if let Some(cylinder) = shapes.rounds[collider_row].as_ref() {
                for contact in rolling_supports(&rolling, cylinder, center) {
                    reduce_support(
                        &mut groups,
                        &mut result.contacts,
                        contact,
                        Opposing::Cylinder(cylinder),
                        center,
                    );
                }
            }
            for (manifold, group) in groups.into_iter().enumerate() {
                group.append_unique(manifold, &mut result.contacts);
            }
        }
        // Bodies of one construction against each other. Terrain triangles above
        // are one-sided surfaces; here both sides are solids, and one separating
        // axis per pair selects the single face or edge that supplies the normal.
        let reach = |row: usize| match kind {
            QueryKind::Surface => margins[row],
            QueryKind::Activation | QueryKind::Recovery | QueryKind::BuriedVertices => {
                PAIR_ACTIVATION_DISTANCE
            }
        };
        let bounds = shapes
            .bounds
            .iter()
            .enumerate()
            .map(|(row, &[minimum, maximum])| {
                [
                    minimum - DVec3::splat(reach(row)),
                    maximum + DVec3::splat(reach(row)),
                ]
            })
            .collect::<Vec<_>>();
        for &[first, second] in machine.candidate_pairs_groups(&bounds, groups).iter() {
            if groups.is_some_and(|g| {
                !g.includes(
                    machine,
                    machine.colliders[first].body,
                    Some(machine.colliders[second].body),
                )
            }) {
                continue;
            }
            result.collider_pair_candidates += 1;
            let reach = match kind {
                // The faster collider's margin already covers its own travel.
                QueryKind::Surface => margins[first].max(margins[second]),
                QueryKind::Activation | QueryKind::Recovery | QueryKind::BuriedVertices => {
                    PAIR_ACTIVATION_DISTANCE
                }
            };
            let first_shape = shapes.shape(machine, poses, first)?;
            let second_shape = shapes.shape(machine, poses, second)?;
            let mut separations = shapes.separations.borrow_mut();
            let cached = separations.get(&[first, second]).copied();
            let separation = match cached {
                Some((_, Some(separation))) => Some(separation),
                Some((margin, None)) if margin >= reach => None,
                _ => {
                    let separation = first_shape
                        .convex_separation_within(second_shape, reach)
                        .map_err(|_| PhysicsError::InvalidCollision)?;
                    separations.insert([first, second], (reach, separation));
                    separation
                }
            };
            drop(separations);
            let Some(separation) = separation else {
                continue;
            };
            if separation.separation > reach {
                continue;
            }
            let (receiving, opposing, points) = pair_points(
                [first_shape, second_shape],
                [first, second],
                separation,
                kind,
                reach,
            )?;
            let collider = &machine.colliders[receiving];
            let pose = poses[collider.body];
            let center = pose.position + pose.rotation * collider.center;
            let response = mixed_response(collider.material, machine.colliders[opposing].material);
            let mut groups = Vec::<SupportGroup>::new();
            let mut activation_recorded = false;
            for (corner, point) in points.into_iter().enumerate() {
                result.unreduced_points += 1;
                let contact = TerrainContact {
                    feature: TerrainContactFeature {
                        topology_generation: machine.generation,
                        collider: receiving,
                        obstacle: ContactObstacle::Collider(opposing),
                        corner,
                    },
                    body: collider.body,
                    other_body: Some(machine.colliders[opposing].body),
                    manifold: 0,
                    terrain_point: point.triangle_point,
                    body_point: point.body_point,
                    normal: point.normal,
                    depth: point.depth,
                    separation: (point.body_point - point.triangle_point).dot(point.normal),
                    response,
                };
                if !activation_recorded && contact.separation <= PAIR_ACTIVATION_DISTANCE {
                    result.activation_features.push(contact.feature);
                    activation_recorded = true;
                }
                if matches!(kind, QueryKind::Recovery | QueryKind::BuriedVertices) {
                    result.contacts.push(contact);
                } else {
                    reduce_support(
                        &mut groups,
                        &mut result.contacts,
                        contact,
                        Opposing::Polytope(shapes.shape(machine, poses, receiving)?),
                        center,
                    );
                }
            }
            for (manifold, group) in groups.into_iter().enumerate() {
                group.append_unique(manifold, &mut result.contacts);
            }
        }
        result.contacts.sort_by_key(|contact| contact.feature);
        Ok(result)
    }
}

// Each moving body the query covers, with the bounds of its colliders' reach.
fn terrain_reaches(
    machine: &MachineCollisionGeometry,
    reaches: &[[DVec3; 2]],
    groups: Option<&ContactGroups>,
) -> Result<Vec<(usize, WorldBounds)>, PhysicsError> {
    let mut bodies = Vec::new();
    for (body, rows) in machine.body_colliders.iter().enumerate() {
        if rows.is_empty()
            || !machine.colliders[rows[0]].moving
            || groups.is_some_and(|g| !g.includes(machine, body, None))
        {
            continue;
        }
        let [minimum, maximum] = rows
            .iter()
            .fold([DVec3::INFINITY, DVec3::NEG_INFINITY], |[lo, hi], &row| {
                [lo.min(reaches[row][0]), hi.max(reaches[row][1])]
            });
        let bounds = WorldBounds {
            minimum: WorldPosition(minimum),
            maximum: WorldPosition(maximum),
        };
        if !valid_bounds(bounds) {
            return Err(PhysicsError::InvalidCollision);
        }
        bodies.push((body, bounds));
    }
    Ok(bodies)
}

// Finite points of one convex against one triangle for the requested query.
fn surface_points(
    shape: &ContactPolytope,
    triangle: [DVec3; 3],
    kind: QueryKind,
    margin: f64,
    window: f64,
) -> Result<Vec<TriangleContactPoint>, PhysicsError> {
    surface_points_with_scratch(
        shape,
        triangle,
        kind,
        margin,
        window,
        &mut mechanic_core::TriangleClipScratch::default(),
    )
}

fn surface_points_with_scratch(
    shape: &ContactPolytope,
    triangle: [DVec3; 3],
    kind: QueryKind,
    margin: f64,
    window: f64,
    scratch: &mut mechanic_core::TriangleClipScratch,
) -> Result<Vec<TriangleContactPoint>, PhysicsError> {
    match kind {
        QueryKind::Activation => activation_points_with_scratch(shape, triangle, window, scratch),
        QueryKind::Surface => shape
            .triangle_proximity_with_scratch(triangle, margin, scratch)
            .map_err(|_| PhysicsError::InvalidCollision),
        QueryKind::BuriedVertices => shape
            .triangle_buried_vertices_with_scratch(triangle, scratch)
            .map_err(|_| PhysicsError::InvalidCollision),
        QueryKind::Recovery => {
            let points = shape
                .triangle_recovery_contacts_with_scratch(triangle, scratch)
                .map_err(|_| PhysicsError::InvalidCollision)?;
            if points.is_empty() {
                activation_points_with_scratch(shape, triangle, window, scratch)
            } else {
                Ok(points)
            }
        }
    }
}

// Points between two colliders from the feature realizing their separating
// axis, as (receiving collider, opposing collider, points). A face clips the
// other solid against that face's triangles, so its outward normal pushes the
// other collider away; crossed edges touch at their single closest pair.
fn pair_points(
    shapes: [&ContactPolytope; 2],
    [first, second]: [usize; 2],
    separation: ConvexSeparation,
    kind: QueryKind,
    margin: f64,
) -> Result<(usize, usize, Vec<TriangleContactPoint>), PhysicsError> {
    let (receiving, opposing, plane) = match separation.feature {
        ConvexFeature::OtherFace(plane) => (first, second, plane),
        ConvexFeature::OwnFace(plane) => (second, first, plane),
        ConvexFeature::Edges([own, other]) => {
            let point = TriangleContactPoint {
                triangle_point: other,
                body_point: own,
                normal: separation.axis,
                depth: (-separation.separation).max(0.0),
            };
            return Ok((first, second, vec![point]));
        }
    };
    let shape = |row| shapes[usize::from(row != first)];
    let mut points = Vec::new();
    for triangle in shape(opposing)
        .face_triangles(plane)
        .map_err(|_| PhysicsError::InvalidCollision)?
    {
        points.extend(surface_points(
            shape(receiving),
            triangle,
            kind,
            margin,
            PAIR_ACTIVATION_DISTANCE,
        )?);
    }
    Ok((receiving, opposing, points))
}

struct SupportGroup {
    normal: DVec3,
    distance: f64,
    response: [f64; 4],
    directions: [DVec3; 5],
    supports: [TerrainContact; 5],
    curved: bool,
    // A cylinder's deepest point, which a tipped cap's lowest rim point can be
    // without winning any corner.
    deepest: Option<TerrainContact>,
}

impl SupportGroup {
    fn append_unique(self, manifold: usize, output: &mut Vec<TerrainContact>) {
        // Points along a level line are one depth; only a clearly deeper point
        // adds a row.
        const DEEPER: f64 = 1e-6;
        for corner in 0..if self.curved { 5 } else { 4 } {
            let mut support = self.supports[corner];
            support.manifold = manifold;
            if self.supports[..corner]
                .iter()
                .all(|previous| previous.terrain_point.distance(support.terrain_point) >= 1e-5)
            {
                output.push(support);
            }
        }
        let corners = &self.supports[..if self.curved { 5 } else { 4 }];
        if let Some(mut deepest) = self.deepest
            && corners
                .iter()
                .all(|support| deepest.separation < support.separation - DEEPER)
        {
            deepest.manifold = manifold;
            output.push(deepest);
        }
    }
}

// The receiving solid, which names the crown direction of a new support group.
#[derive(Clone, Copy)]
enum Opposing<'a> {
    Polytope(&'a ContactPolytope),
    Cylinder(&'a ContactCylinder),
}

impl Opposing<'_> {
    fn normal(self, surface_normal: DVec3) -> DVec3 {
        match self {
            Self::Polytope(shape) => shape.opposing_normal(surface_normal),
            Self::Cylinder(cylinder) => cylinder.opposing_normal(surface_normal),
        }
    }
}

// A cylinder side touches a surface along its lowest line, but a triangle beside
// that line still reports its own nearest point higher up the flank. Such a
// point would win a group corner and hold the wheel ahead of or behind its
// axle, braking it. Keep a flank point only where no lowest-line point of its
// group lies at or below it, as at a kerb or in a crease.
fn rolling_supports(
    points: &[TerrainContact],
    cylinder: &ContactCylinder,
    center: DVec3,
) -> Vec<TerrainContact> {
    let on_line = points
        .iter()
        .map(|point| on_lowest_line(cylinder, point.normal, point.body_point))
        .collect::<Vec<_>>();
    points
        .iter()
        .zip(&on_line)
        .filter(|&(point, &line)| {
            line || !points.iter().zip(&on_line).any(|(other, &other_line)| {
                other_line
                    && other.separation <= point.separation
                    && same_support(Surface::of(other), Surface::of(point), center).is_some()
            })
        })
        .map(|(point, _)| *point)
        .collect()
}

// Whether a point lies on a cylinder side's lowest line toward a surface, or
// the cylinder stands on an end.
fn on_lowest_line(cylinder: &ContactCylinder, normal: DVec3, point: DVec3) -> bool {
    const ON_LINE: f64 = 1e-6;
    cylinder
        .lowest_line_distance(normal, point)
        .is_none_or(|distance| distance <= ON_LINE)
}

// A contact's surface plane and response.
#[derive(Clone, Copy)]
struct Surface {
    normal: DVec3,
    distance: f64,
    response: [f64; 4],
}

impl Surface {
    fn of(contact: &TerrainContact) -> Self {
        Self {
            normal: contact.normal,
            distance: contact.normal.dot(contact.terrain_point),
            response: contact.response,
        }
    }
}

// Whether a contact on `surface` joins the support group of `group`, and if so
// whether only because the surface curves.
fn same_support(group: Surface, surface: Surface, center: DVec3) -> Option<bool> {
    let (normal, distance) = (group.normal, group.distance);
    let own = surface.distance;
    let parallel = (normal - surface.normal).abs().max_element() < 1e-6;
    let separation = ((surface.normal - normal).dot(center) - own + distance).abs();
    let nearby = !parallel && normal.dot(surface.normal) > 0.995 && separation < 0.025;
    (((parallel && (own - distance).abs() < 1e-5) || nearby)
        && group.response.map(f64::to_bits) == surface.response.map(f64::to_bits))
    .then_some(nearby)
}

fn reduce_support(
    groups: &mut Vec<SupportGroup>,
    overflow: &mut Vec<TerrainContact>,
    mut contact: TerrainContact,
    shape: Opposing<'_>,
    center: DVec3,
) {
    let distance = contact.normal.dot(contact.terrain_point);
    for group in groups.iter_mut() {
        let own = Surface {
            normal: group.normal,
            distance: group.distance,
            response: group.response,
        };
        if let Some(nearby) = same_support(own, Surface::of(&contact), center) {
            group.curved |= nearby;
            if let Some(deepest) = &mut group.deepest
                && contact.separation < deepest.separation
            {
                *deepest = contact;
            }
            for (support, direction) in group.supports.iter_mut().zip(group.directions) {
                if contact.terrain_point.dot(direction) > support.terrain_point.dot(direction) {
                    *support = contact;
                }
            }
            return;
        }
    }
    // Keep the existing 16-group/four-corner-plus-crown rule. Extra groups are
    // emitted unreduced; they must never silently disappear at a capacity bound.
    if groups.len() == 16 {
        contact.manifold = 16 + overflow.len();
        overflow.push(contact);
        return;
    }
    let reference = if contact.normal.y.abs() > 0.9 {
        DVec3::X
    } else {
        DVec3::Y
    };
    let u = reference.cross(contact.normal).normalize();
    let v = contact.normal.cross(u);
    groups.push(SupportGroup {
        normal: contact.normal,
        distance,
        response: contact.response,
        directions: [u + v, u - v, -u - v, -u + v, -shape.normal(contact.normal)],
        supports: [contact; 5],
        curved: false,
        deepest: matches!(shape, Opposing::Cylinder(_)).then_some(contact),
    });
}

fn valid_bounds(bounds: WorldBounds) -> bool {
    bounds.minimum.0.is_finite()
        && bounds.maximum.0.is_finite()
        && bounds.minimum.0.cmple(bounds.maximum.0).all()
}

#[allow(clippy::too_many_lines)] // Validate the entire immutable hierarchy once at publication.
fn validate_chunk(chunk: &TerrainCollisionChunk) -> Result<(), PhysicsError> {
    let invalid = PhysicsError::InvalidCollision;
    let bvh = &chunk.triangle_bvh;
    if !chunk.origin.0.is_finite() || !valid_bounds(chunk.bounds) {
        return Err(invalid);
    }
    if bvh.nodes.is_empty() {
        return if bvh.triangles.is_empty() && chunk.indices.is_empty() {
            Ok(())
        } else {
            Err(invalid)
        };
    }
    let mut visited = vec![false; bvh.nodes.len()];
    let mut triangles = vec![false; bvh.triangles.len()];
    let mut stack = vec![0_usize];
    while let Some(index) = stack.pop() {
        let node = bvh.nodes.get(index).ok_or(PhysicsError::InvalidCollision)?;
        if visited[index] || !valid_bounds(node.bounds) {
            return Err(invalid);
        }
        visited[index] = true;
        if node.triangle_count > 0 {
            if node.left_child.is_some() || node.right_child.is_some() {
                return Err(invalid);
            }
            let start = node.first_triangle as usize;
            let end = start
                .checked_add(node.triangle_count as usize)
                .ok_or(PhysicsError::InvalidCollision)?;
            let range = bvh
                .triangles
                .get(start..end)
                .ok_or(PhysicsError::InvalidCollision)?;
            for (offset, triangle) in range.iter().enumerate() {
                if triangles[start + offset] || !node.group_mask.contains(triangle.group_mask) {
                    return Err(invalid);
                }
                triangles[start + offset] = true;
                chunk
                    .triangle_surface_response(triangle.indices)
                    .map_err(|_| PhysicsError::InvalidCollision)?;
                for vertex in triangle.indices {
                    let point = chunk
                        .vertices
                        .get(vertex as usize)
                        .ok_or(PhysicsError::InvalidCollision)?;
                    let point = WorldPosition(chunk.origin.0 + Vec3::from_array(*point).as_dvec3());
                    if !point.0.is_finite() || !node.bounds.contains(point) {
                        return Err(invalid);
                    }
                }
            }
        } else {
            for child in [node.left_child, node.right_child] {
                let child = child.ok_or(PhysicsError::InvalidCollision)? as usize;
                let descendant = bvh.nodes.get(child).ok_or(PhysicsError::InvalidCollision)?;
                if child <= index
                    || !node.bounds.contains(descendant.bounds.minimum)
                    || !node.bounds.contains(descendant.bounds.maximum)
                    || !node.group_mask.contains(descendant.group_mask)
                {
                    return Err(invalid);
                }
                stack.push(child);
            }
        }
    }
    if visited.contains(&false) || triangles.contains(&false) {
        return Err(invalid);
    }
    if !chunk.indices.len().is_multiple_of(3) {
        return Err(invalid);
    }
    let mut active = bvh
        .triangles
        .iter()
        .filter(|triangle| triangle.group_mask.intersects(chunk.active_groups))
        .map(|triangle| triangle.indices)
        .collect::<Vec<_>>();
    let mut indexed = chunk
        .indices
        .chunks_exact(3)
        .map(|row| [row[0], row[1], row[2]])
        .collect::<Vec<_>>();
    active.sort_unstable();
    indexed.sort_unstable();
    if active != indexed {
        return Err(invalid);
    }
    Ok(())
}

#[cfg(test)]
pub(crate) mod tests;

// Existing finite intersections retain their established four-corner rule.
// Extrusion is only for a pair without an intersection, not a perturbation of
// every supporting manifold. Out-of-tolerance/invalid results still fail.
fn activation_points_with_scratch(
    shape: &ContactPolytope,
    triangle: [DVec3; 3],
    window: f64,
    scratch: &mut mechanic_core::TriangleClipScratch,
) -> Result<Vec<mechanic_core::TriangleContactPoint>, PhysicsError> {
    let points = shape
        .triangle_activation_contacts_with_scratch(triangle, window, scratch)
        .map_err(|_| PhysicsError::NotConverged)?;
    if points.iter().any(|point| {
        let gap = (point.body_point - point.triangle_point).dot(point.normal);
        !gap.is_finite() || gap > window
    }) {
        return Err(PhysicsError::NotConverged);
    }
    Ok(points)
}

// Cached construction geometry is relative to the simulation origin, independent
// of terrain publications. Exact body-pose changes invalidate only that body's
// shapes. Topology owns the cache and cannot be changed in place.
type CachedSeparations = HashMap<[usize; 2], (f64, Option<ConvexSeparation>)>;

#[derive(Default)]
struct PoseCache {
    poses: Vec<BodyPose>,
    bounds: Vec<[DVec3; 2]>,
    shapes: Vec<OnceLock<ContactPolytope>>,
    rounds: Vec<Option<ContactCylinder>>,
    separations: std::cell::RefCell<CachedSeparations>,
    spare_shapes: std::cell::RefCell<Vec<ContactPolytope>>,
    clipping: std::cell::RefCell<mechanic_core::TriangleClipScratch>,
    terrain_candidates: Vec<Vec<TerrainNodeId>>,
    terrain_bounds: Vec<[DVec3; 2]>,
}
impl PoseCache {
    fn update(
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
    fn shape(
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
fn overlaps(a: [DVec3; 2], b: [DVec3; 2]) -> bool {
    !a[0].cmpgt(b[1]).any() && !b[0].cmpgt(a[1]).any()
}

struct CandidatePairs<'a> {
    scratch: std::sync::MutexGuard<'a, PairScratch>,
}
impl std::ops::Deref for CandidatePairs<'_> {
    type Target = [[usize; 2]];
    fn deref(&self) -> &Self::Target {
        &self.scratch.pairs
    }
}

#[derive(Default)]
struct PairScratch {
    body_bounds: Vec<[DVec3; 2]>,
    body_nodes: Vec<[DVec3; 2]>,
    body_candidates: Vec<usize>,
    trees: Vec<Vec<[DVec3; 2]>>,
    refitted: Vec<bool>,
    stack: Vec<usize>,
    candidates: Vec<usize>,
    node_pair_tests: usize,
    pairs: Vec<[usize; 2]>,
    pair_stack: Vec<[usize; 2]>,
}

// Every vertex moves at most its integrated point-speed bound. Intersect that
// initial-shape envelope with the origin/radius envelope: both contain the
// entire unwrapped trajectory, including full rotations.
fn swept_bounds(
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
fn path_bounds(
    collider: &Collider,
    path: &crate::MachineMotion<'_>,
    tolerance: f64,
) -> Result<[DVec3; 2], PhysicsError> {
    path_bounds_from(collider, path, tolerance, None)
}

// The optional bound must belong to the exact initial pose and this collider.
fn path_bounds_from(
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
    bounds: Vec<[DVec3; 2]>,
    terrain_generation: u64,
    topology_generation: u64,
    origin: DVec3,
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

impl TerrainContactScene {
    #[cfg(test)]
    pub(crate) fn empty_contact_region(
        &self,
        geometry: &MachineCollisionGeometry,
        motion: &crate::MachineMotion<'_>,
        origin: DVec3,
        padding: f64,
    ) -> Option<EmptyContactRegion> {
        if geometry.generation != motion.generation() || !origin.is_finite() {
            return None;
        }
        let bounds = geometry
            .colliders
            .iter()
            .map(|collider| {
                let b = path_bounds(collider, motion, padding).ok()?;
                (b[0].is_finite() && b[1].is_finite()).then_some(b)
            })
            .collect::<Option<Vec<_>>>()?;
        for (collider, b) in geometry.colliders.iter().zip(&bounds) {
            if collider.moving
                && !self
                    .index
                    .bounds_candidates(WorldBounds {
                        minimum: WorldPosition((origin + b[0]).map(f64::next_down)),
                        maximum: WorldPosition((origin + b[1]).map(f64::next_up)),
                    })
                    .is_empty()
            {
                return None;
            }
        }
        if geometry.candidate_pairs(&bounds).iter().next().is_some() {
            return None;
        }
        Some(EmptyContactRegion {
            bounds,
            terrain_generation: self.generation,
            topology_generation: geometry.generation,
            origin,
        })
    }
}
