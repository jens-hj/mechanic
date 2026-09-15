//! Finite terrain manifolds from immutable colliders and the world's shared BVHs.

use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    sync::{Arc, Mutex, OnceLock},
};

use bevy_math::{DVec3, Vec3};
use mechanic_core::{
    CompiledCreation, ContactPolytope, ConvexFeature, ConvexSeparation, MaterialProperties,
    TriangleContactPoint,
};
use mechanic_world::{
    TerrainCollisionChunk, TerrainNodeId, TerrainSpatialIndex, WorldBounds, WorldPosition,
};

use crate::{BodyPose, PhysicsError};

mod broadphase;
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
    local: ContactPolytope,
    bounds: [DVec3; 2],
    moving: bool,
    radius: f64,
}

impl MachineCollisionGeometry {
    /// Body and conservative radius about the body origin of every collider row,
    /// in row order.
    pub(crate) fn collider_reach(&self) -> impl ExactSizeIterator<Item = (usize, f64)> + '_ {
        self.colliders
            .iter()
            .map(|collider| (collider.body, collider.radius))
    }
}

/// Immutable local collider data compiled once for a construction generation.
pub struct MachineCollisionGeometry {
    generation: u64,
    bodies: usize,
    colliders: Vec<Collider>,
    // Sorted body pairs joined by a bearing, which never collide with each other.
    suppressed: Vec<[usize; 2]>,
    body_colliders: Vec<Vec<usize>>,
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
    /// every trial localizing contacts the solve already carries.
    ///
    /// # Errors
    /// Rejects invalid compiled geometry or body references.
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
        let mut geometry = Self {
            generation: topology_generation,
            bodies: creation.compounds.len(),
            colliders,
            suppressed,
            body_colliders,
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

    // Collider pairs that may touch within `bounds`, in sorted order: on
    // different bodies, at least one moving, and not joined by a bearing.
    // Body traversal rejects suppressed pairs before immutable collider trees.
    fn candidate_pairs(&self, bounds: &[[DVec3; 2]]) -> CandidatePairs<'_> {
        let mut scratch = self
            .pair_scratch
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let PairScratch {
            body_bounds,
            body_nodes,
            body_candidates,
            trees,
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
        for (tree, output) in self.collider_trees.iter().zip(trees.iter_mut()) {
            tree.refit(bounds, output);
        }
        for a in 0..self.bodies {
            if self.body_colliders[a].is_empty() {
                continue;
            }
            body_candidates.clear();
            self.body_tree
                .query(body_nodes, body_bounds[a], stack, body_candidates);
            for &b in body_candidates.iter().filter(|&&b| b > a) {
                let body_pair = [a.min(b), a.max(b)];
                if self.suppressed.binary_search(&body_pair).is_ok()
                    || !(self.colliders[self.body_colliders[a][0]].moving
                        || self.colliders[self.body_colliders[b][0]].moving)
                    || !overlaps(body_bounds[a], body_bounds[b])
                {
                    continue;
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

    /// Like [`Self::proximity`], with one margin per compiled collider row, so a
    /// fast collider reaches far without widening the query for resting ones. A
    /// collider pair uses the larger of its two margins.
    pub(crate) fn proximity_margins(
        &self,
        machine: &MachineCollisionGeometry,
        poses: &[BodyPose],
        origin: DVec3,
        margins: &[f64],
    ) -> Result<TerrainContactQuery, PhysicsError> {
        self.query(machine, poses, origin, margins, QueryKind::Surface)
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
        for (body, rows) in machine.body_colliders.iter().enumerate() {
            if rows.is_empty() || !machine.colliders[rows[0]].moving {
                continue;
            }
            let [minimum, maximum] =
                rows.iter()
                    .fold([DVec3::INFINITY, DVec3::NEG_INFINITY], |[lo, hi], &row| {
                        [
                            lo.min(cache.terrain_bounds[row][0]),
                            hi.max(cache.terrain_bounds[row][1]),
                        ]
                    });
            let bounds = WorldBounds {
                minimum: WorldPosition(minimum),
                maximum: WorldPosition(maximum),
            };
            if !valid_bounds(bounds) {
                return Err(PhysicsError::InvalidCollision);
            }
            let nodes = self.index.bounds_candidates(bounds);
            if nodes.is_empty() {
                continue;
            }
            machine.collider_trees[body].refit(&cache.terrain_bounds, &mut trees[body]);
            for node in nodes {
                let chunk = &self.chunks[&node].geometry;
                let bounds = chunk
                    .triangle_bvh
                    .nodes
                    .first()
                    .map_or(chunk.bounds, |node| node.bounds);
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

    #[allow(clippy::too_many_lines)] // Ordered broadphase, narrowphase, and reduction with explicit counts.
    fn query(
        &self,
        machine: &MachineCollisionGeometry,
        poses: &[BodyPose],
        origin: DVec3,
        margins: &[f64],
        kind: QueryKind,
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
        self.prepare_terrain_candidates(machine, &mut cache, origin, margins)?;
        let shapes = &cache;
        for (collider_row, collider) in machine.colliders.iter().enumerate() {
            if !collider.moving {
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
                    let shape = shapes.shape(machine, poses, collider_row)?;
                    let points = surface_points_with_scratch(
                        shape,
                        triangle,
                        kind,
                        margin,
                        CONTACT_ACTIVATION_DISTANCE,
                        &mut shapes.clipping.borrow_mut(),
                    )?;
                    if points.is_empty() {
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
                        if !activation_recorded && contact.separation <= CONTACT_ACTIVATION_DISTANCE
                        {
                            result.activation_features.push(contact.feature);
                            activation_recorded = true;
                        }
                        if matches!(kind, QueryKind::Recovery) {
                            result.contacts.push(contact);
                        } else {
                            reduce_support(
                                &mut groups,
                                &mut result.contacts,
                                contact,
                                shape,
                                center,
                            );
                        }
                    }
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
            QueryKind::Activation | QueryKind::Recovery => PAIR_ACTIVATION_DISTANCE,
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
        for &[first, second] in machine.candidate_pairs(&bounds).iter() {
            result.collider_pair_candidates += 1;
            let reach = match kind {
                // The faster collider's margin already covers its own travel.
                QueryKind::Surface => margins[first].max(margins[second]),
                QueryKind::Activation | QueryKind::Recovery => PAIR_ACTIVATION_DISTANCE,
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
                if matches!(kind, QueryKind::Recovery) {
                    result.contacts.push(contact);
                } else {
                    reduce_support(
                        &mut groups,
                        &mut result.contacts,
                        contact,
                        shapes.shape(machine, poses, receiving)?,
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
}

impl SupportGroup {
    fn append_unique(self, manifold: usize, output: &mut Vec<TerrainContact>) {
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
    }
}

fn reduce_support(
    groups: &mut Vec<SupportGroup>,
    overflow: &mut Vec<TerrainContact>,
    mut contact: TerrainContact,
    shape: &ContactPolytope,
    center: DVec3,
) {
    let distance = contact.normal.dot(contact.terrain_point);
    for group in groups.iter_mut() {
        let parallel = (group.normal - contact.normal).abs().max_element() < 1e-6;
        let separation =
            ((contact.normal - group.normal).dot(center) - distance + group.distance).abs();
        let nearby = !parallel && group.normal.dot(contact.normal) > 0.995 && separation < 0.025;
        if ((parallel && (distance - group.distance).abs() < 1e-5) || nearby)
            && group.response.map(f64::to_bits) == contact.response.map(f64::to_bits)
        {
            group.curved |= nearby;
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
        directions: [
            u + v,
            u - v,
            -u - v,
            -u + v,
            -shape.opposing_normal(contact.normal),
        ],
        supports: [contact; 5],
        curved: false,
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
) -> Result<[DVec3; 2], PhysicsError> {
    let [minimum, maximum] = collider
        .local
        .transformed_bounds(pose.position, pose.rotation)
        .map_err(|_| PhysicsError::InvalidCollision)?;
    let travel = DVec3::splat((motion.point_speed(collider.radius) + tolerance).next_up());
    let reach = DVec3::splat((motion.origin_speed + collider.radius + tolerance).next_up());
    Ok([
        (minimum - travel)
            .max(pose.position - reach)
            .map(f64::next_down),
        (maximum + travel)
            .min(pose.position + reach)
            .map(f64::next_up),
    ])
}
