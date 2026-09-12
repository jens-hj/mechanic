//! Finite terrain manifolds from immutable colliders and the world's shared BVHs.

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use bevy_math::{DVec3, Vec3};
use mechanic_core::{CompiledCreation, ContactPolytope, MaterialProperties};
use mechanic_world::{
    TerrainCollisionChunk, TerrainNodeId, TerrainSpatialIndex, WorldBounds, WorldPosition,
};

use crate::{BodyPose, PhysicsError};

mod penetration;
pub use penetration::{TerrainPathFailure, TerrainPathOutcome, TerrainPathQuery};
mod sweep;
pub(crate) use sweep::SlowContactMotion;
pub use sweep::{TerrainSweepHit, TerrainSweepOutcome, TerrainSweepQuery};
mod constraints;
pub use constraints::TerrainImpactConstraints;

/// Stable source identity; replacing a chunk or topology invalidates its points.
/// A retained corner is refreshed against geometry before any future reuse.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct TerrainContactFeature {
    /// Compiled construction generation.
    pub topology_generation: u64,
    /// Collider row within that construction.
    pub collider: usize,
    /// Owning terrain node.
    pub node: TerrainNodeId,
    /// Chunk geometry generation supplied by the world.
    pub geometry_generation: u64,
    /// Publication that selected this chunk's active groups/materials.
    pub publication_generation: u64,
    /// Stable triangle row within the immutable BVH.
    pub triangle: usize,
    /// Retained finite polygon corner, requiring geometry refresh before reuse.
    pub corner: usize,
}

// Numerical zero for actual finite opposing points, never a speculative margin.
pub(crate) const CONTACT_ACTIVATION_DISTANCE: f64 = 1e-12;

/// Actual finite terrain/body support and mixed material properties.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TerrainContact {
    /// Generations and geometric source of this point.
    pub feature: TerrainContactFeature,
    /// Compound body receiving the contact impulse.
    pub body: usize,
    /// Query-local manifold number within this collider, for coupled block solving.
    pub manifold: usize,
    /// Surface point, relative to the query's floating origin.
    pub terrain_point: DVec3,
    /// Opposing convex point, relative to the same origin.
    pub body_point: DVec3,
    /// Triangle's outward unit normal.
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
    /// Chunk candidates visited across all moving colliders.
    pub chunk_candidates: usize,
    /// Triangle candidates passed to finite narrowphase.
    pub triangle_candidates: usize,
    /// Point count before cross-triangle manifold reduction.
    pub unreduced_points: usize,
}

struct Collider {
    body: usize,
    center: DVec3,
    material: MaterialProperties,
    local: ContactPolytope,
    moving: bool,
    radius: f64,
}

/// Immutable local collider data compiled once for a construction generation.
pub struct MachineCollisionGeometry {
    generation: u64,
    bodies: usize,
    colliders: Vec<Collider>,
}

impl MachineCollisionGeometry {
    /// Retains the exact box/convex decomposition and material of every collider.
    ///
    /// # Errors
    /// Rejects invalid compiled geometry or body references.
    pub fn new(
        creation: &CompiledCreation,
        topology_generation: u64,
    ) -> Result<Self, PhysicsError> {
        let colliders = creation
            .colliders
            .iter()
            .map(|source| {
                let body = source.compound_index as usize;
                let compound = creation
                    .compounds
                    .get(body)
                    .ok_or(PhysicsError::InvalidCollision)?;
                let local = ContactPolytope::from_collider(source)
                    .map_err(|_| PhysicsError::InvalidCollision)?;
                let radius = local
                    .conservative_radius()
                    .map_err(|_| PhysicsError::InvalidCollision)?;
                Ok(Collider {
                    body,
                    center: source.local_center.as_dvec3(),
                    material: source.material_properties,
                    local,
                    moving: !compound.is_static,
                    radius,
                })
            })
            .collect::<Result<_, PhysicsError>>()?;
        Ok(Self {
            generation: topology_generation,
            bodies: creation.compounds.len(),
            colliders,
        })
    }
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
        self.query(machine, poses, origin, margin, QueryKind::Surface)
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
            CONTACT_ACTIVATION_DISTANCE,
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
            CONTACT_ACTIVATION_DISTANCE,
            QueryKind::Recovery,
        )
    }

    #[allow(clippy::too_many_lines)] // Ordered broadphase, narrowphase, and reduction with explicit counts.
    fn query(
        &self,
        machine: &MachineCollisionGeometry,
        poses: &[BodyPose],
        origin: DVec3,
        margin: f64,
        kind: QueryKind,
    ) -> Result<TerrainContactQuery, PhysicsError> {
        if poses.len() != machine.bodies
            || !origin.is_finite()
            || !margin.is_finite()
            || margin < 0.0
        {
            return Err(PhysicsError::InvalidCollision);
        }
        let mut result = TerrainContactQuery::default();
        for (collider_row, collider) in machine.colliders.iter().enumerate() {
            if !collider.moving {
                continue;
            }
            let pose = poses[collider.body];
            let shape = collider
                .local
                .transformed(pose.position, pose.rotation)
                .map_err(|_| PhysicsError::InvalidCollision)?;
            let center = pose.position + pose.rotation * collider.center;
            let [minimum, maximum] = shape.bounds();
            let bounds = WorldBounds {
                minimum: WorldPosition(origin + minimum - DVec3::splat(margin)),
                maximum: WorldPosition(origin + maximum + DVec3::splat(margin)),
            };
            if !valid_bounds(bounds) {
                return Err(PhysicsError::InvalidCollision);
            }
            let mut groups = Vec::<SupportGroup>::new();
            let candidates = self.index.bounds_candidates(bounds);
            result.chunk_candidates += candidates.len();
            for node in candidates {
                let published = &self.chunks[&node];
                let chunk = &published.geometry;
                for row in chunk.bounds_candidates(bounds) {
                    result.triangle_candidates += 1;
                    let indices = chunk.triangle_bvh.triangles[row].indices;
                    let triangle = indices.map(|index| {
                        chunk.origin.0 - origin
                            + Vec3::from_array(chunk.vertices[index as usize]).as_dvec3()
                    });
                    // Zero-area triangles carry no surface. Small but finite
                    // triangles still count; an area cutoff could open a hole.
                    if (triangle[1] - triangle[0])
                        .cross(triangle[2] - triangle[0])
                        .try_normalize()
                        .is_none()
                    {
                        continue;
                    }
                    let points = match kind {
                        QueryKind::Activation => activation_points(&shape, triangle)?,
                        QueryKind::Surface => shape
                            .triangle_proximity(triangle, margin)
                            .map_err(|_| PhysicsError::InvalidCollision)?,
                        QueryKind::Recovery => {
                            let points = shape
                                .triangle_recovery_contacts(triangle)
                                .map_err(|_| PhysicsError::InvalidCollision)?;
                            if points.is_empty() {
                                activation_points(&shape, triangle)?
                            } else {
                                points
                            }
                        }
                    };
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
                    for (corner, point) in points.into_iter().enumerate() {
                        result.unreduced_points += 1;
                        let contact = TerrainContact {
                            feature: TerrainContactFeature {
                                topology_generation: machine.generation,
                                collider: collider_row,
                                node,
                                geometry_generation: chunk.generation,
                                publication_generation: published.publication,
                                triangle: row,
                                corner,
                            },
                            body: collider.body,
                            manifold: 0,
                            terrain_point: point.triangle_point,
                            body_point: point.body_point,
                            normal: point.normal,
                            depth: point.depth,
                            separation: (point.body_point - point.triangle_point).dot(point.normal),
                            response,
                        };
                        if matches!(kind, QueryKind::Recovery) {
                            result.contacts.push(contact);
                        } else {
                            reduce_support(
                                &mut groups,
                                &mut result.contacts,
                                contact,
                                &shape,
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
        result.contacts.sort_by_key(|contact| contact.feature);
        Ok(result)
    }
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
fn activation_points(
    shape: &ContactPolytope,
    triangle: [DVec3; 3],
) -> Result<Vec<mechanic_core::TriangleContactPoint>, PhysicsError> {
    let points = shape
        .triangle_activation_contacts(triangle, CONTACT_ACTIVATION_DISTANCE)
        .map_err(|_| PhysicsError::NotConverged)?;
    if points.iter().any(|point| {
        let gap = (point.body_point - point.triangle_point).dot(point.normal);
        !gap.is_finite() || gap > CONTACT_ACTIVATION_DISTANCE
    }) {
        return Err(PhysicsError::NotConverged);
    }
    Ok(points)
}
