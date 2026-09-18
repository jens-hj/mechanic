//! Finite terrain manifolds from immutable colliders and the world's shared BVHs.

mod broadphase;
mod constraints;
mod geometry;
mod groups;
mod model;
mod penetration;
mod pose_cache;
mod support;
mod sweep;

pub use constraints::TerrainImpactConstraints;
pub use geometry::MachineCollisionGeometry;
use geometry::terrain_motion;
pub(crate) use groups::{ContactGroups, Measured};
pub(crate) use model::{CONTACT_ACTIVATION_DISTANCE, PAIR_ACTIVATION_DISTANCE};
pub use model::{
    ContactObstacle, ContactTarget, TerrainContact, TerrainContactFeature, TerrainContactQuery,
};
pub use penetration::{TerrainPathFailure, TerrainPathOutcome, TerrainPathQuery};
#[cfg(test)]
pub(crate) use pose_cache::EmptyContactRegion;
use pose_cache::{PairScratch, PoseCache, overlaps, path_bounds, path_bounds_from, swept_bounds};
use support::{
    Opposing, SupportGroup, Surface, activation_points_with_scratch, on_lowest_line, pair_points,
    reduce_support, rolling_supports, same_support, surface_points_with_scratch,
};
pub(crate) use sweep::SlowContactMotion;
pub use sweep::{TerrainSweepHit, TerrainSweepOutcome, TerrainSweepQuery};

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use bevy_math::{DVec3, Vec3};
use mechanic_core::{MaterialProperties, TriangleSupport};
use mechanic_world::{
    TerrainCollisionChunk, TerrainNodeId, TerrainSpatialIndex, WorldBounds, WorldPosition,
};

use crate::{BodyPose, PhysicsError};

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

    #[expect(clippy::too_many_lines)]
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

fn valid_bounds(bounds: WorldBounds) -> bool {
    bounds.minimum.0.is_finite()
        && bounds.maximum.0.is_finite()
        && bounds.minimum.0.cmple(bounds.maximum.0).all()
}

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

#[cfg(test)]
pub(crate) mod tests;
