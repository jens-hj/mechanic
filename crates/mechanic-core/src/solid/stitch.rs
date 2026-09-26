//! Stitching clipped cells into one boundary with stable topology keys and logical edge chains.

use super::model::{
    BoundaryHalfEdge, BoundaryVertex, ConvexVolumeCell, EvaluatedSolid, LogicalEdge, SolidError,
    SurfacePatch, SurfacePatchKey, TopologyKey, TopologySource,
};
use super::polygon::{
    EPSILON, EdgeSegment, FaceSignature, PointKey, PolyCell, face_signature, point_key,
    poly_cell_to_convex, polygon_normal,
};
use bevy_math::DVec3;
use std::collections::{BTreeMap, BTreeSet};

pub(super) fn build_evaluated(cells: &[PolyCell]) -> Result<EvaluatedSolid, SolidError> {
    let (stitched, _) = stitch(cells)?;
    let volume_cells = cells
        .iter()
        .filter_map(|cell| {
            poly_cell_to_convex(cell).map(|piece| ConvexVolumeCell {
                piece,
                band: cell.band,
            })
        })
        .collect::<Vec<_>>();
    if volume_cells.is_empty() {
        return Err(SolidError::ZeroVolume);
    }
    Ok(EvaluatedSolid {
        vertices: stitched.vertices,
        half_edges: stitched.half_edges,
        surfaces: stitched.surfaces,
        logical_edges: stitched.logical_edges,
        cells: volume_cells,
    })
}

pub(super) struct Stitched {
    pub(super) vertices: Vec<BoundaryVertex>,
    pub(super) half_edges: Vec<BoundaryHalfEdge>,
    pub(super) surfaces: Vec<SurfacePatch>,
    pub(super) logical_edges: Vec<LogicalEdge>,
}

#[expect(clippy::too_many_lines)]
pub(super) fn stitch(cells: &[PolyCell]) -> Result<(Stitched, Vec<EdgeSegment>), SolidError> {
    let mut occurrences = BTreeMap::<FaceSignature, Vec<(usize, usize)>>::new();
    for (cell_index, cell) in cells.iter().enumerate() {
        for (face_index, face) in cell.faces.iter().enumerate() {
            if face.interior {
                continue;
            }
            occurrences
                .entry(face_signature(&face.vertices))
                .or_default()
                .push((cell_index, face_index));
        }
    }
    let boundary = occurrences
        .values()
        .filter(|uses| uses.len() == 1)
        .map(|uses| uses[0])
        .collect::<Vec<_>>();
    let mut vertex_map = BTreeMap::<PointKey, u32>::new();
    let mut vertices = Vec::<BoundaryVertex>::new();
    let mut half_edges = Vec::<BoundaryHalfEdge>::new();
    let mut surfaces = Vec::<SurfacePatch>::new();
    let mut surface_families = Vec::<SurfacePatchKey>::new();
    let mut surface_continuity = Vec::<Vec<SurfacePatchKey>>::new();
    let mut directed = BTreeMap::<(u32, u32), u32>::new();
    let mut half_edge_cells = Vec::<usize>::new();
    for (cell_index, face_index) in boundary {
        let face = &cells[cell_index].faces[face_index];
        let surface_index = u32::try_from(surfaces.len()).map_err(|_| SolidError::NonManifold)?;
        let first_edge = u32::try_from(half_edges.len()).map_err(|_| SolidError::NonManifold)?;
        let indices = face
            .vertices
            .iter()
            .map(|&point| {
                let key = point_key(point);
                *vertex_map.entry(key).or_insert_with(|| {
                    let index = u32::try_from(vertices.len()).unwrap_or(u32::MAX);
                    vertices.push(BoundaryVertex {
                        position: point.as_vec3(),
                        outgoing: None,
                    });
                    index
                })
            })
            .collect::<Vec<_>>();
        for index in 0..indices.len() {
            let origin = indices[index];
            let destination = indices[(index + 1) % indices.len()];
            let edge = u32::try_from(half_edges.len()).map_err(|_| SolidError::NonManifold)?;
            vertices[origin as usize].outgoing.get_or_insert(edge);
            half_edges.push(BoundaryHalfEdge {
                origin,
                twin: u32::MAX,
                next: first_edge
                    + u32::try_from((index + 1) % indices.len())
                        .map_err(|_| SolidError::NonManifold)?,
                face: surface_index,
                logical_edge: None,
            });
            half_edge_cells.push(cell_index);
            if directed.insert((origin, destination), edge).is_some() {
                return Err(SolidError::NonManifold);
            }
        }
        surfaces.push(SurfacePatch {
            key: face.patch,
            normal: polygon_normal(&face.vertices).as_vec3(),
            half_edge: first_edge,
            smoothing_group: face.smoothing_group,
            uv_provenance: face.uv_provenance,
            band: cells[cell_index].band,
        });
        surface_families.push(face.family);
        surface_continuity.push(face.smooth_with.clone());
    }
    for (&(origin, destination), &edge) in &directed {
        let Some(&twin) = directed.get(&(destination, origin)) else {
            return Err(SolidError::NonManifold);
        };
        half_edges[edge as usize].twin = twin;
    }
    let mut logical = BTreeMap::<TopologyKey, Vec<u32>>::new();
    let mut segments = Vec::new();
    for edge_index in 0..half_edges.len() {
        let edge = half_edges[edge_index];
        if edge_index as u32 > edge.twin {
            continue;
        }
        let twin_index = edge.twin;
        let first_face = edge.face as usize;
        let second_face = half_edges[twin_index as usize].face as usize;
        let first_surface = &surfaces[first_face];
        let second_surface = &surfaces[second_face];
        let first_family = surface_families[first_face];
        let second_family = surface_families[second_face];
        if first_family == second_family
            || (first_surface.smoothing_group != 0
                && first_surface.smoothing_group == second_surface.smoothing_group)
            || surface_continuity[first_face].contains(&second_family)
            || surface_continuity[second_face].contains(&first_family)
        {
            continue;
        }
        let key = topology_key(first_family, second_family);
        let (canonical_edge_index, canonical_twin_index) =
            if (first_family, first_surface.key) <= (second_family, second_surface.key) {
                (edge_index as u32, twin_index)
            } else {
                (twin_index, edge_index as u32)
            };
        logical.entry(key).or_default().push(canonical_edge_index);
        let canonical_edge = half_edges[canonical_edge_index as usize];
        let canonical_twin = half_edges[canonical_twin_index as usize];
        let canonical_first = &surfaces[canonical_edge.face as usize];
        let canonical_second = &surfaces[canonical_twin.face as usize];
        let a = DVec3::from(vertices[canonical_edge.origin as usize].position);
        let b = DVec3::from(
            vertices[half_edges[canonical_edge.next as usize].origin as usize].position,
        );
        let first_normal = DVec3::from(canonical_first.normal);
        let second_normal = DVec3::from(canonical_second.normal);
        let tangent = (b - a).normalize();
        let convex = first_normal.cross(second_normal).dot(tangent) > EPSILON;
        segments.push(EdgeSegment {
            key,
            half_edge: canonical_edge_index,
            a,
            b,
            first_normal,
            second_normal,
            first_family: surface_families[canonical_edge.face as usize],
            second_family: surface_families[canonical_twin.face as usize],
            uv_provenance: canonical_first.uv_provenance,
            cell: half_edge_cells[canonical_edge_index as usize],
            convex,
        });
    }
    let mut logical_edges = Vec::new();
    for (candidate_key, half_edges_for_key) in logical {
        let chains = split_logical_chains(&half_edges, half_edges_for_key);
        let split = chains.len() > 1;
        for (ordinal, half_edges_for_key) in chains.into_iter().enumerate() {
            let key = if split {
                split_topology_key(candidate_key, ordinal)
            } else {
                candidate_key
            };
            let mut degree = BTreeMap::<u32, usize>::new();
            for &edge in &half_edges_for_key {
                half_edges[edge as usize].logical_edge = Some(key);
                let twin = half_edges[edge as usize].twin;
                half_edges[twin as usize].logical_edge = Some(key);
                let start = half_edges[edge as usize].origin;
                let end = half_edges[half_edges[edge as usize].next as usize].origin;
                *degree.entry(start).or_default() += 1;
                *degree.entry(end).or_default() += 1;
                if let Some(segment) = segments
                    .iter_mut()
                    .find(|segment| segment.half_edge == edge)
                {
                    segment.key = key;
                }
            }
            let convex = half_edges_for_key.iter().all(|edge| {
                segments
                    .iter()
                    .find(|segment| segment.half_edge == *edge)
                    .is_none_or(|segment| segment.convex)
            });
            logical_edges.push(LogicalEdge {
                key,
                half_edges: half_edges_for_key,
                closed: !degree.is_empty() && degree.values().all(|degree| *degree == 2),
                convex,
            });
        }
    }
    logical_edges.sort_by_key(|edge| edge.key);
    Ok((
        Stitched {
            vertices,
            half_edges,
            surfaces,
            logical_edges,
        },
        segments,
    ))
}

pub(super) fn split_logical_chains(
    half_edges: &[BoundaryHalfEdge],
    half_edges_for_key: Vec<u32>,
) -> Vec<Vec<u32>> {
    let endpoints = |edge: u32| {
        let start = half_edges[edge as usize].origin;
        let end = half_edges[half_edges[edge as usize].next as usize].origin;
        (start, end)
    };
    let mut incident = BTreeMap::<u32, Vec<u32>>::new();
    for &edge in &half_edges_for_key {
        let (start, end) = endpoints(edge);
        incident.entry(start).or_default().push(edge);
        incident.entry(end).or_default().push(edge);
    }
    let mut unvisited = half_edges_for_key.into_iter().collect::<BTreeSet<_>>();
    let mut chains = Vec::new();

    for (&vertex, edges) in &incident {
        if edges.len() == 2 {
            continue;
        }
        for &edge in edges {
            if !unvisited.remove(&edge) {
                continue;
            }
            chains.push(walk_logical_chain(
                half_edges,
                &incident,
                &mut unvisited,
                edge,
                vertex,
            ));
        }
    }
    while let Some(&edge) = unvisited.first() {
        unvisited.remove(&edge);
        let (start, _) = endpoints(edge);
        chains.push(walk_logical_chain(
            half_edges,
            &incident,
            &mut unvisited,
            edge,
            start,
        ));
    }
    chains
}

pub(super) fn walk_logical_chain(
    half_edges: &[BoundaryHalfEdge],
    incident: &BTreeMap<u32, Vec<u32>>,
    unvisited: &mut BTreeSet<u32>,
    first_edge: u32,
    start_vertex: u32,
) -> Vec<u32> {
    let mut chain = vec![first_edge];
    let first = half_edges[first_edge as usize];
    let first_end = half_edges[first.next as usize].origin;
    let mut vertex = if first.origin == start_vertex {
        first_end
    } else {
        first.origin
    };
    while let Some(edges) = incident.get(&vertex)
        && edges.len() == 2
        && let Some(&next) = edges.iter().find(|edge| unvisited.contains(edge))
    {
        unvisited.remove(&next);
        chain.push(next);
        let edge = half_edges[next as usize];
        let end = half_edges[edge.next as usize].origin;
        vertex = if edge.origin == vertex {
            end
        } else {
            edge.origin
        };
    }
    chain
}

pub(super) fn split_topology_key(candidate: TopologyKey, ordinal: usize) -> TopologyKey {
    let mut hash = 2_166_136_261_u32;
    for value in [
        0x5350_4c54,
        candidate.local,
        u32::try_from(ordinal).unwrap_or(u32::MAX),
    ] {
        hash ^= value;
        hash = hash.wrapping_mul(16_777_619);
    }
    TopologyKey {
        source: candidate.source,
        local: hash,
    }
}

pub(super) fn topology_key(first: SurfacePatchKey, second: SurfacePatchKey) -> TopologyKey {
    let (a, b) = if first <= second {
        (first, second)
    } else {
        (second, first)
    };
    let source = match (a.source, b.source) {
        (TopologySource::Base, TopologySource::Base) => TopologySource::Base,
        (TopologySource::Feature(feature), TopologySource::Base)
        | (TopologySource::Base, TopologySource::Feature(feature)) => {
            TopologySource::Feature(feature)
        }
        (TopologySource::Feature(first), TopologySource::Feature(second)) => {
            TopologySource::Feature(first.max(second))
        }
    };
    let mut hash = 2_166_136_261_u32;
    for value in [patch_word(a), a.local, patch_word(b), b.local] {
        hash ^= value;
        hash = hash.wrapping_mul(16_777_619);
    }
    TopologyKey {
        source,
        local: hash,
    }
}

pub(super) const fn patch_word(key: SurfacePatchKey) -> u32 {
    match key.source {
        TopologySource::Base => 0,
        TopologySource::Feature(feature) => feature.index().wrapping_add(1),
    }
}

pub(super) const fn topology_word(key: TopologyKey) -> u32 {
    match key.source {
        TopologySource::Base => 0,
        TopologySource::Feature(feature) => feature.index().wrapping_add(1),
    }
}
