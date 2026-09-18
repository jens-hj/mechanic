//! What changes between published cuts, and which faces may publish without caps.

use super::faces::{TerrainFace, TerrainTransitionMask};
use super::selection::ActiveTerrainNode;
use crate::TerrainNodeId;
use std::collections::{BTreeMap, BTreeSet};

/// Resolved startup-region nodes versus the current critical total.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TerrainReadiness {
    /// Current-generation critical nodes already active, including empty chunks.
    pub resolved: usize,
    /// Current critical nodes in the desired cut.
    pub total: usize,
}

/// One current-generation chunk whose mesh or face readiness changed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TerrainPublicationUpsert {
    /// Current active node and generation.
    pub node: ActiveTerrainNode,
    /// Faces whose desired neighbors are ready for uncapped publication.
    pub ready_faces: TerrainTransitionMask,
}

/// Incremental publication work produced by [`TerrainStreamer`](crate::TerrainStreamer).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TerrainPublicationDelta {
    /// Exact desired-cut generation represented by this publication.
    pub generation: u64,
    /// Current chunks that need upload or face-index replacement.
    pub upserts: Vec<TerrainPublicationUpsert>,
    /// Chunk identities no longer present in the authoritative active cut.
    pub removals: Vec<TerrainNodeId>,
}

impl TerrainPublicationDelta {
    /// True when the delta carries no publication work.
    pub fn is_empty(&self) -> bool {
        self.upserts.is_empty() && self.removals.is_empty()
    }
}

impl TerrainReadiness {
    /// True once every critical node has a current active result.
    pub const fn is_complete(self) -> bool {
        self.resolved == self.total
    }
}

/// Faces that can publish without a temporary cap.
///
/// A face is ready when its desired neighbor is active, or when the desired
/// cut contains no neighbor on that face. The latter covers generator-proven
/// solid/empty space and prevents permanent cap slabs at sparse-cut edges.
pub fn publication_face_mask(
    node: TerrainNodeId,
    active: &BTreeSet<TerrainNodeId>,
    desired: &BTreeSet<TerrainNodeId>,
) -> TerrainTransitionMask {
    let mut mask = TerrainTransitionMask::NONE;
    for face in TerrainFace::ALL {
        let candidates = neighbour_candidates(node, face);
        let has_desired = candidates
            .iter()
            .any(|candidate| desired.contains(candidate));
        let has_active = candidates
            .iter()
            .any(|candidate| active.contains(candidate));
        if !has_desired || has_active {
            mask.insert(face);
        }
    }
    mask
}

pub(super) fn publication_face_mask_maps(
    node: TerrainNodeId,
    active: &BTreeMap<TerrainNodeId, ActiveTerrainNode>,
    desired: &BTreeMap<TerrainNodeId, ActiveTerrainNode>,
) -> TerrainTransitionMask {
    let mut mask = TerrainTransitionMask::NONE;
    for face in TerrainFace::ALL {
        let candidates = neighbour_candidates(node, face);
        let has_desired = candidates
            .iter()
            .any(|candidate| desired.contains_key(candidate));
        let has_active = candidates.iter().any(|candidate| {
            active.get(candidate).is_some_and(|active_node| {
                desired
                    .get(candidate)
                    .is_some_and(|desired_node| active_node == desired_node)
            })
        });
        if !has_desired || has_active {
            mask.insert(face);
        }
    }
    mask
}

pub(super) fn neighbour_candidates(
    node: TerrainNodeId,
    face: TerrainFace,
) -> BTreeSet<TerrainNodeId> {
    let edge = i32::try_from(node.edge_bricks()).expect("streamed node edge fits i32");
    let mut adjacent = node.coordinates;
    match face {
        TerrainFace::NegativeX => adjacent.x -= 1,
        TerrainFace::PositiveX => adjacent.x += edge,
        TerrainFace::NegativeY => adjacent.y -= 1,
        TerrainFace::PositiveY => adjacent.y += edge,
        TerrainFace::NegativeZ => adjacent.z -= 1,
        TerrainFace::PositiveZ => adjacent.z += edge,
    }
    let mut candidates = BTreeSet::new();
    if let Some(equal) = TerrainNodeId::containing(adjacent, node.level) {
        candidates.insert(equal);
    }
    if let Some(coarse_level) = node.level.checked_add(1)
        && let Some(coarse) = TerrainNodeId::containing(adjacent, coarse_level)
    {
        candidates.insert(coarse);
    }
    if let Some(fine_level) = node.level.checked_sub(1) {
        let half = edge / 2;
        for first in [0, half] {
            for second in [0, half] {
                let mut coordinate = adjacent;
                match face {
                    TerrainFace::NegativeX | TerrainFace::PositiveX => {
                        coordinate.y = node.coordinates.y + first;
                        coordinate.z = node.coordinates.z + second;
                    }
                    TerrainFace::NegativeY | TerrainFace::PositiveY => {
                        coordinate.x = node.coordinates.x + first;
                        coordinate.z = node.coordinates.z + second;
                    }
                    TerrainFace::NegativeZ | TerrainFace::PositiveZ => {
                        coordinate.x = node.coordinates.x + first;
                        coordinate.y = node.coordinates.y + second;
                    }
                }
                if let Some(fine) = TerrainNodeId::containing(coordinate, fine_level) {
                    candidates.insert(fine);
                }
            }
        }
    }
    candidates.remove(&node);
    candidates
}
