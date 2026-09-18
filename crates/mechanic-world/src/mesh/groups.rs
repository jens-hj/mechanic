//! Index groups: interior triangles, and the per-face caps and transitions that stitch neighbours.

use crate::{TerrainFace, TerrainTransitionMask};

/// Bit mask for regular, per-face transition, and per-face cap triangles.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct TerrainTriangleGroupMask(pub(super) u16);

impl TerrainTriangleGroupMask {
    /// Regular modified-Marching-Cubes triangles.
    pub const REGULAR: Self = Self(1);

    pub(super) const fn transition(face: TerrainFace) -> Self {
        Self(1 << (1 + face as u16))
    }

    pub(super) const fn cap(face: TerrainFace) -> Self {
        Self(1 << (7 + face as u16))
    }

    /// True when the masks enable at least one common geometry group.
    pub const fn intersects(self, other: Self) -> bool {
        self.0 & other.0 != 0
    }

    /// True when every group in `other` is included in this mask.
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    pub(super) const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

/// Separately activatable regular, transition, and temporary cap triangles.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TerrainIndexGroups {
    /// Ordinary modified-Marching-Cubes triangles.
    pub regular: Vec<u32>,
    /// Transvoxel seam triangles on each face.
    pub transitions: [Vec<u32>; 6],
    /// Marching-squares temporary closure triangles on each face.
    pub caps: [Vec<u32>; 6],
}

impl TerrainIndexGroups {
    /// Number of final indices without allocating the combined index vector.
    pub fn final_index_count(&self, transitions: TerrainTransitionMask) -> usize {
        self.regular.len()
            + TerrainFace::ALL
                .into_iter()
                .filter(|&face| transitions.contains(face))
                .map(|face| self.transitions[face.index()].len())
                .sum::<usize>()
    }

    /// Final indices with every requested transition active and no caps.
    pub fn final_indices(&self, transitions: TerrainTransitionMask) -> Vec<u32> {
        let mut indices = Vec::with_capacity(self.final_index_count(transitions));
        indices.extend_from_slice(&self.regular);
        for face in TerrainFace::ALL {
            if transitions.contains(face) {
                indices.extend_from_slice(&self.transitions[face.index()]);
            }
        }
        indices
    }

    /// Sealed indices, adding a cap beyond any requested seam or neighbor whose
    /// generation is not yet ready.
    pub fn sealed_indices(
        &self,
        requested_transitions: TerrainTransitionMask,
        ready_faces: TerrainTransitionMask,
    ) -> Vec<u32> {
        let mut indices =
            Vec::with_capacity(self.sealed_index_count(requested_transitions, ready_faces));
        indices.extend_from_slice(&self.regular);
        for face in TerrainFace::ALL {
            if requested_transitions.contains(face) {
                indices.extend_from_slice(&self.transitions[face.index()]);
            }
            if !ready_faces.contains(face) {
                indices.extend_from_slice(&self.caps[face.index()]);
            }
        }
        indices
    }

    /// Number of sealed indices without allocating the combined index vector.
    pub fn sealed_index_count(
        &self,
        requested_transitions: TerrainTransitionMask,
        ready_faces: TerrainTransitionMask,
    ) -> usize {
        self.regular.len()
            + TerrainFace::ALL
                .into_iter()
                .map(|face| {
                    usize::from(requested_transitions.contains(face))
                        * self.transitions[face.index()].len()
                        + usize::from(!ready_faces.contains(face)) * self.caps[face.index()].len()
                })
                .sum::<usize>()
    }

    pub(super) fn final_group_mask(transitions: TerrainTransitionMask) -> TerrainTriangleGroupMask {
        let mut mask = TerrainTriangleGroupMask::REGULAR;
        for face in TerrainFace::ALL {
            if transitions.contains(face) {
                mask = mask.union(TerrainTriangleGroupMask::transition(face));
            }
        }
        mask
    }

    pub(super) fn sealed_group_mask(
        requested_transitions: TerrainTransitionMask,
        ready_faces: TerrainTransitionMask,
    ) -> TerrainTriangleGroupMask {
        let mut mask = TerrainTriangleGroupMask::REGULAR;
        for face in TerrainFace::ALL {
            if requested_transitions.contains(face) {
                mask = mask.union(TerrainTriangleGroupMask::transition(face));
            }
            if !ready_faces.contains(face) {
                mask = mask.union(TerrainTriangleGroupMask::cap(face));
            }
        }
        mask
    }
}

#[derive(Clone, Copy)]
pub(super) enum IndexGroup {
    Regular,
    Transition(TerrainFace),
    Cap(TerrainFace),
}
