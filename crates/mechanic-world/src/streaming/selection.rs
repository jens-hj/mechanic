//! Choosing the balanced active cut of the terrain octree around the points of interest.

use super::bounds_cache::TerrainBoundsCache;
use super::faces::{TerrainFace, TerrainTransitionMask};
use crate::{
    BRICK_EDGE_CELLS, BrickCoord, TerrainDensityClass, TerrainField, TerrainNodeId,
    TerrainOctreeSnapshot, WorldPosition,
};
use bevy_math::DVec3;
use std::collections::{BTreeMap, BTreeSet, HashSet};

/// One selected node and the exact mesh generation it requires.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ActiveTerrainNode {
    /// Selected octree node.
    pub id: TerrainNodeId,
    /// Latest promoted-descendant revision.
    pub generation: u64,
    /// Faces bordering the next coarser LOD.
    pub transition_mask: TerrainTransitionMask,
}

/// Observable selection counters used by streaming diagnostics and benchmarks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TerrainSelectionStats {
    /// Selected mixed nodes by LOD level.
    pub selected_by_lod: [usize; 6],
    /// Procedurally proven empty nodes omitted from the cut.
    pub rejected_empty: usize,
    /// Procedurally proven solid nodes omitted from the cut.
    pub rejected_solid: usize,
    /// Approximate bounds-cache memory after selection.
    pub cache_memory_bytes: usize,
    /// Cache hits performed by this selection.
    pub cache_hits: u64,
    /// Cache misses performed by this selection.
    pub cache_misses: u64,
}

/// Selected terrain cut and its classification counters.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TerrainSelection {
    /// Non-overlapping 2:1 mixed-node cut.
    pub nodes: Vec<ActiveTerrainNode>,
    /// Classification and cache counters.
    pub stats: TerrainSelectionStats,
}

#[derive(Debug, Default)]
pub(super) struct TerrainEditFootprints {
    pub(super) nodes_by_level: [HashSet<TerrainNodeId>; 6],
}

#[derive(Debug, Default)]
pub(super) struct TerrainSelectionState {
    pub(super) stats: TerrainSelectionStats,
    pub(super) selected: BTreeSet<TerrainNodeId>,
}

impl TerrainEditFootprints {
    pub(super) fn new(terrain: &TerrainOctreeSnapshot) -> Self {
        let mut footprints = Self::default();
        for brick in terrain.bricks() {
            let coordinate = brick.coordinate();
            for level in 1..=5 {
                let containing = TerrainNodeId::containing(coordinate, level)
                    .expect("promoted brick belongs to every streamed level");
                let edge =
                    i32::try_from(containing.edge_bricks()).expect("streamed node edge fits i32");
                let x_origins = affected_node_origins(coordinate.x, containing.coordinates.x, edge);
                let y_origins = affected_node_origins(coordinate.y, containing.coordinates.y, edge);
                let z_origins = affected_node_origins(coordinate.z, containing.coordinates.z, edge);
                for x in x_origins.into_iter().flatten() {
                    for y in y_origins.into_iter().flatten() {
                        for z in z_origins.into_iter().flatten() {
                            footprints.nodes_by_level[usize::from(level)].insert(TerrainNodeId {
                                coordinates: BrickCoord::new(x, y, z),
                                level,
                            });
                        }
                    }
                }
            }
        }
        footprints
    }

    pub(super) fn contains(&self, id: TerrainNodeId) -> bool {
        self.nodes_by_level[usize::from(id.level)].contains(&id)
    }
}

pub(super) fn affected_node_origins(
    coordinate: i32,
    containing: i32,
    edge: i32,
) -> [Option<i32>; 3] {
    let negative = (coordinate == containing)
        .then(|| containing.checked_sub(edge))
        .flatten();
    let positive_boundary = i64::from(coordinate) == i64::from(containing) + i64::from(edge) - 1;
    let positive = positive_boundary
        .then(|| containing.checked_add(edge))
        .flatten();
    [negative, Some(containing), positive]
}

/// Selects the 1 km balanced terrain cut around `focus`.
///
/// The requested bands are 20 cm through 64 m, 40 cm through 160 m,
/// 80 cm through 400 m, and 160 cm through the horizon. Promoted edits and
/// adjacent sampling footprints within 32 m select 5 cm leaves. Level one is
/// otherwise introduced only by 2:1 balancing.
pub fn select_active_nodes(
    field: &TerrainField,
    terrain: &TerrainOctreeSnapshot,
    focus: WorldPosition,
) -> Vec<ActiveTerrainNode> {
    select_active_nodes_cached(field, terrain, focus, &mut TerrainBoundsCache::default()).nodes
}

/// Selects terrain while retaining procedural bounds across reselections.
pub fn select_active_nodes_cached(
    field: &TerrainField,
    terrain: &TerrainOctreeSnapshot,
    focus: WorldPosition,
    cache: &mut TerrainBoundsCache,
) -> TerrainSelection {
    select_active_nodes_with_interests(field, terrain, focus, &[], cache)
}

/// Selects one balanced cut covering the player and additional moving bodies.
pub fn select_active_nodes_with_interests(
    field: &TerrainField,
    terrain: &TerrainOctreeSnapshot,
    focus: WorldPosition,
    interests: &[WorldPosition],
    cache: &mut TerrainBoundsCache,
) -> TerrainSelection {
    let mut focuses = vec![focus.0];
    for interest in interests {
        if interest.is_inside_world()
            && !focuses
                .iter()
                .any(|point| point.distance_squared(interest.0) < 64.0)
        {
            focuses.push(interest.0);
        }
    }
    let before = cache.access_counts();
    let mut state = TerrainSelectionState::default();
    let edit_footprints = TerrainEditFootprints::new(terrain);
    select_recursive(
        field,
        terrain,
        TerrainNodeId::ROOT,
        &focuses,
        cache,
        &edit_footprints,
        &mut state,
    );
    balance_cut(field, terrain, cache, &mut state.stats, &mut state.selected);
    let mut transition_masks = state
        .selected
        .iter()
        .copied()
        .map(|id| (id, transition_mask(id, &state.selected)))
        .collect::<BTreeMap<_, _>>();
    propagate_transition_boundary_sync(&state.selected, &mut transition_masks);
    let nodes = transition_masks
        .into_iter()
        .map(|(id, transition_mask)| ActiveTerrainNode {
            generation: mesh_dependency_generation(terrain, id, transition_mask),
            transition_mask,
            id,
        })
        .collect::<Vec<_>>();
    for node in &nodes {
        state.stats.selected_by_lod[usize::from(node.id.level)] += 1;
    }
    cache.evict_far_from(focus.0);
    let after = cache.access_counts();
    state.stats.cache_hits = after.0.saturating_sub(before.0);
    state.stats.cache_misses = after.1.saturating_sub(before.1);
    state.stats.cache_memory_bytes = cache.memory_bytes();
    TerrainSelection {
        nodes,
        stats: state.stats,
    }
}

pub(super) fn mesh_dependency_generation(
    terrain: &TerrainOctreeSnapshot,
    id: TerrainNodeId,
    transition_mask: TerrainTransitionMask,
) -> u64 {
    let stride = 1_i64 << id.level;
    let minimum = id.minimum_cell_i64();
    let maximum = id.maximum_cell_exclusive_i64();
    let (lower_halo, upper_halo) = if transition_mask == TerrainTransitionMask::NONE {
        (stride + 1, 2 * stride)
    } else {
        // Boundary synchronization derives a coarse gradient one coarse sample
        // beyond the node, and each sample conservatively covers its full cell range.
        (2 * stride + 1, 4 * stride)
    };
    let cell_to_brick = |cell: i64| {
        i32::try_from(cell.div_euclid(i64::from(BRICK_EDGE_CELLS)))
            .expect("streamed mesh dependency lies in brick coordinate space")
    };
    terrain.latest_revision_between(
        BrickCoord::new(
            cell_to_brick(minimum[0] - lower_halo),
            cell_to_brick(minimum[1] - lower_halo),
            cell_to_brick(minimum[2] - lower_halo),
        ),
        BrickCoord::new(
            cell_to_brick(maximum[0] + upper_halo - 1),
            cell_to_brick(maximum[1] + upper_halo - 1),
            cell_to_brick(maximum[2] + upper_halo - 1),
        ),
    )
}

pub(super) fn select_recursive(
    field: &TerrainField,
    terrain: &TerrainOctreeSnapshot,
    id: TerrainNodeId,
    focus: &[DVec3],
    cache: &mut TerrainBoundsCache,
    edit_footprints: &TerrainEditFootprints,
    state: &mut TerrainSelectionState,
) {
    let (minimum, maximum) = node_bounds(id);
    let distance_squared = focus
        .iter()
        .map(|&point| horizontal_distance_squared_to_bounds(point, minimum, maximum))
        .fold(f64::INFINITY, f64::min);
    if distance_squared > 1_000_000.0 || maximum.y < -128.0 || minimum.y > 256.0 {
        return;
    }

    if id.level > 5 {
        for child in id.children().expect("root descendants have children") {
            select_recursive(field, terrain, child, focus, cache, edit_footprints, state);
        }
        return;
    }

    match classify_node(field, terrain, id, cache) {
        TerrainDensityClass::Empty => {
            state.stats.rejected_empty += 1;
            return;
        }
        TerrainDensityClass::Solid => {
            state.stats.rejected_solid += 1;
            return;
        }
        TerrainDensityClass::Mixed => {}
    }
    let has_edits = edit_footprints.contains(id);
    let target_level = if has_edits && distance_squared <= 1_024.0 {
        0
    } else if distance_squared <= 4_096.0 {
        2
    } else if distance_squared <= 25_600.0 {
        3
    } else if distance_squared <= 160_000.0 {
        4
    } else {
        5
    };
    if id.level > target_level {
        for child in id.children().expect("a refined node has children") {
            select_recursive(field, terrain, child, focus, cache, edit_footprints, state);
        }
    } else {
        state.selected.insert(id);
    }
}

pub(super) fn classify_node(
    field: &TerrainField,
    terrain: &TerrainOctreeSnapshot,
    id: TerrainNodeId,
    cache: &mut TerrainBoundsCache,
) -> TerrainDensityClass {
    let edit_summary = terrain.node(id);
    let edited = edit_summary.is_some_and(|node| node.promoted_descendants != 0);
    if let Some(summary) = edit_summary
        && id
            .edge_bricks()
            .checked_pow(3)
            .and_then(|count| u64::try_from(count).ok())
            .is_some_and(|count| count == summary.promoted_descendants)
    {
        return if summary.maximum_density <= 0.0 {
            TerrainDensityClass::Empty
        } else if summary.minimum_density > 0.0 {
            TerrainDensityClass::Solid
        } else {
            TerrainDensityClass::Mixed
        };
    }
    let bounds = cache.bounds(field, id);
    let minimum_cell = id.minimum_cell_i64();
    let maximum_cell = id.maximum_cell_exclusive_i64();
    let minimum =
        DVec3::from_array(minimum_cell.map(|cell| cell as f64 * crate::TERRAIN_CELL_METERS));
    let maximum =
        DVec3::from_array(maximum_cell.map(|cell| cell as f64 * crate::TERRAIN_CELL_METERS));
    let sample_minimum_y = minimum.y + crate::TERRAIN_CELL_METERS * 0.5;
    let sample_maximum_y = maximum.y - crate::TERRAIN_CELL_METERS * 0.5;
    let margin = bounds.margin;
    if bounds.maximum_surface - sample_minimum_y < -margin {
        TerrainDensityClass::Empty
    } else if bounds.minimum_surface - sample_maximum_y > margin
        && !edited
        && !field.cave_intersects_bounds(minimum, maximum)
    {
        TerrainDensityClass::Solid
    } else {
        TerrainDensityClass::Mixed
    }
}

pub(super) fn node_bounds(id: TerrainNodeId) -> (DVec3, DVec3) {
    let minimum = id
        .minimum_cell_i64()
        .map(|cell| cell as f64 * crate::TERRAIN_CELL_METERS);
    let maximum = id
        .maximum_cell_exclusive_i64()
        .map(|cell| cell as f64 * crate::TERRAIN_CELL_METERS);
    (DVec3::from_array(minimum), DVec3::from_array(maximum))
}

pub(super) fn horizontal_distance_squared_to_bounds(
    focus: DVec3,
    minimum: DVec3,
    maximum: DVec3,
) -> f64 {
    let dx = if focus.x < minimum.x {
        minimum.x - focus.x
    } else if focus.x > maximum.x {
        focus.x - maximum.x
    } else {
        0.0
    };
    let dz = if focus.z < minimum.z {
        minimum.z - focus.z
    } else if focus.z > maximum.z {
        focus.z - maximum.z
    } else {
        0.0
    };
    dx.mul_add(dx, dz * dz)
}

pub(super) fn balance_cut(
    field: &TerrainField,
    terrain: &TerrainOctreeSnapshot,
    cache: &mut TerrainBoundsCache,
    stats: &mut TerrainSelectionStats,
    selected: &mut BTreeSet<TerrainNodeId>,
) {
    loop {
        let mut split = BTreeSet::new();
        for &node in selected.iter() {
            for face in TerrainFace::ALL {
                let Some(neighbour) = adjacent_leaf(node, face) else {
                    continue;
                };
                let Some(owner) = owner_of_leaf(selected, neighbour) else {
                    continue;
                };
                if owner.level > node.level + 1 {
                    split.insert(owner);
                }
            }
        }
        if split.is_empty() {
            break;
        }
        for coarse in split {
            selected.remove(&coarse);
            for child in coarse.children().expect("a coarse neighbour can split") {
                match classify_node(field, terrain, child, cache) {
                    TerrainDensityClass::Mixed => {
                        selected.insert(child);
                    }
                    TerrainDensityClass::Empty => stats.rejected_empty += 1,
                    TerrainDensityClass::Solid => stats.rejected_solid += 1,
                }
            }
        }
    }
}

pub(super) fn transition_mask(
    node: TerrainNodeId,
    selected: &BTreeSet<TerrainNodeId>,
) -> TerrainTransitionMask {
    let mut mask = TerrainTransitionMask::NONE;
    for face in TerrainFace::ALL {
        let Some(neighbour) = adjacent_leaf(node, face) else {
            continue;
        };
        if owner_of_leaf(selected, neighbour).is_some_and(|owner| owner.level == node.level + 1) {
            mask.insert(face);
        }
    }
    mask
}

pub(super) fn propagate_transition_boundary_sync(
    selected: &BTreeSet<TerrainNodeId>,
    masks: &mut BTreeMap<TerrainNodeId, TerrainTransitionMask>,
) {
    let transitions = masks
        .iter()
        .flat_map(|(&node, mask)| {
            TerrainFace::ALL
                .into_iter()
                .filter(move |&face| mask.contains(face))
                .map(move |face| (node, face))
        })
        .collect::<Vec<_>>();
    for (node, transition_face) in transitions {
        let transition_bit = 1 << transition_face as u8;
        masks
            .get_mut(&node)
            .expect("selected transition node exists")
            .insert_face_boundary_features(transition_face);

        for side in TerrainFace::ALL {
            if face_axis(side) == face_axis(transition_face) {
                continue;
            }
            if let Some(neighbor) = equal_lod_neighbor(node, side)
                && selected.contains(&neighbor)
            {
                masks
                    .get_mut(&neighbor)
                    .expect("selected equal neighbor exists")
                    .insert_boundary_feature(transition_bit | (1 << side.opposite() as u8));
            }
        }
        for first in TerrainFace::ALL {
            for second in TerrainFace::ALL {
                if face_axis(first) == face_axis(transition_face)
                    || face_axis(second) == face_axis(transition_face)
                    || face_axis(first) >= face_axis(second)
                {
                    continue;
                }
                let Some(first_neighbor) = equal_lod_neighbor(node, first) else {
                    continue;
                };
                let Some(diagonal) = equal_lod_neighbor(first_neighbor, second) else {
                    continue;
                };
                if selected.contains(&diagonal) {
                    masks
                        .get_mut(&diagonal)
                        .expect("selected diagonal neighbor exists")
                        .insert_boundary_feature(
                            transition_bit
                                | (1 << first.opposite() as u8)
                                | (1 << second.opposite() as u8),
                        );
                }
            }
        }
    }
}

pub(super) const fn face_axis(face: TerrainFace) -> u8 {
    face as u8 / 2
}

pub(super) fn equal_lod_neighbor(node: TerrainNodeId, face: TerrainFace) -> Option<TerrainNodeId> {
    let edge = i32::try_from(node.edge_bricks()).ok()?;
    let mut coordinates = node.coordinates;
    match face {
        TerrainFace::NegativeX => coordinates.x = coordinates.x.checked_sub(edge)?,
        TerrainFace::PositiveX => coordinates.x = coordinates.x.checked_add(edge)?,
        TerrainFace::NegativeY => coordinates.y = coordinates.y.checked_sub(edge)?,
        TerrainFace::PositiveY => coordinates.y = coordinates.y.checked_add(edge)?,
        TerrainFace::NegativeZ => coordinates.z = coordinates.z.checked_sub(edge)?,
        TerrainFace::PositiveZ => coordinates.z = coordinates.z.checked_add(edge)?,
    }
    Some(TerrainNodeId {
        coordinates,
        level: node.level,
    })
}

pub(super) fn owner_of_leaf(
    selected: &BTreeSet<TerrainNodeId>,
    leaf: BrickCoord,
) -> Option<TerrainNodeId> {
    (0..=5).find_map(|level| {
        let id = TerrainNodeId::containing(leaf, level)?;
        selected.contains(&id).then_some(id)
    })
}

pub(super) fn adjacent_leaf(node: TerrainNodeId, face: TerrainFace) -> Option<BrickCoord> {
    let edge = i32::try_from(node.edge_bricks()).ok()?;
    let middle = edge / 2;
    let mut coordinate = node.coordinates;
    match face {
        TerrainFace::NegativeX => coordinate.x = coordinate.x.checked_sub(1)?,
        TerrainFace::PositiveX => coordinate.x = coordinate.x.checked_add(edge)?,
        TerrainFace::NegativeY => coordinate.y = coordinate.y.checked_sub(1)?,
        TerrainFace::PositiveY => coordinate.y = coordinate.y.checked_add(edge)?,
        TerrainFace::NegativeZ => coordinate.z = coordinate.z.checked_sub(1)?,
        TerrainFace::PositiveZ => coordinate.z = coordinate.z.checked_add(edge)?,
    }
    match face {
        TerrainFace::NegativeX | TerrainFace::PositiveX => {
            coordinate.y = coordinate.y.checked_add(middle)?;
            coordinate.z = coordinate.z.checked_add(middle)?;
        }
        TerrainFace::NegativeY | TerrainFace::PositiveY => {
            coordinate.x = coordinate.x.checked_add(middle)?;
            coordinate.z = coordinate.z.checked_add(middle)?;
        }
        TerrainFace::NegativeZ | TerrainFace::PositiveZ => {
            coordinate.x = coordinate.x.checked_add(middle)?;
            coordinate.y = coordinate.y.checked_add(middle)?;
        }
    }
    Some(coordinate)
}
