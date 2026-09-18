//! Balanced adaptive-octree selection and generation-aware streaming state.

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    time::{Duration, Instant},
};

use bevy_math::DVec3;

use crate::{
    BRICK_EDGE_CELLS, BRICK_EDGE_METERS, BrickCoord, TerrainDensityClass, TerrainField,
    TerrainNodeId, TerrainOctreeSnapshot, WorldPosition,
};

/// One of the six axis-aligned node faces.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum TerrainFace {
    /// Negative X.
    NegativeX = 0,
    /// Positive X.
    PositiveX = 1,
    /// Negative Y.
    NegativeY = 2,
    /// Positive Y.
    PositiveY = 3,
    /// Negative Z.
    NegativeZ = 4,
    /// Positive Z.
    PositiveZ = 5,
}

impl TerrainFace {
    /// Stable face order used by masks and geometry arrays.
    pub const ALL: [Self; 6] = [
        Self::NegativeX,
        Self::PositiveX,
        Self::NegativeY,
        Self::PositiveY,
        Self::NegativeZ,
        Self::PositiveZ,
    ];

    /// Array index.
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Opposite face.
    #[must_use]
    pub const fn opposite(self) -> Self {
        match self {
            Self::NegativeX => Self::PositiveX,
            Self::PositiveX => Self::NegativeX,
            Self::NegativeY => Self::PositiveY,
            Self::PositiveY => Self::NegativeY,
            Self::NegativeZ => Self::PositiveZ,
            Self::PositiveZ => Self::NegativeZ,
        }
    }
}

/// Faces bordering a coarser node plus boundary features that must share its
/// nested scalar samples with neighboring fine chunks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TerrainTransitionMask(u32);

const BOUNDARY_FEATURES: [u8; 26] = [
    0b00_0001, 0b00_0010, 0b00_0100, 0b00_1000, 0b01_0000, 0b10_0000, 0b00_0101, 0b00_1001,
    0b00_0110, 0b00_1010, 0b01_0001, 0b10_0001, 0b01_0010, 0b10_0010, 0b01_0100, 0b10_0100,
    0b01_1000, 0b10_1000, 0b01_0101, 0b10_0101, 0b01_1001, 0b10_1001, 0b01_0110, 0b10_0110,
    0b01_1010, 0b10_1010,
];

impl TerrainTransitionMask {
    /// Empty mask.
    pub const NONE: Self = Self(0);

    /// Creates a validated six-face mask.
    pub fn from_bits(bits: u8) -> Self {
        let mut mask = Self(u32::from(bits & 0x3f));
        for face in TerrainFace::ALL {
            if mask.contains(face) {
                mask.insert_face_boundary_features(face);
            }
        }
        mask
    }

    /// Raw six bits.
    pub const fn bits(self) -> u8 {
        (self.0 & 0x3f) as u8
    }

    /// True when `face` needs a transition cell.
    pub const fn contains(self, face: TerrainFace) -> bool {
        self.0 & (1 << face as u8) != 0
    }

    fn insert(&mut self, face: TerrainFace) {
        self.0 |= 1 << face as u8;
    }

    fn insert_boundary_feature(&mut self, faces: u8) {
        if let Some(index) = BOUNDARY_FEATURES
            .iter()
            .position(|&feature| feature == faces)
        {
            self.0 |= 1 << (6 + index);
        }
    }

    fn insert_face_boundary_features(&mut self, transition_face: TerrainFace) {
        let transition_bit = 1 << transition_face as u8;
        self.insert_boundary_feature(transition_bit);
        for side in TerrainFace::ALL {
            if face_axis(side) == face_axis(transition_face) {
                continue;
            }
            self.insert_boundary_feature(transition_bit | (1 << side as u8));
        }
        for first in TerrainFace::ALL {
            for second in TerrainFace::ALL {
                if face_axis(first) != face_axis(transition_face)
                    && face_axis(second) != face_axis(transition_face)
                    && face_axis(first) < face_axis(second)
                {
                    self.insert_boundary_feature(
                        transition_bit | (1 << first as u8) | (1 << second as u8),
                    );
                }
            }
        }
    }

    pub(crate) fn synchronizes_boundary_feature(self, faces: u8) -> bool {
        BOUNDARY_FEATURES
            .iter()
            .position(|&feature| feature == faces)
            .is_some_and(|index| self.0 & (1 << (6 + index)) != 0)
    }
}

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

const PROCEDURAL_BOUNDS_CACHE_BYTES: usize = 32 * 1024 * 1024;
const PROCEDURAL_BOUNDS_ENTRY_BYTES: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct ProceduralBoundsKey {
    level: u8,
    x: i32,
    z: i32,
}

#[derive(Clone, Copy, Debug)]
struct ProceduralBounds {
    minimum_surface: f64,
    maximum_surface: f64,
    margin: f64,
}

/// Per-world cache of conservative procedural surface bounds used by octree
/// selection. Entries are independent of vertical coordinate and therefore
/// serve every y node in the same horizontal octree column.
#[derive(Clone, Debug, Default)]
pub struct TerrainBoundsCache {
    entries: HashMap<ProceduralBoundsKey, ProceduralBounds>,
    hits: u64,
    misses: u64,
}

impl TerrainBoundsCache {
    /// Approximate retained cache memory, capped at 32 MiB.
    pub fn memory_bytes(&self) -> usize {
        self.entries
            .len()
            .saturating_mul(PROCEDURAL_BOUNDS_ENTRY_BYTES)
    }

    /// Cumulative cache hits and misses.
    pub const fn access_counts(&self) -> (u64, u64) {
        (self.hits, self.misses)
    }

    fn bounds(&mut self, field: &TerrainField, id: TerrainNodeId) -> ProceduralBounds {
        let key = ProceduralBoundsKey {
            level: id.level,
            x: id.coordinates.x,
            z: id.coordinates.z,
        };
        if let Some(bounds) = self.entries.get(&key).copied() {
            self.hits = self.hits.saturating_add(1);
            return bounds;
        }
        self.misses = self.misses.saturating_add(1);
        let minimum = id.minimum_cell_i64();
        let maximum = id.maximum_cell_exclusive_i64();
        let extent = maximum[0] - minimum[0] - 1;
        let x_coordinates = [
            minimum[0],
            minimum[0] + extent / 4,
            minimum[0] + extent / 2,
            minimum[0] + extent * 3 / 4,
            maximum[0] - 1,
        ];
        let extent = maximum[2] - minimum[2] - 1;
        let z_coordinates = [
            minimum[2],
            minimum[2] + extent / 4,
            minimum[2] + extent / 2,
            minimum[2] + extent * 3 / 4,
            maximum[2] - 1,
        ];
        let mut bounds = ProceduralBounds {
            minimum_surface: f64::INFINITY,
            maximum_surface: f64::NEG_INFINITY,
            margin: 0.0,
        };
        let mut surfaces = [[0.0_f64; 5]; 5];
        for (z_index, z) in z_coordinates.into_iter().enumerate() {
            for (x_index, x) in x_coordinates.into_iter().enumerate() {
                let x = x as f64 * crate::TERRAIN_CELL_METERS + crate::TERRAIN_CELL_METERS * 0.5;
                let z = z as f64 * crate::TERRAIN_CELL_METERS + crate::TERRAIN_CELL_METERS * 0.5;
                let surface = field.surface_height(x, z);
                surfaces[z_index][x_index] = surface;
                bounds.minimum_surface = bounds.minimum_surface.min(surface);
                bounds.maximum_surface = bounds.maximum_surface.max(surface);
            }
        }
        let mut maximum_step = 0.0_f64;
        for z in 0..5 {
            for x in 0..5 {
                if x + 1 < 5 {
                    maximum_step = maximum_step.max((surfaces[z][x + 1] - surfaces[z][x]).abs());
                }
                if z + 1 < 5 {
                    maximum_step = maximum_step.max((surfaces[z + 1][x] - surfaces[z][x]).abs());
                }
            }
        }
        let maximum_margin = id.edge_bricks() as f64 * BRICK_EDGE_METERS * 0.25;
        bounds.margin = maximum_step.mul_add(1.5, 0.25).min(maximum_margin);
        self.entries.insert(key, bounds);
        bounds
    }

    fn evict_far_from(&mut self, focus: DVec3) {
        let maximum_entries = PROCEDURAL_BOUNDS_CACHE_BYTES / PROCEDURAL_BOUNDS_ENTRY_BYTES;
        if self.entries.len() <= maximum_entries {
            return;
        }
        let target = maximum_entries * 7 / 8;
        let mut keys = self.entries.keys().copied().collect::<Vec<_>>();
        keys.sort_by(|first, second| {
            let distance = |key: ProceduralBoundsKey| {
                let edge = (1_i64 << key.level) as f64 * BRICK_EDGE_METERS;
                let x = f64::from(key.x) * BRICK_EDGE_METERS + edge * 0.5;
                let z = f64::from(key.z) * BRICK_EDGE_METERS + edge * 0.5;
                (x - focus.x).mul_add(x - focus.x, (z - focus.z) * (z - focus.z))
            };
            distance(*second).total_cmp(&distance(*first))
        });
        for key in keys.into_iter().take(self.entries.len() - target) {
            self.entries.remove(&key);
        }
    }
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
struct TerrainEditFootprints {
    nodes_by_level: [HashSet<TerrainNodeId>; 6],
}

#[derive(Debug, Default)]
struct TerrainSelectionState {
    stats: TerrainSelectionStats,
    selected: BTreeSet<TerrainNodeId>,
}

impl TerrainEditFootprints {
    fn new(terrain: &TerrainOctreeSnapshot) -> Self {
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

    fn contains(&self, id: TerrainNodeId) -> bool {
        self.nodes_by_level[usize::from(id.level)].contains(&id)
    }
}

fn affected_node_origins(coordinate: i32, containing: i32, edge: i32) -> [Option<i32>; 3] {
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

fn mesh_dependency_generation(
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

fn select_recursive(
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

fn classify_node(
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

fn node_bounds(id: TerrainNodeId) -> (DVec3, DVec3) {
    let minimum = id
        .minimum_cell_i64()
        .map(|cell| cell as f64 * crate::TERRAIN_CELL_METERS);
    let maximum = id
        .maximum_cell_exclusive_i64()
        .map(|cell| cell as f64 * crate::TERRAIN_CELL_METERS);
    (DVec3::from_array(minimum), DVec3::from_array(maximum))
}

fn horizontal_distance_squared_to_bounds(focus: DVec3, minimum: DVec3, maximum: DVec3) -> f64 {
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

fn balance_cut(
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

fn transition_mask(
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

fn propagate_transition_boundary_sync(
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

const fn face_axis(face: TerrainFace) -> u8 {
    face as u8 / 2
}

fn equal_lod_neighbor(node: TerrainNodeId, face: TerrainFace) -> Option<TerrainNodeId> {
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

fn owner_of_leaf(selected: &BTreeSet<TerrainNodeId>, leaf: BrickCoord) -> Option<TerrainNodeId> {
    (0..=5).find_map(|level| {
        let id = TerrainNodeId::containing(leaf, level)?;
        selected.contains(&id).then_some(id)
    })
}

fn adjacent_leaf(node: TerrainNodeId, face: TerrainFace) -> Option<BrickCoord> {
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

/// Generation-aware node streaming state shared by render and collision owners.
#[derive(Clone, Debug, Default)]
pub struct TerrainStreamer {
    cut_generation: u64,
    desired: BTreeMap<TerrainNodeId, ActiveTerrainNode>,
    pending: BTreeMap<TerrainNodeId, PendingTerrainNode>,
    staged: BTreeMap<TerrainNodeId, ActiveTerrainNode>,
    active: BTreeMap<TerrainNodeId, ActiveTerrainNode>,
    pinned: BTreeSet<TerrainNodeId>,
    critical: BTreeSet<TerrainNodeId>,
    seam_dependencies: BTreeSet<TerrainNodeId>,
    dirty_publication: BTreeSet<TerrainNodeId>,
    replacement_needs: BTreeMap<TerrainNodeId, BTreeSet<TerrainNodeId>>,
    replacement_owners: BTreeMap<TerrainNodeId, BTreeSet<TerrainNodeId>>,
}

#[derive(Clone, Debug)]
struct PendingTerrainNode {
    node: ActiveTerrainNode,
    queued_at: Instant,
}

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

/// Incremental publication work produced by [`TerrainStreamer`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TerrainPublicationDelta {
    /// Exact desired-cut generation represented by this publication.
    pub generation: u64,
    /// Current chunks that need upload or face-index replacement.
    pub upserts: Vec<TerrainPublicationUpsert>,
    /// Chunk identities no longer present in the authoritative active cut.
    pub removals: Vec<TerrainNodeId>,
}

/// Incremental difference between two selected terrain cuts.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TerrainSelectionDelta {
    /// Exact terrain edit generation used by the selector.
    pub generation: u64,
    /// Added or generation/transition-modified nodes.
    pub upserts: Vec<ActiveTerrainNode>,
    /// Nodes absent from the replacement cut.
    pub removals: Vec<TerrainNodeId>,
}

impl TerrainSelectionDelta {
    /// Computes a stable delta without changing either complete cut.
    pub fn between(
        generation: u64,
        previous: impl IntoIterator<Item = ActiveTerrainNode>,
        current: impl IntoIterator<Item = ActiveTerrainNode>,
    ) -> Self {
        let previous = previous
            .into_iter()
            .map(|node| (node.id, node))
            .collect::<BTreeMap<_, _>>();
        let current = current
            .into_iter()
            .map(|node| (node.id, node))
            .collect::<BTreeMap<_, _>>();
        Self {
            generation,
            upserts: current
                .iter()
                .filter_map(|(&id, &node)| (previous.get(&id) != Some(&node)).then_some(node))
                .collect(),
            removals: previous
                .keys()
                .filter(|id| !current.contains_key(id))
                .copied()
                .collect(),
        }
    }

    /// True when the selected cut is unchanged.
    pub fn is_empty(&self) -> bool {
        self.upserts.is_empty() && self.removals.is_empty()
    }
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

impl TerrainStreamer {
    /// Reconciles a newly selected cut and queues missing or stale generations.
    pub fn set_desired(&mut self, cut: impl IntoIterator<Item = ActiveTerrainNode>) {
        let desired = cut.into_iter().map(|node| (node.id, node)).collect();
        if self.desired != desired {
            self.cut_generation = self.cut_generation.wrapping_add(1).max(1);
        }
        self.desired = desired;
        self.pending
            .retain(|id, pending| self.desired.get(id) == Some(&pending.node));
        self.staged
            .retain(|id, staged| self.desired.get(id) == Some(staged));
        for (&id, &node) in &self.desired {
            if self.active.get(&id) != Some(&node) && self.staged.get(&id) != Some(&node) {
                self.pending
                    .entry(id)
                    .or_insert_with(|| PendingTerrainNode {
                        node,
                        queued_at: Instant::now(),
                    });
            }
        }
        let obsolete = self
            .active
            .iter()
            .filter_map(|(&id, active)| (self.desired.get(&id) != Some(active)).then_some(id))
            .collect::<BTreeSet<_>>();
        self.replacement_needs.clear();
        self.replacement_owners.clear();
        for &replacement in self.desired.keys() {
            let mut ancestor = Some(replacement);
            while let Some(candidate) = ancestor {
                if obsolete.contains(&candidate) {
                    self.replacement_needs
                        .entry(candidate)
                        .or_default()
                        .insert(replacement);
                    self.replacement_owners
                        .entry(replacement)
                        .or_default()
                        .insert(candidate);
                }
                ancestor = candidate.parent();
            }
        }
        for &old in &obsolete {
            let mut ancestor = old.parent();
            while let Some(candidate) = ancestor {
                if self.desired.contains_key(&candidate) {
                    self.replacement_needs
                        .entry(old)
                        .or_default()
                        .insert(candidate);
                    self.replacement_owners
                        .entry(candidate)
                        .or_default()
                        .insert(old);
                    break;
                }
                ancestor = candidate.parent();
            }
        }
        for id in obsolete {
            if !self.replacement_needs.contains_key(&id) && !self.pinned.contains(&id) {
                self.active.remove(&id);
                self.mark_publication_dirty(id);
            }
        }
        self.rebuild_seam_dependencies();
    }

    /// Pins nodes overlapped by construction bodies.
    pub fn set_pinned(&mut self, nodes: impl IntoIterator<Item = TerrainNodeId>) {
        self.pinned.clear();
        self.pinned.extend(nodes);
    }

    /// Sets nodes that must resolve before local world entry.
    pub fn set_critical_nodes(&mut self, nodes: impl IntoIterator<Item = TerrainNodeId>) {
        self.critical.clear();
        self.critical.extend(nodes);
    }

    /// Highest-priority pending request not already in flight.
    pub fn next_request(
        &self,
        in_flight: &BTreeSet<TerrainNodeId>,
        focus: WorldPosition,
    ) -> Option<ActiveTerrainNode> {
        self.pending
            .values()
            .map(|pending| pending.node)
            .filter(|node| !in_flight.contains(&node.id))
            .min_by(|first, second| {
                let priority = |node: &ActiveTerrainNode| {
                    let centre = DVec3::new(
                        (f64::from(node.id.coordinates.x) + node.id.edge_bricks() as f64 * 0.5)
                            * BRICK_EDGE_METERS,
                        (f64::from(node.id.coordinates.y) + node.id.edge_bricks() as f64 * 0.5)
                            * BRICK_EDGE_METERS,
                        (f64::from(node.id.coordinates.z) + node.id.edge_bricks() as f64 * 0.5)
                            * BRICK_EDGE_METERS,
                    );
                    (
                        !(self.pinned.contains(&node.id) || self.critical.contains(&node.id)),
                        std::cmp::Reverse(node.generation),
                        !self.seam_dependencies.contains(&node.id),
                        centre.distance_squared(focus.0),
                    )
                };
                let first_priority = priority(first);
                let second_priority = priority(second);
                first_priority
                    .0
                    .cmp(&second_priority.0)
                    .then_with(|| first_priority.1.cmp(&second_priority.1))
                    .then_with(|| first_priority.2.cmp(&second_priority.2))
                    .then_with(|| first_priority.3.total_cmp(&second_priority.3))
                    .then_with(|| first.id.cmp(&second.id))
            })
    }

    /// Removes a request from the pending queue once a worker accepts it.
    pub fn mark_started(&mut self, node: ActiveTerrainNode) {
        if self.pending.get(&node.id).map(|pending| pending.node) == Some(node) {
            self.pending.remove(&node.id);
        }
    }

    /// Stages a result if it still matches the desired generation and seam mask.
    pub fn stage(&mut self, node: ActiveTerrainNode) -> bool {
        if self.desired.get(&node.id) != Some(&node) {
            return false;
        }
        self.staged.insert(node.id, node);
        true
    }

    /// Atomically activates a complete local replacement at a safe
    /// render/physics boundary.
    pub fn activate(&mut self, id: TerrainNodeId) -> Vec<ActiveTerrainNode> {
        if !self.staged.contains_key(&id) {
            return Vec::new();
        }
        let mut old_nodes = self
            .replacement_owners
            .get(&id)
            .cloned()
            .unwrap_or_default();
        if old_nodes.is_empty() {
            let Some(node) = self.staged.remove(&id) else {
                return Vec::new();
            };
            self.active.insert(id, node);
            self.mark_publication_dirty(id);
            return vec![node];
        }

        let mut replacements = BTreeSet::new();
        loop {
            let previous_old_count = old_nodes.len();
            for old in old_nodes.clone() {
                replacements.extend(
                    self.replacement_needs
                        .get(&old)
                        .into_iter()
                        .flatten()
                        .copied(),
                );
            }
            for replacement in replacements.clone() {
                old_nodes.extend(
                    self.replacement_owners
                        .get(&replacement)
                        .into_iter()
                        .flatten()
                        .copied(),
                );
            }
            if old_nodes.len() == previous_old_count {
                break;
            }
        }
        let complete = replacements.iter().all(|replacement| {
            self.desired.get(replacement) == self.active.get(replacement)
                || self.staged.get(replacement) == self.desired.get(replacement)
        });
        if !complete {
            return Vec::new();
        }

        for old in &old_nodes {
            self.active.remove(old);
            self.mark_publication_dirty(*old);
            if let Some(needs) = self.replacement_needs.remove(old) {
                for replacement in needs {
                    if let Some(owners) = self.replacement_owners.get_mut(&replacement) {
                        owners.remove(old);
                        if owners.is_empty() {
                            self.replacement_owners.remove(&replacement);
                        }
                    }
                }
            }
        }
        let mut activated = Vec::new();
        for replacement in replacements {
            if let Some(node) = self.staged.remove(&replacement) {
                self.active.insert(replacement, node);
                self.mark_publication_dirty(replacement);
                activated.push(node);
            }
        }
        activated
    }

    /// Pending plus staged work count.
    pub fn backlog(&self) -> usize {
        self.pending.len() + self.staged.len()
    }

    /// Current active cut.
    pub fn active(&self) -> impl Iterator<Item = ActiveTerrainNode> + '_ {
        self.active.values().copied()
    }

    /// Active nodes whose generation and transition dependency exactly match
    /// the current desired cut. Faces touching any other active generation
    /// must remain capped.
    pub fn current_active(&self) -> impl Iterator<Item = ActiveTerrainNode> + '_ {
        self.active
            .values()
            .filter(|node| self.desired.get(&node.id) == Some(*node))
            .copied()
    }

    /// Current desired cut, including work not active yet.
    pub fn desired(&self) -> impl Iterator<Item = ActiveTerrainNode> + '_ {
        self.desired.values().copied()
    }

    /// Current startup-region completion. Empty active chunks count as resolved.
    pub fn local_readiness(&self) -> TerrainReadiness {
        let mut readiness = TerrainReadiness::default();
        for id in &self.critical {
            let Some(desired) = self.desired.get(id) else {
                continue;
            };
            readiness.total += 1;
            readiness.resolved += usize::from(self.active.get(id) == Some(desired));
        }
        readiness
    }

    /// Age of the oldest request still waiting for a worker.
    pub fn oldest_queue_age(&self) -> Duration {
        self.pending
            .values()
            .map(|pending| pending.queued_at.elapsed())
            .max()
            .unwrap_or_default()
    }

    /// Nodes whose mesh or face readiness may need republishing.
    pub fn take_dirty_publication(&mut self) -> BTreeSet<TerrainNodeId> {
        core::mem::take(&mut self.dirty_publication)
    }

    /// Takes the incremental render/collision publication delta.
    ///
    /// Face readiness is computed directly against streamer-owned maps, so the
    /// app does not need to rebuild complete active and desired sets merely to
    /// publish a handful of changed chunks.
    pub fn take_publication_delta(&mut self) -> TerrainPublicationDelta {
        let dirty = core::mem::take(&mut self.dirty_publication);
        let mut delta = TerrainPublicationDelta {
            generation: self.cut_generation,
            ..TerrainPublicationDelta::default()
        };
        for id in dirty {
            if let Some(&node) = self.active.get(&id)
                && self.desired.get(&id) == Some(&node)
            {
                delta.upserts.push(TerrainPublicationUpsert {
                    node,
                    ready_faces: publication_face_mask_maps(id, &self.active, &self.desired),
                });
            } else {
                delta.removals.push(id);
            }
        }
        delta
    }

    /// True when publication bookkeeping has work for the main thread.
    pub fn has_dirty_publication(&self) -> bool {
        !self.dirty_publication.is_empty()
    }

    /// Returns publication work that did not fit the current main-thread budget.
    pub fn defer_publication(&mut self, nodes: impl IntoIterator<Item = TerrainNodeId>) {
        self.dirty_publication.extend(nodes);
    }

    fn rebuild_seam_dependencies(&mut self) {
        self.seam_dependencies.clear();
        let selected = self.desired.keys().copied().collect::<BTreeSet<_>>();
        for node in self.desired.values() {
            if node.transition_mask == TerrainTransitionMask::NONE {
                continue;
            }
            self.seam_dependencies.insert(node.id);
            for face in TerrainFace::ALL {
                if !node.transition_mask.contains(face) {
                    continue;
                }
                if let Some(neighbour) = adjacent_leaf(node.id, face)
                    && let Some(owner) = owner_of_leaf(&selected, neighbour)
                {
                    self.seam_dependencies.insert(owner);
                }
            }
        }
    }

    fn mark_publication_dirty(&mut self, id: TerrainNodeId) {
        self.dirty_publication.insert(id);
        for face in TerrainFace::ALL {
            for neighbour in neighbour_candidates(id, face) {
                if self.desired.contains_key(&neighbour) || self.active.contains_key(&neighbour) {
                    self.dirty_publication.insert(neighbour);
                }
            }
        }
    }
}

/// Terrain worker count during interactive play.
///
/// Four logical cores remain available for Bevy's main, render, IO, and async
/// work. This yields six terrain workers on the reference 10-core M1 Pro.
pub fn terrain_worker_count() -> usize {
    std::thread::available_parallelism()
        .map_or(4, usize::from)
        .saturating_sub(4)
        .clamp(1, 8)
}

/// Terrain worker count while the opaque world-loading screen is active.
pub fn terrain_loading_worker_count() -> usize {
    std::thread::available_parallelism()
        .map_or(4, usize::from)
        .saturating_sub(2)
        .clamp(1, 8)
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

fn publication_face_mask_maps(
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

fn neighbour_candidates(node: TerrainNodeId, face: TerrainFace) -> BTreeSet<TerrainNodeId> {
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

#[cfg(test)]
mod tests;
