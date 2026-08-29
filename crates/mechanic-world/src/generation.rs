//! Versioned deterministic terrain and guaranteed cave generation.

#![allow(clippy::cast_possible_truncation)] // Noise and seed contracts intentionally narrow.

use std::{
    collections::{HashMap, VecDeque},
    sync::{Arc, Mutex},
};

use bevy_math::{DVec2, DVec3};
use fastnoise_lite::{DomainWarpType, FastNoiseLite, FractalType, NoiseType};
use serde::{Deserialize, Serialize};

use crate::{
    TERRAIN_CELL_METERS, TerrainNodeId, WorldCell, WorldGeneratorVersion, WorldPosition, WorldSeed,
};

const MESH_COLUMN_CACHE_BYTES: usize = 16 * 1024 * 1024;

/// Material assigned to an occupied terrain cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TerrainMaterial {
    /// Thin, fibrous alien ground cover.
    SurfaceCover,
    /// Compressible near-surface mineral soil.
    Soil,
    /// Competent underlying rock.
    Rock,
}

impl TerrainMaterial {
    /// Stable binary representation used in edited-brick files.
    pub const fn code(self) -> u8 {
        match self {
            Self::SurfaceCover => 0,
            Self::Soil => 1,
            Self::Rock => 2,
        }
    }

    /// Decodes the stable binary representation.
    pub const fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(Self::SurfaceCover),
            1 => Some(Self::Soil),
            2 => Some(Self::Rock),
            _ => None,
        }
    }
}

/// Density and material at one sample. Positive density is solid.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TerrainSample {
    /// Signed distance-like density in metres.
    pub density: f32,
    /// Material the cell carries when density is positive.
    pub material: TerrainMaterial,
}

#[derive(Clone, Copy)]
pub(crate) struct TerrainColumnSample {
    surface: f64,
    geology: f32,
    cave_edge_mask: u16,
    cave_node_mask: u16,
}

#[derive(Default)]
struct MeshColumnCache {
    columns: HashMap<MeshColumnKey, Arc<Vec<TerrainColumnSample>>>,
    insertion_order: VecDeque<MeshColumnKey>,
    bytes: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct MeshColumnKey {
    level: u8,
    x: i32,
    z: i32,
}

impl TerrainSample {
    /// True when the sample lies in occupied terrain.
    pub const fn is_solid(self) -> bool {
        self.density > 0.0
    }
}

/// Node in the authored-by-generator cave graph.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CaveNode {
    /// Stable node identifier.
    pub id: u8,
    /// Global node position in metres.
    pub position: WorldPosition,
    /// Tunnel or chamber radius at this node.
    pub radius: f64,
}

/// One connection carved as a variable-radius capsule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CaveEdge {
    /// First node identifier.
    pub first: u8,
    /// Second node identifier.
    pub second: u8,
}

/// Connected cave recipe guaranteed near spawn.
#[derive(Clone, Debug, PartialEq)]
pub struct CaveGraph {
    /// Entrance, turns, branch ends, and chamber centres.
    pub nodes: Vec<CaveNode>,
    /// Connections between nodes.
    pub edges: Vec<CaveEdge>,
    /// Node at the traversable entrance.
    pub entrance: u8,
    /// Large chamber node.
    pub chamber: u8,
}

impl CaveGraph {
    fn node(&self, id: u8) -> CaveNode {
        self.nodes[usize::from(id)]
    }

    /// True when all nodes can be reached from the entrance.
    pub fn is_connected(&self) -> bool {
        let mut seen = vec![false; self.nodes.len()];
        let mut stack = vec![self.entrance];
        while let Some(node) = stack.pop() {
            if seen[usize::from(node)] {
                continue;
            }
            seen[usize::from(node)] = true;
            for edge in &self.edges {
                if edge.first == node {
                    stack.push(edge.second);
                } else if edge.second == node {
                    stack.push(edge.first);
                }
            }
        }
        seen.into_iter().all(core::convert::identity)
    }

    /// Number of graph nodes with more than two incident tunnels.
    pub fn branch_count(&self) -> usize {
        self.nodes
            .iter()
            .filter(|node| {
                self.edges
                    .iter()
                    .filter(|edge| edge.first == node.id || edge.second == node.id)
                    .count()
                    >= 3
            })
            .count()
    }
}

/// Untouched deterministic terrain field for one seed and generator version.
pub struct TerrainField {
    seed: WorldSeed,
    version: WorldGeneratorVersion,
    warp: FastNoiseLite,
    elevation: FastNoiseLite,
    ridges: FastNoiseLite,
    geology: FastNoiseLite,
    cave: CaveGraph,
    cave_horizontal_bounds: (DVec2, DVec2),
    mesh_column_cache: Mutex<MeshColumnCache>,
}

impl core::fmt::Debug for TerrainField {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("TerrainField")
            .field("seed", &self.seed)
            .field("version", &self.version)
            .field("cave", &self.cave)
            .finish_non_exhaustive()
    }
}

impl TerrainField {
    /// Creates the current generator for `seed`.
    pub fn new(seed: WorldSeed) -> Self {
        Self::with_version(seed, WorldGeneratorVersion::CURRENT)
    }

    /// Creates a specific supported generation recipe.
    ///
    /// # Panics
    ///
    /// Panics for an unknown version. Persistence validates versions before
    /// constructing a field, so unsupported worlds are never guessed at.
    pub fn with_version(seed: WorldSeed, version: WorldGeneratorVersion) -> Self {
        assert_eq!(version, WorldGeneratorVersion::CURRENT);
        let base_seed = fold_seed(seed.0);

        let mut warp = FastNoiseLite::with_seed(base_seed.wrapping_add(11));
        warp.set_domain_warp_type(Some(DomainWarpType::OpenSimplex2));
        warp.set_fractal_type(Some(FractalType::DomainWarpProgressive));
        warp.set_fractal_octaves(Some(3));
        warp.set_frequency(Some(0.000_32));
        warp.set_domain_warp_amp(Some(180.0));

        let mut elevation = FastNoiseLite::with_seed(base_seed.wrapping_add(23));
        elevation.set_noise_type(Some(NoiseType::OpenSimplex2S));
        elevation.set_fractal_type(Some(FractalType::FBm));
        elevation.set_fractal_octaves(Some(5));
        elevation.set_frequency(Some(0.000_55));

        let mut ridges = FastNoiseLite::with_seed(base_seed.wrapping_add(47));
        ridges.set_noise_type(Some(NoiseType::OpenSimplex2));
        ridges.set_fractal_type(Some(FractalType::Ridged));
        ridges.set_fractal_octaves(Some(4));
        ridges.set_frequency(Some(0.000_24));

        let mut geology = FastNoiseLite::with_seed(base_seed.wrapping_add(71));
        geology.set_noise_type(Some(NoiseType::OpenSimplex2S));
        geology.set_fractal_type(Some(FractalType::FBm));
        geology.set_fractal_octaves(Some(3));
        geology.set_frequency(Some(0.006));

        let cave = cave_for(seed);
        let cave_horizontal_bounds = cave.nodes.iter().fold(
            (DVec2::splat(f64::INFINITY), DVec2::splat(f64::NEG_INFINITY)),
            |(minimum, maximum), node| {
                let position = DVec2::new(node.position.0.x, node.position.0.z);
                (
                    minimum.min(position - DVec2::splat(node.radius)),
                    maximum.max(position + DVec2::splat(node.radius)),
                )
            },
        );
        Self {
            seed,
            version,
            warp,
            elevation,
            ridges,
            geology,
            cave,
            cave_horizontal_bounds,
            mesh_column_cache: Mutex::new(MeshColumnCache::default()),
        }
    }

    /// Seed used by this field.
    pub const fn seed(&self) -> WorldSeed {
        self.seed
    }

    /// Generator recipe used by this field.
    pub const fn version(&self) -> WorldGeneratorVersion {
        self.version
    }

    /// Guaranteed cave graph near the safe spawn.
    pub fn cave(&self) -> &CaveGraph {
        &self.cave
    }

    /// Safe meadow-like spawn at the centre of the finite world.
    pub fn safe_spawn(&self) -> WorldPosition {
        WorldPosition(DVec3::new(0.0, self.surface_height(0.0, 0.0) + 0.05, 0.0))
    }

    /// Generated surface elevation at horizontal coordinates.
    pub fn surface_height(&self, x: f64, z: f64) -> f64 {
        self.surface_and_geology(x, z).0
    }

    fn surface_and_geology(&self, x: f64, z: f64) -> (f64, f32) {
        let (warped_x, warped_z) = self.warp.domain_warp_2d(x, z);
        let broad = f64::from(self.elevation.get_noise_2d(warped_x, warped_z));
        let ridge = f64::from(self.ridges.get_noise_2d(warped_x, warped_z));
        let geology = self.geology.get_noise_2d(x, z);
        let detail = f64::from(geology);
        let wild_height = 18.0 + broad * 65.0 + ridge.max(0.0).powi(3) * 120.0 + detail * 2.5;

        // The spawn guarantee is part of the generator recipe, not a search
        // heuristic: a smooth radial blend yields a broad, low-slope meadow.
        let meadow_weight = smoothstep(90.0, 240.0, x.hypot(z));
        (
            4.0_f64.mul_add(1.0 - meadow_weight, wild_height * meadow_weight),
            geology,
        )
    }

    /// Samples untouched terrain at an exact cell centre.
    pub fn sample_cell(&self, cell: WorldCell) -> TerrainSample {
        let position = cell.centre();
        let column = self.sample_column(position.0.x, position.0.z);
        self.sample_position_in_column(position, column)
    }

    /// Samples untouched terrain at a continuous global position.
    pub fn sample_position(&self, position: WorldPosition) -> TerrainSample {
        let column = self.sample_column(position.0.x, position.0.z);
        self.sample_position_in_column(position, column)
    }

    pub(crate) fn sample_column(&self, x: f64, z: f64) -> TerrainColumnSample {
        let (surface, geology) = self.surface_and_geology(x, z);
        let (cave_edge_mask, cave_node_mask) = self.cave_masks_at(x, z);
        TerrainColumnSample {
            surface,
            geology,
            cave_edge_mask,
            cave_node_mask,
        }
    }

    pub(crate) fn cached_mesh_columns(
        &self,
        node: TerrainNodeId,
        prepare: impl FnOnce() -> Vec<TerrainColumnSample>,
    ) -> Arc<Vec<TerrainColumnSample>> {
        let key = MeshColumnKey {
            level: node.level,
            x: node.coordinates.x,
            z: node.coordinates.z,
        };
        if let Some(columns) = self
            .mesh_column_cache
            .lock()
            .expect("terrain column cache is not poisoned")
            .columns
            .get(&key)
            .cloned()
        {
            return columns;
        }
        let prepared = Arc::new(prepare());
        let entry_bytes = prepared
            .len()
            .saturating_mul(core::mem::size_of::<TerrainColumnSample>());
        let mut cache = self
            .mesh_column_cache
            .lock()
            .expect("terrain column cache is not poisoned");
        if let Some(existing) = cache.columns.get(&key).cloned() {
            return existing;
        }
        while cache.bytes.saturating_add(entry_bytes) > MESH_COLUMN_CACHE_BYTES {
            let Some(oldest) = cache.insertion_order.pop_front() else {
                break;
            };
            if let Some(removed) = cache.columns.remove(&oldest) {
                cache.bytes = cache.bytes.saturating_sub(
                    removed
                        .len()
                        .saturating_mul(core::mem::size_of::<TerrainColumnSample>()),
                );
            }
        }
        cache.bytes = cache.bytes.saturating_add(entry_bytes);
        cache.insertion_order.push_back(key);
        cache.columns.insert(key, Arc::clone(&prepared));
        prepared
    }

    pub(crate) fn sample_cell_in_column(
        &self,
        cell: WorldCell,
        column: TerrainColumnSample,
    ) -> TerrainSample {
        self.sample_position_in_column(cell.centre(), column)
    }

    fn sample_position_in_column(
        &self,
        position: WorldPosition,
        column: TerrainColumnSample,
    ) -> TerrainSample {
        if !position.is_inside_world() {
            return TerrainSample {
                density: -1.0,
                material: TerrainMaterial::Rock,
            };
        }
        let point = position.0;
        let surface = column.surface;
        let mut density = surface - point.y;
        if column.cave_edge_mask != 0 || column.cave_node_mask != 0 {
            for (index, edge) in self.cave.edges.iter().enumerate() {
                if column.cave_edge_mask & (1 << index) == 0 {
                    continue;
                }
                let first = self.cave.node(edge.first);
                let second = self.cave.node(edge.second);
                let (distance, along) =
                    distance_to_segment(point, first.position.0, second.position.0);
                let radius = first.radius + (second.radius - first.radius) * along;
                density = density.min(distance - radius);
            }
            for (index, node) in self.cave.nodes.iter().enumerate() {
                if column.cave_node_mask & (1 << index) == 0 {
                    continue;
                }
                density = density.min(point.distance(node.position.0) - node.radius);
            }
        }

        let depth = surface - point.y;
        let material = if depth <= TERRAIN_CELL_METERS * 3.0 {
            TerrainMaterial::SurfaceCover
        } else if depth <= 1.5 + f64::from(column.geology) * 0.25 {
            TerrainMaterial::Soil
        } else {
            TerrainMaterial::Rock
        };
        TerrainSample {
            density: density as f32,
            material,
        }
    }

    fn cave_masks_at(&self, x: f64, z: f64) -> (u16, u16) {
        let point_2d = DVec2::new(x, z);
        if point_2d.cmplt(self.cave_horizontal_bounds.0).any()
            || point_2d.cmpgt(self.cave_horizontal_bounds.1).any()
        {
            return (0, 0);
        }
        let point = DVec3::new(x, 0.0, z);
        debug_assert!(self.cave.edges.len() <= u16::BITS as usize);
        debug_assert!(self.cave.nodes.len() <= u16::BITS as usize);
        let edge_mask = self
            .cave
            .edges
            .iter()
            .enumerate()
            .filter(|(_, edge)| {
                let first = self.cave.node(edge.first);
                let second = self.cave.node(edge.second);
                horizontal_distance_to_segment(point, first.position.0, second.position.0)
                    <= first.radius.max(second.radius)
            })
            .fold(0_u16, |mask, (index, _)| mask | (1 << index));
        let node_mask = self
            .cave
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, node)| (x - node.position.0.x).hypot(z - node.position.0.z) <= node.radius)
            .fold(0_u16, |mask, (index, _)| mask | (1 << index));
        (edge_mask, node_mask)
    }

    pub(crate) fn cave_intersects_bounds(&self, minimum: DVec3, maximum: DVec3) -> bool {
        self.cave.edges.iter().any(|edge| {
            let first = self.cave.node(edge.first);
            let second = self.cave.node(edge.second);
            let radius = first.radius.max(second.radius);
            let edge_minimum = first.position.0.min(second.position.0) - DVec3::splat(radius);
            let edge_maximum = first.position.0.max(second.position.0) + DVec3::splat(radius);
            bounds_overlap(edge_minimum, edge_maximum, minimum, maximum)
        }) || self.cave.nodes.iter().any(|node| {
            let node_minimum = node.position.0 - DVec3::splat(node.radius);
            let node_maximum = node.position.0 + DVec3::splat(node.radius);
            bounds_overlap(node_minimum, node_maximum, minimum, maximum)
        })
    }
}

fn bounds_overlap(
    first_minimum: DVec3,
    first_maximum: DVec3,
    second_minimum: DVec3,
    second_maximum: DVec3,
) -> bool {
    first_minimum.cmple(second_maximum).all() && second_minimum.cmple(first_maximum).all()
}

fn fold_seed(seed: u64) -> i32 {
    let mixed = seed ^ seed.rotate_left(29) ^ 0x9e37_79b9_7f4a_7c15;
    (mixed ^ (mixed >> 32)) as i32
}

fn smoothstep(edge0: f64, edge1: f64, value: f64) -> f64 {
    let value = ((value - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    value * value * (3.0 - 2.0 * value)
}

fn distance_to_segment(point: DVec3, first: DVec3, second: DVec3) -> (f64, f64) {
    let segment = second - first;
    let along = ((point - first).dot(segment) / segment.length_squared()).clamp(0.0, 1.0);
    (point.distance(first + segment * along), along)
}

fn horizontal_distance_to_segment(point: DVec3, first: DVec3, second: DVec3) -> f64 {
    let segment_x = second.x - first.x;
    let segment_z = second.z - first.z;
    let length_squared = segment_x.mul_add(segment_x, segment_z * segment_z);
    if length_squared <= f64::EPSILON {
        return (point.x - first.x).hypot(point.z - first.z);
    }
    let along = (((point.x - first.x) * segment_x + (point.z - first.z) * segment_z)
        / length_squared)
        .clamp(0.0, 1.0);
    let closest_x = first.x + segment_x * along;
    let closest_z = first.z + segment_z * along;
    (point.x - closest_x).hypot(point.z - closest_z)
}

fn cave_for(seed: WorldSeed) -> CaveGraph {
    // Seed affects handedness and small lateral offsets while the topology and
    // clearance guarantees remain invariant across every world.
    let handedness = if seed.0.count_ones().is_multiple_of(2) {
        1.0
    } else {
        -1.0
    };
    let offset = f64::from(((seed.0 >> 8) & 15) as u8) * 0.35 - 2.625;
    let positions = [
        (DVec3::new(58.0, 7.0, 0.0), 2.0),
        (DVec3::new(66.0, -3.0, handedness * 4.0), 1.65),
        (DVec3::new(80.0, -8.0, handedness * (10.0 + offset)), 1.6),
        (DVec3::new(96.0, -10.0, handedness * 5.0), 1.55),
        (DVec3::new(110.0, -9.0, handedness * 18.0), 1.5),
        (DVec3::new(111.0, -13.0, handedness * -9.0), 1.5),
        (DVec3::new(128.0, -12.0, handedness * 2.0), 4.2),
        (DVec3::new(145.0, -10.0, handedness * 13.0), 1.5),
    ];
    let nodes = positions
        .into_iter()
        .enumerate()
        .map(|(id, (position, radius))| CaveNode {
            id: u8::try_from(id).expect("the cave has fewer than 256 nodes"),
            position: WorldPosition(position),
            radius,
        })
        .collect();
    CaveGraph {
        nodes,
        edges: vec![
            CaveEdge {
                first: 0,
                second: 1,
            },
            CaveEdge {
                first: 1,
                second: 2,
            },
            CaveEdge {
                first: 2,
                second: 3,
            },
            CaveEdge {
                first: 3,
                second: 4,
            },
            CaveEdge {
                first: 3,
                second: 5,
            },
            CaveEdge {
                first: 3,
                second: 6,
            },
            CaveEdge {
                first: 6,
                second: 7,
            },
        ],
        entrance: 0,
        chamber: 6,
    }
}

#[cfg(test)]
mod tests {
    use bevy_math::DVec3;

    use super::{TerrainField, TerrainMaterial};
    use crate::{TERRAIN_CELL_METERS, WorldCell, WorldPosition, WorldSeed};

    #[test]
    fn generation_is_deterministic_and_seeded() {
        let first = TerrainField::new(WorldSeed(42));
        let second = TerrainField::new(WorldSeed(42));
        let other = TerrainField::new(WorldSeed(43));
        let probes = [
            WorldPosition(DVec3::new(900.25, 17.0, -441.75)),
            WorldPosition(DVec3::new(-3_200.0, -20.0, 2_750.0)),
        ];
        for probe in probes {
            assert_eq!(first.sample_position(probe), second.sample_position(probe));
        }
        assert!(
            (first.surface_height(900.25, -441.75) - other.surface_height(900.25, -441.75)).abs()
                > f64::EPSILON
        );
    }

    #[test]
    fn finite_world_is_empty_beyond_horizontal_bounds() {
        let field = TerrainField::new(WorldSeed(1));
        assert!(
            !field
                .sample_position(WorldPosition(DVec3::new(8_000.1, -100.0, 0.0)))
                .is_solid()
        );
    }

    #[test]
    fn spawn_meadow_has_safe_clearance_and_shallow_slope() {
        let field = TerrainField::new(WorldSeed(u64::MAX));
        let spawn = field.safe_spawn();
        assert!(!field.sample_position(spawn).is_solid());
        let centre = field.surface_height(0.0, 0.0);
        for point in [(5.0, 0.0), (-5.0, 0.0), (0.0, 5.0), (0.0, -5.0)] {
            assert!((field.surface_height(point.0, point.1) - centre).abs() < 0.1);
        }
    }

    #[test]
    fn cave_is_connected_branched_and_has_walking_clearance() {
        for seed in [WorldSeed(0), WorldSeed(1), WorldSeed(u64::MAX)] {
            let field = TerrainField::new(seed);
            assert!(field.cave().is_connected());
            assert!(field.cave().branch_count() >= 1);
            let chamber = field.cave().nodes[usize::from(field.cave().chamber)];
            assert!(chamber.radius * 2.0 >= 2.4);
            assert!(!field.sample_position(chamber.position).is_solid());
        }
    }

    #[test]
    fn strata_resolve_at_cell_centres() {
        let field = TerrainField::new(WorldSeed(99));
        let surface = field.surface_height(500.0, 500.0);
        let material_at_depth = |depth: f64| {
            field
                .sample_position(WorldPosition(DVec3::new(500.0, surface - depth, 500.0)))
                .material
        };
        assert_eq!(
            material_at_depth(TERRAIN_CELL_METERS),
            TerrainMaterial::SurfaceCover
        );
        assert_eq!(material_at_depth(0.5), TerrainMaterial::Soil);
        assert_eq!(material_at_depth(3.0), TerrainMaterial::Rock);
    }

    #[test]
    fn cell_sampling_uses_exact_centres() {
        let field = TerrainField::new(WorldSeed(7));
        let cell = WorldCell::new(20_000, 0, -10_000);
        assert_eq!(
            field.sample_cell(cell),
            field.sample_position(cell.centre())
        );
    }
}
