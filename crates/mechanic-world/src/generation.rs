//! Declarative, versioned, deterministic 3D terrain generation.
//!
//! A world is a set of biomes placed in climate space. Each biome compiles its
//! authored RON expressions to tapes: a planar `height` and a full 3D
//! `density`. At every column the climate picks blend weights; the weighted
//! biome densities are summed, then carved by the world's river valleys and
//! carve layers: caves, their entrances, and ravines. Surface rules paint the result from the dominant biome's
//! palette. See `docs/world-generation.md`.

mod compile;
mod fields;
mod interval;
mod load;
mod noise;
mod rivers;
mod scatter;
mod spec;
mod surfaces;
mod tape;

use std::sync::{Arc, Mutex};

use bevy_math::{DVec3, IVec3};
use serde::{Deserialize, Serialize};

use self::compile::{Scope, compile, compile_planar, compile_varying};
use self::fields::WorldFields;
use self::interval::Interval;
use self::noise::NoiseGen;
use self::rivers::{DRAINAGE_CELL_METRES, RIVER_LIFT_METRES, RiverNetwork, drainage_side};
use self::scatter::mix;
use self::spec::{BiomeDoc, CarveDoc, Dims, Expr, Fractal, NoiseDoc, NoiseKind};
use self::surfaces::{SurfaceProbe, SurfaceRules};
use self::tape::{PlanarCache, Tape, smoothstep};
use crate::{
    TERRAIN_CELL_METERS, TerrainDensityClass, WORLD_HALF_EXTENT_METERS, WorldCell,
    WorldGeneratorVersion, WorldPosition, WorldSeed,
};

pub use self::load::{WorldgenError, WorldgenSpec};
pub use self::spec::TextureSet;
pub use self::surfaces::{SurfaceId, SurfaceLook, SurfacePalette};

/// Material assigned to an occupied terrain cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TerrainMaterial {
    /// Thin, fibrous alien ground cover.
    SurfaceCover,
    /// Compressible near-surface mineral soil.
    Soil,
    /// Competent underlying rock.
    Rock,
    /// Loose granular silica.
    Sand,
    /// Raw ferrous ore.
    Iron,
    /// Raw carbon mineral.
    Graphite,
}

impl TerrainMaterial {
    /// Every material terrain cells can carry, in selector order.
    pub const ALL: [Self; 6] = [
        Self::SurfaceCover,
        Self::Soil,
        Self::Sand,
        Self::Rock,
        Self::Iron,
        Self::Graphite,
    ];

    /// Every material in `code()` order.
    pub const BY_CODE: [Self; 6] = [
        Self::SurfaceCover,
        Self::Soil,
        Self::Rock,
        Self::Sand,
        Self::Iron,
        Self::Graphite,
    ];

    /// Number of independently represented terrain materials.
    pub const COUNT: usize = 6;

    /// Stable binary representation used in edited-brick files.
    pub const fn code(self) -> u8 {
        match self {
            Self::SurfaceCover => 0,
            Self::Soil => 1,
            Self::Rock => 2,
            Self::Sand => 3,
            Self::Iron => 4,
            Self::Graphite => 5,
        }
    }

    /// Decodes the stable binary representation.
    pub const fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(Self::SurfaceCover),
            1 => Some(Self::Soil),
            2 => Some(Self::Rock),
            3 => Some(Self::Sand),
            4 => Some(Self::Iron),
            5 => Some(Self::Graphite),
            _ => None,
        }
    }

    /// Palette name of the material's plain surface.
    pub const fn name(self) -> &'static str {
        match self {
            Self::SurfaceCover => "surface_cover",
            Self::Soil => "soil",
            Self::Rock => "rock",
            Self::Sand => "sand",
            Self::Iron => "iron",
            Self::Graphite => "graphite",
        }
    }

    /// Texture family of the material's plain surface.
    pub const fn plain_texture(self) -> TextureSet {
        match self {
            Self::SurfaceCover => TextureSet::Grass,
            Self::Soil => TextureSet::Dirt,
            Self::Rock => TextureSet::Stone,
            Self::Sand => TextureSet::Sand,
            Self::Iron => TextureSet::Iron,
            Self::Graphite => TextureSet::Graphite,
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
    /// How the cell looks.
    pub surface: SurfaceId,
    /// Accumulated plastic compaction over half a cell.
    pub compaction: u8,
    /// How much less than a cell of undisturbed ground this cell holds, in
    /// quanta. Undisturbed ground is 0; spoil laid back down is loose.
    pub looseness: u8,
}

impl TerrainSample {
    /// True when the sample lies in occupied terrain.
    pub const fn is_solid(self) -> bool {
        self.density > 0.0
    }

    /// Untouched ground of a material with its plain look.
    pub const fn plain(density: f32, material: TerrainMaterial) -> Self {
        Self {
            density,
            material,
            surface: SurfaceId::plain(material),
            compaction: 0,
            looseness: 0,
        }
    }
}

const MAX_BIOMES: usize = 16;

/// Most carve layers a world may declare.
const MAX_CARVES: usize = 8;

/// Height added to a carve where its layer is disallowed or fading.
const CARVE_LIFT_METRES: f64 = 64.0;

/// Rock over which a carve opens once its roof is satisfied.
const ROOF_FADE_METRES: f64 = 4.0;

/// Height over which a carve opens above its floor and closes below its top.
const BAND_FADE_METRES: f64 = 8.0;

/// A carve only shapes ground within this much of its void, so blocks the
/// bounds put farther from every void skip evaluating it. Beyond it the
/// density is the biome's alone, which meshing never reads that far from a
/// surface.
const CARVE_REACH_METRES: f64 = 16.0;

/// Seen from far away, a layer only shows where its roof lies below this, so
/// voids under a solid roof are skipped.
const DISTANT_ROOF_METRES: f64 = 2.0;

/// Samples per edge of the blocks meshing culls by interval bounds.
const CULL_BLOCK: usize = 8;

/// Extra void per unit of carve factor above one.
const CARVE_WIDENING_METRES: f64 = 1.5;

/// Worlds compiled recently, shared by fields for the same seed and spec.
const COMPILED_CACHE: usize = 8;

struct CompiledBiome {
    name: String,
    target: [Option<f64>; 4],
    rarity: f64,
    height: Tape,
    density: Tape,
    rivers: f64,
    /// Multiplier per carve layer.
    carves: [f64; MAX_CARVES],
    rules: SurfaceRules,
}

struct CompiledCarve {
    name: String,
    void: Tape,
    roof: Tape,
    floor: Tape,
    top: Option<Tape>,
    visible: f64,
}

struct CompiledWorld {
    /// Unique per compiled world, so per-thread caches never mix worlds.
    id: u64,
    climate: [Tape; 4],
    biomes: Vec<CompiledBiome>,
    spawn: usize,
    spawn_radius: f64,
    blend: f64,
    carves: Vec<CompiledCarve>,
    fields: WorldFields,
    rivers: RiverNetwork,
    /// Biomes with any weight around each drainage-grid point, as bit sets.
    /// Lower and upper weight bounds per drainage point and biome, in 1/255.
    weight_low: Vec<u8>,
    weight_high: Vec<u8>,
    palette: SurfacePalette,
    vertical: (f64, f64),
    sea_level: f64,
    dither: NoiseGen,
}

/// Biome blend weights at one column, in biome order.
#[derive(Clone, Copy, Debug)]
struct Weights {
    biome: [u8; MAX_BIOMES],
    weight: [f64; MAX_BIOMES],
    len: usize,
}

impl Weights {
    fn iter(&self) -> impl Iterator<Item = (usize, f64)> + '_ {
        self.biome[..self.len]
            .iter()
            .zip(&self.weight[..self.len])
            .map(|(biome, weight)| (usize::from(*biome), *weight))
    }

    fn of(&self, biome: usize) -> f64 {
        self.iter()
            .find(|(candidate, _)| *candidate == biome)
            .map_or(0.0, |(_, weight)| weight)
    }
}

/// Source of [`CompiledWorld::id`].
static NEXT_WORLD_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// Slots in each thread's column and cell caches. Point queries such as
/// breakage, soil, and collision probes revisit the same few cells and
/// columns tick after tick, and each costs a full climate and biome
/// evaluation.
const COLUMN_CACHE_SLOTS: usize = 1_024;
const CELL_CACHE_SLOTS: usize = 32_768;

type ColumnSlot = Option<(u64, [u64; 2], Column)>;
type CellSlot = Option<(u64, WorldCell, TerrainSample)>;

thread_local! {
    /// Direct-mapped: a collision simply recomputes, which is deterministic.
    static COLUMNS: std::cell::RefCell<Vec<ColumnSlot>> =
        std::cell::RefCell::new(vec![None; COLUMN_CACHE_SLOTS]);
    static CELLS: std::cell::RefCell<Vec<CellSlot>> =
        std::cell::RefCell::new(vec![None; CELL_CACHE_SLOTS]);
}

/// Everything about a column that does not depend on height.
#[derive(Clone, Copy, Debug)]
struct Column {
    inside: bool,
    weights: Weights,
    ground: f64,
    valley: f64,
    river_distance: f64,
    /// Temperature, humidity, continentalness, and weirdness.
    climate: [f64; 4],
    carves: [CarveColumn; MAX_CARVES],
    dominant: usize,
}

/// One carve layer over a column.
#[derive(Clone, Copy, Debug, Default)]
struct CarveColumn {
    /// Blended biome multiplier; zero keeps the layer out.
    factor: f64,
    roof: f64,
    floor: f64,
    top: f64,
}

/// Regular sample lattice in cell units, x fastest.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Lattice {
    /// Cell index of the first sample on each axis.
    pub(crate) origin: IVec3,
    /// Cells between neighbouring samples.
    pub(crate) stride: i32,
    /// Samples per axis.
    pub(crate) dims: [usize; 3],
    /// Samples lie at cell centres rather than cell corners.
    pub(crate) centred: bool,
}

impl Lattice {
    fn coordinate(&self, axis: usize, index: usize) -> f64 {
        let cell =
            self.origin[axis] + i32::try_from(index).expect("lattice fits i32") * self.stride;
        if self.centred {
            (f64::from(cell) + 0.5) * TERRAIN_CELL_METERS
        } else {
            f64::from(cell) * TERRAIN_CELL_METERS
        }
    }

    pub(crate) const fn len(&self) -> usize {
        self.dims[0] * self.dims[1] * self.dims[2]
    }
}

/// Planar values of each biome and carve tape, kept between blocks.
struct BlockCaches {
    biomes: Vec<PlanarCache>,
    carves: Vec<PlanarCache>,
}

/// One block of a lattice being sampled, with the lattice's columns.
#[derive(Clone, Copy)]
struct LatticeBlock<'a> {
    lattice: &'a Lattice,
    columns: &'a [Column],
    start: [usize; 3],
    dims: [usize; 3],
}

/// Per-column data of a lattice, kept so its points can be painted later,
/// with the carve layer that shaped each point.
pub(crate) struct LatticeColumns {
    columns: Vec<Column>,
    carved: Vec<u8>,
    dims: [usize; 3],
}

/// World position of a cell corner, rounded exactly as [`Lattice`] rounds.
pub(crate) fn corner_position(cell: WorldCell) -> DVec3 {
    DVec3::new(
        f64::from(cell.x) * TERRAIN_CELL_METERS,
        f64::from(cell.y) * TERRAIN_CELL_METERS,
        f64::from(cell.z) * TERRAIN_CELL_METERS,
    )
}

/// Expands each drainage point's biome set by its eight neighbours.
/// Bounds on `Σ weight·value` when each weight lies in its bound and the
/// weights sum to one: fill the lower bounds, then spend what remains on the
/// largest (or smallest) values first.
fn blend_bounds(weights: &[Interval], values: &[Interval]) -> Interval {
    let extreme = |largest: bool| {
        let mut order = (0..weights.len()).collect::<Vec<_>>();
        let key = |index: usize| {
            if largest {
                values[index].hi
            } else {
                values[index].lo
            }
        };
        order.sort_by(|&a, &b| {
            if largest {
                key(b).total_cmp(&key(a))
            } else {
                key(a).total_cmp(&key(b))
            }
        });
        let mut remaining = 1.0 - weights.iter().map(|weight| weight.lo).sum::<f64>();
        let mut sum = 0.0;
        for index in order {
            let extra = (weights[index].hi - weights[index].lo).clamp(0.0, remaining.max(0.0));
            remaining -= extra;
            let weight = weights[index].lo + extra;
            if weight > 0.0 {
                sum += weight * key(index);
            }
        }
        sum
    };
    Interval::new(extreme(false), extreme(true))
}

/// Per-cell weight bounds from weights sampled on the drainage grid: each
/// point takes the extremes of itself and its eight neighbours, quantised
/// outward to 1/255.
fn weight_grid_bounds(samples: &[f64], biomes: usize) -> (Vec<u8>, Vec<u8>) {
    let side = drainage_side();
    let mut high = vec![0_u8; samples.len()];
    let mut low = vec![0_u8; samples.len()];
    for row in 0..side {
        for column in 0..side {
            let point = column + row * side;
            for biome in 0..biomes {
                let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
                for neighbour_row in row.saturating_sub(1)..=(row + 1).min(side - 1) {
                    for neighbour_column in column.saturating_sub(1)..=(column + 1).min(side - 1) {
                        let weight =
                            samples[(neighbour_column + neighbour_row * side) * biomes + biome];
                        lo = lo.min(weight);
                        hi = hi.max(weight);
                    }
                }
                #[expect(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "weights lie in [0, 1]"
                )]
                {
                    high[point * biomes + biome] = (hi * 255.0).ceil().clamp(0.0, 255.0) as u8;
                    low[point * biomes + biome] = (lo * 255.0).floor().clamp(0.0, 255.0) as u8;
                }
            }
        }
    }
    (low, high)
}

/// Untouched deterministic terrain field for one seed and generator version.
pub struct TerrainField {
    seed: WorldSeed,
    version: WorldGeneratorVersion,
    spec: Arc<WorldgenSpec>,
    world: Arc<CompiledWorld>,
}

impl core::fmt::Debug for TerrainField {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("TerrainField")
            .field("seed", &self.seed)
            .field("version", &self.version)
            .field("spec", &self.spec.hash())
            .finish_non_exhaustive()
    }
}

impl TerrainField {
    /// Creates the current generator for `seed` from the embedded definition.
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
        Self::from_spec(seed, WorldgenSpec::embedded())
            .unwrap_or_else(|error| panic!("embedded worldgen does not compile: {error}"))
    }

    /// Compiles a field from any definition, such as one being authored.
    ///
    /// # Errors
    ///
    /// Returns the first expression or rule that cannot be compiled.
    ///
    /// # Panics
    ///
    /// Panics if another thread panicked while holding the compile cache.
    pub fn from_spec(seed: WorldSeed, spec: Arc<WorldgenSpec>) -> Result<Self, WorldgenError> {
        static CACHE: Mutex<Vec<(u64, u64, Arc<CompiledWorld>)>> = Mutex::new(Vec::new());
        let cached = CACHE
            .lock()
            .expect("compiled world cache is not poisoned")
            .iter()
            .find(|(cached_seed, hash, _)| *cached_seed == seed.0 && *hash == spec.hash())
            .map(|(.., world)| Arc::clone(world));
        let world = if let Some(world) = cached {
            world
        } else {
            let world = Arc::new(CompiledWorld::new(seed, &spec)?);
            let mut cache = CACHE.lock().expect("compiled world cache is not poisoned");
            if cache.len() >= COMPILED_CACHE {
                cache.remove(0);
            }
            cache.push((seed.0, spec.hash(), Arc::clone(&world)));
            world
        };
        Ok(Self {
            seed,
            version: WorldGeneratorVersion::CURRENT,
            spec,
            world,
        })
    }

    /// Seed used by this field.
    pub const fn seed(&self) -> WorldSeed {
        self.seed
    }

    /// Generator recipe used by this field.
    pub const fn version(&self) -> WorldGeneratorVersion {
        self.version
    }

    /// Definition this field was compiled from.
    pub fn spec(&self) -> &Arc<WorldgenSpec> {
        &self.spec
    }

    /// Every surface this world can show.
    pub fn palette(&self) -> &SurfacePalette {
        &self.world.palette
    }

    /// Lowest and highest generated heights.
    pub fn vertical_range(&self) -> (f64, f64) {
        self.world.vertical
    }

    /// Height rivers drain to.
    pub fn sea_level(&self) -> f64 {
        self.world.sea_level
    }

    /// Number of traced river segments.
    pub fn river_segment_count(&self) -> usize {
        self.world.rivers.segment_count()
    }

    /// Name of the biome that paints a column.
    pub fn biome_at(&self, x: f64, z: f64) -> &str {
        let column = self.world.column(x, z);
        &self.world.biomes[column.dominant].name
    }

    /// Blended biome ground height before 3D features, rivers, and caves.
    pub fn ground_height(&self, x: f64, z: f64) -> f64 {
        self.world.column(x, z).ground
    }

    /// Horizontal distance to the nearest river centre line, if one is near.
    pub fn river_distance(&self, x: f64, z: f64) -> Option<f64> {
        let distance = self.world.column(x, z).river_distance;
        distance.is_finite().then_some(distance)
    }

    /// A flat, open, dry spot near the centre of the world.
    pub fn safe_spawn(&self) -> WorldPosition {
        let world = &self.world;
        let ring_spacing = 12.0;
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a few dozen rings"
        )]
        let rings = (world.spawn_radius * 2.0 / ring_spacing).ceil() as u32;
        for ring in 0..=rings {
            let radius = f64::from(ring) * ring_spacing;
            let points = (ring * 8).max(1);
            for point in 0..points {
                let angle = f64::from(point) / f64::from(points) * core::f64::consts::TAU;
                let (x, z) = (radius * angle.cos(), radius * angle.sin());
                if let Some(height) = self.spawnable_height(x, z) {
                    return WorldPosition(DVec3::new(x, height + 0.05, z));
                }
            }
        }
        WorldPosition(DVec3::new(0.0, self.surface_height(0.0, 0.0) + 0.05, 0.0))
    }

    fn spawnable_height(&self, x: f64, z: f64) -> Option<f64> {
        let height = self.topmost_surface(x, z)?;
        if height < self.world.sea_level + 0.5 {
            return None;
        }
        for (dx, dz) in [(3.0, 0.0), (-3.0, 0.0), (0.0, 3.0), (0.0, -3.0)] {
            let neighbour = self.topmost_surface(x + dx, z + dz)?;
            if (neighbour - height).abs() > 0.6 {
                return None;
            }
        }
        [0.3, 1.2, 2.2]
            .into_iter()
            .all(|lift| self.density(DVec3::new(x, height + lift, z)) < 0.0)
            .then_some(height)
    }

    /// Height of the highest ground in a column, or the bottom of the world
    /// when the column is open all the way down.
    pub fn surface_height(&self, x: f64, z: f64) -> f64 {
        self.topmost_surface(x, z).unwrap_or(self.world.vertical.0)
    }

    /// Height where a ray cast straight down from the sky first meets ground.
    pub fn topmost_surface(&self, x: f64, z: f64) -> Option<f64> {
        let (bottom, top) = self.world.vertical;
        let column = self.world.column(x, z);
        if !column.inside {
            return None;
        }
        let density = |y: f64| self.world.density_in_column(&column, DVec3::new(x, y, z));
        let mut y = top;
        let mut value = density(y);
        if value > 0.0 {
            return Some(top);
        }
        while y > bottom {
            let step = (-value * 0.5).clamp(0.1, 4.0);
            let next = y - step;
            let next_value = density(next);
            if next_value > 0.0 {
                let (mut empty, mut solid) = (y, next);
                for _ in 0..24 {
                    let middle = 0.5 * (empty + solid);
                    if density(middle) > 0.0 {
                        solid = middle;
                    } else {
                        empty = middle;
                    }
                }
                return Some(0.5 * (empty + solid));
            }
            y = next;
            value = next_value;
        }
        None
    }

    /// Untouched density at a continuous position.
    pub fn density(&self, position: DVec3) -> f64 {
        self.world.density(position)
    }

    /// The carve layer whose void shapes untouched ground at a position,
    /// if one does: inside its void or in the rock just around it.
    pub fn carve_at(&self, position: DVec3) -> Option<&str> {
        let column = self.world.cached_column(position.x, position.z);
        if !column.inside {
            return None;
        }
        let (_, carved) = self.world.density_and_carve(&column, position);
        carved
            .checked_sub(1)
            .map(|layer| self.world.carves[usize::from(layer)].name.as_str())
    }

    /// Names of the world's carve layers.
    pub fn carve_names(&self) -> impl Iterator<Item = &str> {
        self.world.carves.iter().map(|carve| carve.name.as_str())
    }

    /// Untouched density at a cell centre, equal to `sample_cell(cell).density`.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "stored densities are f32 by contract"
    )]
    pub fn cell_density(&self, cell: WorldCell) -> f32 {
        self.world.density(cell.centre().0) as f32
    }

    /// Samples untouched terrain at an exact cell centre.
    pub fn sample_cell(&self, cell: WorldCell) -> TerrainSample {
        let world = &self.world;
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "cell coordinates are hashed bit for bit"
        )]
        let slot = (mix(world.id
            ^ (cell.x as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
            ^ (cell.y as u64).wrapping_mul(0xc2b2_ae3d_27d4_eb4f)
            ^ (cell.z as u64).wrapping_mul(0x1656_67b1_9e37_79f9)) as usize)
            % CELL_CACHE_SLOTS;
        if let Some(sample) = CELLS.with(|cache| {
            cache.borrow()[slot].and_then(|(id, cached, sample)| {
                (id == world.id && cached == cell).then_some(sample)
            })
        }) {
            return sample;
        }
        let sample = self.sample_cell_uncached(cell);
        CELLS.with(|cache| cache.borrow_mut()[slot] = Some((world.id, cell, sample)));
        sample
    }

    fn sample_cell_uncached(&self, cell: WorldCell) -> TerrainSample {
        let world = &self.world;
        let centre = cell.centre().0;
        let column = world.cached_column(centre.x, centre.z);
        if !column.inside {
            return TerrainSample::plain(-1.0, TerrainMaterial::Rock);
        }
        let (density, carved) = world.density_and_carve(&column, centre);
        let neighbour = |offset: IVec3| {
            let position = WorldCell::new(cell.x + offset.x, cell.y + offset.y, cell.z + offset.z)
                .centre()
                .0;
            if offset.x == 0 && offset.z == 0 {
                world.density_in_column(&column, position)
            } else {
                world.density(position)
            }
        };
        let gradient = [
            (neighbour(IVec3::X) - neighbour(IVec3::NEG_X)) / (2.0 * TERRAIN_CELL_METERS),
            (neighbour(IVec3::Y) - neighbour(IVec3::NEG_Y)) / (2.0 * TERRAIN_CELL_METERS),
            (neighbour(IVec3::Z) - neighbour(IVec3::NEG_Z)) / (2.0 * TERRAIN_CELL_METERS),
        ];
        world.sample(&column, centre, density, gradient, carved)
    }

    /// Samples untouched terrain at a continuous global position.
    pub fn sample_position(&self, position: WorldPosition) -> TerrainSample {
        let world = &self.world;
        let point = position.0;
        let column = world.column(point.x, point.z);
        if !column.inside {
            return TerrainSample::plain(-1.0, TerrainMaterial::Rock);
        }
        let (density, carved) = world.density_and_carve(&column, point);
        let h = TERRAIN_CELL_METERS;
        let along = |axis: DVec3| {
            (world.density(point + axis * h) - world.density(point - axis * h)) / (2.0 * h)
        };
        let gradient = [along(DVec3::X), along(DVec3::Y), along(DVec3::Z)];
        world.sample(&column, point, density, gradient, carved)
    }

    /// Samples every cell in `[minimum, minimum + dims)`, x fastest. Each
    /// sample equals [`Self::sample_cell`] bit for bit.
    pub(crate) fn sample_cells(&self, minimum: WorldCell, dims: [usize; 3]) -> Vec<TerrainSample> {
        let halo = Lattice {
            origin: IVec3::new(minimum.x - 1, minimum.y - 1, minimum.z - 1),
            stride: 1,
            dims: dims.map(|edge| edge + 2),
            centred: true,
        };
        let (densities, columns, carved) = self.world.density_lattice(&halo, false, true);
        let index = |x: usize, y: usize, z: usize| x + halo.dims[0] * (y + halo.dims[1] * z);
        let mut samples = Vec::with_capacity(dims[0] * dims[1] * dims[2]);
        let step = 2.0 * TERRAIN_CELL_METERS;
        for z in 1..=dims[2] {
            for y in 1..=dims[1] {
                for x in 1..=dims[0] {
                    let column = &columns[x + halo.dims[0] * z];
                    if !column.inside {
                        samples.push(TerrainSample::plain(-1.0, TerrainMaterial::Rock));
                        continue;
                    }
                    let gradient = [
                        (densities[index(x + 1, y, z)] - densities[index(x - 1, y, z)]) / step,
                        (densities[index(x, y + 1, z)] - densities[index(x, y - 1, z)]) / step,
                        (densities[index(x, y, z + 1)] - densities[index(x, y, z - 1)]) / step,
                    ];
                    let position = DVec3::new(
                        halo.coordinate(0, x),
                        halo.coordinate(1, y),
                        halo.coordinate(2, z),
                    );
                    samples.push(self.world.sample(
                        column,
                        position,
                        densities[index(x, y, z)],
                        gradient,
                        carved[index(x, y, z)],
                    ));
                }
            }
        }
        samples
    }

    /// Densities over a lattice, each equal to [`Self::density`] at the
    /// lattice position bit for bit.
    pub(crate) fn density_lattice(&self, lattice: &Lattice) -> Vec<f64> {
        self.world.density_lattice(lattice, false, true).0
    }

    /// Densities for meshing a lattice, with the columns needed to paint its
    /// points. With `cull`, blocks far from any untouched surface carry only a
    /// bound of the right sign, so edited regions must not cull. Without
    /// `enclosed`, voids that cannot reach the surface count as rock, as
    /// distant streaming shows them.
    pub(crate) fn mesh_lattice(
        &self,
        lattice: &Lattice,
        cull: bool,
        enclosed: bool,
    ) -> (Vec<f64>, LatticeColumns) {
        let (densities, columns, carved) = self.world.density_lattice(lattice, cull, enclosed);
        (
            densities,
            LatticeColumns {
                columns,
                carved,
                dims: lattice.dims,
            },
        )
    }

    /// Whether a meshing lattice evidently holds no surface: every point of a
    /// lattice four times coarser has the same sign, and lies at least two
    /// coarse spacings from zero. A surface crossing the coarse lattice flips
    /// a sign; only a feature narrower than the margin could pass between its
    /// points unseen. Lets a chunk that selection could not prove empty skip
    /// full sampling.
    pub(crate) fn lattice_is_clear(&self, lattice: &Lattice, enclosed: bool) -> bool {
        const COARSENING: usize = 4;
        let coarse = Lattice {
            origin: lattice.origin,
            stride: lattice.stride * i32::try_from(COARSENING).expect("small factor"),
            dims: lattice.dims.map(|edge| (edge - 1).div_ceil(COARSENING) + 1),
            centred: lattice.centred,
        };
        let margin = 2.0 * f64::from(coarse.stride) * TERRAIN_CELL_METERS;
        let (densities, ..) = self.world.density_lattice(&coarse, true, enclosed);
        let solid = densities[0] > 0.0;
        densities
            .iter()
            .all(|&density| (density > 0.0) == solid && density.abs() > margin)
    }

    /// Paints lattice point `(x, y, z)` of a lattice sampled with columns.
    pub(crate) fn paint_lattice(
        &self,
        columns: &LatticeColumns,
        [x, y, z]: [usize; 3],
        position: DVec3,
        density: f64,
        gradient: [f64; 3],
    ) -> (TerrainMaterial, SurfaceId) {
        let [width, height, _] = columns.dims;
        let column = &columns.columns[x + width * z];
        if !column.inside {
            return (
                TerrainMaterial::Rock,
                SurfaceId::plain(TerrainMaterial::Rock),
            );
        }
        let carved = columns.carved[x + width * (y + height * z)];
        let sample = self
            .world
            .sample(column, position, density, gradient, carved);
        (sample.material, sample.surface)
    }

    /// Material and surface a lattice point shows, given its density gradient.
    pub(crate) fn paint(
        &self,
        position: DVec3,
        density: f64,
        gradient: [f64; 3],
    ) -> (TerrainMaterial, SurfaceId) {
        let column = self.world.column(position.x, position.z);
        if !column.inside {
            return (
                TerrainMaterial::Rock,
                SurfaceId::plain(TerrainMaterial::Rock),
            );
        }
        let (_, carved) = self.world.density_and_carve(&column, position);
        let sample = self
            .world
            .sample(&column, position, density, gradient, carved);
        (sample.material, sample.surface)
    }

    /// Conservatively classifies untouched terrain in a box.
    pub fn classify(&self, minimum: DVec3, maximum: DVec3) -> TerrainDensityClass {
        Self::class_of(self.world.interval(minimum, maximum, true))
    }

    /// Classifies a box as seen from far away: voids that cannot reach the
    /// surface count as closed, so a box wholly inside rock is solid even
    /// when tunnels run through it. Empty stays exact, since carves only ever
    /// remove ground.
    pub fn classify_distant(&self, minimum: DVec3, maximum: DVec3) -> TerrainDensityClass {
        Self::class_of(self.world.interval(minimum, maximum, false))
    }

    fn class_of(bounds: Interval) -> TerrainDensityClass {
        if bounds.hi <= 0.0 {
            TerrainDensityClass::Empty
        } else if bounds.lo > 0.0 {
            TerrainDensityClass::Solid
        } else {
            TerrainDensityClass::Mixed
        }
    }
}

impl CompiledWorld {
    fn new(seed: WorldSeed, spec: &WorldgenSpec) -> Result<Self, WorldgenError> {
        let base = mix(seed.0 ^ 0x6d65_6368_616e_6963);
        let library = &spec.library.definitions;
        let empty = std::collections::BTreeMap::new();
        let world_scope = Scope {
            local: &empty,
            library,
            fields: None,
        };
        let world = &spec.world;
        let climate_tape = |expr: &Expr, salt: u64, name: &str| {
            compile_planar(
                expr,
                world_scope,
                mix(base ^ salt),
                &format!("world.ron climate {name}"),
            )
        };
        let climate = [
            climate_tape(&world.climate.temperature, 1, "temperature")?,
            climate_tape(&world.climate.humidity, 2, "humidity")?,
            climate_tape(&world.climate.continentalness, 3, "continentalness")?,
            climate_tape(&world.climate.weirdness, 4, "weirdness")?,
        ];
        if spec.biomes.len() > MAX_BIOMES {
            return Err(WorldgenError::Invalid {
                context: "world.ron biomes".to_owned(),
                message: format!("at most {MAX_BIOMES} biomes are supported"),
            });
        }
        let palette = SurfacePalette::new(&world.palette)?;
        if world.carves.len() > MAX_CARVES {
            return Err(WorldgenError::Invalid {
                context: "world.ron carves".to_owned(),
                message: format!("at most {MAX_CARVES} carve layers are supported"),
            });
        }
        let carve_names = world
            .carves
            .iter()
            .map(|carve| (carve.name.clone(), carve.unlisted))
            .collect::<Vec<_>>();
        let biomes = spec
            .biomes
            .iter()
            .map(|doc| compile_biome(doc, library, &palette, &carve_names, base))
            .collect::<Result<Vec<_>, _>>()?;
        let spawn = spec
            .biomes
            .iter()
            .position(|biome| biome.name == world.spawn.biome)
            .expect("the loader checks the spawn biome");
        let fields = WorldFields::new();
        let carves = compile_carves(&world.carves, library, &fields, base)?;
        #[expect(
            clippy::cast_possible_truncation,
            reason = "noise seeds are folded hashes"
        )]
        let dither = NoiseGen::new(
            &NoiseDoc {
                kind: NoiseKind::Simplex,
                fractal: Fractal::Fbm,
                octaves: 2,
                freq: 1.0 / 18.0,
                lacunarity: 2.0,
                gain: 0.5,
                amp: 1.0,
                offset: 0.0,
                dims: Dims::Two,
                seed: 0,
            },
            mix(base ^ 6) as i32,
        );
        let mut compiled = Self {
            id: NEXT_WORLD_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            climate,
            biomes,
            spawn,
            spawn_radius: world.spawn.radius,
            blend: world.blend.max(1.0e-3),
            carves,
            fields,
            rivers: RiverNetwork::none(),
            weight_low: Vec::new(),
            weight_high: Vec::new(),
            palette,
            vertical: world.vertical,
            sea_level: world.sea_level,
            dither,
        };
        let (heights, weights) = compiled.drainage_heights();
        compiled.fields.fill(&heights);
        (compiled.weight_low, compiled.weight_high) =
            weight_grid_bounds(&weights, compiled.biomes.len());
        #[expect(
            clippy::cast_possible_truncation,
            reason = "noise seeds are folded hashes"
        )]
        let river_seed = mix(base ^ 7) as i32;
        compiled.rivers = RiverNetwork::trace(&world.rivers, world.sea_level, &heights, river_seed);
        Ok(compiled)
    }

    /// Blended biome heights on the drainage grid, and which biomes weigh on
    /// each grid point, computed in parallel.
    #[expect(clippy::cast_precision_loss, reason = "grid indices are a few hundred")]
    fn drainage_heights(&self) -> (Vec<f64>, Vec<f64>) {
        let side = drainage_side();
        let biomes = self.biomes.len();
        let mut heights = vec![0.0; side * side];
        let mut present = vec![0.0; side * side * biomes];
        let threads = std::thread::available_parallelism().map_or(4, usize::from);
        let rows_per_thread = side.div_ceil(threads);
        std::thread::scope(|scope| {
            for (chunk_index, (heights, present)) in heights
                .chunks_mut(rows_per_thread * side)
                .zip(present.chunks_mut(rows_per_thread * side * biomes))
                .enumerate()
            {
                scope.spawn(move || {
                    for (offset, (height, present)) in heights
                        .iter_mut()
                        .zip(present.chunks_mut(biomes))
                        .enumerate()
                    {
                        let index = chunk_index * rows_per_thread * side + offset;
                        let x =
                            (index % side) as f64 * DRAINAGE_CELL_METRES - WORLD_HALF_EXTENT_METERS;
                        let z =
                            (index / side) as f64 * DRAINAGE_CELL_METRES - WORLD_HALF_EXTENT_METERS;
                        let weights = self.weights(x, z);
                        *height = weights
                            .iter()
                            .map(|(biome, weight)| {
                                weight * self.biomes[biome].height.eval([x, 0.0, z], &[])
                            })
                            .sum();
                        for (biome, weight) in weights.iter() {
                            present[biome] = weight;
                        }
                    }
                });
            }
        });
        (heights, present)
    }

    /// Bounds on every biome's blend weight over the columns of a box: the
    /// extremes sampled at the drainage points in and one cell around it.
    /// Climate varies over hundreds of metres, so a weight cannot rise and
    /// fall again between neighbouring points.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "clamped to the drainage grid"
    )]
    fn weight_bounds(&self, x: Interval, z: Interval) -> [Interval; MAX_BIOMES] {
        let side = drainage_side();
        let biomes = self.biomes.len();
        let index = |value: f64, round_up: bool| {
            let cell = (value + WORLD_HALF_EXTENT_METERS) / DRAINAGE_CELL_METRES;
            let cell = if round_up { cell.ceil() } else { cell.floor() };
            (cell.max(0.0) as usize).min(side - 1)
        };
        let mut low = [u8::MAX; MAX_BIOMES];
        let mut high = [0_u8; MAX_BIOMES];
        for row in index(z.lo, false)..=index(z.hi, true) {
            for column in index(x.lo, false)..=index(x.hi, true) {
                let point = (column + row * side) * biomes;
                for biome in 0..biomes {
                    low[biome] = low[biome].min(self.weight_low[point + biome]);
                    high[biome] = high[biome].max(self.weight_high[point + biome]);
                }
            }
        }
        let mut bounds = [Interval::point(0.0); MAX_BIOMES];
        for biome in 0..biomes {
            bounds[biome] = Interval::new(
                f64::from(low[biome]) / 255.0,
                f64::from(high[biome]) / 255.0,
            );
        }
        bounds
    }

    /// Bounds on every biome's blend weight over a box, following
    /// [`Self::weights`] step by step in interval arithmetic and intersected
    /// with the drainage-grid bounds.
    fn blend_weight_bounds(
        &self,
        x: Interval,
        z: Interval,
        climate: [Interval; 4],
    ) -> [Interval; MAX_BIOMES] {
        let mut bounds = self.weight_bounds(x, z);
        let mut distances = [Interval::point(0.0); MAX_BIOMES];
        let mut nearest = Interval::point(f64::INFINITY);
        for (index, biome) in self.biomes.iter().enumerate() {
            let mut squared = Interval::point(0.0);
            for (target, value) in biome.target.iter().zip(climate) {
                if let Some(target) = target {
                    let offset = value.sub(Interval::point(*target)).abs();
                    squared = squared.add(offset.mul(offset));
                }
            }
            distances[index] = squared
                .monotone(|value| value.max(0.0).sqrt())
                .add(Interval::point(biome.rarity));
            nearest = nearest.min(distances[index]);
        }
        let mut raw = [Interval::point(0.0); MAX_BIOMES];
        for (index, slot) in raw.iter_mut().enumerate().take(self.biomes.len()) {
            let behind = distances[index].sub(nearest);
            let closeness = |gap: f64| (1.0 - gap.max(0.0) / self.blend).max(0.0);
            let (lo, hi) = (closeness(behind.hi), closeness(behind.lo));
            *slot = Interval::new(lo * lo * lo, hi * hi * hi);
        }
        // The nearest biome's raw weight is exactly one, so the total of the
        // raw weights never falls below one.
        let total_lo: f64 = raw.iter().map(|value| value.lo).sum();
        let total_hi: f64 = raw.iter().map(|value| value.hi).sum();
        let axis = |range: Interval, farthest: bool| {
            if farthest {
                range.lo.abs().max(range.hi.abs())
            } else {
                (range.lo.max(0.0) - range.hi.min(0.0)).max(0.0)
            }
        };
        let spawn_of =
            |distance: f64| 1.0 - smoothstep(self.spawn_radius * 0.5, self.spawn_radius, distance);
        let spawn = Interval::new(
            spawn_of(axis(x, true).hypot(axis(z, true))),
            spawn_of(axis(x, false).hypot(axis(z, false))),
        );
        for (index, slot) in bounds.iter_mut().enumerate().take(self.biomes.len()) {
            let share = Interval::new(
                raw[index].lo / (total_hi - raw[index].hi + raw[index].lo).max(1.0),
                (raw[index].hi / (total_lo - raw[index].lo + raw[index].hi).max(1.0)).min(1.0),
            );
            let weight = if index == self.spawn {
                Interval::new(
                    spawn.lo.mul_add(1.0 - share.lo, share.lo),
                    spawn.hi.mul_add(1.0 - share.hi, share.hi),
                )
            } else {
                Interval::new(share.lo * (1.0 - spawn.hi), share.hi * (1.0 - spawn.lo))
            };
            *slot = slot.intersect(weight);
        }
        bounds
    }

    fn climate_at(&self, x: f64, z: f64) -> [f64; 4] {
        self.climate
            .each_ref()
            .map(|tape| tape.eval([x, 0.0, z], &[]))
    }

    fn weights(&self, x: f64, z: f64) -> Weights {
        self.weights_in(self.climate_at(x, z), x, z)
    }

    fn weights_in(&self, climate: [f64; 4], x: f64, z: f64) -> Weights {
        let mut distances = [0.0_f64; MAX_BIOMES];
        let mut nearest = f64::INFINITY;
        for (index, biome) in self.biomes.iter().enumerate() {
            let squared: f64 = biome
                .target
                .iter()
                .zip(climate)
                .filter_map(|(target, value)| {
                    target.map(|target| (value - target) * (value - target))
                })
                .sum();
            distances[index] = squared.sqrt() + biome.rarity;
            nearest = nearest.min(distances[index]);
        }
        let mut raw = [0.0_f64; MAX_BIOMES];
        let mut total = 0.0;
        for (index, slot) in raw.iter_mut().enumerate().take(self.biomes.len()) {
            let closeness = (1.0 - (distances[index] - nearest) / self.blend).max(0.0);
            *slot = closeness * closeness * closeness;
            total += *slot;
        }
        let spawn = 1.0 - smoothstep(self.spawn_radius * 0.5, self.spawn_radius, x.hypot(z));
        let mut weights = Weights {
            biome: [0; MAX_BIOMES],
            weight: [0.0; MAX_BIOMES],
            len: 0,
        };
        for (index, value) in raw.iter().enumerate().take(self.biomes.len()) {
            let mut weight = value / total * (1.0 - spawn);
            if index == self.spawn {
                weight += spawn;
            }
            if weight > 0.0 {
                weights.biome[weights.len] = u8::try_from(index).expect("biome count is capped");
                weights.weight[weights.len] = weight;
                weights.len += 1;
            }
        }
        weights
    }

    fn column(&self, x: f64, z: f64) -> Column {
        if x.abs() >= WORLD_HALF_EXTENT_METERS || z.abs() >= WORLD_HALF_EXTENT_METERS {
            return Column {
                inside: false,
                weights: Weights {
                    biome: [0; MAX_BIOMES],
                    weight: [0.0; MAX_BIOMES],
                    len: 0,
                },
                ground: self.vertical.0,
                valley: f64::INFINITY,
                river_distance: f64::INFINITY,
                climate: [0.0; 4],
                carves: [CarveColumn::default(); MAX_CARVES],
                dominant: self.spawn,
            };
        }
        let climate = self.climate_at(x, z);
        let weights = self.weights_in(climate, x, z);
        let mut ground = 0.0;
        let mut river_factor = 0.0;
        let mut carves = [CarveColumn::default(); MAX_CARVES];
        let mut dominant = (self.spawn, f64::NEG_INFINITY);
        for (biome, weight) in weights.iter() {
            let compiled = &self.biomes[biome];
            ground += weight * compiled.height.eval([x, 0.0, z], &[]);
            river_factor += weight * compiled.rivers;
            for (carve, factor) in carves.iter_mut().zip(compiled.carves) {
                carve.factor += weight * factor;
            }
            let score = if weights.len > 1 {
                #[expect(clippy::cast_precision_loss, reason = "biome index is tiny")]
                let shift = biome as f64 * 1_000.0;
                0.2f64.mul_add(self.dither.sample(x + shift, 0.0, z), weight)
            } else {
                weight
            };
            if score > dominant.1 {
                dominant = (biome, score);
            }
        }
        let (valley, river_distance) = if river_factor > 0.0 {
            self.rivers
                .valley(x, z)
                .map_or((f64::INFINITY, f64::INFINITY), |valley| {
                    (
                        (1.0 - river_factor.min(1.0)).mul_add(RIVER_LIFT_METRES, valley.height),
                        valley.distance,
                    )
                })
        } else {
            (f64::INFINITY, f64::INFINITY)
        };
        for (carve, compiled) in carves.iter_mut().zip(&self.carves) {
            if carve.factor > 0.0 {
                let point = [x, 0.0, z];
                carve.roof = compiled.roof.eval(point, &climate);
                carve.floor = compiled.floor.eval(point, &climate);
                carve.top = compiled
                    .top
                    .as_ref()
                    .map_or(f64::INFINITY, |top| top.eval(point, &climate));
            }
        }
        Column {
            inside: true,
            weights,
            ground,
            valley,
            river_distance,
            climate,
            carves,
            dominant: dominant.0,
        }
    }

    /// How far carve layer `layer` opens at a point with biome density
    /// `blended`: 0 closes it, 1 opens it as authored, and larger factors widen
    /// every void by a few metres. Without `enclosed`, voids the roof keeps
    /// from the surface stay closed, as do voids deeper than the layer shows.
    fn carve_open(
        &self,
        layer: usize,
        column: &Column,
        y: f64,
        blended: f64,
        enclosed: bool,
    ) -> f64 {
        let carve = &column.carves[layer];
        if carve.factor <= 0.0 {
            return 0.0;
        }
        let mut open = carve.factor
            * smoothstep(carve.roof, carve.roof + ROOF_FADE_METRES, blended)
            * smoothstep(carve.floor, carve.floor + BAND_FADE_METRES, y);
        if carve.top.is_finite() {
            open *= 1.0 - smoothstep(carve.top - BAND_FADE_METRES, carve.top, y);
        }
        if !enclosed {
            let visible = self.carves[layer].visible;
            open *= (1.0 - smoothstep(0.0, DISTANT_ROOF_METRES, carve.roof))
                * (1.0 - smoothstep(visible - ROOF_FADE_METRES, visible, blended));
        }
        open
    }

    /// Applies rivers, carve layers, and the vertical limits to a biome
    /// blend. Returns the density and the carve layer that shaped it, plus
    /// one, or zero when none did.
    fn compose(
        &self,
        column: &Column,
        y: f64,
        blended: f64,
        enclosed: bool,
        mut void: impl FnMut(usize) -> f64,
    ) -> (f64, u8) {
        let mut density = blended.min(column.valley - y);
        let mut carved = 0;
        for layer in 0..self.carves.len() {
            let open = self.carve_open(layer, column, y, blended, enclosed);
            if open > 0.0 {
                let closed = (1.0 - open.min(1.0)) * CARVE_LIFT_METRES;
                let widened = (open - 1.0).max(0.0) * CARVE_WIDENING_METRES;
                let carve = closed - widened - void(layer);
                if carve < CARVE_REACH_METRES && carve < density {
                    density = carve;
                    carved = u8::try_from(layer + 1).expect("carve layers are capped");
                }
            }
        }
        (
            density.max(self.vertical.0 - y).min(self.vertical.1 - y),
            carved,
        )
    }

    fn density(&self, position: DVec3) -> f64 {
        let column = self.cached_column(position.x, position.z);
        if !column.inside {
            return -1.0;
        }
        self.density_in_column(&column, position)
    }

    /// [`Self::column`] through this thread's column cache.
    fn cached_column(&self, x: f64, z: f64) -> Column {
        let key = [x.to_bits(), z.to_bits()];
        #[expect(
            clippy::cast_possible_truncation,
            reason = "coordinates are hashed bit for bit"
        )]
        let slot = (mix(self.id
            ^ key[0].wrapping_mul(0x9e37_79b9_7f4a_7c15)
            ^ key[1].rotate_left(29)) as usize)
            % COLUMN_CACHE_SLOTS;
        if let Some(column) = COLUMNS.with(|cache| {
            cache.borrow()[slot]
                .and_then(|(id, cached, column)| (id == self.id && cached == key).then_some(column))
        }) {
            return column;
        }
        let column = self.column(x, z);
        COLUMNS.with(|cache| cache.borrow_mut()[slot] = Some((self.id, key, column)));
        column
    }

    fn density_in_column(&self, column: &Column, position: DVec3) -> f64 {
        self.density_and_carve(column, position).0
    }

    /// Density at a point with the carve layer that shaped it.
    fn density_and_carve(&self, column: &Column, position: DVec3) -> (f64, u8) {
        let point = position.to_array();
        let mut blended = 0.0;
        for (biome, weight) in column.weights.iter() {
            blended += weight * self.biomes[biome].density.eval(point, &[]);
        }
        let [temperature, humidity, continentalness, weirdness] = column.climate;
        let inputs = [blended, temperature, humidity, continentalness, weirdness];
        self.compose(column, position.y, blended, true, |layer| {
            self.carves[layer].void.eval(point, &inputs)
        })
    }

    /// Densities over a lattice, with its columns and the carve layer that
    /// shaped each point. With `cull`, blocks the interval bounds prove far
    /// from any surface hold a bound of the right sign instead of exact
    /// values; meshing only needs their sign. Without `enclosed`, voids that
    /// cannot reach the surface are left closed.
    fn density_lattice(
        &self,
        lattice: &Lattice,
        cull: bool,
        enclosed: bool,
    ) -> (Vec<f64>, Vec<Column>, Vec<u8>) {
        let [nx, ny, nz] = lattice.dims;
        let columns: Vec<Column> = (0..nz)
            .flat_map(|k| (0..nx).map(move |i| (i, k)))
            .map(|(i, k)| self.column(lattice.coordinate(0, i), lattice.coordinate(2, k)))
            .collect();
        let mut densities = vec![0.0; lattice.len()];
        let mut carved = vec![0; lattice.len()];
        let block = if cull { CULL_BLOCK } else { nx.max(ny).max(nz) };
        let spacing = f64::from(lattice.stride) * TERRAIN_CELL_METERS;
        // Blocks stacked over the same columns share their planar values.
        let mut caches = BlockCaches {
            biomes: self.biomes.iter().map(|_| PlanarCache::default()).collect(),
            carves: self.carves.iter().map(|_| PlanarCache::default()).collect(),
        };
        for k0 in (0..nz).step_by(block) {
            for i0 in (0..nx).step_by(block) {
                for j0 in (0..ny).step_by(block) {
                    let start = [i0, j0, k0];
                    let dims = [block.min(nx - i0), block.min(ny - j0), block.min(nz - k0)];
                    if cull {
                        // Two samples of margin keep every crossing and the
                        // gradients around it inside evaluated blocks.
                        let margin = DVec3::splat(2.0 * spacing);
                        let corner = |offset: [usize; 3]| {
                            DVec3::new(
                                lattice.coordinate(0, offset[0]),
                                lattice.coordinate(1, offset[1]),
                                lattice.coordinate(2, offset[2]),
                            )
                        };
                        let bounds = self.interval(
                            corner(start) - margin,
                            corner([i0 + dims[0] - 1, j0 + dims[1] - 1, k0 + dims[2] - 1]) + margin,
                            enclosed,
                        );
                        let fill = if bounds.hi < 0.0 {
                            Some(bounds.hi)
                        } else if bounds.lo > 0.0 {
                            Some(bounds.lo)
                        } else {
                            None
                        };
                        if let Some(fill) = fill {
                            for k in k0..k0 + dims[2] {
                                for j in j0..j0 + dims[1] {
                                    let row = nx * (j + ny * k);
                                    densities[row + i0..row + i0 + dims[0]].fill(fill);
                                }
                            }
                            continue;
                        }
                    }
                    let block = LatticeBlock {
                        lattice,
                        columns: &columns,
                        start,
                        dims,
                    };
                    self.density_block(&block, enclosed, &mut caches, &mut densities, &mut carved);
                }
            }
        }
        (densities, columns, carved)
    }

    /// Exact densities of one block of a lattice, written in place with the
    /// carve layer that shaped each point.
    fn density_block(
        &self,
        block: &LatticeBlock<'_>,
        enclosed: bool,
        caches: &mut BlockCaches,
        densities: &mut [f64],
        carved: &mut [u8],
    ) {
        let LatticeBlock {
            lattice,
            columns,
            start,
            dims,
        } = *block;
        let [nx, ny, _] = lattice.dims;
        let coordinate = |axis: usize, index: usize| lattice.coordinate(axis, start[axis] + index);
        let column = |i: usize, k: usize| &columns[start[0] + i + nx * (start[2] + k)];
        let local = |i: usize, j: usize, k: usize| i + dims[0] * (j + dims[1] * k);
        let len = dims[0] * dims[1] * dims[2];
        let mut blended = vec![0.0; len];
        let mut values = Vec::new();
        for (biome_index, biome) in self.biomes.iter().enumerate() {
            let present = (0..dims[2])
                .any(|k| (0..dims[0]).any(|i| column(i, k).weights.of(biome_index) > 0.0));
            if !present {
                continue;
            }
            biome.density.eval_grid_varying(
                dims,
                &coordinate,
                &[],
                Some(&mut caches.biomes[biome_index]),
                &mut values,
            );
            for k in 0..dims[2] {
                for i in 0..dims[0] {
                    let weight = column(i, k).weights.of(biome_index);
                    if weight <= 0.0 {
                        continue;
                    }
                    for j in 0..dims[1] {
                        let index = local(i, j, k);
                        blended[index] += weight * values[index];
                    }
                }
            }
        }
        let voids = self.block_voids(block, &blended, enclosed, &mut caches.carves);
        for k in 0..dims[2] {
            for j in 0..dims[1] {
                let y = coordinate(1, j);
                for i in 0..dims[0] {
                    let column = column(i, k);
                    let index = local(i, j, k);
                    let target = start[0] + i + nx * (start[1] + j + ny * (start[2] + k));
                    (densities[target], carved[target]) = if column.inside {
                        self.compose(column, y, blended[index], enclosed, |layer| {
                            // A layer skipped above is closed at every point.
                            voids[layer]
                                .get(index)
                                .copied()
                                .unwrap_or(f64::NEG_INFINITY)
                        })
                    } else {
                        (-1.0, 0)
                    };
                }
            }
        }
    }

    /// Each carve layer's void over a block, empty for layers closed there:
    /// only layers open somewhere in the block, and whose bounds bring them
    /// within reach, are evaluated.
    fn block_voids(
        &self,
        block: &LatticeBlock<'_>,
        blended: &[f64],
        enclosed: bool,
        caches: &mut [PlanarCache],
    ) -> [Vec<f64>; MAX_CARVES] {
        let LatticeBlock {
            lattice,
            columns,
            start,
            dims,
        } = *block;
        let nx = lattice.dims[0];
        let coordinate = |axis: usize, index: usize| lattice.coordinate(axis, start[axis] + index);
        let column = |i: usize, k: usize| &columns[start[0] + i + nx * (start[2] + k)];
        let local = |i: usize, j: usize, k: usize| i + dims[0] * (j + dims[1] * k);
        let len = blended.len();
        let mut voids: [Vec<f64>; MAX_CARVES] = Default::default();
        let rock = blended.iter().fold(
            Interval::new(f64::INFINITY, f64::NEG_INFINITY),
            |bounds, &value| Interval::new(bounds.lo.min(value), bounds.hi.max(value)),
        );
        let last = dims.map(|edge| edge - 1);
        let domain = [0, 1, 2].map(|axis| {
            let (a, b) = (coordinate(axis, 0), coordinate(axis, last[axis]));
            Interval::new(a.min(b), a.max(b))
        });
        let mut climate: [Vec<f64>; 4] = Default::default();
        let mut climate_bounds = [Interval::new(f64::INFINITY, f64::NEG_INFINITY); 4];
        for (channel, (buffer, bounds)) in climate.iter_mut().zip(&mut climate_bounds).enumerate() {
            buffer.reserve(len);
            for k in 0..dims[2] {
                for _ in 0..dims[1] {
                    for i in 0..dims[0] {
                        let value = column(i, k).climate[channel];
                        buffer.push(value);
                        *bounds = Interval::new(bounds.lo.min(value), bounds.hi.max(value));
                    }
                }
            }
        }
        let inputs: [&[f64]; 5] = [blended, &climate[0], &climate[1], &climate[2], &climate[3]];
        let mut input_bounds = [rock; 5];
        input_bounds[1..].copy_from_slice(&climate_bounds);
        for (layer, carve) in self.carves.iter().enumerate() {
            let widest = (0..dims[2])
                .flat_map(|k| (0..dims[0]).map(move |i| (i, k)))
                .map(|(i, k)| (column(i, k).carves[layer].factor - 1.0).max(0.0))
                .fold(0.0, f64::max)
                * CARVE_WIDENING_METRES;
            let needed = (0..dims[2]).any(|k| {
                (0..dims[0]).any(|i| {
                    let column = column(i, k);
                    column.inside
                        && column.carves[layer].factor > 0.0
                        && (0..dims[1]).any(|j| {
                            self.carve_open(
                                layer,
                                column,
                                coordinate(1, j),
                                blended[local(i, j, k)],
                                enclosed,
                            ) > 0.0
                        })
                })
            }) && -carve.void.interval(domain, &input_bounds).hi - widest
                < CARVE_REACH_METRES;
            if needed {
                carve.void.eval_grid_varying(
                    dims,
                    &coordinate,
                    &inputs,
                    Some(&mut caches[layer]),
                    &mut voids[layer],
                );
            }
        }
        voids
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "stored densities are f32 by contract"
    )]
    fn sample(
        &self,
        column: &Column,
        position: DVec3,
        density: f64,
        gradient: [f64; 3],
        carved: u8,
    ) -> TerrainSample {
        let slope = gradient
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        let (up, depth) = if slope > 1.0e-9 {
            (-gradient[1] / slope, density.max(0.0) / slope.max(0.25))
        } else {
            (1.0, density.max(0.0))
        };
        let probe = SurfaceProbe {
            position: position.to_array(),
            depth,
            up,
            river_distance: column.river_distance,
            carved,
        };
        let (material, surface) = self.biomes[column.dominant].rules.paint(&probe);
        TerrainSample {
            density: density as f32,
            material,
            surface,
            compaction: 0,
            looseness: 0,
        }
    }

    fn interval(&self, minimum: DVec3, maximum: DVec3, enclosed: bool) -> Interval {
        let (bottom, top) = self.vertical;
        let y = Interval::new(minimum.y, maximum.y);
        if minimum.y >= top {
            return Interval::new(top - maximum.y, top - minimum.y);
        }
        if minimum.x >= WORLD_HALF_EXTENT_METERS
            || maximum.x <= -WORLD_HALF_EXTENT_METERS
            || minimum.z >= WORLD_HALF_EXTENT_METERS
            || maximum.z <= -WORLD_HALF_EXTENT_METERS
        {
            return Interval::point(-1.0);
        }
        if maximum.x.abs().max(minimum.x.abs()) >= WORLD_HALF_EXTENT_METERS
            || maximum.z.abs().max(minimum.z.abs()) >= WORLD_HALF_EXTENT_METERS
        {
            return Interval::EVERYTHING;
        }
        let x = Interval::new(minimum.x, maximum.x);
        let z = Interval::new(minimum.z, maximum.z);
        let domain = [x, y, z];
        let climate = self
            .climate
            .each_ref()
            .map(|tape| tape.interval([x, Interval::point(0.0), z], &[]));
        let weights = self.blend_weight_bounds(x, z, climate);
        let mut present_weights = Vec::with_capacity(self.biomes.len());
        let mut densities = Vec::with_capacity(self.biomes.len());
        let mut rivers_possible = false;
        let mut factors = [0.0_f64; MAX_CARVES];
        for (index, biome) in self.biomes.iter().enumerate() {
            if weights[index].hi <= 0.0 {
                continue;
            }
            present_weights.push(weights[index]);
            densities.push(biome.density.interval(domain, &[]));
            rivers_possible |= biome.rivers > 0.0;
            for (factor, biome_factor) in factors.iter_mut().zip(biome.carves) {
                *factor = factor.max(biome_factor);
            }
        }
        let blended = blend_bounds(&present_weights, &densities);
        let mut density = blended;
        if rivers_possible && let Some(lowest) = self.rivers.lowest_valley(x, z) {
            density = Interval::new(density.lo.min(lowest - y.hi), density.hi);
        }
        let planar = [x, Interval::point(0.0), z];
        for (carve, factor) in self.carves.iter().zip(factors) {
            if factor <= 0.0 {
                continue;
            }
            let roof = carve.roof.interval(planar, &climate);
            if blended.hi <= roof.lo
                || y.hi <= carve.floor.interval(planar, &climate).lo
                || carve
                    .top
                    .as_ref()
                    .is_some_and(|top| y.lo >= top.interval(planar, &climate).hi)
                || (!enclosed && (roof.lo >= DISTANT_ROOF_METRES || blended.lo >= carve.visible))
            {
                continue;
            }
            let mut inputs = [blended; 5];
            inputs[1..].copy_from_slice(&climate);
            let void = carve.void.interval(domain, &inputs);
            let widest = (factor - 1.0).max(0.0) * CARVE_WIDENING_METRES;
            density = Interval::new(density.lo.min(-void.hi - widest), density.hi);
        }
        density
            .max(Interval::point(bottom).sub(y))
            .min(Interval::point(top).sub(y))
    }
}

fn compile_carves(
    docs: &[CarveDoc],
    library: &std::collections::BTreeMap<String, Expr>,
    fields: &WorldFields,
    base: u64,
) -> Result<Vec<CompiledCarve>, WorldgenError> {
    let empty = std::collections::BTreeMap::new();
    let scope = Scope {
        local: &empty,
        library,
        fields: Some(fields),
    };
    docs.iter()
        .zip(0_u64..)
        .map(|(doc, index)| compile_carve(doc, scope, mix(base ^ 5 ^ (index << 8))))
        .collect()
}

/// Climate channels in the order columns store them, readable by name in
/// carve layers.
const CLIMATE_NAMES: [&str; 4] = ["temperature", "humidity", "continentalness", "weirdness"];

fn compile_carve(
    doc: &CarveDoc,
    scope: Scope<'_>,
    seed: u64,
) -> Result<CompiledCarve, WorldgenError> {
    let context = format!("world.ron carve `{}`", doc.name);
    let climate = CLIMATE_NAMES.map(str::to_owned);
    // The column's climate is a constant input to its planar parts, and a
    // per-point one to the void beside the rock depth.
    let planar = |expr: &Expr, part: &str| {
        let context = format!("{context} {part}");
        let tape = compile(expr, scope, &climate, seed, &context)?;
        if tape.result_axes() & tape::AXIS_Y != 0 {
            return Err(WorldgenError::Invalid {
                context,
                message: "must depend on x and z only; use 2D noise and no Y".to_owned(),
            });
        }
        Ok(tape)
    };
    let varying = std::iter::once("rock".to_owned())
        .chain(climate.iter().cloned())
        .collect::<Vec<_>>();
    Ok(CompiledCarve {
        name: doc.name.clone(),
        void: compile_varying(&doc.void, scope, &varying, seed, &format!("{context} void"))?,
        roof: planar(&doc.roof, "roof")?,
        floor: planar(&doc.floor, "floor")?,
        top: doc.top.as_ref().map(|top| planar(top, "top")).transpose()?,
        visible: doc.visible,
    })
}

fn compile_biome(
    doc: &BiomeDoc,
    library: &std::collections::BTreeMap<String, Expr>,
    palette: &SurfacePalette,
    carve_names: &[(String, f64)],
    base: u64,
) -> Result<CompiledBiome, WorldgenError> {
    let seed = mix(base
        ^ doc
            .name
            .bytes()
            .fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
                (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
            }));
    let context = format!("biome `{}`", doc.name);
    let height_scope = Scope {
        local: &doc.definitions,
        library,
        fields: None,
    };
    let height = compile_planar(
        &doc.height,
        height_scope,
        seed,
        &format!("{context} height"),
    )?;
    let mut local = doc.definitions.clone();
    local.insert("height".to_owned(), doc.height.clone());
    let scope = Scope {
        local: &local,
        library,
        fields: None,
    };
    let density_expr = doc
        .density
        .clone()
        .unwrap_or_else(|| Expr::Height(Box::new(Expr::Ref("height".to_owned()))));
    let density = compile(
        &density_expr,
        scope,
        &[],
        seed,
        &format!("{context} density"),
    )?;
    let names = carve_names
        .iter()
        .map(|(name, _)| name.clone())
        .collect::<Vec<_>>();
    let rules = SurfaceRules::new(&doc.surface, palette, &names, scope, seed, &context)?;
    let mut carves = [0.0; MAX_CARVES];
    for (slot, (name, unlisted)) in carves.iter_mut().zip(carve_names) {
        *slot = doc.carves.get(name).copied().unwrap_or(*unlisted);
    }
    if let Some(unknown) = doc.carves.keys().find(|name| !names.contains(name)) {
        return Err(WorldgenError::Invalid {
            context: format!("{context} carves"),
            message: format!("unknown carve layer `{unknown}`"),
        });
    }
    Ok(CompiledBiome {
        name: doc.name.clone(),
        target: [
            doc.climate.temperature,
            doc.climate.humidity,
            doc.climate.continentalness,
            doc.climate.weirdness,
        ],
        rarity: doc.rarity,
        height,
        density,
        rivers: doc.rivers,
        carves,
        rules,
    })
}

#[cfg(test)]
mod tests;
