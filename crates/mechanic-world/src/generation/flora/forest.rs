//! Trees placed in a world: the compiled species and biome layers, where each
//! tree stands, and the caches that keep grown trees between samples.
//!
//! Every tree is a pure function of the world seed, its layer, and its grid
//! cell, so any thread may place or grow one; the caches only save the work.
//! Placing a tree is cheap and is cached per thread. Growing one costs about a
//! millisecond and up to a megabyte, so grown trees live in one shared map
//! that forgets its oldest beyond a memory budget, behind a small table of
//! recent trees per thread. Far away, where the terrain is sampled too coarsely
//! for twigs, a tree is drawn as an impostor: its trunk and a crown-shaped
//! ellipsoid, which needs no growing.

use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, PoisonError};

use bevy_math::DVec3;

use super::super::grid::{JitterGrid, cache_slot};
use super::super::interval::Interval;
use super::super::scatter::mix;
use super::super::surfaces::SurfaceId;
use super::super::tape::Tape;
use super::grow::{crown_shape, tree_height};
use super::model::{Parts, SAMPLE_MARGIN};
use super::{FLARE_HEIGHT, FLARE_WIDENING, Part, SpeciesSpec, TreeModel, grow_tree};
use crate::TerrainMaterial;

/// Bytes of grown trees the shared cache keeps before it forgets the oldest.
const SHARED_BUDGET_BYTES: usize = 192 << 20;

/// Bytes an empty cell costs the shared cache.
const EMPTY_CELL_BYTES: usize = 64;

/// Recent grown trees each thread keeps in front of the shared cache.
const LOCAL_TREE_SLOTS: usize = 128;

/// Recent placements each thread keeps.
const LOCAL_INSTANCE_SLOTS: usize = 16_384;

/// Extra reach beyond what the genome allows, for wobble and leaning stems.
const REACH_SLACK: f64 = 2.0;

/// Coarsest lattice stride, in cells, that samples grown trees. Coarser
/// levels of detail draw impostors: twigs are finer than their samples.
pub(crate) const MAX_GROWN_STRIDE: i32 = 4;

/// Densest a tree gets: no trunk or leaf ball is wider than this.
pub(crate) const TREE_DENSITY_CEILING: f64 = 4.0;

/// An impostor's crown fills this much of the crown the genome aims for, as
/// foliage with holes reads smaller from afar.
const IMPOSTOR_CROWN: f64 = 0.8;

/// A species as the world grows it, with its looks and how far its trees can
/// reach from their origins.
#[derive(Debug)]
pub(crate) struct ForestSpecies {
    pub(crate) spec: SpeciesSpec,
    pub(crate) bark: SurfaceId,
    pub(crate) foliage: SurfaceId,
    /// Horizontal reach from the origin of anything above ground, at the
    /// tallest height, in metres.
    reach: f64,
    /// Horizontal reach of the roots, in metres.
    root_reach: f64,
    /// Deepest point below the origin.
    depth: f64,
}

impl ForestSpecies {
    pub(crate) fn new(spec: SpeciesSpec, bark: SurfaceId, foliage: SurfaceId) -> Self {
        let tallest = spec.height.1;
        let crown = spec.width * tallest * 0.5;
        let trunk = spec.girth * tallest * 0.5;
        let disc = 1.5 * trunk * f64::from(spec.stems.1).sqrt();
        let slack = disc + SAMPLE_MARGIN + REACH_SLACK;
        let reach = crown + spec.foliage.size * 1.3 + slack;
        let root_reach = spec.roots.spread * crown + trunk * FLARE_WIDENING + slack;
        let depth = spec.roots.depth + trunk + FLARE_HEIGHT + SAMPLE_MARGIN + REACH_SLACK;
        Self {
            spec,
            bark,
            foliage,
            reach,
            root_reach,
            depth,
        }
    }

    /// Whether a tree drawn `height` tall at `origin` can reach into a box:
    /// its crown above ground, or with `roots`, its roots below.
    fn could_reach(&self, origin: DVec3, height: f64, domain: &[Interval; 3], roots: bool) -> bool {
        let scale = height / self.spec.height.1;
        let crown = DVec3::new(
            self.reach * scale + REACH_SLACK,
            0.0,
            self.reach * scale + REACH_SLACK,
        );
        let root_reach = DVec3::new(self.root_reach, 0.0, self.root_reach);
        let flare = DVec3::Y * (FLARE_HEIGHT + SAMPLE_MARGIN);
        overlaps(
            domain,
            origin - crown - flare,
            origin + crown + DVec3::Y * self.rise(height),
        ) || roots
            && overlaps(
                domain,
                origin - root_reach - DVec3::Y * self.depth,
                origin + root_reach + flare,
            )
    }

    /// Highest point above the origin of a tree drawn this tall.
    fn rise(&self, height: f64) -> f64 {
        height * 1.2 + self.spec.foliage.size * 1.3 + SAMPLE_MARGIN
    }
}

/// One biome's trees of one species.
#[derive(Debug)]
pub(crate) struct ForestLayer {
    pub(crate) biome: usize,
    pub(crate) species: usize,
    pub(crate) grid: JitterGrid,
    pub(crate) mask: Option<Tape>,
}

/// Where a tree stands, before it is grown.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct TreeInstance {
    origin: DVec3,
    seed: u64,
    height: f64,
    species: usize,
}

impl TreeInstance {
    /// Base of the tree, where its stems leave the ground.
    #[cfg(test)]
    pub(crate) const fn origin(&self) -> DVec3 {
        self.origin
    }

    /// Height drawn for the tree.
    #[cfg(test)]
    pub(crate) const fn height(&self) -> f64 {
        self.height
    }
}

/// A grown tree and its species.
#[derive(Debug)]
pub(crate) struct Tree {
    pub(crate) model: TreeModel,
    pub(crate) species: usize,
}

impl Tree {
    /// Whether the tree can reach into a box.
    fn touches(&self, domain: &[Interval; 3]) -> bool {
        overlaps(
            domain,
            self.model.min - SAMPLE_MARGIN,
            self.model.max + SAMPLE_MARGIN,
        )
    }
}

fn overlaps(domain: &[Interval; 3], low: DVec3, high: DVec3) -> bool {
    domain[0].hi >= low.x
        && domain[0].lo <= high.x
        && domain[1].hi >= low.y
        && domain[1].lo <= high.y
        && domain[2].hi >= low.z
        && domain[2].lo <= high.z
}

/// The part of a tree that dominates a point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct TreeHit {
    pub(crate) density: f64,
    pub(crate) part: Part,
    pub(crate) species: usize,
}

type Key = (u64, u32, i64, i64);

type TreeSlot = Option<(Key, Option<Arc<Tree>>)>;

type InstanceSlot = Option<(Key, Option<TreeInstance>)>;

thread_local! {
    static LOCAL_TREES: RefCell<Vec<TreeSlot>> = RefCell::new(vec![None; LOCAL_TREE_SLOTS]);
    static LOCAL_INSTANCES: RefCell<Vec<InstanceSlot>> =
        RefCell::new(vec![None; LOCAL_INSTANCE_SLOTS]);
}

#[derive(Debug, Default)]
struct SharedTrees {
    map: HashMap<Key, Option<Arc<Tree>>>,
    order: VecDeque<(Key, usize)>,
    bytes: usize,
}

/// Every tree a world can grow.
#[derive(Debug)]
pub(crate) struct Forest {
    id: u64,
    species: Vec<ForestSpecies>,
    layers: Vec<ForestLayer>,
    shared: Mutex<SharedTrees>,
}

/// Finds where a layer's tree at `(x, z)` stands, if it may grow there.
pub(crate) type Placement<'a> = &'a dyn Fn(&ForestLayer, f64, f64) -> Option<DVec3>;

impl Forest {
    pub(crate) fn new(id: u64, species: Vec<ForestSpecies>, layers: Vec<ForestLayer>) -> Self {
        Self {
            id,
            species,
            layers,
            shared: Mutex::new(SharedTrees::default()),
        }
    }

    /// Tallest tree a layer grows, in metres.
    pub(crate) fn tallest(&self, layer: &ForestLayer) -> f64 {
        self.species[layer.species].spec.height.1
    }

    /// Whether the world grows any trees at all.
    pub(crate) fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// How far any tree reaches from its origin: sideways, up, and down.
    pub(crate) fn reach(&self) -> (f64, f64, f64) {
        self.layers
            .iter()
            .map(|layer| &self.species[layer.species])
            .fold((0.0, 0.0, 0.0), |(reach, rise, depth), species| {
                (
                    reach
                        .max(species.reach + REACH_SLACK)
                        .max(species.root_reach),
                    rise.max(species.rise(species.spec.height.1)),
                    depth.max(species.depth),
                )
            })
    }

    /// Material and look of a tree part.
    pub(crate) fn paint(&self, hit: TreeHit) -> (TerrainMaterial, SurfaceId) {
        let species = &self.species[hit.species];
        match hit.part {
            Part::Wood | Part::Root => (TerrainMaterial::Wood, species.bark),
            Part::Foliage => (TerrainMaterial::Foliage, species.foliage),
        }
    }

    fn key(&self, layer: usize, cell_x: i64, cell_z: i64) -> Key {
        (
            self.id,
            u32::try_from(layer).unwrap_or(u32::MAX),
            cell_x,
            cell_z,
        )
    }

    fn slot(&self, layer: usize, cell_x: i64, cell_z: i64, slots: usize) -> usize {
        cache_slot(self.id ^ (layer as u64) << 48, cell_x, cell_z, slots)
    }

    /// Where a layer's tree in a grid cell stands, if it has one.
    fn instance(
        &self,
        layer: usize,
        cell_x: i64,
        cell_z: i64,
        place: Placement<'_>,
    ) -> Option<TreeInstance> {
        let key = self.key(layer, cell_x, cell_z);
        let slot = self.slot(layer, cell_x, cell_z, LOCAL_INSTANCE_SLOTS);
        if let Some(found) = LOCAL_INSTANCES.with(|local| {
            local.borrow()[slot].and_then(|(cached, instance)| (cached == key).then_some(instance))
        }) {
            return found;
        }
        let found = self.place(layer, cell_x, cell_z, place);
        LOCAL_INSTANCES.with(|local| local.borrow_mut()[slot] = Some((key, found)));
        found
    }

    fn place(
        &self,
        layer: usize,
        cell_x: i64,
        cell_z: i64,
        place: Placement<'_>,
    ) -> Option<TreeInstance> {
        let forest_layer = &self.layers[layer];
        let (x, z, _) = forest_layer.grid.place(cell_x, cell_z)?;
        if let Some(mask) = &forest_layer.mask
            && mask.eval([x, 0.0, z], &[]) <= 0.0
        {
            return None;
        }
        let origin = place(forest_layer, x, z)?;
        #[expect(clippy::cast_sign_loss, reason = "cell indices are hashed bit for bit")]
        let seed = mix(forest_layer.grid.seed
            ^ (cell_x as u64).wrapping_mul(0xa076_1d64_78bd_642f)
            ^ (cell_z as u64).wrapping_mul(0xe703_7ed1_a0b4_28db));
        let species = forest_layer.species;
        Some(TreeInstance {
            origin,
            seed,
            height: tree_height(&self.species[species].spec, seed),
            species,
        })
    }

    /// The grown tree a layer has in a grid cell, if any.
    fn tree(
        &self,
        layer: usize,
        cell_x: i64,
        cell_z: i64,
        place: Placement<'_>,
    ) -> Option<Arc<Tree>> {
        let key = self.key(layer, cell_x, cell_z);
        let slot = self.slot(layer, cell_x, cell_z, LOCAL_TREE_SLOTS);
        if let Some(found) = LOCAL_TREES.with(|local| {
            local.borrow()[slot]
                .as_ref()
                .and_then(|(cached, tree)| (*cached == key).then(|| tree.clone()))
        }) {
            return found;
        }
        let found = self.shared_tree(key, layer, cell_x, cell_z, place);
        LOCAL_TREES.with(|local| local.borrow_mut()[slot] = Some((key, found.clone())));
        found
    }

    fn shared_tree(
        &self,
        key: Key,
        layer: usize,
        cell_x: i64,
        cell_z: i64,
        place: Placement<'_>,
    ) -> Option<Arc<Tree>> {
        let lock = || self.shared.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(found) = lock().map.get(&key) {
            return found.clone();
        }
        // Grown outside the lock: two threads may grow the same tree, which
        // is the same tree either way.
        let grown = self.instance(layer, cell_x, cell_z, place).map(|instance| {
            Arc::new(Tree {
                model: grow_tree(
                    &self.species[instance.species].spec,
                    instance.seed,
                    instance.origin,
                ),
                species: instance.species,
            })
        });
        let bytes = grown
            .as_ref()
            .map_or(EMPTY_CELL_BYTES, |tree| tree.model.memory_bytes());
        let mut shared = lock();
        if shared.map.insert(key, grown.clone()).is_none() {
            shared.order.push_back((key, bytes));
            shared.bytes += bytes;
        }
        while shared.bytes > SHARED_BUDGET_BYTES {
            let Some((old, old_bytes)) = shared.order.pop_front() else {
                break;
            };
            shared.map.remove(&old);
            shared.bytes -= old_bytes;
        }
        grown
    }

    /// Calls `visit` with each layer and grid cell whose tree could reach a
    /// box.
    fn cells_near(&self, domain: &[Interval; 3], mut visit: impl FnMut(usize, i64, i64)) {
        if domain
            .iter()
            .any(|axis| !axis.lo.is_finite() || !axis.hi.is_finite())
        {
            return;
        }
        for (index, layer) in self.layers.iter().enumerate() {
            let species = &self.species[layer.species];
            let reach = (species.reach + REACH_SLACK).max(species.root_reach);
            let (x0, x1, z0, z1) = layer.grid.cells_near(domain[0], domain[2], reach);
            for cell_z in z0..=z1 {
                for cell_x in x0..=x1 {
                    visit(index, cell_x, cell_z);
                }
            }
        }
    }

    fn could_reach(&self, instance: &TreeInstance, domain: &[Interval; 3], parts: Parts) -> bool {
        self.species[instance.species].could_reach(
            instance.origin,
            instance.height,
            domain,
            parts == Parts::All,
        )
    }

    /// Number of grid cells a search of a box would visit.
    pub(crate) fn cells_in(&self, domain: &[Interval; 3]) -> f64 {
        self.layers
            .iter()
            .map(|layer| {
                let species = &self.species[layer.species];
                let reach = (species.reach + REACH_SLACK).max(species.root_reach);
                let span =
                    |axis: Interval| (axis.hi - axis.lo + 2.0 * reach) / layer.grid.cell + 2.0;
                span(domain[0]) * span(domain[2])
            })
            .sum()
    }

    /// Whether any tree's bounds reach into a box, from placements alone.
    pub(crate) fn any_near(&self, domain: [Interval; 3], place: Placement<'_>) -> bool {
        let mut found = false;
        self.cells_near(&domain, |layer, cell_x, cell_z| {
            if found {
                return;
            }
            if let Some(instance) = self.instance(layer, cell_x, cell_z, place) {
                found = self.could_reach(&instance, &domain, Parts::Solid);
            }
        });
        found
    }

    /// Grown trees whose `parts` can reach into a box, appended to `found`.
    pub(crate) fn trees_near(
        &self,
        domain: [Interval; 3],
        parts: Parts,
        place: Placement<'_>,
        found: &mut Vec<Arc<Tree>>,
    ) {
        self.cells_near(&domain, |layer, cell_x, cell_z| {
            if self
                .instance(layer, cell_x, cell_z, place)
                .is_some_and(|instance| self.could_reach(&instance, &domain, parts))
                && let Some(tree) = self.tree(layer, cell_x, cell_z, place)
                && tree.touches(&domain)
            {
                found.push(tree);
            }
        });
    }

    /// Placed trees that can reach into a box, appended to `found`.
    pub(crate) fn instances_near(
        &self,
        domain: [Interval; 3],
        place: Placement<'_>,
        found: &mut Vec<TreeInstance>,
    ) {
        self.cells_near(&domain, |layer, cell_x, cell_z| {
            if let Some(instance) = self.instance(layer, cell_x, cell_z, place)
                && self.could_reach(&instance, &domain, Parts::Solid)
            {
                found.push(instance);
            }
        });
    }

    /// The grown tree whose `parts` dominate a point, if any reaches it, with
    /// wood at least `floor` metres thick.
    pub(crate) fn sample(
        &self,
        point: DVec3,
        parts: Parts,
        floor: f64,
        place: Placement<'_>,
    ) -> Option<TreeHit> {
        if self.is_empty() {
            return None;
        }
        let domain = [
            Interval::point(point.x),
            Interval::point(point.y),
            Interval::point(point.z),
        ];
        let mut best: Option<TreeHit> = None;
        self.cells_near(&domain, |layer, cell_x, cell_z| {
            if !self
                .instance(layer, cell_x, cell_z, place)
                .is_some_and(|instance| self.could_reach(&instance, &domain, parts))
            {
                return;
            }
            let Some(tree) = self.tree(layer, cell_x, cell_z, place) else {
                return;
            };
            if let Some((density, part)) = tree.model.sample_parts(point, parts, floor) {
                let density = f64::from(density);
                if best.is_none_or(|hit| density > hit.density) {
                    best = Some(TreeHit {
                        density,
                        part,
                        species: tree.species,
                    });
                }
            }
        });
        best
    }

    /// Bounds of an impostor whose trunk is at least `floor` metres thick:
    /// its crown and trunk.
    pub(crate) fn impostor_bounds(&self, instance: &TreeInstance, floor: f64) -> (DVec3, DVec3) {
        let spec = &self.species[instance.species].spec;
        let height = instance.height;
        let widest = (spec.width * height * 0.5 * IMPOSTOR_CROWN)
            .max(0.5)
            .max(floor)
            + 0.5;
        let side = DVec3::new(widest, 0.0, widest);
        (
            instance.origin - side - DVec3::Y * 0.5,
            instance.origin + side + DVec3::Y * (height + 0.5),
        )
    }

    /// An impostor's density at a point, for terrain too coarse to show
    /// twigs: a trunk at least `floor` metres thick under a solid crown of
    /// the species' own envelope.
    pub(crate) fn impostor(&self, instance: &TreeInstance, point: DVec3, floor: f64) -> TreeHit {
        let spec = &self.species[instance.species].spec;
        let height = instance.height;
        let crown_from = spec.crown_base * height;
        let crown_height = (height - crown_from).max(1.0);
        let widest = (spec.width * height * 0.5 * IMPOSTOR_CROWN).max(0.5);
        let rise = point.y - instance.origin.y;
        let t = ((rise - crown_from) / crown_height).clamp(0.0, 1.0);
        let across = (point - instance.origin).with_y(0.0).length();
        let crown = (crown_shape(t, spec.dominance) * widest - across)
            .min(rise - crown_from)
            .min(height - rise);
        let trunk_radius = (spec.girth * height * 0.5).max(floor);
        let trunk = (trunk_radius - across)
            .min(rise + 0.5)
            .min(crown_from + 0.5 * crown_height - rise);
        if crown >= trunk {
            TreeHit {
                density: crown,
                part: Part::Foliage,
                species: instance.species,
            }
        } else {
            TreeHit {
                density: trunk,
                part: Part::Wood,
                species: instance.species,
            }
        }
    }
}
