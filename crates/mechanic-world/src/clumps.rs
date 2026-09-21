//! Persistent loose material and prepared, conservative ownership changes.

use std::collections::BTreeMap;

use bevy_math::{DQuat, DVec3};
use serde::{Deserialize, Serialize};

use crate::{
    BreakageResponse, ExtractionCell, MATERIAL_QUANTUM_M3, TerrainEditOutcome, TerrainField,
    TerrainMaterial, TerrainOctree, WorldCell, WorldPosition,
};

/// Awake loose bodies the world budgets for. Past it, freshly cut soft ground
/// is laid straight back down as spoil instead of taking flight.
pub const MAX_ACTIVE_CLUMPS: usize = 4_096;

/// How long soft spoil rests on the ground before it becomes ground, in seconds.
pub const SETTLE_SECONDS: f64 = 0.25;

/// A movable, world-owned quantity of material. Positions are global doubles.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MaterialClump {
    /// Stable identity, independent of physics row allocation.
    pub id: u64,
    /// Original terrain material.
    pub material: TerrainMaterial,
    /// Exact amount of material, in 1/510ths of what one solid cell holds.
    pub quanta: u32,
    /// Box-shaped convex fragment half extents, in metres.
    pub half_extents: DVec3,
    /// Global centre of mass.
    pub position: WorldPosition,
    /// Orientation of the fragment.
    pub rotation: DQuat,
    /// Linear velocity in m/s.
    pub linear_velocity: DVec3,
    /// Angular velocity in rad/s.
    pub angular_velocity: DVec3,
    /// Consecutive stable, terrain-supported seconds.
    pub settled_seconds: f64,
    /// Whether the body is asleep, while remaining collidable.
    pub sleeping: bool,
}

impl MaterialClump {
    /// Material mass in kg.
    pub fn mass_kg(&self) -> f64 {
        f64::from(self.quanta)
            * MATERIAL_QUANTUM_M3
            * BreakageResponse::for_material(self.material).density_kg_m3
    }

    /// Records support and motion. Machinery-only support cannot deposit soil.
    pub fn update_settling(&mut self, terrain_supported: bool, seconds: f64) {
        if !seconds.is_finite() || !(0.0..=1.0).contains(&seconds) {
            return;
        }
        if terrain_supported
            && self.linear_velocity.length() < 0.05
            && self.angular_velocity.length() < 0.1
        {
            self.settled_seconds += seconds;
        } else {
            self.settled_seconds = 0.0;
        }
    }

    /// Whether this soft clump has settled long enough to try depositing.
    pub fn can_deposit(&self) -> bool {
        BreakageResponse::for_material(self.material).deposits
            && self.quanta >= crate::CELL_QUANTA - u32::from(u8::MAX)
            && self.settled_seconds >= SETTLE_SECONDS
    }

    /// Validates data before a saved or prepared body reaches physics.
    pub fn is_valid(&self) -> bool {
        self.id > 0
            && self.quanta > 0
            && self.quanta <= 510 * 125
            && self.half_extents.is_finite()
            && self.half_extents.min_element() > 0.0
            && self.half_extents.max_element() <= 0.125 + 1e-9
            && self.position.is_inside_world()
            && self.position.0.is_finite()
            && ((self.half_extents.element_product() * 8.0)
                / (f64::from(self.quanta) * MATERIAL_QUANTUM_M3)
                - 1.0)
                .abs()
                < 1e-6
            && self.rotation.is_finite()
            && (self.rotation.length_squared() - 1.0).abs() < 1e-6
            && self.linear_velocity.is_finite()
            && self.angular_velocity.is_finite()
            && self.settled_seconds.is_finite()
            && self.settled_seconds >= 0.0
    }
}

/// Committed material ownership, persisted together with the matching terrain.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClumpCollection {
    /// Next unused stable identity.
    pub next_id: u64,
    /// Bodies in identity order.
    pub bodies: BTreeMap<u64, MaterialClump>,
}

impl Default for ClumpCollection {
    fn default() -> Self {
        Self {
            next_id: 1,
            bodies: BTreeMap::new(),
        }
    }
}

impl ClumpCollection {
    /// Available capacity, including bodies that a publication would wake.
    pub fn available(&self) -> usize {
        MAX_ACTIVE_CLUMPS.saturating_sub(self.bodies.values().filter(|body| !body.sleeping).count())
    }

    /// Checks identity and body data on load.
    pub fn is_valid(&self) -> bool {
        self.next_id > 0
            && self
                .bodies
                .iter()
                .all(|(&id, body)| id == body.id && id < self.next_id && body.is_valid())
    }

    /// Breaks the cells out of the terrain into clumps that leave along their
    /// throw. Stale, repeated or malformed sources change nothing. With
    /// `laid_down`, soft spoil is made ready to settle at once where it lies.
    pub fn extract(
        &mut self,
        terrain: &mut TerrainOctree,
        field: &TerrainField,
        sources: &[ExtractionCell],
        laid_down: bool,
    ) -> Option<TerrainEditOutcome> {
        if sources
            .iter()
            .any(|source| source.cell.y > i32::MAX - 5 || !source.sample.density.is_finite())
        {
            return None;
        }
        let mut bodies = Vec::new();
        let mut next_id = self.next_id;
        for cells in fragment_boxes(sources) {
            let id = next_id;
            next_id = id.checked_add(1)?;
            let mut body = fragment(id, &cells)?;
            if laid_down && BreakageResponse::for_material(body.material).deposits {
                body.linear_velocity = DVec3::ZERO;
                body.settled_seconds = SETTLE_SECONDS;
            }
            if !body.is_valid() {
                return None;
            }
            bodies.push(body);
        }
        let outcome = terrain.extract_cells(field, sources)?;
        self.next_id = next_id;
        self.bodies
            .extend(bodies.into_iter().map(|body| (body.id, body)));
        Some(outcome)
    }

    /// Takes ownership of ground that was pressed flat: each cell's material
    /// becomes spoil lying where the cell was. Whatever pressed it out pushes it
    /// aside, and it settles where it comes to rest: the berm along a rut.
    pub fn heave(&mut self, pressed_out: &[(WorldCell, TerrainMaterial, u32)]) {
        for &(cell, material, quanta) in pressed_out {
            let id = self.next_id;
            let Some(next_id) = id.checked_add(1) else {
                return;
            };
            self.next_id = next_id;
            self.bodies.insert(
                id,
                MaterialClump {
                    id,
                    material,
                    quanta,
                    half_extents: DVec3::splat(
                        (f64::from(quanta) * MATERIAL_QUANTUM_M3).cbrt() * 0.5,
                    ),
                    position: cell.centre(),
                    rotation: DQuat::IDENTITY,
                    linear_velocity: DVec3::ZERO,
                    angular_velocity: DVec3::ZERO,
                    settled_seconds: 0.0,
                    sleeping: false,
                },
            );
        }
    }

    /// Lays a settled soft clump back down as loose ground, running downhill
    /// from where it lies to its repose. What cannot be laid stays loose.
    pub fn settle(
        &mut self,
        terrain: &mut TerrainOctree,
        field: &TerrainField,
        id: u64,
        occupied: &mut dyn FnMut(WorldCell) -> bool,
        steps: &mut usize,
    ) -> Option<TerrainEditOutcome> {
        let body = self.bodies.get_mut(&id)?;
        if !body.can_deposit() {
            return None;
        }
        let start = body.position.cell().ok()?;
        let (outcome, left) =
            terrain.lay_spoil(field, start, body.material, body.quanta, occupied, steps);
        if left == body.quanta {
            return None;
        }
        if left == 0 {
            self.bodies.remove(&id);
        } else {
            body.half_extents *= (f64::from(left) / f64::from(body.quanta)).cbrt();
            body.quanta = left;
        }
        Some(outcome)
    }

    /// One exchange of material between the ground and the clumps: settled
    /// spoil is laid down, loose ground left too steep slides, and ground a tool
    /// has broken comes out. Returns the edits for the caller to publish.
    pub fn transfer(
        &mut self,
        terrain: &mut TerrainOctree,
        field: &TerrainField,
        breakage: &mut crate::BreakageAccumulator,
        slump: &mut crate::SpoilSlump,
        occupied: &mut dyn FnMut(WorldCell) -> bool,
        limits: TransferLimits,
    ) -> Vec<TerrainEditOutcome> {
        let mut outcomes = Vec::new();
        let mut steps = limits.steps;
        let settled = self
            .bodies
            .values()
            .filter(|body| body.can_deposit())
            .map(|body| body.id)
            .take(limits.clumps)
            .collect::<Vec<_>>();
        for id in settled {
            if steps == 0 {
                break;
            }
            if let Some(outcome) = self.settle(terrain, field, id, occupied, &mut steps) {
                for &cell in &outcome.laid_cells {
                    slump.disturb(cell);
                }
                outcomes.push(outcome);
            }
        }
        let sliding = slump.take_unstable(terrain, field, occupied, limits.columns);
        if let Some(outcome) = self.extract(terrain, field, &sliding, true) {
            for source in &sliding {
                slump.disturb(source.cell);
            }
            outcomes.push(outcome);
        }
        breakage.discard_stale(terrain, field);
        let broken = breakage.ready(terrain, field, limits.cells);
        if !broken.is_empty() {
            let laid_down = self.available() == 0;
            if let Some(outcome) = self.extract(terrain, field, &broken, laid_down) {
                for source in &broken {
                    slump.disturb(source.cell);
                }
                outcomes.push(outcome);
            }
            // Refused sources were stale; either way they are done with.
            breakage.committed(&broken);
        }
        outcomes
    }
}

/// How much one [`ClumpCollection::transfer`] may do.
#[derive(Clone, Copy, Debug)]
pub struct TransferLimits {
    /// Settled clumps laid down.
    pub clumps: usize,
    /// Columns of loose ground looked over for sliding.
    pub columns: usize,
    /// Cells of broken ground taken out.
    pub cells: usize,
    /// Columns spoil may run across in all, laying down.
    pub steps: usize,
}

impl Default for TransferLimits {
    fn default() -> Self {
        Self {
            clumps: 64,
            columns: 256,
            cells: 512,
            steps: 2_000,
        }
    }
}

// The clump a group of broken cells makes.
fn fragment(id: u64, cells: &[ExtractionCell]) -> Option<MaterialClump> {
    let first = cells.first()?;
    let mut minimum = first.cell.centre().0;
    let mut maximum = minimum;
    let mut quanta = 0_u32;
    let mut throw = DVec3::ZERO;
    for source in cells {
        throw += source.throw;
        minimum = minimum.min(source.cell.centre().0);
        maximum = maximum.max(source.cell.centre().0);
        quanta += u32::try_from(source.material_quanta()).ok()?;
    }
    let count = f64::from(u32::try_from(cells.len()).ok()?);
    let span = (maximum - minimum) / crate::TERRAIN_CELL_METERS + DVec3::ONE;
    let whole = quanta == crate::CELL_QUANTA * u32::try_from(cells.len()).ok()?;
    let (half_extents, centre) = if whole && (span.element_product() - count).abs() < 0.5 {
        // A whole cuboid keeps its shape.
        (
            (maximum - minimum + DVec3::splat(crate::TERRAIN_CELL_METERS)) * 0.5,
            (minimum + maximum) * 0.5,
        )
    } else {
        // Scattered cells gather into one clod of their volume, where they lay
        // on average.
        let volume = f64::from(quanta) * MATERIAL_QUANTUM_M3;
        let mean = cells
            .iter()
            .map(|source| source.cell.centre().0)
            .sum::<DVec3>()
            / count;
        (DVec3::splat(volume.cbrt() * 0.5), mean)
    };
    Some(MaterialClump {
        id,
        material: first.sample.material,
        quanta,
        half_extents,
        position: WorldPosition(centre),
        rotation: DQuat::IDENTITY,
        linear_velocity: throw / count,
        angular_velocity: DVec3::ZERO,
        settled_seconds: 0.0,
        sleeping: false,
    })
}

// Greedy cuboids contain only ready cells of one material, however each was
// packed: pressed ground never packs two cells alike, and a fragment per cell
// is a body per cell. They
// never bridge air, unbroken rock, or a different material to complete a hull.
fn fragment_boxes(sources: &[ExtractionCell]) -> Vec<Vec<ExtractionCell>> {
    let mut cells: BTreeMap<_, _> = sources
        .iter()
        .map(|source| (source.cell, *source))
        .collect();
    let mut groups = Vec::new();
    while let Some((&start, &first)) = cells.first_key_value() {
        let mut size = [1_i32; 3];
        for axis in 0..3 {
            while size[axis] < 5 {
                let mut next = size;
                next[axis] += 1;
                let contains = (0..next[2]).all(|z| {
                    (0..next[1]).all(|y| {
                        (0..next[0]).all(|x| {
                            cells
                                .get(&WorldCell::new(start.x + x, start.y + y, start.z + z))
                                .is_some_and(|source| {
                                    source.sample.material == first.sample.material
                                })
                        })
                    })
                });
                if !contains {
                    break;
                }
                size = next;
            }
        }
        let mut group = Vec::new();
        for z in 0..size[2] {
            for y in 0..size[1] {
                for x in 0..size[0] {
                    if let Some(source) =
                        cells.remove(&WorldCell::new(start.x + x, start.y + y, start.z + z))
                    {
                        group.push(source);
                    }
                }
            }
        }
        groups.push(group);
    }
    // Ground broken under a tool comes away in scattered cells that form no
    // cuboid. A body per cell is a body too many: loose cells of one material
    // in the same three-cell block make one clod.
    let mut clods = BTreeMap::<_, Vec<ExtractionCell>>::new();
    let mut whole = Vec::new();
    for group in groups {
        if group.len() >= 8 {
            whole.push(group);
            continue;
        }
        let first = group[0];
        let block = [first.cell.x, first.cell.y, first.cell.z].map(|axis| axis.div_euclid(3));
        clods
            .entry((block, first.sample.material.code()))
            .or_default()
            .extend(group);
    }
    whole.extend(clods.into_values());
    whole
}
