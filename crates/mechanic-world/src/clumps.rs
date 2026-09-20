//! Persistent loose material and prepared, conservative ownership changes.

use std::collections::BTreeMap;

use bevy_math::{DQuat, DVec3};
use serde::{Deserialize, Serialize};

use crate::{
    BreakageResponse, ExtractionCell, MATERIAL_QUANTUM_M3, TerrainEditOutcome, TerrainField,
    TerrainMaterial, TerrainOctree, WorldCell, WorldPosition,
};

/// Maximum awake loose bodies in the initial CPU implementation.
pub const MAX_ACTIVE_CLUMPS: usize = 256;

/// A movable, world-owned quantity of material. Positions are global doubles.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MaterialClump {
    /// Stable identity, independent of physics row allocation.
    pub id: u64,
    /// Original terrain material.
    pub material: TerrainMaterial,
    /// Exact material volume in 1/510-cell units.
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
        BreakageResponse::for_material(self.material).deposits && self.settled_seconds >= 1.0
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

    /// Checks identity, body data and the active budget on load/publication.
    pub fn is_valid(&self) -> bool {
        self.next_id > 0
            && self
                .bodies
                .iter()
                .all(|(&id, body)| id == body.id && id < self.next_id && body.is_valid())
            && self.bodies.values().filter(|body| !body.sleeping).count() <= MAX_ACTIVE_CLUMPS
    }

    /// Removes clumps lost inside solid ground and returns how many. Terrain
    /// collides from outside only, so a clump that gets under the surface falls
    /// without end. One whose centre and the space above its top are both solid
    /// is back in the ground it came from.
    pub fn absorb_buried(&mut self, terrain: &TerrainOctree, field: &TerrainField) -> usize {
        let before = self.bodies.len();
        self.bodies.retain(|_, body| {
            let above = body.half_extents.max_element() + crate::TERRAIN_CELL_METERS;
            ![DVec3::ZERO, DVec3::Y * above].into_iter().all(|offset| {
                WorldPosition(body.position.0 + offset)
                    .cell()
                    .is_ok_and(|cell| terrain.sample_cell(field, cell).is_solid())
            })
        });
        before - self.bodies.len()
    }

    /// Prepares a complete ownership change on copies. Failure leaves both
    /// inputs untouched. The caller installs all outputs at one boundary.
    pub fn prepare_extraction(
        &self,
        terrain: &TerrainOctree,
        field: &TerrainField,
        sources: &[ExtractionCell],
    ) -> Option<MaterialTransfer> {
        if !self.is_valid()
            || sources
                .iter()
                .any(|source| source.cell.y > i32::MAX - 5 || !source.sample.density.is_finite())
        {
            return None;
        }
        let mut terrain = terrain.clone();
        let mut clumps = self.clone();
        let groups = fragment_boxes(sources);
        if groups.len() > clumps.available() || !crate::breakage::unique_sources(sources) {
            return None;
        }
        let outcome = terrain.extract_cells(field, sources)?;
        for cells in groups {
            let id = clumps.next_id;
            clumps.next_id = id.checked_add(1)?;
            let first = cells.first()?;
            let mut minimum = first.cell.centre().0;
            let mut maximum = minimum;
            let mut quanta = 0_u32;
            let mut throw = DVec3::ZERO;
            for source in &cells {
                throw += source.throw;
                minimum = minimum.min(source.cell.centre().0);
                maximum = maximum.max(source.cell.centre().0);
                quanta += u32::try_from(source.material_quanta()).ok()?;
            }
            let mut half_extents =
                (maximum - minimum + DVec3::splat(crate::TERRAIN_CELL_METERS)) * 0.5;
            let mut centre = (minimum + maximum) * 0.5;
            let shrink = half_extents.y * f64::from(first.sample.compaction) / 510.0;
            half_extents.y -= shrink;
            centre.y -= shrink;
            let body = MaterialClump {
                id,
                material: first.sample.material,
                quanta,
                half_extents,
                position: WorldPosition(centre),
                rotation: DQuat::IDENTITY,
                linear_velocity: throw
                    / f64::from(u32::try_from(cells.len()).unwrap_or(u32::MAX).max(1)),
                angular_velocity: DVec3::ZERO,
                settled_seconds: 0.0,
                sleeping: false,
            };
            if !body.is_valid() {
                return None;
            }
            clumps.bodies.insert(id, body);
        }
        Some(MaterialTransfer {
            terrain,
            clumps,
            outcome,
        })
    }

    /// Attempts supported whole-cell deposition; hard materials remain bodies.
    pub fn prepare_deposition(
        &self,
        terrain: &TerrainOctree,
        field: &TerrainField,
        id: u64,
        targets: &[WorldCell],
    ) -> Option<MaterialTransfer> {
        let body = self.bodies.get(&id)?;
        if !body.can_deposit() {
            return None;
        }
        let mut terrain = terrain.clone();
        let (outcome, remaining) =
            terrain.deposit_material(field, targets, body.material, u64::from(body.quanta));
        if remaining == u64::from(body.quanta) {
            return None;
        }
        let mut clumps = self.clone();
        if remaining == 0 {
            clumps.bodies.remove(&id);
        } else {
            let body = clumps.bodies.get_mut(&id)?;
            let remaining = u32::try_from(remaining).ok()?;
            let scale = (f64::from(remaining) / f64::from(body.quanta)).cbrt();
            body.half_extents *= scale;
            body.quanta = remaining;
            body.settled_seconds = 0.0;
        }
        Some(MaterialTransfer {
            terrain,
            clumps,
            outcome,
        })
    }
}

/// Prepared terrain and bodies; neither is authoritative until jointly installed.
pub struct MaterialTransfer {
    /// Replacement terrain.
    pub terrain: TerrainOctree,
    /// Replacement material ownership.
    pub clumps: ClumpCollection,
    /// Dirty terrain footprint for meshing and foundation invalidation.
    pub outcome: TerrainEditOutcome,
}

// Greedy cuboids contain only ready cells of one material and compaction. They
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
                                        && source.sample.compaction == first.sample.compaction
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
    groups
}
