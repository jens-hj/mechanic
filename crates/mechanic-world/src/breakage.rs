//! Local mechanical breakage and conservative terrain/material transfers.
//!
//! These are game parameters, not engineering material measurements. Weight
//! resting on the ground never mines: a loaded contact must deliver mechanical
//! work, or be driven sideways into soft ground hard enough to crush it.

use std::collections::{BTreeMap, BTreeSet};

use bevy_math::DVec3;

use crate::{
    TerrainField, TerrainMaterial, TerrainOctree, TerrainSample, WorldCell, WorldPosition,
};

/// Material volume quantum: one 510th of a terrain cell. A compaction step
/// removes one quantum, matching the existing half-cell/255 displacement law.
pub const MATERIAL_QUANTUM_M3: f64 = crate::REMOVED_CELL_CUBIC_METERS / 510.0;
/// How far past its strength soft ground is pushed sideways before it crushes.
/// A parked machine leaning on a slope stays well below this.
const CRUSH_OVERLOAD: f64 = 4.0;
/// Share of the tool's surface speed broken material leaves with.
const THROW_SHARE: f64 = 0.5;
/// Speed broken material is pushed off the surface it left, in m/s.
const THROW_LIFT_M_S: f64 = 0.5;
/// Fastest broken material leaves, in m/s.
const MAX_THROW_M_S: f64 = 4.0;
/// Maximum number of independently accumulated loaded cells.
const MAX_PENDING_CELLS: usize = 16_384;

/// Game-tuned resistance and handling properties of terrain material.
#[derive(Clone, Copy, Debug)]
pub struct BreakageResponse {
    /// Minimum applied stress before work can damage the material, in Pa.
    pub stress_pa: f64,
    /// Work required to extract one cubic metre, in J/m³.
    pub work_j_m3: f64,
    /// Mass per material volume, in kg/m³.
    pub density_kg_m3: f64,
    /// Whether resting fragments can return to low-compaction terrain.
    pub deposits: bool,
}

impl BreakageResponse {
    /// Initial ordering for sand, soil, weaker minerals, rock and iron ore.
    pub const fn for_material(material: TerrainMaterial) -> Self {
        let (stress_pa, work_j_m3, density_kg_m3, deposits) = match material {
            TerrainMaterial::Sand => (12_000.0, 8_000.0, 1_600.0, true),
            TerrainMaterial::SurfaceCover => (20_000.0, 16_000.0, 1_200.0, true),
            TerrainMaterial::Soil => (35_000.0, 32_000.0, 1_700.0, true),
            TerrainMaterial::Graphite => (600_000.0, 800_000.0, 2_200.0, false),
            TerrainMaterial::Rock => (2_000_000.0, 4_000_000.0, 2_600.0, false),
            TerrainMaterial::Iron => (6_000_000.0, 12_000_000.0, 4_000.0, false),
        };
        Self {
            stress_pa,
            work_j_m3,
            density_kg_m3,
            deposits,
        }
    }
}

/// A measured contact's physical work and stress in global coordinates.
#[derive(Clone, Copy, Debug)]
pub struct BreakagePatch {
    /// Centre of the loaded footprint.
    pub centre: WorldPosition,
    /// Outward terrain normal.
    pub normal: DVec3,
    /// Ground the load presses on.
    pub footprint: crate::LoadFootprint,
    /// Applied normal/tangential resultant stress in Pa.
    pub stress_pa: f64,
    /// Dissipated mechanical work, excluding position stabilization, in J.
    pub work_j: f64,
    /// Horizontal share of the normal stress, in Pa: how hard the body is
    /// driven sideways into the ground rather than resting on it.
    pub crush_pa: f64,
    /// Duration of the load, in seconds.
    pub seconds: f64,
    /// Velocity of the tool's surface over the ground, in m/s. Broken
    /// material leaves along it.
    pub throw: DVec3,
}

impl BreakagePatch {
    fn valid(self) -> bool {
        self.centre.is_inside_world()
            && self.centre.0.is_finite()
            && self.normal.is_finite()
            && (self.normal.length_squared() - 1.0).abs() < 0.01
            && self.footprint.is_valid(self.normal)
            && self.stress_pa.is_finite()
            && self.stress_pa >= 0.0
            && self.work_j.is_finite()
            && self.work_j >= 0.0
            && self.crush_pa.is_finite()
            && self.crush_pa >= 0.0
            && self.seconds.is_finite()
            && (0.0..=1.0).contains(&self.seconds)
            && self.throw.is_finite()
    }

    // Work soft ground of this strength absorbs while it is crushed aside, per
    // cell under the footprint. The soil law, turned sideways: a blade, tooth or
    // bumper driven into a bank advances through it instead of stalling.
    fn crush_work_j(self, material: TerrainMaterial, strength_pa: f64) -> f64 {
        let response = BreakageResponse::for_material(material);
        let limit = CRUSH_OVERLOAD * strength_pa;
        if !response.deposits || self.crush_pa <= limit {
            return 0.0;
        }
        let rate = f64::from(crate::SoilResponse::for_material(material).yield_rate_m_s);
        let depth = (rate * (self.crush_pa / limit - 1.0) * self.seconds)
            .min(f64::from(crate::soil::MAX_SOIL_DEPTH_METRES));
        depth * crate::TERRAIN_CELL_METERS.powi(2) * response.work_j_m3
    }
}

/// One source cell, including the exact sample that authorized its removal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExtractionCell {
    /// Global source cell.
    pub cell: WorldCell,
    /// Expected source value. Changed sources invalidate a transfer.
    pub sample: TerrainSample,
    /// Velocity the broken material leaves with, in m/s.
    pub throw: DVec3,
}

/// Material in one solid cell, in quanta. Pressing a cell packs it without
/// taking anything away, so however compacted, a cell holds this much until it
/// is pressed flat and leaves as spoil.
pub const CELL_QUANTA: u32 = 510;

impl ExtractionCell {
    /// Material the cell holds.
    pub const fn material_quanta(self) -> u64 {
        CELL_QUANTA as u64
    }
}

#[derive(Clone, Copy)]
struct Damage {
    sample: TerrainSample,
    work_j: f64,
    normal: DVec3,
    // Tool velocity summed by the work it delivered, and that work uncapped.
    throw: DVec3,
    thrown_j: f64,
}

impl Damage {
    fn new(sample: TerrainSample, normal: DVec3) -> Self {
        Self {
            sample,
            work_j: 0.0,
            normal,
            throw: DVec3::ZERO,
            thrown_j: 0.0,
        }
    }

    // Along the tool's motion, with a push off the surface so spoil clears the
    // cut. Bounded: fast fragments cost the solver more than they add.
    fn launch(&self) -> DVec3 {
        let along = if self.thrown_j > 0.0 {
            self.throw / self.thrown_j
        } else {
            DVec3::ZERO
        };
        (along * THROW_SHARE + self.normal * THROW_LIFT_M_S).clamp_length_max(MAX_THROW_M_S)
    }
}

/// Bounded work accumulation; ready cells retain their work until a transfer
/// commits, so a full body budget cannot consume terrain or lose material.
#[derive(Default)]
pub struct BreakageAccumulator {
    pending: BTreeMap<WorldCell, Damage>,
}

impl BreakageAccumulator {
    /// Drops damage on sources changed by another edit.
    pub fn discard_stale(&mut self, terrain: &TerrainOctree, field: &TerrainField) {
        self.pending.retain(|cell, damage| {
            terrain.sample_cell(field, *cell) == damage.sample
                && exposed(terrain, field, *cell, damage.normal)
        });
    }

    /// Distributes a contact's work across exposed cells in its footprint.
    /// Invalid measurements are ignored without promoting terrain.
    pub fn accumulate(
        &mut self,
        terrain: &TerrainOctree,
        field: &TerrainField,
        patch: BreakagePatch,
    ) {
        if !patch.valid() || (patch.work_j == 0.0 && patch.crush_pa == 0.0) {
            return;
        }
        // Cells from one cell inside the surface to half a cell outside it.
        let (below, above) = (crate::TERRAIN_CELL_METERS, 0.025);
        let margin = DVec3::splat(crate::TERRAIN_CELL_METERS * 0.5);
        let [low, high] = patch.footprint.bounds(patch.normal, below, above);
        let (Ok(min), Ok(max)) = (
            WorldPosition(patch.centre.0 + low - margin).cell(),
            WorldPosition(patch.centre.0 + high + margin).cell(),
        ) else {
            return;
        };
        let mut loaded = Vec::new();
        for z in min.z..=max.z {
            for y in min.y..=max.y {
                for x in min.x..=max.x {
                    let cell = WorldCell::new(x, y, z);
                    if !cell.is_editable() {
                        continue;
                    }
                    let delta = cell.centre().0 - patch.centre.0;
                    let depth = delta.dot(patch.normal);
                    if !(-below..=above).contains(&depth)
                        || !patch.footprint.reaches(
                            patch.normal,
                            delta,
                            crate::TERRAIN_CELL_METERS * 0.5,
                        )
                    {
                        continue;
                    }
                    let sample = terrain.sample_cell(field, cell);
                    if !sample.is_solid() {
                        continue;
                    }
                    let Ok(outside) =
                        WorldPosition(cell.centre().0 + patch.normal * crate::TERRAIN_CELL_METERS)
                            .cell()
                    else {
                        continue;
                    };
                    if terrain.sample_cell(field, outside).is_solid() {
                        continue;
                    }
                    loaded.push((cell, sample));
                }
            }
        }
        let share =
            patch.work_j / f64::from(u32::try_from(loaded.len()).unwrap_or(u32::MAX).max(1));
        for (cell, sample) in loaded {
            let response = BreakageResponse::for_material(sample.material);
            let hardening = if response.deposits {
                1.0 + f64::from(sample.compaction) / 255.0
            } else {
                1.0
            };
            let strength = response.stress_pa * hardening;
            if patch.stress_pa < strength {
                continue;
            }
            let share = share + patch.crush_work_j(sample.material, strength);
            if share == 0.0 {
                continue;
            }
            if self.pending.len() >= MAX_PENDING_CELLS && !self.pending.contains_key(&cell) {
                continue;
            }
            let damage = self
                .pending
                .entry(cell)
                .or_insert(Damage::new(sample, patch.normal));
            if damage.sample != sample {
                *damage = Damage::new(sample, patch.normal);
            }
            let required = extraction_work(sample);
            damage.work_j = (damage.work_j + share).min(required);
            damage.throw += patch.throw * share;
            damage.thrown_j += share;
        }
    }

    /// Ready source cells, ordered deterministically and bounded by capacity.
    pub fn ready(
        &self,
        terrain: &TerrainOctree,
        field: &TerrainField,
        capacity: usize,
    ) -> Vec<ExtractionCell> {
        self.pending
            .iter()
            .filter(|&(&cell, damage)| {
                terrain.sample_cell(field, cell) == damage.sample
                    && exposed(terrain, field, cell, damage.normal)
                    && damage.work_j >= extraction_work(damage.sample)
            })
            .map(|(&cell, damage)| ExtractionCell {
                cell,
                sample: damage.sample,
                throw: damage.launch(),
            })
            .take(capacity)
            .collect()
    }

    /// Forgets only sources belonging to a successfully committed transfer.
    pub fn committed(&mut self, cells: &[ExtractionCell]) {
        for source in cells {
            self.pending.remove(&source.cell);
        }
    }
}

fn extraction_work(sample: TerrainSample) -> f64 {
    // A cell has at most 510 quanta, exactly representable as f64.
    f64::from(510 - u32::from(sample.compaction))
        * MATERIAL_QUANTUM_M3
        * BreakageResponse::for_material(sample.material).work_j_m3
}

fn exposed(terrain: &TerrainOctree, field: &TerrainField, cell: WorldCell, normal: DVec3) -> bool {
    WorldPosition(cell.centre().0 + normal * crate::TERRAIN_CELL_METERS)
        .cell()
        .is_ok_and(|outside| !terrain.sample_cell(field, outside).is_solid())
}

/// Rejects duplicate source cells before an all-or-nothing extraction.
pub(crate) fn unique_sources(cells: &[ExtractionCell]) -> bool {
    let unique: BTreeSet<_> = cells.iter().map(|source| source.cell).collect();
    unique.len() == cells.len()
}
