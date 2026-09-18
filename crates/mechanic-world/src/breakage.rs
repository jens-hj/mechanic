//! Local mechanical breakage and conservative terrain/material transfers.
//!
//! These are game parameters, not engineering material measurements. Pressure
//! alone never mines: a loaded contact must also deliver mechanical work.

use std::collections::{BTreeMap, BTreeSet};

use bevy_math::DVec3;

use crate::{
    TerrainField, TerrainMaterial, TerrainOctree, TerrainSample, WorldCell, WorldPosition,
};

/// Material volume quantum: one 510th of a terrain cell. A compaction step
/// removes one quantum, matching the existing half-cell/255 displacement law.
pub const MATERIAL_QUANTUM_M3: f64 = crate::REMOVED_CELL_CUBIC_METERS / 510.0;
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
    /// Circular footprint radius in metres.
    pub radius: f64,
    /// Applied normal/tangential resultant stress in Pa.
    pub stress_pa: f64,
    /// Dissipated mechanical work, excluding position stabilization, in J.
    pub work_j: f64,
}

impl BreakagePatch {
    fn valid(self) -> bool {
        self.centre.is_inside_world()
            && self.centre.0.is_finite()
            && self.normal.is_finite()
            && (self.normal.length_squared() - 1.0).abs() < 0.01
            && self.radius.is_finite()
            && (0.025..=0.3).contains(&self.radius)
            && self.stress_pa.is_finite()
            && self.stress_pa >= 0.0
            && self.work_j.is_finite()
            && self.work_j >= 0.0
    }
}

/// One source cell, including the exact sample that authorized its removal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExtractionCell {
    /// Global source cell.
    pub cell: WorldCell,
    /// Expected source value. Changed sources invalidate a transfer.
    pub sample: TerrainSample,
}

impl ExtractionCell {
    /// Remaining material after prior plastic compression.
    pub const fn material_quanta(self) -> u64 {
        510 - self.sample.compaction as u64
    }
}

#[derive(Clone, Copy)]
struct Damage {
    sample: TerrainSample,
    work_j: f64,
    normal: DVec3,
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
        if !patch.valid() || patch.work_j == 0.0 {
            return;
        }
        let radius = DVec3::splat(patch.radius + crate::TERRAIN_CELL_METERS);
        let (Ok(min), Ok(max)) = (
            WorldPosition(patch.centre.0 - radius).cell(),
            WorldPosition(patch.centre.0 + radius).cell(),
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
                    if !(-crate::TERRAIN_CELL_METERS..=0.025).contains(&depth)
                        || (delta - depth * patch.normal).length_squared() > patch.radius.powi(2)
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
            if patch.stress_pa < response.stress_pa * hardening {
                continue;
            }
            if self.pending.len() >= MAX_PENDING_CELLS && !self.pending.contains_key(&cell) {
                continue;
            }
            let damage = self.pending.entry(cell).or_insert(Damage {
                sample,
                work_j: 0.0,
                normal: patch.normal,
            });
            if damage.sample != sample {
                *damage = Damage {
                    sample,
                    work_j: 0.0,
                    normal: patch.normal,
                };
            }
            let required = extraction_work(ExtractionCell { cell, sample });
            damage.work_j = (damage.work_j + share).min(required);
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
            .filter_map(|(&cell, damage)| {
                let source = ExtractionCell {
                    cell,
                    sample: damage.sample,
                };
                (terrain.sample_cell(field, cell) == damage.sample
                    && exposed(terrain, field, cell, damage.normal)
                    && damage.work_j >= extraction_work(source))
                .then_some(source)
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

fn extraction_work(source: ExtractionCell) -> f64 {
    // A cell has at most 510 quanta, exactly representable as f64.
    f64::from(u32::try_from(source.material_quanta()).unwrap_or(510))
        * MATERIAL_QUANTUM_M3
        * BreakageResponse::for_material(source.sample.material).work_j_m3
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
