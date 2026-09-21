//! Game-tuned plastic compaction, not measured soil data. Pressing packs a
//! cell without taking material from it; a cell pressed flat is squeezed out,
//! and its material is owed to the world as spoil.
#![expect(clippy::cast_possible_truncation)]

use std::collections::BTreeMap;

use bevy_math::DVec3;

use crate::{
    TERRAIN_CELL_METERS, TerrainEditError, TerrainField, TerrainMaterial, TerrainOctree,
    TerrainSample, WorldCell, WorldPosition,
};

/// Smallest persisted plastic displacement.
pub const COMPACTION_STEP_METRES: f32 = TERRAIN_CELL_METERS as f32 * 0.5 / 255.0;
/// Maximum density displacement delivered in one commit.
pub const MAX_SOIL_DEPTH_METRES: f32 = TERRAIN_CELL_METERS as f32 * 0.25;
/// Minimum accumulated displacement before requesting a remesh.
pub const SOIL_COMMIT_DEPTH_METRES: f32 = 0.002;

/// Least upward component of a terrain normal that counts as ground. Steeper
/// faces are walls: they carry no weight and never compact.
pub const GROUND_NORMAL_MIN_Y: f64 = 0.25;

/// Material's plastic bearing response.
#[derive(Clone, Copy, Debug)]
pub struct SoilResponse {
    /// Undisturbed bearing capacity in pascals.
    pub bearing_capacity_pa: f32,
    /// Additional capacity multiplier at full compaction.
    pub hardening: f32,
    /// Depth per second at twice the current capacity.
    pub yield_rate_m_s: f32,
}

impl SoilResponse {
    /// Game-tuned material parameters; minerals remain rigid.
    pub const fn for_material(material: TerrainMaterial) -> Self {
        let (bearing_capacity_pa, hardening, yield_rate_m_s) = match material {
            TerrainMaterial::SurfaceCover => (15_000.0, 12.0, 0.015),
            TerrainMaterial::Sand => (25_000.0, 10.0, 0.012),
            TerrainMaterial::Soil => (40_000.0, 12.0, 0.010),
            TerrainMaterial::Rock | TerrainMaterial::Iron | TerrainMaterial::Graphite => {
                (f32::INFINITY, 0.0, 0.0)
            }
        };
        Self {
            bearing_capacity_pa,
            hardening,
            yield_rate_m_s,
        }
    }

    /// Pressure ground compacted this far carries before it yields, in pascals.
    pub fn capacity_pa(self, compaction: u8) -> f32 {
        self.bearing_capacity_pa * (1.0 + self.hardening * f32::from(compaction) / 255.0)
    }

    /// Plastic displacement of a cell under a constant pressure, bounded per
    /// delivery. Loose ground carries less.
    pub fn depth(self, sample: TerrainSample, pressure_pa: f32, seconds: f32) -> f32 {
        if !pressure_pa.is_finite() || !seconds.is_finite() || pressure_pa <= 0.0 || seconds <= 0.0
        {
            return 0.0;
        }
        let capacity = self.capacity_pa(sample.compaction) * loose_strength(sample.looseness);
        (self.yield_rate_m_s * (pressure_pa / capacity - 1.0).max(0.0) * seconds)
            .min(MAX_SOIL_DEPTH_METRES)
    }
}

/// How much of undisturbed ground's strength ground this loose keeps.
pub fn loose_strength(looseness: u8) -> f32 {
    // A cell holds 510 quanta, exactly representable.
    let held = 1.0 - f32::from(looseness) / 510.0;
    held * held
}

/// Pressure on an upward-facing terrain footprint in global coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SoilPatch {
    /// Global contact centre.
    pub centre: WorldPosition,
    /// Unit outward terrain normal. Walls and ceilings do not compact downward.
    pub normal: DVec3,
    /// Ground the load presses on.
    pub footprint: crate::LoadFootprint,
    /// Average normal pressure over the patch, in pascals.
    pub pressure_pa: f32,
    /// Duration of the applied pressure.
    pub seconds: f32,
}

impl SoilPatch {
    pub(crate) fn validate(self) -> Result<(), TerrainEditError> {
        if !self.centre.0.is_finite()
            || !self.normal.is_finite()
            || (self.normal.length_squared() - 1.0).abs() > 0.01
            || !self.footprint.is_valid(self.normal)
            || !self.pressure_pa.is_finite()
            || self.pressure_pa < 0.0
            || !self.seconds.is_finite()
            || !(0.0..=1.0).contains(&self.seconds)
        {
            return Err(TerrainEditError::InvalidSoilPatch);
        }
        let reach = self.footprint.reach();
        if self.centre.0.x.abs() + reach >= crate::WORLD_HALF_EXTENT_METERS
            || self.centre.0.z.abs() + reach >= crate::WORLD_HALF_EXTENT_METERS
        {
            return Err(TerrainEditError::UnbreakableBoundary);
        }
        Ok(())
    }
}

/// Accumulated displacement for one cell, tied to the sample that received it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SoilCompression {
    /// Target cell.
    pub cell: WorldCell,
    /// Sample when the load was accumulated; stale loads are discarded.
    pub sample: TerrainSample,
    /// Undelivered displacement in metres.
    pub depth: f32,
}

/// Bounded, sub-threshold soil memory shared by app and headless replay.
#[derive(Default)]
pub struct SoilAccumulator {
    pending: BTreeMap<WorldCell, SoilCompression>,
}

impl SoilAccumulator {
    /// Adds a tick's pressure without promoting bricks or requesting meshes.
    ///
    /// # Errors
    /// Rejects malformed patches before changing pending state.
    pub fn accumulate(
        &mut self,
        terrain: &TerrainOctree,
        field: &TerrainField,
        patch: SoilPatch,
    ) -> Result<(), TerrainEditError> {
        let cells = terrain.soil_compressions(field, patch)?;
        for compression in cells {
            if self.pending.len() >= 16_384 && !self.pending.contains_key(&compression.cell) {
                self.pending.clear();
            }
            let pending = self
                .pending
                .entry(compression.cell)
                .or_insert(SoilCompression {
                    depth: 0.0,
                    ..compression
                });
            if pending.sample != compression.sample {
                *pending = SoilCompression {
                    depth: 0.0,
                    ..compression
                };
            }
            pending.depth = (pending.depth + compression.depth).min(MAX_SOIL_DEPTH_METRES);
        }
        Ok(())
    }

    /// Takes cells ready for a commit. Call at most every six simulation ticks.
    pub fn take_ready(&mut self) -> Vec<SoilCompression> {
        let mut ready = Vec::new();
        self.pending.retain(|_, pending| {
            if pending.depth >= SOIL_COMMIT_DEPTH_METRES {
                ready.push(*pending);
                false
            } else {
                true
            }
        });
        ready
    }

    /// Drops pending pressure when the edit worker cannot keep up or the world changes.
    pub fn clear(&mut self) {
        self.pending.clear();
    }
}
