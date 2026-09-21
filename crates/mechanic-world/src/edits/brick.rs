//! One promoted 5 cm brick of explicit samples, and its byte encoding.

use crate::soil::{COMPACTION_STEP_METRES, MAX_SOIL_DEPTH_METRES};
use crate::{
    BRICK_EDGE_CELLS, BrickCoord, TERRAIN_CELL_METERS, TerrainField, TerrainMaterial,
    TerrainSample, WorldCell,
};
use bevy_math::IVec3;
use thiserror::Error;

pub(super) const BRICK_CELL_COUNT: usize = 32 * 32 * 32;

pub(super) const EMPTY_DENSITY: f32 = -0.5 * TERRAIN_CELL_METERS as f32;

pub(super) const BRICK_MAGIC: [u8; 4] = *b"MECB";

pub(super) const BRICK_FORMAT_VERSION: u16 = 4;

/// Fully promoted 32³-cell brick and its density acceleration bounds.
#[derive(Clone, Debug, PartialEq)]
pub struct TerrainBrick {
    pub(super) coordinate: BrickCoord,
    pub(super) cells: Vec<TerrainSample>,
    pub(super) minimum_density: f32,
    pub(super) maximum_density: f32,
    pub(super) revision: u64,
}

impl TerrainBrick {
    /// Brick coordinate.
    pub const fn coordinate(&self) -> BrickCoord {
        self.coordinate
    }

    /// Minimum density in the brick, used to skip known-empty hierarchy nodes.
    pub const fn minimum_density(&self) -> f32 {
        self.minimum_density
    }

    /// Maximum density in the brick, used to skip known-solid hierarchy nodes.
    pub const fn maximum_density(&self) -> f32 {
        self.maximum_density
    }

    /// Latest terrain revision represented by this leaf.
    pub const fn revision(&self) -> u64 {
        self.revision
    }

    /// Returns the promoted cell at a zero-based local coordinate.
    pub fn sample(&self, local: IVec3) -> Option<TerrainSample> {
        local_index(local).map(|index| self.cells[index])
    }

    pub(super) fn promote(field: &TerrainField, coordinate: BrickCoord) -> Self {
        let minimum = coordinate.minimum_cell();
        let mut columns = Vec::with_capacity(usize::try_from(BRICK_EDGE_CELLS.pow(2)).unwrap());
        for z in 0..BRICK_EDGE_CELLS {
            for x in 0..BRICK_EDGE_CELLS {
                let position = WorldCell::new(minimum.x + x, minimum.y, minimum.z + z).centre();
                columns.push(field.sample_column(position.0.x, position.0.z));
            }
        }
        let mut cells = Vec::with_capacity(BRICK_CELL_COUNT);
        let mut minimum_density = f32::INFINITY;
        let mut maximum_density = f32::NEG_INFINITY;
        for z in 0..BRICK_EDGE_CELLS {
            for y in 0..BRICK_EDGE_CELLS {
                for x in 0..BRICK_EDGE_CELLS {
                    let cell = WorldCell::new(minimum.x + x, minimum.y + y, minimum.z + z);
                    let column = columns[usize::try_from(x + z * BRICK_EDGE_CELLS)
                        .expect("local index is positive")];
                    let sample = field.sample_cell_in_column(cell, column);
                    minimum_density = minimum_density.min(sample.density);
                    maximum_density = maximum_density.max(sample.density);
                    cells.push(sample);
                }
            }
        }
        Self {
            coordinate,
            cells,
            minimum_density,
            maximum_density,
            revision: 0,
        }
    }

    #[expect(
        clippy::cast_sign_loss,
        reason = "depth is finite and positive before quantisation"
    )]
    pub(super) fn compress(&mut self, local: IVec3, depth: f32) -> Option<f32> {
        let index = local_index(local)?;
        let sample = &mut self.cells[index];
        let before = sample.density;
        let bounded = before;
        let depth = depth
            .min(MAX_SOIL_DEPTH_METRES)
            .min(bounded - EMPTY_DENSITY);
        if !depth.is_finite() || depth < COMPACTION_STEP_METRES {
            return None;
        }
        let steps = (depth / COMPACTION_STEP_METRES).floor();
        let depth = steps * COMPACTION_STEP_METRES;
        sample.density = (bounded - depth).max(EMPTY_DENSITY);
        sample.compaction = sample.compaction.saturating_add(steps as u8);
        self.minimum_density = self.minimum_density.min(sample.density);
        if before >= self.maximum_density {
            self.maximum_density = self
                .cells
                .iter()
                .map(|cell| cell.density)
                .fold(f32::NEG_INFINITY, f32::max);
        }
        Some(depth)
    }

    pub(super) fn set_empty(&mut self, local: IVec3) -> Option<TerrainMaterial> {
        let index = local_index(local)?;
        let sample = &mut self.cells[index];
        if !sample.is_solid() {
            return None;
        }
        let removed = sample.material;
        let removed_density = sample.density;
        sample.density = EMPTY_DENSITY;
        sample.compaction = 0;
        sample.looseness = 0;
        self.minimum_density = self.minimum_density.min(EMPTY_DENSITY);
        if removed_density >= self.maximum_density {
            self.maximum_density = self
                .cells
                .iter()
                .map(|cell| cell.density)
                .fold(f32::NEG_INFINITY, f32::max);
        }
        Some(removed)
    }

    pub(super) fn set_solid(
        &mut self,
        local: IVec3,
        material: TerrainMaterial,
        density: f32,
        looseness: u8,
    ) -> bool {
        let Some(index) = local_index(local) else {
            return false;
        };
        let sample = &mut self.cells[index];
        if sample.is_solid() {
            return false;
        }
        let previous_density = sample.density;
        *sample = TerrainSample {
            density,
            material,
            compaction: 0,
            looseness,
        };
        self.maximum_density = self.maximum_density.max(density);
        if previous_density <= self.minimum_density {
            self.minimum_density = self
                .cells
                .iter()
                .map(|cell| cell.density)
                .fold(f32::INFINITY, f32::min);
        }
        true
    }
}

/// Corrupt or unsupported edited-brick payload.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum BrickDecodeError {
    /// Header or version is not recognized.
    #[error("edited terrain brick has an unsupported header or version")]
    UnsupportedHeader,
    /// Payload ended before a complete record.
    #[error("edited terrain brick is truncated")]
    Truncated,
    /// Material byte is not a v1 material.
    #[error("edited terrain brick contains unknown material code {0}")]
    UnknownMaterial(u8),
    /// Runs did not expand to exactly 32³ cells.
    #[error("edited terrain brick expands to {0} cells instead of {BRICK_CELL_COUNT}")]
    InvalidCellCount(usize),
}

/// Encodes one promoted brick using versioned run-length compression.
pub fn encode_brick(brick: &TerrainBrick) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(BRICK_CELL_COUNT / 2);
    bytes.extend_from_slice(&BRICK_MAGIC);
    bytes.extend_from_slice(&BRICK_FORMAT_VERSION.to_le_bytes());
    for coordinate in [brick.coordinate.x, brick.coordinate.y, brick.coordinate.z] {
        bytes.extend_from_slice(&coordinate.to_le_bytes());
    }
    bytes.extend_from_slice(&brick.revision.to_le_bytes());
    let mut index = 0;
    while index < brick.cells.len() {
        let sample = brick.cells[index];
        let mut run = 1_usize;
        while index + run < brick.cells.len()
            && brick.cells[index + run] == sample
            && run < usize::from(u16::MAX)
        {
            run += 1;
        }
        bytes.extend_from_slice(&u16::try_from(run).unwrap_or(u16::MAX).to_le_bytes());
        bytes.extend_from_slice(&sample.density.to_bits().to_le_bytes());
        bytes.push(sample.material.code());
        bytes.push(sample.compaction);
        bytes.push(sample.looseness);
        index += run;
    }
    bytes
}

/// Decodes a saved edited brick without falling back to procedural data.
///
/// # Errors
///
/// Returns a precise corruption error; callers must preserve the source file.
pub fn decode_brick(bytes: &[u8]) -> Result<TerrainBrick, BrickDecodeError> {
    if bytes.get(..4) != Some(&BRICK_MAGIC) || read_u16(bytes, 4)? != BRICK_FORMAT_VERSION {
        return Err(BrickDecodeError::UnsupportedHeader);
    }
    let coordinate = BrickCoord::new(
        read_i32(bytes, 6)?,
        read_i32(bytes, 10)?,
        read_i32(bytes, 14)?,
    );
    let mut cells = Vec::with_capacity(BRICK_CELL_COUNT);
    let revision = read_u64(bytes, 18)?;
    let mut cursor = 26;
    while cursor < bytes.len() {
        let run = usize::from(read_u16(bytes, cursor)?);
        let density = f32::from_bits(read_u32(bytes, cursor + 2)?);
        let compaction = *bytes.get(cursor + 7).ok_or(BrickDecodeError::Truncated)?;
        let looseness = *bytes.get(cursor + 8).ok_or(BrickDecodeError::Truncated)?;
        let code = *bytes.get(cursor + 6).ok_or(BrickDecodeError::Truncated)?;
        let material =
            TerrainMaterial::from_code(code).ok_or(BrickDecodeError::UnknownMaterial(code))?;
        if run == 0 || cells.len() + run > BRICK_CELL_COUNT {
            return Err(BrickDecodeError::InvalidCellCount(cells.len() + run));
        }
        cells.extend(std::iter::repeat_n(
            TerrainSample {
                density,
                material,
                compaction,
                looseness,
            },
            run,
        ));
        cursor += 9;
    }
    if cells.len() != BRICK_CELL_COUNT {
        return Err(BrickDecodeError::InvalidCellCount(cells.len()));
    }
    let minimum_density = cells
        .iter()
        .map(|cell| cell.density)
        .fold(f32::INFINITY, f32::min);
    let maximum_density = cells
        .iter()
        .map(|cell| cell.density)
        .fold(f32::NEG_INFINITY, f32::max);
    Ok(TerrainBrick {
        coordinate,
        cells,
        minimum_density,
        maximum_density,
        revision,
    })
}

pub(super) fn local_index(local: IVec3) -> Option<usize> {
    if local.cmplt(IVec3::ZERO).any() || local.cmpge(IVec3::splat(BRICK_EDGE_CELLS)).any() {
        return None;
    }
    let index = local.x + local.y * BRICK_EDGE_CELLS + local.z * BRICK_EDGE_CELLS.pow(2);
    usize::try_from(index).ok()
}

pub(super) fn read_u16(bytes: &[u8], cursor: usize) -> Result<u16, BrickDecodeError> {
    let raw = bytes
        .get(cursor..cursor + 2)
        .ok_or(BrickDecodeError::Truncated)?;
    Ok(u16::from_le_bytes([raw[0], raw[1]]))
}

pub(super) fn read_u32(bytes: &[u8], cursor: usize) -> Result<u32, BrickDecodeError> {
    let raw = bytes
        .get(cursor..cursor + 4)
        .ok_or(BrickDecodeError::Truncated)?;
    Ok(u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]))
}

pub(super) fn read_u64(bytes: &[u8], cursor: usize) -> Result<u64, BrickDecodeError> {
    let raw = bytes
        .get(cursor..cursor + 8)
        .ok_or(BrickDecodeError::Truncated)?;
    Ok(u64::from_le_bytes(
        raw.try_into().expect("slice length checked"),
    ))
}

pub(super) fn read_i32(bytes: &[u8], cursor: usize) -> Result<i32, BrickDecodeError> {
    Ok(i32::from_le_bytes(read_u32(bytes, cursor)?.to_le_bytes()))
}
