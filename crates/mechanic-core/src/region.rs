//! Shape regions: a selected solid cuboid of blocks, and the control cage that
//! deforms it.
//!
//! A region is the answer to "what am I editing". Nothing can be shaped until
//! an area is chosen, and once one is, only that area moves. The blocks inside
//! keep their identity for material, mass, and welds, but their geometry is
//! replaced by the region's: one merged shape rather than a pile of boxes.
//!
//! The cage is a grid of control vertices. A fresh region has two planes per
//! axis — eight corners, one hexahedron. Subdividing inserts a whole plane, so
//! the cage is always a valid grid of hexahedra and the decomposition, its
//! watertightness, and the bounding-box clamp all keep holding.
//!
//! Every vertex is clamped to the region's original bounding box. A corner can
//! therefore only ever move inward, which is what stops one region from growing
//! into its neighbours.

use std::collections::BTreeMap;

use bevy_math::IVec3;
use thiserror::Error;

use crate::shape::{CellGrid, STEPS_PER_CELL, STEPS_PER_HALF_UNIT};
use crate::{ConstructionMaterial, MaterialAppearance};

/// A cage vertex, indexed by its plane along each axis.
pub type CageIndex = [u16; 3];

/// Something a region cannot be asked to do.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum RegionError {
    /// A region must span at least one cell on every axis.
    #[error("a region must be at least one cell on every axis; got {0:?}")]
    EmptyExtent(IVec3),
    /// The cage has no vertex there.
    #[error("cage vertex {0:?} is outside the region")]
    UnknownVertex(CageIndex),
    /// The move would take a vertex out of the region's bounding box.
    #[error("a cage vertex cannot leave the region's bounding box")]
    OutsideBounds,
    /// Subdivision was asked for somewhere it cannot go.
    #[error("cannot subdivide axis {axis} at cell {position}")]
    BadSubdivision {
        /// Axis asked for.
        axis: usize,
        /// Cell position asked for.
        position: i32,
    },
}

/// A solid cuboid of blocks and the cage that shapes it.
#[derive(Clone, Debug, PartialEq)]
pub struct ShapeRegion {
    origin_steps: IVec3,
    size_cells: IVec3,
    material: ConstructionMaterial,
    appearance: MaterialAppearance,
    /// Cage plane positions in cells from the origin, ascending. The first is
    /// always 0 and the last always `size_cells[axis]`.
    planes: [Vec<i32>; 3],
    /// Displacement in steps, for the vertices that have moved.
    offsets: BTreeMap<CageIndex, [i16; 3]>,
}

impl ShapeRegion {
    /// Creates an unshaped region covering `size_cells` cells from `origin`.
    ///
    /// # Errors
    ///
    /// Returns [`RegionError::EmptyExtent`] unless every axis spans at least one
    /// cell.
    pub fn new(
        origin_half_units: IVec3,
        size_cells: IVec3,
        material: ConstructionMaterial,
    ) -> Result<Self, RegionError> {
        Self::from_origin_steps(
            origin_half_units * STEPS_PER_HALF_UNIT,
            size_cells,
            material,
        )
    }

    /// Creates an unshaped region at an exact shape-step origin.
    ///
    /// # Errors
    ///
    /// Returns [`RegionError::EmptyExtent`] unless every axis spans at least one
    /// cell.
    pub fn from_origin_steps(
        origin_steps: IVec3,
        size_cells: IVec3,
        material: ConstructionMaterial,
    ) -> Result<Self, RegionError> {
        if size_cells.cmplt(IVec3::ONE).any() {
            return Err(RegionError::EmptyExtent(size_cells));
        }
        Ok(Self {
            origin_steps,
            size_cells,
            material,
            appearance: MaterialAppearance::BAKED,
            planes: core::array::from_fn(|axis| vec![0, size_cells[axis]]),
            offsets: BTreeMap::new(),
        })
    }

    /// Minimum corner, in shape steps.
    pub const fn origin_steps(&self) -> IVec3 {
        self.origin_steps
    }

    /// Extent in construction cells.
    pub const fn size_cells(&self) -> IVec3 {
        self.size_cells
    }

    /// Material every block in the region shares.
    pub const fn material(&self) -> ConstructionMaterial {
        self.material
    }

    /// Color and finish shared by every block in the region.
    pub const fn appearance(&self) -> MaterialAppearance {
        self.appearance
    }

    /// Uses an explicit construction appearance.
    #[must_use]
    pub const fn with_appearance(mut self, appearance: MaterialAppearance) -> Self {
        self.appearance = appearance;
        self
    }

    pub(crate) const fn set_appearance(&mut self, appearance: MaterialAppearance) {
        self.appearance = appearance;
    }

    /// Whether the region has been shaped at all.
    pub fn is_unshaped(&self) -> bool {
        self.offsets.is_empty()
    }

    /// Number of cage planes along each axis.
    pub fn plane_counts(&self) -> [usize; 3] {
        core::array::from_fn(|axis| self.planes[axis].len())
    }

    /// Every cage vertex index, in deterministic order.
    pub fn vertices(&self) -> impl Iterator<Item = CageIndex> + '_ {
        let [x, y, z] = self.plane_counts();
        (0..z).flat_map(move |k| {
            (0..y).flat_map(move |j| {
                (0..x).map(move |i| {
                    [
                        u16::try_from(i).unwrap_or(u16::MAX),
                        u16::try_from(j).unwrap_or(u16::MAX),
                        u16::try_from(k).unwrap_or(u16::MAX),
                    ]
                })
            })
        })
    }

    /// Displaced vertices only, in deterministic order.
    pub fn offsets(&self) -> impl Iterator<Item = (CageIndex, [i16; 3])> + '_ {
        self.offsets.iter().map(|(&index, &offset)| (index, offset))
    }

    /// The cell grid the cage describes, in half-grid units.
    pub fn grid(&self) -> CellGrid {
        CellGrid::from_cell_planes(self.origin_steps, &self.planes)
    }

    /// Where a cage vertex sits with nothing moving it.
    pub fn base_steps(&self, index: CageIndex) -> Option<IVec3> {
        let mut position = IVec3::ZERO;
        for axis in 0..3 {
            let cells = *self.planes[axis].get(usize::from(index[axis]))?;
            position[axis] = self.origin_steps[axis] + cells * STEPS_PER_CELL;
        }
        Some(position)
    }

    /// Where a cage vertex sits now.
    pub fn vertex_steps(&self, index: CageIndex) -> Option<IVec3> {
        let base = self.base_steps(index)?;
        let offset = self.offset(index);
        Some(base + IVec3::new(offset[0].into(), offset[1].into(), offset[2].into()))
    }

    /// This vertex's displacement, or zero when it rests on the grid.
    pub fn offset(&self, index: CageIndex) -> [i16; 3] {
        self.offsets.get(&index).copied().unwrap_or_default()
    }

    /// Corner positions for one cage cell, for the decomposition.
    ///
    /// # Panics
    ///
    /// Never in practice: cell indices come from the grid the cage produced.
    pub fn corner_steps(&self, cell: IVec3, corner: usize) -> IVec3 {
        let bit = |axis: usize| i32::from(u8::try_from((corner >> axis) & 1).unwrap_or(0));
        let index = [
            u16::try_from(cell.x + bit(0)).expect("cage indices are small"),
            u16::try_from(cell.y + bit(1)).expect("cage indices are small"),
            u16::try_from(cell.z + bit(2)).expect("cage indices are small"),
        ];
        self.vertex_steps(index)
            .expect("a cage cell corner is always a cage vertex")
    }

    /// The region's original bounding box, in steps.
    pub fn bounds_steps(&self) -> (IVec3, IVec3) {
        (
            self.origin_steps,
            self.origin_steps + self.size_cells * STEPS_PER_CELL,
        )
    }

    /// Moves one cage vertex.
    ///
    /// # Errors
    ///
    /// Returns [`RegionError::UnknownVertex`] for an index the cage does not
    /// have, or [`RegionError::OutsideBounds`] when the move would leave the
    /// region's original bounding box.
    pub fn set_offset(&mut self, index: CageIndex, offset: [i16; 3]) -> Result<(), RegionError> {
        let base = self
            .base_steps(index)
            .ok_or(RegionError::UnknownVertex(index))?;
        let moved = base + IVec3::new(offset[0].into(), offset[1].into(), offset[2].into());
        let (minimum, maximum) = self.bounds_steps();
        if moved.cmplt(minimum).any() || moved.cmpgt(maximum).any() {
            return Err(RegionError::OutsideBounds);
        }
        if offset == [0; 3] {
            self.offsets.remove(&index);
        } else {
            self.offsets.insert(index, offset);
        }
        Ok(())
    }

    /// Inserts a cage plane at `position` cells along `axis`.
    ///
    /// Every vertex on the new plane is interpolated from its neighbours, so
    /// the surface does not move — the cage simply gains a row of handles.
    ///
    /// # Errors
    ///
    /// Returns [`RegionError::BadSubdivision`] when the position is outside the
    /// region or the axis is not 0, 1, or 2. Asking for a plane that already
    /// exists succeeds and changes nothing.
    ///
    /// # Panics
    ///
    /// Never in practice: cage indices are bounded by the region's extent.
    pub fn subdivide(&mut self, axis: usize, position: i32) -> Result<(), RegionError> {
        let bad = RegionError::BadSubdivision { axis, position };
        if axis > 2 || position <= 0 || position >= self.size_cells[axis] {
            return Err(bad);
        }
        // Already a plane: nothing to do, and not an error worth raising to the
        // user who merely clicked the same spot twice.
        let Err(insert_at) = self.planes[axis].binary_search(&position) else {
            return Ok(());
        };
        let low = self.planes[axis][insert_at - 1];
        let high = self.planes[axis][insert_at];
        let blend = f64::from(position - low) / f64::from(high - low);

        let inserted = u16::try_from(insert_at).expect("cage indices are small");
        let mut moved: BTreeMap<CageIndex, [i16; 3]> = BTreeMap::new();
        for (&index, &offset) in &self.offsets {
            let mut shifted = index;
            if index[axis] >= inserted {
                shifted[axis] += 1;
            }
            moved.insert(shifted, offset);
        }

        // Interpolate the new plane from the two it sits between, reading the
        // already-shifted indices so both sides are the post-insert ones.
        let mut created: BTreeMap<CageIndex, [i16; 3]> = BTreeMap::new();
        let counts = self.plane_counts();
        let others: [usize; 2] = [(axis + 1) % 3, (axis + 2) % 3];
        for first in 0..counts[others[0]] {
            for second in 0..counts[others[1]] {
                let mut low_index = [0_u16; 3];
                low_index[axis] = inserted - 1;
                low_index[others[0]] = u16::try_from(first).expect("cage indices are small");
                low_index[others[1]] = u16::try_from(second).expect("cage indices are small");
                let mut high_index = low_index;
                high_index[axis] = inserted + 1;

                let low_offset = moved.get(&low_index).copied().unwrap_or_default();
                let high_offset = moved.get(&high_index).copied().unwrap_or_default();
                if low_offset == [0; 3] && high_offset == [0; 3] {
                    continue;
                }
                let mut new_index = low_index;
                new_index[axis] = inserted;
                let lerp = |low: i16, high: i16| {
                    let value = f64::from(low).mul_add(1.0 - blend, f64::from(high) * blend);
                    #[allow(clippy::cast_possible_truncation)] // Offsets are tiny.
                    let rounded = value.round() as i16;
                    rounded
                };
                created.insert(
                    new_index,
                    [
                        lerp(low_offset[0], high_offset[0]),
                        lerp(low_offset[1], high_offset[1]),
                        lerp(low_offset[2], high_offset[2]),
                    ],
                );
            }
        }

        self.planes[axis].insert(insert_at, position);
        moved.extend(created);
        self.offsets = moved;
        Ok(())
    }

    /// Whether one outer face of the region is still flat.
    ///
    /// Flat means every cage vertex on that face rests on the grid, so the face
    /// is a true axis-aligned rectangle something can be mounted flush against.
    /// Shaping a face makes it unplaceable; bringing those vertices back onto
    /// the grid makes it placeable again, which is how a mounting surface is
    /// made where there was none.
    pub fn face_is_flat(&self, axis: usize, positive: bool) -> bool {
        let last = u16::try_from(self.planes[axis].len() - 1).unwrap_or(0);
        let wanted = if positive { last } else { 0 };
        self.offsets.iter().all(|(index, _)| index[axis] != wanted)
    }

    /// Whether this region covers the cell whose minimum corner is at
    /// `cell_min_steps`.
    pub fn covers_cell(&self, cell_min_steps: IVec3) -> bool {
        let relative = cell_min_steps - self.origin_steps;
        relative.cmpge(IVec3::ZERO).all()
            && (relative % STEPS_PER_CELL).cmpeq(IVec3::ZERO).all()
            && (relative / STEPS_PER_CELL).cmplt(self.size_cells).all()
    }

    /// Whether two regions claim any of the same space.
    pub fn overlaps(&self, other: &Self) -> bool {
        let (low, high) = (
            self.origin_steps,
            self.origin_steps + self.size_cells * STEPS_PER_CELL,
        );
        let (other_low, other_high) = (
            other.origin_steps,
            other.origin_steps + other.size_cells * STEPS_PER_CELL,
        );
        low.cmplt(other_high).all() && other_low.cmplt(high).all()
    }
}
