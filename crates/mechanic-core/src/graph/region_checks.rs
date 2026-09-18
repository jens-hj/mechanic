//! Shape-region validity: inverted cells and claimed area.

use super::ConstructionGraph;
use super::error::GraphError;
use crate::{
    ConstructionMaterial, MaterialAppearance, PartId, PartSpec, RegionId, ShapeRegion, SolidOwner,
};
use bevy_math::IVec3;
use std::collections::BTreeMap;

impl ConstructionGraph {
    /// Whether a region's cage has turned any of its cells inside out.
    pub(super) fn reject_inverted_region(&self, region: RegionId) -> Result<(), GraphError> {
        let shape = self
            .regions
            .get(region)
            .ok_or(GraphError::MissingRegion(region))?;
        let grid = shape.grid();
        if crate::shape::has_inverted_cell(&grid, &|cell, corner| shape.corner_steps(cell, corner))
        {
            return Err(GraphError::InvertedCell(region));
        }
        Ok(())
    }

    /// Checks the rules an area must satisfy before it can become a region,
    /// without adding one, so a drag can preview whether it would be accepted.
    ///
    /// # Errors
    ///
    /// Reports the first rule the area breaks.
    pub fn check_region_area(&self, region: &ShapeRegion) -> Result<(), GraphError> {
        self.validate_region_area(region, self.edit_frame_id())
    }

    /// Checks the rules an area must satisfy before it can become a region:
    /// every cell filled, one material, one rigid body, whole blocks only, and
    /// nothing already claiming the space.
    pub(super) fn validate_region_area(
        &self,
        region: &ShapeRegion,
        frame: crate::ConstructionFrameId,
    ) -> Result<(), GraphError> {
        if let Some((id, _)) = self.regions.iter().find(|(id, existing)| {
            self.region_frame_id(*id) == Some(frame) && existing.overlaps(region)
        }) {
            return Err(GraphError::RegionOverlaps(id));
        }

        let mut occupants: BTreeMap<[i32; 3], (PartId, ConstructionMaterial, MaterialAppearance)> =
            BTreeMap::new();
        for (id, spec) in self.parts.iter() {
            if self.part_frame_id(id) != Some(frame) {
                continue;
            }
            let Some(cuboid) = spec.as_cuboid() else {
                continue;
            };
            // Only ordinary blocks make up a region; the fixed authored machine
            // parts are components, not material.
            if !matches!(spec, PartSpec::Cuboid(_)) {
                continue;
            }
            let cells = crate::part_cells(cuboid);
            let counts = cells.counts();
            for z in 0..counts.z {
                for y in 0..counts.y {
                    for x in 0..counts.x {
                        let corner = cells.corner_steps(IVec3::new(x, y, z), 0);
                        occupants
                            .insert(corner.to_array(), (id, cuboid.material, cuboid.appearance));
                    }
                }
            }
        }

        let size = region.size_cells();
        let origin = region.origin_steps();
        let mut material: Option<ConstructionMaterial> = None;
        let mut appearance: Option<MaterialAppearance> = None;
        let mut members: Vec<PartId> = Vec::new();
        let mut claimed: BTreeMap<PartId, i32> = BTreeMap::new();
        let mut empty = 0_usize;
        for z in 0..size.z {
            for y in 0..size.y {
                for x in 0..size.x {
                    let corner = origin + IVec3::new(x, y, z) * crate::POSITION_TICKS_PER_GRID_UNIT;
                    let Some(&(part, cell_material, cell_appearance)) =
                        occupants.get(&corner.to_array())
                    else {
                        empty += 1;
                        continue;
                    };
                    if *material.get_or_insert(cell_material) != cell_material {
                        return Err(GraphError::RegionMixedMaterials);
                    }
                    if *appearance.get_or_insert(cell_appearance) != cell_appearance {
                        return Err(GraphError::RegionMixedAppearances);
                    }
                    if !members.contains(&part) {
                        members.push(part);
                    }
                    *claimed.entry(part).or_default() += 1;
                }
            }
        }
        if empty > 0 {
            return Err(GraphError::RegionNotSolid(empty));
        }
        if material != Some(region.material()) {
            return Err(GraphError::RegionMixedMaterials);
        }
        if appearance != Some(region.appearance()) {
            return Err(GraphError::RegionMixedAppearances);
        }

        // A member hands its whole surface and mass to the region, so an area
        // that holds only some of a block's cells would lose the rest.
        for (&part, &cells) in &claimed {
            if self.owner_has_shape_features(SolidOwner::Part(part)) {
                return Err(GraphError::FeaturedPartInRegion(part));
            }
            if self.parts.get(part).is_some_and(|spec| spec.is_layered()) {
                return Err(GraphError::LayeredPartInRegion(part));
            }
            let whole = self
                .parts
                .get(part)
                .and_then(|spec| spec.as_cuboid())
                .map_or(0, |cuboid| {
                    crate::part_cells(cuboid).counts().element_product()
                });
            if cells != whole {
                return Err(GraphError::RegionSplitsPart);
            }
        }

        // One rigid body: walking the welds from any member must reach them all.
        if let Some(&seed) = members.first() {
            let body = self.rigid_group(seed);
            if !members.iter().all(|part| body.contains(part)) {
                return Err(GraphError::RegionSpansBodies);
            }
        }
        Ok(())
    }
}
