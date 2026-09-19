pub(crate) mod bearings;
pub(crate) mod bounds;
pub(crate) mod candidates;
pub(crate) mod faces;
pub(crate) mod grid;
pub(crate) mod layers;
pub(crate) mod pipes;
pub(crate) mod raycast;
pub(crate) mod snap;
pub(crate) mod staging;
pub(crate) mod welds;

pub(crate) use bearings::{
    LinearAttachment, bearing_anchor_from_hit_with_grid, bearing_attachment_candidate,
    bearing_overlaps_candidate, bearing_overlaps_cylinder_candidate, bearing_support_face,
    bearing_support_face_excluding, center_cylinder_candidate_on_bearing, linear_block_candidate,
    linear_cylinder_candidate, linear_mount_overlaps_face, linear_support_face_excluding,
    newly_locked_bearings, plate_block_candidate, plate_cylinder_candidate,
    stage_bearing_attachment_in_bounds, stage_linear_block_batch_in_bounds,
    stage_linear_block_volume_in_bounds, stage_linear_cylinder_in_bounds, stage_plate_block,
    stage_plate_cylinder,
};
#[cfg(test)]
pub(crate) use bearings::{bearing_anchor_from_hit, stage_bearing_attachment};
#[cfg(test)]
use bounds::validate_part;
pub(crate) use bounds::{
    composed_part_world_bounds, part_world_bounds, parts_overlap, validate_block_batch_in_bounds,
    validate_block_volume_in_bounds, validate_indexed_block_batch_in_bounds, validate_world_bounds,
};
pub(crate) use candidates::{
    candidate_from_hit_with_grid_and_supports, cylinder_candidate_from_hit_with_grid,
    free_cuboid_candidate, free_cylinder_candidate, oriented_cuboid_candidate_from_hit_with_grid,
};
#[cfg(test)]
pub(crate) use candidates::{
    cuboid_candidate_from_hit, cylinder_candidate_from_hit, oriented_cuboid_candidate_from_hit,
};
pub(crate) use faces::{
    face_geometry_from_ref, face_is_flat, primitive_surface_patch, try_face_geometry_from_ref,
};
#[cfg(test)]
pub(crate) use grid::block_sheet_specs;
use grid::cardinal_axis;
pub(crate) use grid::{block_box_bounds, block_box_specs, block_span_from_rays};
pub(crate) use layers::{
    LayerDrag, LayerTarget, layer_target_from_hit, layered_parts, stage_layer,
    validate_layered_parts,
};
#[cfg(test)]
pub(crate) use pipes::stage_pipe_run;
pub(crate) use pipes::{
    PipeBranch, PipeNode, PipeRunAttachment, PipeRunPiece, apply_pipe_branch,
    pipe_branch_candidate, pipe_run_pieces, stage_pipe_run_in_bounds, validate_pipe_run_in_bounds,
};
pub(crate) use raycast::{
    raycast_construction, raycast_construction_filtered_with_ground,
    raycast_construction_for_annulus, raycast_construction_for_annulus_filtered_with_ground,
    raycast_construction_for_annulus_with_ground, raycast_evaluated_surface,
    raycast_part_in_construction, raycast_placement_plane_point, region_pieces,
};
pub(crate) use snap::{
    PlacementSnapIndex, SmartGuide, smart_snap_anchor, smart_snap_block_span,
    smart_snap_cuboid_candidate, smart_snap_cuboid_candidate_with_supports,
    smart_snap_cylinder_candidate, smart_snap_free_cuboid_candidate,
    smart_snap_free_cylinder_candidate,
};
#[cfg(test)]
pub(crate) use staging::{
    stage_bearing_block_batch, stage_block_batch, stage_block_batch_from_source,
    stage_block_batch_from_source_in_bounds, stage_block_batch_in_bounds, stage_cuboid,
    stage_cylinder_from_source, stage_engine_from_source, transmission_candidate_from_hit,
};
pub(crate) use staging::{
    stage_bearing_cylinder_in_bounds, stage_block_volume_in_bounds, stage_controller_in_bounds,
    stage_dimension_link_in_bounds, stage_engine_in_bounds, stage_input_in_bounds,
    stage_seat_in_bounds, stage_servo_in_bounds, stage_transmission,
    transmission_candidate_from_hit_in_bounds, validate_cylinder_candidate_in_bounds,
};
#[cfg(test)]
pub(crate) use welds::begin_weld;
pub(crate) use welds::{rigid_body_parts, stage_weld_objects};

use std::fmt;

use bevy::{math::DVec2, prelude::*};
use mechanic_core::{
    BuildPose, ConstructionGraph, CuboidSpec, CylinderSpec, FaceKind, FaceOwner, FaceRef,
    GridRotation, POSITION_TICKS_PER_GRID_UNIT, PartId,
};

pub(crate) const GROUND_HALF_SIZE: f32 = 10.0;
const CONTACT_EPSILON: f32 = 1.0e-5;
const GRID_UNIT_METERS: f32 = 0.25;
pub(crate) const BEARING_DEPTH: f32 = 0.10;
pub(crate) const MAX_DRAG_BLOCKS: usize = 4_096;
pub(crate) const BLOCK_SIZE_METERS: f32 = GRID_UNIT_METERS;
const BLOCK_SIZE_UNITS: u8 = 1;

/// Fixed global placement-grid resolution selected by keyboard modifiers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub(crate) enum PlacementGrid {
    #[default]
    Centimetres25,
    Centimetres5,
    Centimetres1,
}

impl PlacementGrid {
    pub(crate) const fn from_modifiers(shift: bool, control: bool) -> Self {
        match (shift, control) {
            (true, true) => Self::Centimetres1,
            (true, false) => Self::Centimetres5,
            (false, _) => Self::Centimetres25,
        }
    }

    pub(crate) const fn step_ticks(self) -> i32 {
        match self {
            Self::Centimetres25 => POSITION_TICKS_PER_GRID_UNIT,
            Self::Centimetres5 => 20,
            Self::Centimetres1 => 4,
        }
    }

    pub(crate) const fn step_meters(self) -> f32 {
        match self {
            Self::Centimetres25 => 0.25,
            Self::Centimetres5 => 0.05,
            Self::Centimetres1 => 0.01,
        }
    }

    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Centimetres25 => "25 cm",
            Self::Centimetres5 => "5 cm",
            Self::Centimetres1 => "1 cm",
        }
    }
}

const ALL_FACES: [FaceKind; 6] = [
    FaceKind::PositiveX,
    FaceKind::NegativeX,
    FaceKind::PositiveY,
    FaceKind::NegativeY,
    FaceKind::PositiveZ,
    FaceKind::NegativeZ,
];

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) enum PlacementBounds {
    #[default]
    Garage,
    GarageBuild,
    /// Candidates use a creation's local grid; limits remain in garage space.
    GarageBuildFrame {
        to_garage: mechanic_core::ConstructionFrame,
    },
    World {
        origin: DVec2,
    },
}

impl PlacementBounds {
    pub(crate) fn in_edit_frame(self, to_build: mechanic_core::ConstructionFrame) -> Self {
        match self {
            Self::GarageBuild => Self::GarageBuildFrame {
                to_garage: to_build,
            },
            Self::World { .. } => Self::World {
                origin: DVec2::ZERO,
            },
            _ => self,
        }
    }

    pub(crate) const fn is_world(self) -> bool {
        matches!(self, Self::World { .. })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PlacementPlane {
    Xy,
    Xz,
    Yz,
}

impl PlacementPlane {
    pub(crate) fn from_normal(normal: Vec3) -> Self {
        match cardinal_axis(normal).0 {
            0 => Self::Yz,
            1 => Self::Xz,
            _ => Self::Xy,
        }
    }

    pub(crate) const fn cycle(self) -> Self {
        match self {
            Self::Xz => Self::Xy,
            Self::Xy => Self::Yz,
            Self::Yz => Self::Xz,
        }
    }

    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Xy => "XY",
            Self::Xz => "XZ",
            Self::Yz => "YZ",
        }
    }

    pub(crate) const fn normal_axis(self) -> usize {
        match self {
            Self::Xy => 2,
            Self::Xz => 1,
            Self::Yz => 0,
        }
    }

    pub(crate) const fn tangent_axes(self) -> [usize; 2] {
        match self {
            Self::Xy => [0, 1],
            Self::Xz => [0, 2],
            Self::Yz => [1, 2],
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct SurfaceHit {
    pub(crate) distance: f32,
    pub(crate) point: Vec3,
    pub(crate) face: FaceRef,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct OrientedCuboidHit {
    pub(crate) distance: f32,
    pub(crate) point: Vec3,
    pub(crate) local_normal: Vec3,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct PlacementCandidate {
    pub(crate) spec: CuboidSpec,
    pub(crate) attached_face: FaceKind,
    pub(crate) anchor: Option<Vec3>,
    pub(crate) support: PlacementSupport,
}

/// Compact description of one solid drag-created block volume.
///
/// `span` counts blocks beyond the starting block and retains its sign, so the
/// descriptor also preserves which block was initially attached to a surface
/// or bearing. Individual authored specs are materialized only when the volume
/// is committed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct BlockVolume {
    start: CuboidSpec,
    span: IVec3,
    counts: UVec3,
    count: usize,
    minimum: Vec3,
    maximum: Vec3,
}

impl BlockVolume {
    pub(crate) fn new(start: CuboidSpec, span: IVec3) -> Result<Self, PlacementError> {
        let counts = UVec3::from_array(span.to_array().map(|steps| steps.unsigned_abs() + 1));
        let count = (counts.x as usize)
            .saturating_mul(counts.y as usize)
            .saturating_mul(counts.z as usize);
        if count > MAX_DRAG_BLOCKS {
            return Err(PlacementError::TooManyBlocks {
                count,
                maximum: MAX_DRAG_BLOCKS,
            });
        }
        let (minimum, maximum) = block_box_bounds(start, span);
        Ok(Self {
            start,
            span,
            counts,
            count,
            minimum,
            maximum,
        })
    }

    pub(crate) const fn start(self) -> CuboidSpec {
        self.start
    }

    pub(crate) const fn dimensions(self) -> UVec3 {
        self.counts
    }

    pub(crate) const fn count(self) -> usize {
        self.count
    }

    pub(crate) const fn bounds(self) -> (Vec3, Vec3) {
        (self.minimum, self.maximum)
    }

    pub(crate) fn specs(self) -> BlockVolumeSpecs {
        BlockVolumeSpecs {
            volume: self,
            next: 0,
        }
    }

    fn spec_at_logical(self, logical: UVec3) -> CuboidSpec {
        let direction = self.span.signum();
        let steps = IVec3::new(
            i32::try_from(logical.x).expect("block volume axis fits i32") * direction.x,
            i32::try_from(logical.y).expect("block volume axis fits i32") * direction.y,
            i32::try_from(logical.z).expect("block volume axis fits i32") * direction.z,
        );
        self.spec_at_steps(steps)
    }

    fn spec_at_physical(self, physical: UVec3) -> CuboidSpec {
        let logical = UVec3::new(
            if self.span.x < 0 {
                self.counts.x - 1 - physical.x
            } else {
                physical.x
            },
            if self.span.y < 0 {
                self.counts.y - 1 - physical.y
            } else {
                physical.y
            },
            if self.span.z < 0 {
                self.counts.z - 1 - physical.z
            } else {
                physical.z
            },
        );
        self.spec_at_logical(logical)
    }

    fn spec_at_steps(self, steps: IVec3) -> CuboidSpec {
        let dimension_units = self.start.dimensions[0].units();
        let block_ticks = i32::from(dimension_units) * POSITION_TICKS_PER_GRID_UNIT;
        let center = self.start.pose.translation_position_ticks() + steps * block_ticks;
        CuboidSpec::new(
            [dimension_units; 3],
            BuildPose::from_position_ticks(center, GridRotation::default()),
        )
        .expect("dragged blocks retain the selected valid size")
        .with_material(self.start.material)
        .with_appearance(self.start.appearance)
    }

    fn linear_index(self, logical: UVec3) -> usize {
        ((logical.x as usize * self.counts.y as usize) + logical.y as usize)
            * self.counts.z as usize
            + logical.z as usize
    }

    fn part_at_physical(self, parts: &[PartId], physical: UVec3) -> PartId {
        let logical = UVec3::new(
            if self.span.x < 0 {
                self.counts.x - 1 - physical.x
            } else {
                physical.x
            },
            if self.span.y < 0 {
                self.counts.y - 1 - physical.y
            } else {
                physical.y
            },
            if self.span.z < 0 {
                self.counts.z - 1 - physical.z
            } else {
                physical.z
            },
        );
        parts[self.linear_index(logical)]
    }
}

pub(crate) struct BlockVolumeSpecs {
    volume: BlockVolume,
    next: usize,
}

impl Iterator for BlockVolumeSpecs {
    type Item = CuboidSpec;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next >= self.volume.count {
            return None;
        }
        let yz = self.volume.counts.y as usize * self.volume.counts.z as usize;
        let x = self.next / yz;
        let remainder = self.next % yz;
        let y = remainder / self.volume.counts.z as usize;
        let z = remainder % self.volume.counts.z as usize;
        self.next += 1;
        Some(self.volume.spec_at_logical(UVec3::new(
            u32::try_from(x).expect("block volume axis fits u32"),
            u32::try_from(y).expect("block volume axis fits u32"),
            u32::try_from(z).expect("block volume axis fits u32"),
        )))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.volume.count - self.next;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for BlockVolumeSpecs {}

/// One synchronously committed authored volume and its publication metadata.
pub(crate) struct BlockVolumePlacement {
    pub(crate) graph: ConstructionGraph,
    pub(crate) new_parts: Vec<PartId>,
    pub(crate) weld_count: usize,
    pub(crate) bounds: (Vec3, Vec3),
    pub(crate) publication_generation: u64,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct CylinderPlacementCandidate {
    pub(crate) spec: CylinderSpec,
    pub(crate) attached_face: FaceKind,
    pub(crate) anchor: Option<Vec3>,
    pub(crate) support: PlacementSupport,
}

/// What makes a placement candidate available at its current position.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PlacementSupport {
    Surface(FaceOwner),
    Bearing,
    Free,
}

impl PlacementSupport {
    const fn auto_weld_source(self) -> Option<FaceOwner> {
        match self {
            Self::Surface(source) => Some(source),
            Self::Bearing | Self::Free => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum PlacementError {
    /// The surface has been shaped, so nothing can sit flush on it.
    SurfaceNotFlat,
    OutsidePlatform,
    NoFaceOverlap,
    OverlapsPart(PartId),
    BearingOnGround,
    BearingOutsideFace,
    SameObject,
    ObjectsDoNotTouch,
    CurvedSurface,
    NotLayerSurface,
    TransmissionOutputOnly,
    EmptyBlockBatch,
    BlocksOverlap,
    DragPlaneUnavailable,
    PipeRun(String),
    TooManyBlocks {
        count: usize,
        maximum: usize,
    },
    Graph(String),
}

impl fmt::Display for PlacementError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SurfaceNotFlat => {
                formatter.write_str("that surface is not flat — flatten it before building on it")
            }
            Self::OutsidePlatform => formatter.write_str("part would extend beyond the platform"),
            Self::NoFaceOverlap => formatter.write_str("cube does not overlap the selected face"),
            Self::OverlapsPart(part) => write!(formatter, "part overlaps {part:?}"),
            Self::BearingOnGround => formatter.write_str("bearings cannot attach to the ground"),
            Self::BearingOutsideFace => {
                formatter.write_str("the bearing anchor lies outside this face")
            }
            Self::SameObject => formatter.write_str("select two different objects"),
            Self::ObjectsDoNotTouch => {
                formatter.write_str("weld contact must contain a continuous 5 × 5 cm square")
            }
            Self::CurvedSurface => {
                formatter.write_str("curved cylinder walls are not connection faces")
            }
            Self::NotLayerSurface => {
                formatter.write_str("point at a flat face, or a full cylinder's wall or bore")
            }
            Self::TransmissionOutputOnly => formatter.write_str(
                "transmissions attach only to an engine or chain-tail positive-Z output",
            ),
            Self::EmptyBlockBatch => formatter.write_str("block drag did not produce any blocks"),
            Self::BlocksOverlap => formatter.write_str("dragged blocks overlap one another"),
            Self::DragPlaneUnavailable => {
                formatter.write_str("camera ray does not reach the selected drag plane")
            }
            Self::PipeRun(reason) => formatter.write_str(reason),
            Self::TooManyBlocks { count, maximum } => {
                write!(
                    formatter,
                    "drag would place {count} blocks; maximum is {maximum}"
                )
            }
            Self::Graph(error) => formatter.write_str(error),
        }
    }
}

#[cfg(test)]
mod tests;
