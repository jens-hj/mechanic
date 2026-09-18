use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
    fmt,
};

use bevy::{math::DVec2, prelude::*};
use mechanic_core::{
    BearingDimensions, BearingId, BearingKind, BearingSpec, BuildCommand, BuildOutcome, BuildPose,
    ConstructionGraph, ControllerSpec, ConvexPiece, CuboidSpec, CylinderDimensions, CylinderSpec,
    DimensionLinkId, DimensionLinkSpec, EngineKind, EngineSpec, FaceKind, FaceOwner, FaceRef,
    GridDimension, GridRotation, InputSpec, LinearBearing, LinearBearingDimensions,
    POSITION_TICK_METERS, POSITION_TICKS_PER_GRID_UNIT, POSITION_TICKS_PER_HALF_GRID_UNIT, PartId,
    PartPiece, PartSpec, PipeArms, PipeBendDimensions, PipeBendSpec, PipeJunctionDimensions,
    PipeJunctionSpec, RigidLinkSpec, SeatSpec, ServoSpec, ShapeRegion, TransmissionSpec, WeldSpec,
};
use mechanic_world::WORLD_HALF_EXTENT_METERS;

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

const SNAP_BIN_METERS: f32 = 1.0;
pub(crate) const SMART_SNAP_CAPTURE_METERS: f32 = 0.025;

#[derive(Clone, Copy, Debug)]
struct SnapTarget {
    part: PartId,
    spec: PartSpec,
    frame: mechanic_core::ConstructionFrame,
    minimum: Vec3,
    maximum: Vec3,
}

/// Spatially binned committed solid-part bounds used by placement previews.
#[derive(Resource, Default)]
pub(crate) struct PlacementSnapIndex {
    targets: Vec<SnapTarget>,
    bins: HashMap<IVec3, Vec<usize>>,
}

impl PlacementSnapIndex {
    pub(crate) fn rebuild(&mut self, graph: &ConstructionGraph) {
        self.targets.clear();
        self.bins.clear();
        for (part, spec) in graph.parts() {
            let (minimum, maximum) = composed_part_world_bounds(graph, part)
                .expect("indexed parts have construction frames");
            let target_index = self.targets.len();
            self.targets.push(SnapTarget {
                part,
                spec: *spec,
                frame: graph
                    .part_frame(part)
                    .expect("indexed parts have construction frames"),
                minimum,
                maximum,
            });
            let low = (minimum / SNAP_BIN_METERS).floor().as_ivec3();
            let high = (maximum / SNAP_BIN_METERS).floor().as_ivec3();
            for x in low.x..=high.x {
                for y in low.y..=high.y {
                    for z in low.z..=high.z {
                        self.bins
                            .entry(IVec3::new(x, y, z))
                            .or_default()
                            .push(target_index);
                    }
                }
            }
        }
    }

    fn nearby(&self, minimum: Vec3, maximum: Vec3, radius: f32) -> Vec<SnapTarget> {
        if self.targets.len() <= 16_384 {
            return self
                .targets
                .iter()
                .copied()
                .filter(|target| {
                    aabb_distance(minimum, maximum, target.minimum, target.maximum) <= radius
                })
                .collect();
        }
        let low = ((minimum - Vec3::splat(radius)) / SNAP_BIN_METERS)
            .floor()
            .as_ivec3();
        let high = ((maximum + Vec3::splat(radius)) / SNAP_BIN_METERS)
            .floor()
            .as_ivec3();
        let mut indices = HashSet::new();
        for x in low.x..=high.x {
            for y in low.y..=high.y {
                for z in low.z..=high.z {
                    if let Some(bin) = self.bins.get(&IVec3::new(x, y, z)) {
                        indices.extend(bin.iter().copied());
                    }
                }
            }
        }
        let mut targets = indices
            .into_iter()
            .filter_map(|index| self.targets.get(index).copied())
            .filter(|target| {
                aabb_distance(minimum, maximum, target.minimum, target.maximum) <= radius
            })
            .collect::<Vec<_>>();
        targets.sort_by_key(|target| target.part);
        targets
    }

    pub(crate) fn overlaps(&self, spec: PartSpec) -> bool {
        let (minimum, maximum) = part_world_bounds(spec);
        self.nearby(minimum, maximum, 0.0)
            .into_iter()
            .any(|target| parts_overlap_with_frame(spec, target.spec, target.frame))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum GuideKind {
    Center,
    Edge,
}

/// One active object-alignment line in local build coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct SmartGuide {
    pub(crate) axis: usize,
    pub(crate) coordinate: f32,
    pub(crate) from: Vec3,
    pub(crate) to: Vec3,
}

#[derive(Clone, Copy, Debug)]
struct AxisGuide {
    delta: f32,
    coordinate: f32,
    kind: GuideKind,
    part: PartId,
    target_center: Vec3,
}

#[derive(Clone, Copy, Debug)]
struct BlockEndpointGuide {
    span: i32,
    pointer_delta: f32,
    guide: AxisGuide,
}

/// Lets nearby committed object centers and edges override the global grid.
pub(crate) fn smart_snap_cuboid_candidate(
    graph: &ConstructionGraph,
    index: &PlacementSnapIndex,
    hit: SurfaceHit,
    gridded: PlacementCandidate,
    grid: PlacementGrid,
    range: f32,
    valid: impl FnMut(PlacementCandidate) -> bool,
) -> (PlacementCandidate, Vec<SmartGuide>) {
    let supports = support_geometries_from_hit(graph, hit);
    smart_snap_cuboid_candidate_with_supports(index, hit, gridded, grid, range, &supports, valid)
}

pub(crate) fn smart_snap_cuboid_candidate_with_supports(
    index: &PlacementSnapIndex,
    hit: SurfaceHit,
    gridded: PlacementCandidate,
    grid: PlacementGrid,
    range: f32,
    supports: &[FaceGeometry],
    mut valid: impl FnMut(PlacementCandidate) -> bool,
) -> (PlacementCandidate, Vec<SmartGuide>) {
    let Some(support) = support_at_hit(supports, hit.point) else {
        return (gridded, Vec::new());
    };
    let (normal_axis, _) = cardinal_axis(support.normal);
    let tangent_axes = match normal_axis {
        0 => [1, 2],
        1 => [0, 2],
        _ => [0, 1],
    };
    let (grid_minimum, grid_maximum) = cuboid_world_bounds(gridded.spec);
    let cell_margin = smart_snap_cell_margin(grid);
    let targets = index.nearby(grid_minimum, grid_maximum, range + cell_margin);
    let choices = tangent_axes.map(|axis| {
        axis_guides(
            axis,
            grid_minimum,
            grid_maximum,
            &targets,
            SMART_SNAP_CAPTURE_METERS + cell_margin,
        )
    });
    for axis_guides in guide_combinations(&choices) {
        let mut ticks = gridded.spec.pose.translation_position_ticks();
        for (choice_index, guide) in axis_guides.iter().enumerate() {
            if let Some(guide) = guide {
                let axis = tangent_axes[choice_index];
                ticks[axis] += rounded_position_tick(guide.delta);
            }
        }
        let spec = CuboidSpec::new(
            gridded.spec.dimensions.map(GridDimension::units),
            BuildPose::from_position_ticks(ticks, gridded.spec.pose.rotation),
        )
        .expect("an aligned candidate remains dimensionally valid")
        .with_material(gridded.spec.material)
        .with_appearance(gridded.spec.appearance);
        let candidate_face = face_geometry(spec, gridded.attached_face);
        let candidate = PlacementCandidate {
            spec,
            attached_face: gridded.attached_face,
            anchor: supports
                .iter()
                .find_map(|support| overlap_center(support, &candidate_face)),
            support: gridded.support,
        };
        if valid(candidate) {
            let center = candidate.spec.pose.translation();
            let rendered =
                render_smart_guides(axis_guides, &choices, tangent_axes, normal_axis, center);
            return (candidate, rendered);
        }
    }
    (gridded, Vec::new())
}

/// Cylinder counterpart of [`smart_snap_cuboid_candidate`].
pub(crate) fn smart_snap_cylinder_candidate(
    graph: &ConstructionGraph,
    index: &PlacementSnapIndex,
    hit: SurfaceHit,
    gridded: CylinderPlacementCandidate,
    grid: PlacementGrid,
    range: f32,
    mut valid: impl FnMut(CylinderPlacementCandidate) -> bool,
) -> (CylinderPlacementCandidate, Vec<SmartGuide>) {
    let supports = support_geometries_from_hit(graph, hit);
    let Some(support) = support_at_hit(&supports, hit.point) else {
        return (gridded, Vec::new());
    };
    let (normal_axis, _) = cardinal_axis(support.normal);
    let tangent_axes = match normal_axis {
        0 => [1, 2],
        1 => [0, 2],
        _ => [0, 1],
    };
    let (grid_minimum, grid_maximum) = part_world_bounds(PartSpec::Cylinder(gridded.spec));
    let cell_margin = smart_snap_cell_margin(grid);
    let targets = index.nearby(grid_minimum, grid_maximum, range + cell_margin);
    let choices = tangent_axes.map(|axis| {
        axis_guides(
            axis,
            grid_minimum,
            grid_maximum,
            &targets,
            SMART_SNAP_CAPTURE_METERS + cell_margin,
        )
    });
    for axis_guides in guide_combinations(&choices) {
        let mut ticks = gridded.spec.pose.translation_position_ticks();
        for (choice_index, guide) in axis_guides.iter().enumerate() {
            if let Some(guide) = guide {
                let axis = tangent_axes[choice_index];
                ticks[axis] += rounded_position_tick(guide.delta);
            }
        }
        let spec = CylinderSpec::new(
            gridded.spec.dimensions,
            BuildPose::from_position_ticks(ticks, gridded.spec.pose.rotation),
        )
        .with_material(gridded.spec.material)
        .with_appearance(gridded.spec.appearance);
        let candidate_face = cylinder_face_geometry(spec, gridded.attached_face)
            .expect("a cylinder placement uses a flat axial face");
        let candidate = CylinderPlacementCandidate {
            spec,
            attached_face: gridded.attached_face,
            anchor: supports
                .iter()
                .find_map(|support| overlap_center(support, &candidate_face))
                .or_else(|| supporting_face_overlap(graph, support, &candidate_face)),
            support: gridded.support,
        };
        if valid(candidate) {
            let center = candidate.spec.pose.translation();
            let rendered =
                render_smart_guides(axis_guides, &choices, tangent_axes, normal_axis, center);
            return (candidate, rendered);
        }
    }
    (gridded, Vec::new())
}

/// Three-axis object snapping for candidates that are not constrained to a face.
pub(crate) fn smart_snap_free_cuboid_candidate(
    index: &PlacementSnapIndex,
    gridded: PlacementCandidate,
    grid: PlacementGrid,
    range: f32,
    mut valid: impl FnMut(PlacementCandidate) -> bool,
) -> (PlacementCandidate, Vec<SmartGuide>) {
    let (minimum, maximum) = cuboid_world_bounds(gridded.spec);
    let cell_margin = smart_snap_cell_margin(grid);
    let targets = index.nearby(minimum, maximum, range + cell_margin);
    let choices = [0, 1, 2].map(|axis| {
        axis_guides(
            axis,
            minimum,
            maximum,
            &targets,
            SMART_SNAP_CAPTURE_METERS + cell_margin,
        )
    });
    for selected in guide_combinations_3(&choices) {
        let mut ticks = gridded.spec.pose.translation_position_ticks();
        for (axis, guide) in selected.iter().enumerate() {
            if let Some(guide) = guide {
                ticks[axis] += rounded_position_tick(guide.delta);
            }
        }
        let spec = CuboidSpec::new(
            gridded.spec.dimensions.map(GridDimension::units),
            BuildPose::from_position_ticks(ticks, gridded.spec.pose.rotation),
        )
        .expect("an aligned free candidate remains dimensionally valid")
        .with_material(gridded.spec.material)
        .with_appearance(gridded.spec.appearance);
        let candidate = PlacementCandidate { spec, ..gridded };
        if valid(candidate) {
            return (
                candidate,
                render_free_smart_guides(selected, &choices, spec.pose.translation()),
            );
        }
    }
    (gridded, Vec::new())
}

/// Cylinder counterpart of [`smart_snap_free_cuboid_candidate`].
pub(crate) fn smart_snap_free_cylinder_candidate(
    index: &PlacementSnapIndex,
    gridded: CylinderPlacementCandidate,
    grid: PlacementGrid,
    range: f32,
    mut valid: impl FnMut(CylinderPlacementCandidate) -> bool,
) -> (CylinderPlacementCandidate, Vec<SmartGuide>) {
    let (minimum, maximum) = part_world_bounds(PartSpec::Cylinder(gridded.spec));
    let cell_margin = smart_snap_cell_margin(grid);
    let targets = index.nearby(minimum, maximum, range + cell_margin);
    let choices = [0, 1, 2].map(|axis| {
        axis_guides(
            axis,
            minimum,
            maximum,
            &targets,
            SMART_SNAP_CAPTURE_METERS + cell_margin,
        )
    });
    for selected in guide_combinations_3(&choices) {
        let mut ticks = gridded.spec.pose.translation_position_ticks();
        for (axis, guide) in selected.iter().enumerate() {
            if let Some(guide) = guide {
                ticks[axis] += rounded_position_tick(guide.delta);
            }
        }
        let spec = CylinderSpec::new(
            gridded.spec.dimensions,
            BuildPose::from_position_ticks(ticks, gridded.spec.pose.rotation),
        )
        .with_material(gridded.spec.material)
        .with_appearance(gridded.spec.appearance);
        let candidate = CylinderPlacementCandidate { spec, ..gridded };
        if valid(candidate) {
            return (
                candidate,
                render_free_smart_guides(selected, &choices, spec.pose.translation()),
            );
        }
    }
    (gridded, Vec::new())
}

/// Point-sized smart snapping for bearing anchors constrained to a support face.
pub(crate) fn smart_snap_anchor(
    index: &PlacementSnapIndex,
    gridded: Vec3,
    normal_axis: usize,
    grid: PlacementGrid,
    range: f32,
    mut valid: impl FnMut(Vec3) -> bool,
) -> (Vec3, Vec<SmartGuide>) {
    let tangent_axes = match normal_axis {
        0 => [1, 2],
        1 => [0, 2],
        _ => [0, 1],
    };
    let cell_margin = smart_snap_cell_margin(grid);
    let targets = index.nearby(gridded, gridded, range + cell_margin);
    let choices = tangent_axes.map(|axis| {
        axis_guides(
            axis,
            gridded,
            gridded,
            &targets,
            SMART_SNAP_CAPTURE_METERS + cell_margin,
        )
    });
    for axis_guides in guide_combinations(&choices) {
        let mut anchor = gridded;
        for (choice_index, guide) in axis_guides.iter().enumerate() {
            if let Some(guide) = guide {
                let axis = tangent_axes[choice_index];
                anchor[axis] += guide.delta;
            }
        }
        anchor = (anchor / POSITION_TICK_METERS).round() * POSITION_TICK_METERS;
        anchor[normal_axis] = gridded[normal_axis];
        if valid(anchor) {
            return (
                anchor,
                render_smart_guides(axis_guides, &choices, tangent_axes, normal_axis, anchor),
            );
        }
    }
    (gridded, Vec::new())
}

/// Lets the opposite block of a plane drag acquire nearby object alignments.
///
/// The starting block remains sovereign: a guide may only select another whole
/// block span on its 25 cm lattice. Guides are then rebuilt from that final span
/// so their visibility cannot change while the selected plane stays unchanged.
pub(crate) fn smart_snap_block_span(
    index: &PlacementSnapIndex,
    start: CuboidSpec,
    plane: PlacementPlane,
    gridded_span: IVec3,
    pointer: Vec3,
    range: f32,
    mut valid: impl FnMut(IVec3) -> bool,
) -> (IVec3, Vec<SmartGuide>) {
    let block_size = f32::from(start.dimensions[0].units()) * GRID_UNIT_METERS;
    let targets = index.nearby(pointer, pointer, range);
    let tangent_axes = plane.tangent_axes();
    let choices = tangent_axes
        .map(|axis| block_endpoint_axis_guides(axis, start, pointer, block_size, &targets));

    for selected in block_endpoint_guide_combinations(&choices) {
        let mut span = gridded_span;
        for (choice_index, guide) in selected.iter().enumerate() {
            if let Some(guide) = guide {
                span[tangent_axes[choice_index]] = guide.span;
            }
        }
        if valid(span) {
            return (
                span,
                block_endpoint_guides(index, start, span, plane, range),
            );
        }
    }

    if valid(gridded_span) {
        (
            gridded_span,
            block_endpoint_guides(index, start, gridded_span, plane, range),
        )
    } else {
        (gridded_span, Vec::new())
    }
}

fn block_endpoint_axis_guides(
    axis: usize,
    start: CuboidSpec,
    pointer: Vec3,
    block_size: f32,
    targets: &[SnapTarget],
) -> Vec<BlockEndpointGuide> {
    let start_center = start.pose.translation()[axis];
    let mut choices = Vec::new();
    for target in targets {
        let target_center = (target.minimum + target.maximum) * 0.5;
        if let Some(span) = integral_block_span(start_center, target_center[axis], block_size) {
            push_block_endpoint_choice(
                &mut choices,
                span,
                pointer[axis],
                target_center[axis],
                GuideKind::Center,
                *target,
            );
        }
        for target_edge in [target.minimum[axis], target.maximum[axis]] {
            for direction in [-1.0, 1.0] {
                let endpoint_center = target_edge - direction * block_size * 0.5;
                let Some(span) = integral_block_span(start_center, endpoint_center, block_size)
                else {
                    continue;
                };
                if span != 0 && (span.is_positive() != direction.is_sign_positive()) {
                    continue;
                }
                push_block_endpoint_choice(
                    &mut choices,
                    span,
                    pointer[axis],
                    target_edge,
                    GuideKind::Edge,
                    *target,
                );
            }
        }
    }
    choices.sort_by(|left, right| {
        left.pointer_delta
            .abs()
            .total_cmp(&right.pointer_delta.abs())
            .then_with(|| left.guide.kind.cmp(&right.guide.kind))
            .then_with(|| left.guide.part.cmp(&right.guide.part))
            .then_with(|| left.span.cmp(&right.span))
    });
    let mut distinct = Vec::new();
    for choice in choices {
        if distinct
            .iter()
            .all(|existing: &BlockEndpointGuide| existing.span != choice.span)
        {
            distinct.push(choice);
        }
    }
    distinct
}

fn push_block_endpoint_choice(
    choices: &mut Vec<BlockEndpointGuide>,
    span: i32,
    pointer_coordinate: f32,
    guide_coordinate: f32,
    kind: GuideKind,
    target: SnapTarget,
) {
    let pointer_delta = guide_coordinate - pointer_coordinate;
    if pointer_delta.abs() > SMART_SNAP_CAPTURE_METERS {
        return;
    }
    choices.push(BlockEndpointGuide {
        span,
        pointer_delta,
        guide: AxisGuide {
            delta: pointer_delta,
            coordinate: guide_coordinate,
            kind,
            part: target.part,
            target_center: (target.minimum + target.maximum) * 0.5,
        },
    });
}

#[allow(clippy::cast_possible_truncation)]
fn integral_block_span(start: f32, endpoint: f32, block_size: f32) -> Option<i32> {
    let steps = ((endpoint - start) / block_size).round();
    let span = steps as i32;
    ((start + steps * block_size - endpoint).abs() <= POSITION_TICK_METERS * 0.5).then_some(span)
}

fn block_endpoint_guide_combinations(
    choices: &[Vec<BlockEndpointGuide>; 2],
) -> Vec<[Option<BlockEndpointGuide>; 2]> {
    let mut combinations = Vec::new();
    for first in choices[0].iter().copied().map(Some).chain([None]) {
        for second in choices[1].iter().copied().map(Some).chain([None]) {
            if first.is_none() && second.is_none() {
                continue;
            }
            let guides = [first, second];
            let displacement = guides
                .iter()
                .flatten()
                .map(|guide| guide.pointer_delta * guide.pointer_delta)
                .sum::<f32>()
                .sqrt();
            let center_count = guides
                .iter()
                .flatten()
                .filter(|guide| guide.guide.kind == GuideKind::Center)
                .count();
            let identities = guides.map(|guide| {
                guide.map_or((u32::MAX, u32::MAX), |guide| {
                    (guide.guide.part.index(), guide.guide.part.generation())
                })
            });
            combinations.push((guides, displacement, center_count, identities));
        }
    }
    combinations.sort_by(|left, right| {
        let left_count = left.0.iter().flatten().count();
        let right_count = right.0.iter().flatten().count();
        right_count
            .cmp(&left_count)
            .then_with(|| left.1.total_cmp(&right.1))
            .then_with(|| right.2.cmp(&left.2))
            .then_with(|| left.3.cmp(&right.3))
    });
    combinations
        .into_iter()
        .map(|(guides, _, _, _)| guides)
        .collect()
}

fn block_endpoint_guides(
    index: &PlacementSnapIndex,
    start: CuboidSpec,
    span: IVec3,
    plane: PlacementPlane,
    range: f32,
) -> Vec<SmartGuide> {
    let block_size = f32::from(start.dimensions[0].units()) * GRID_UNIT_METERS;
    let center = start.pose.translation() + span.as_vec3() * block_size;
    let half = Vec3::splat(block_size * 0.5);
    let targets = index.nearby(center - half, center + half, range);
    let tangent_axes = plane.tangent_axes();
    let normal_axis = plane.normal_axis();
    let mut rendered = Vec::new();

    for (choice_index, axis) in tangent_axes.into_iter().enumerate() {
        let line_axis = tangent_axes[1 - choice_index];
        let endpoint_edges: &[f32] = match span[axis].cmp(&0) {
            Ordering::Less => &[center[axis] - block_size * 0.5],
            Ordering::Greater => &[center[axis] + block_size * 0.5],
            Ordering::Equal => &[
                center[axis] - block_size * 0.5,
                center[axis] + block_size * 0.5,
            ],
        };
        for target in &targets {
            let target_center = (target.minimum + target.maximum) * 0.5;
            let mut coordinates = Vec::new();
            if (target_center[axis] - center[axis]).abs() <= POSITION_TICK_METERS * 0.5 {
                coordinates.push(target_center[axis]);
            }
            for endpoint_edge in endpoint_edges {
                for target_edge in [target.minimum[axis], target.maximum[axis]] {
                    if (target_edge - endpoint_edge).abs() <= POSITION_TICK_METERS * 0.5 {
                        coordinates.push(target_edge);
                    }
                }
            }
            for coordinate in coordinates {
                let mut from = center;
                let mut to = center;
                from[axis] = coordinate;
                to[axis] = coordinate;
                to[line_axis] = target_center[line_axis];
                to[normal_axis] = from[normal_axis];
                let guide = SmartGuide {
                    axis,
                    coordinate,
                    from,
                    to,
                };
                if !rendered.contains(&guide) {
                    rendered.push(guide);
                }
            }
        }
    }
    rendered
}

fn smart_snap_cell_margin(grid: PlacementGrid) -> f32 {
    grid.step_meters() * 0.5
}

fn guide_combinations(choices: &[Vec<AxisGuide>; 2]) -> Vec<[Option<AxisGuide>; 2]> {
    let mut combinations = Vec::new();
    for first in choices[0].iter().copied().map(Some).chain([None]) {
        for second in choices[1].iter().copied().map(Some).chain([None]) {
            if first.is_none() && second.is_none() {
                continue;
            }
            let guides = [first, second];
            let displacement = guides
                .iter()
                .flatten()
                .map(|guide| guide.delta * guide.delta)
                .sum::<f32>()
                .sqrt();
            let center_count = guides
                .iter()
                .flatten()
                .filter(|guide| guide.kind == GuideKind::Center)
                .count();
            let identities = guides.map(|guide| {
                guide.map_or((u32::MAX, u32::MAX), |guide| {
                    (guide.part.index(), guide.part.generation())
                })
            });
            combinations.push((guides, displacement, center_count, identities));
        }
    }
    combinations.sort_by(|left, right| {
        let left_count = left.0.iter().flatten().count();
        let right_count = right.0.iter().flatten().count();
        right_count
            .cmp(&left_count)
            .then_with(|| left.1.total_cmp(&right.1))
            .then_with(|| right.2.cmp(&left.2))
            .then_with(|| left.3.cmp(&right.3))
    });
    // A multi-block region can contribute the same guide once per block. Keep
    // the best-ranked guide for each resulting pose; rendering still receives
    // the original choices and can show every coincident alignment line.
    let mut positions = HashSet::new();
    combinations
        .into_iter()
        .filter(|(guides, _, _, _)| {
            positions.insert(
                guides.map(|guide| guide.map_or(0, |guide| rounded_position_tick(guide.delta))),
            )
        })
        .map(|(guides, _, _, _)| guides)
        .collect()
}

fn guide_combinations_3(choices: &[Vec<AxisGuide>; 3]) -> Vec<[Option<AxisGuide>; 3]> {
    let mut combinations = Vec::new();
    for x in choices[0].iter().copied().map(Some).chain([None]) {
        for y in choices[1].iter().copied().map(Some).chain([None]) {
            for z in choices[2].iter().copied().map(Some).chain([None]) {
                if x.is_none() && y.is_none() && z.is_none() {
                    continue;
                }
                let guides = [x, y, z];
                let displacement = guides
                    .iter()
                    .flatten()
                    .map(|guide| guide.delta * guide.delta)
                    .sum::<f32>()
                    .sqrt();
                let center_count = guides
                    .iter()
                    .flatten()
                    .filter(|guide| guide.kind == GuideKind::Center)
                    .count();
                let identities = guides.map(|guide| {
                    guide.map_or((u32::MAX, u32::MAX), |guide| {
                        (guide.part.index(), guide.part.generation())
                    })
                });
                combinations.push((guides, displacement, center_count, identities));
            }
        }
    }
    combinations.sort_by(|left, right| {
        let left_count = left.0.iter().flatten().count();
        let right_count = right.0.iter().flatten().count();
        right_count
            .cmp(&left_count)
            .then_with(|| left.1.total_cmp(&right.1))
            .then_with(|| right.2.cmp(&left.2))
            .then_with(|| left.3.cmp(&right.3))
    });
    // Free placement has the same duplicate-guide multiplier on three axes.
    let mut positions = HashSet::new();
    combinations
        .into_iter()
        .filter(|(guides, _, _, _)| {
            positions.insert(
                guides.map(|guide| guide.map_or(0, |guide| rounded_position_tick(guide.delta))),
            )
        })
        .map(|(guides, _, _, _)| guides)
        .collect()
}

fn render_free_smart_guides(
    selected: [Option<AxisGuide>; 3],
    choices: &[Vec<AxisGuide>; 3],
    center: Vec3,
) -> Vec<SmartGuide> {
    let mut rendered = Vec::new();
    for (axis, selected) in selected.iter().enumerate() {
        let Some(selected) = selected else {
            continue;
        };
        for guide in &choices[axis] {
            if guide.kind != selected.kind
                || (guide.delta - selected.delta).abs() > POSITION_TICK_METERS * 0.5
                || (guide.coordinate - selected.coordinate).abs() > POSITION_TICK_METERS * 0.5
            {
                continue;
            }
            let mut from = center;
            from[axis] = guide.coordinate;
            let mut to = guide.target_center;
            to[axis] = guide.coordinate;
            let delta = (to - from).abs();
            let varying_axes = [delta.x, delta.y, delta.z]
                .into_iter()
                .filter(|component| *component > POSITION_TICK_METERS * 0.5)
                .count();
            if varying_axes != 1 {
                continue;
            }
            rendered.push(SmartGuide {
                axis,
                coordinate: guide.coordinate,
                from,
                to,
            });
        }
    }
    rendered
}

fn render_smart_guides(
    selected: [Option<AxisGuide>; 2],
    choices: &[Vec<AxisGuide>; 2],
    tangent_axes: [usize; 2],
    normal_axis: usize,
    center: Vec3,
) -> Vec<SmartGuide> {
    let mut rendered = Vec::new();
    for (choice_index, selected) in selected.iter().enumerate() {
        let Some(selected) = selected else {
            continue;
        };
        let axis = tangent_axes[choice_index];
        let line_axis = tangent_axes[1 - choice_index];
        for guide in &choices[choice_index] {
            if guide.kind != selected.kind
                || (guide.delta - selected.delta).abs() > POSITION_TICK_METERS * 0.5
                || (guide.coordinate - selected.coordinate).abs() > POSITION_TICK_METERS * 0.5
            {
                continue;
            }
            let mut from = center;
            let mut to = center;
            from[axis] = guide.coordinate;
            to[axis] = guide.coordinate;
            to[line_axis] = guide.target_center[line_axis];
            to[normal_axis] = from[normal_axis];
            rendered.push(SmartGuide {
                axis,
                coordinate: guide.coordinate,
                from,
                to,
            });
        }
    }
    rendered
}

fn axis_guides(
    axis: usize,
    raw_minimum: Vec3,
    raw_maximum: Vec3,
    targets: &[SnapTarget],
    capture: f32,
) -> Vec<AxisGuide> {
    let raw_center = f32::midpoint(raw_minimum[axis], raw_maximum[axis]);
    let mut guides = Vec::new();
    for target in targets {
        let target_center = (target.minimum + target.maximum) * 0.5;
        let center_delta = target_center[axis] - raw_center;
        if center_delta.abs() <= capture {
            guides.push(AxisGuide {
                delta: center_delta,
                coordinate: target_center[axis],
                kind: GuideKind::Center,
                part: target.part,
                target_center,
            });
        }
        for (raw_edge, target_edge) in
            [raw_minimum[axis], raw_maximum[axis]]
                .into_iter()
                .flat_map(|raw_edge| {
                    [target.minimum[axis], target.maximum[axis]]
                        .into_iter()
                        .map(move |target_edge| (raw_edge, target_edge))
                })
        {
            let delta = target_edge - raw_edge;
            if delta.abs() <= capture {
                guides.push(AxisGuide {
                    delta,
                    coordinate: target_edge,
                    kind: GuideKind::Edge,
                    part: target.part,
                    target_center,
                });
            }
        }
    }
    guides.sort_by(|left, right| {
        left.delta
            .abs()
            .total_cmp(&right.delta.abs())
            .then_with(|| left.kind.cmp(&right.kind))
            .then_with(|| left.part.cmp(&right.part))
    });
    guides.dedup_by(|left, right| {
        (left.delta - right.delta).abs() <= POSITION_TICK_METERS * 0.5
            && left.kind == right.kind
            && left.part == right.part
    });
    guides
}

fn aabb_distance(first_min: Vec3, first_max: Vec3, second_min: Vec3, second_max: Vec3) -> f32 {
    let separation = (first_min - second_max)
        .max(second_min - first_max)
        .max(Vec3::ZERO);
    separation.length()
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

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum PipeRunAttachment<'a> {
    AutoWeld {
        source: FaceOwner,
    },
    Free,
    Linear(LinearAttachment<'a>),
    Bearing {
        source: FaceRef,
        anchor: Vec3,
        dimensions: BearingDimensions,
        rigid_targets: &'a [PartId],
    },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct PipeRunPiece {
    pub(crate) spec: PartSpec,
    pub(crate) inlet: FaceKind,
    pub(crate) outlet: FaceKind,
}

#[derive(Clone, Debug)]
pub(crate) struct FaceGeometry {
    pub(crate) center: Vec3,
    pub(crate) normal: Vec3,
    pub(crate) tangent_u: Vec3,
    pub(crate) tangent_v: Vec3,
    profile: FaceProfile,
}

#[derive(Clone, Debug)]
enum FaceProfile {
    Rectangle {
        half_u: f32,
        half_v: f32,
    },
    Annulus {
        inner_radius: f32,
        outer_radius: f32,
    },
    AnnularSector {
        inner_radius: f32,
        outer_radius: f32,
        half_angle: f32,
    },
    Polygon {
        vertices: Vec<Vec2>,
    },
    Ground,
}

pub(crate) fn raycast_construction(
    graph: &ConstructionGraph,
    origin: Vec3,
    direction: Vec3,
) -> Option<SurfaceHit> {
    raycast_construction_with_ground(graph, origin, direction, raycast_ground(origin, direction))
}

pub(crate) fn raycast_construction_with_ground(
    graph: &ConstructionGraph,
    origin: Vec3,
    direction: Vec3,
    ground: Option<SurfaceHit>,
) -> Option<SurfaceHit> {
    raycast_construction_filtered_with_ground(graph, origin, direction, ground, |_| true)
}

/// Restricts eligible parts before choosing a representative region or nearest hit.
pub(crate) fn raycast_construction_filtered_with_ground(
    graph: &ConstructionGraph,
    origin: Vec3,
    direction: Vec3,
    ground: Option<SurfaceHit>,
    accepts_part: impl Fn(PartId) -> bool,
) -> Option<SurfaceHit> {
    if !origin.is_finite() || !direction.is_finite() || direction.length_squared() < f32::EPSILON {
        return None;
    }
    let direction = direction.normalize();
    raycast_sources(graph, accepts_part)
        .filter_map(|(part, _, _)| raycast_part_in_construction(graph, part, origin, direction))
        .chain(ground)
        .filter(|hit| hit.distance >= 0.0 && hit.distance.is_finite())
        .min_by(|left, right| left.distance.total_cmp(&right.distance))
}

/// Exact authored surface for one part, including its shared region and frame.
pub(crate) fn raycast_part_in_construction(
    graph: &ConstructionGraph,
    part: PartId,
    origin: Vec3,
    direction: Vec3,
) -> Option<SurfaceHit> {
    if !origin.is_finite() || !direction.is_finite() || direction.length_squared() < f32::EPSILON {
        return None;
    }
    let direction = direction.normalize();
    let spec = *graph.part(part)?;
    let region = graph.region_of(part);
    if let Some(id) = region {
        let region = graph.region(id)?;
        if graph.owner_has_shape_features(mechanic_core::SolidOwner::Region(id)) {
            let solid = graph
                .evaluated_solid_shared(mechanic_core::SolidOwner::Region(id))
                .ok()?;
            return raycast_evaluated_solid(
                origin,
                direction,
                part,
                &solid,
                graph.part_frame(part)?,
            );
        }
        let frame = graph.part_frame(part)?;
        let inverse = frame.inverse();
        return raycast_region(
            inverse.point(origin),
            inverse.vector(direction),
            part,
            region,
        )
        .map(|hit| composed_surface_hit(hit, frame));
    }
    if graph.owner_has_shape_features(mechanic_core::SolidOwner::Part(part)) {
        let solid = graph
            .evaluated_solid_shared(mechanic_core::SolidOwner::Part(part))
            .ok()?;
        return raycast_evaluated_solid(origin, direction, part, &solid, graph.part_frame(part)?);
    }
    let frame = graph.part_frame(part)?;
    let inverse = frame.inverse();
    raycast_part(inverse.point(origin), inverse.vector(direction), part, spec)
        .map(|hit| composed_surface_hit(hit, frame))
}

fn composed_surface_hit(
    mut hit: SurfaceHit,
    frame: mechanic_core::ConstructionFrame,
) -> SurfaceHit {
    hit.point = frame.point(hit.point);
    // A rigid frame preserves ray distance and the identity of the local face.
    hit
}

/// One representative part for each region, plus every standalone part.
///
/// A region owns one shared surface even when hundreds of blocks fill it. The
/// representative part only supplies the legacy [`FaceOwner::Part`] returned
/// by picking; the region geometry itself must be tested exactly once.
fn raycast_sources<'a>(
    graph: &'a ConstructionGraph,
    accepts_part: impl Fn(PartId) -> bool + 'a,
) -> impl Iterator<Item = (PartId, PartSpec, Option<mechanic_core::RegionId>)> + 'a {
    let mut seen_regions = HashSet::new();
    graph.parts().filter_map(move |(part, spec)| {
        if !accepts_part(part) {
            return None;
        }
        let region = graph.region_of(part);
        if region.is_some_and(|region| !seen_regions.insert(region)) {
            return None;
        }
        Some((part, *spec, region))
    })
}

pub(crate) fn raycast_construction_for_annulus(
    graph: &ConstructionGraph,
    origin: Vec3,
    direction: Vec3,
    inner_diameter: f32,
    outer_diameter: f32,
) -> Option<SurfaceHit> {
    let ground = raycast_ground(origin, direction);
    raycast_construction_for_annulus_with_ground(
        graph,
        origin,
        direction,
        inner_diameter,
        outer_diameter,
        ground,
    )
}

pub(crate) fn raycast_construction_for_annulus_with_ground(
    graph: &ConstructionGraph,
    origin: Vec3,
    direction: Vec3,
    inner_diameter: f32,
    outer_diameter: f32,
    ground: Option<SurfaceHit>,
) -> Option<SurfaceHit> {
    raycast_construction_for_annulus_filtered_with_ground(
        graph,
        origin,
        direction,
        inner_diameter,
        outer_diameter,
        ground,
        |_| true,
    )
}

pub(crate) fn raycast_construction_for_annulus_filtered_with_ground(
    graph: &ConstructionGraph,
    origin: Vec3,
    direction: Vec3,
    inner_diameter: f32,
    outer_diameter: f32,
    ground: Option<SurfaceHit>,
    accepts_part: impl Fn(PartId) -> bool,
) -> Option<SurfaceHit> {
    if !origin.is_finite()
        || !direction.is_finite()
        || direction.length_squared() < f32::EPSILON
        || !inner_diameter.is_finite()
        || !outer_diameter.is_finite()
        || inner_diameter < 0.0
        || outer_diameter <= inner_diameter
    {
        return None;
    }
    let direction = direction.normalize();
    let placement_profile = FaceProfile::Annulus {
        inner_radius: inner_diameter * 0.5,
        outer_radius: outer_diameter * 0.5,
    };
    raycast_construction_filtered_with_ground(graph, origin, direction, ground, &accepts_part)
        .into_iter()
        .chain(
            graph
                .parts()
                .filter(|(part, _)| accepts_part(*part))
                .filter_map(|(part, spec)| match spec {
                    PartSpec::Cylinder(spec) => {
                        let frame = graph.part_frame(part)?;
                        let inverse = frame.inverse();
                        raycast_cylinder_bore_obstruction(
                            inverse.point(origin),
                            inverse.vector(direction),
                            part,
                            *spec,
                            &placement_profile,
                        )
                        .map(|hit| composed_surface_hit(hit, frame))
                    }
                    PartSpec::PipeBend(_)
                    | PartSpec::PipeJunction(_)
                    | PartSpec::Cuboid(_)
                    | PartSpec::Controller(_)
                    | PartSpec::Engine(_)
                    | PartSpec::Transmission(_)
                    | PartSpec::Servo(_)
                    | PartSpec::Seat(_)
                    | PartSpec::Input(_)
                    | PartSpec::DimensionLink(_) => None,
                }),
        )
        .min_by(|left, right| left.distance.total_cmp(&right.distance))
}

fn raycast_cylinder_bore_obstruction(
    origin: Vec3,
    direction: Vec3,
    part: PartId,
    spec: CylinderSpec,
    placement_profile: &FaceProfile,
) -> Option<SurfaceHit> {
    let rotation = spec.pose.rotation.quaternion();
    let inverse = rotation.inverse();
    let local_origin = inverse * (origin - spec.pose.translation());
    let local_direction = inverse * direction;
    if local_direction.y.abs() <= f32::EPSILON {
        return None;
    }
    let outer_radius = spec.dimensions.outer_diameter() * 0.5;
    let half_length = spec.dimensions.axial_length() * 0.5;
    [
        (half_length, FaceKind::PositiveY),
        (-half_length, FaceKind::NegativeY),
    ]
    .into_iter()
    .filter_map(|(y, face_kind)| {
        let distance = (y - local_origin.y) / local_direction.y;
        if distance < 0.0 {
            return None;
        }
        let local_point = local_origin + local_direction * distance;
        let radial_squared = local_point
            .x
            .mul_add(local_point.x, local_point.z * local_point.z);
        if radial_squared > (outer_radius + CONTACT_EPSILON).powi(2) {
            return None;
        }
        let support = cylinder_face_geometry(spec, face_kind)
            .expect("cylinder end faces always expose flat geometry");
        if point_in_profile(local_point.x, local_point.z, &support.profile) {
            return None;
        }
        let placement = FaceGeometry {
            center: origin + direction * distance,
            normal: -support.normal,
            tangent_u: support.tangent_u,
            tangent_v: support.tangent_v,
            profile: placement_profile.clone(),
        };
        profiles_overlap(&support, &placement).then_some(SurfaceHit {
            distance,
            point: placement.center,
            face: FaceRef::part(part, face_kind),
        })
    })
    .min_by(|left, right| left.distance.total_cmp(&right.distance))
}

pub(crate) fn candidate_from_hit(graph: &ConstructionGraph, hit: SurfaceHit) -> PlacementCandidate {
    candidate_from_hit_with_grid(
        graph,
        hit,
        PlacementGrid::Centimetres25,
        PlacementBounds::Garage,
    )
}

pub(crate) fn candidate_from_hit_with_grid(
    graph: &ConstructionGraph,
    hit: SurfaceHit,
    grid: PlacementGrid,
    bounds: PlacementBounds,
) -> PlacementCandidate {
    candidate_from_hit_with_grid_and_supports(graph, hit, grid, bounds).0
}

pub(crate) fn candidate_from_hit_with_grid_and_supports(
    graph: &ConstructionGraph,
    hit: SurfaceHit,
    grid: PlacementGrid,
    bounds: PlacementBounds,
) -> (PlacementCandidate, Vec<FaceGeometry>) {
    let supports = support_geometries_from_hit(graph, hit);
    let candidate = oriented_cuboid_candidate_from_supports(
        graph,
        hit,
        [BLOCK_SIZE_UNITS; 3],
        GridRotation::default(),
        grid,
        bounds,
        &supports,
    );
    (candidate, supports)
}

/// Places a fixed-size authored cuboid flush with the face under the pointer.
#[cfg(test)]
pub(crate) fn cuboid_candidate_from_hit(
    graph: &ConstructionGraph,
    hit: SurfaceHit,
    dimensions: [u8; 3],
) -> PlacementCandidate {
    oriented_cuboid_candidate_from_hit_with_grid(
        graph,
        hit,
        dimensions,
        GridRotation::default(),
        PlacementGrid::Centimetres25,
        PlacementBounds::Garage,
    )
}

/// Places a fixed-size cuboid with a grid-aligned orientation flush with a face.
#[cfg(test)]
pub(crate) fn oriented_cuboid_candidate_from_hit(
    graph: &ConstructionGraph,
    hit: SurfaceHit,
    dimensions: [u8; 3],
    rotation: GridRotation,
) -> PlacementCandidate {
    oriented_cuboid_candidate_from_hit_with_grid(
        graph,
        hit,
        dimensions,
        rotation,
        PlacementGrid::Centimetres25,
        PlacementBounds::Garage,
    )
}

pub(crate) fn oriented_cuboid_candidate_from_hit_with_grid(
    graph: &ConstructionGraph,
    hit: SurfaceHit,
    dimensions: [u8; 3],
    rotation: GridRotation,
    grid: PlacementGrid,
    bounds: PlacementBounds,
) -> PlacementCandidate {
    let supports = support_geometries_from_hit(graph, hit);
    oriented_cuboid_candidate_from_supports(
        graph, hit, dimensions, rotation, grid, bounds, &supports,
    )
}

fn oriented_cuboid_candidate_from_supports(
    _graph: &ConstructionGraph,
    hit: SurfaceHit,
    dimensions: [u8; 3],
    rotation: GridRotation,
    grid: PlacementGrid,
    bounds: PlacementBounds,
    supports: &[FaceGeometry],
) -> PlacementCandidate {
    let world_dimensions = oriented_grid_dimensions(dimensions, rotation);
    let support = support_at_hit(supports, hit.point)
        .expect("cuboid placement requires a flat support surface");
    let support_center_ticks = snap_world_to_position_ticks(support.center);
    let mut center_ticks = snap_global_center_ticks(
        snap_world_to_position_ticks(hit.point),
        world_dimensions,
        grid,
        bounds,
    );
    let (axis, sign) = cardinal_axis(support.normal);
    center_ticks[axis] = support_center_ticks[axis]
        + sign * i32::from(world_dimensions[axis]) * POSITION_TICKS_PER_HALF_GRID_UNIT;

    let spec = CuboidSpec::new(
        dimensions,
        BuildPose::from_position_ticks(center_ticks, rotation),
    )
    .expect("the fixed block size is a valid core dimension");
    let attached_face = face_for_normal(rotation.quaternion().inverse() * -support.normal);
    let candidate_face = face_geometry(spec, attached_face);
    let anchor = supports
        .iter()
        .find_map(|support| overlap_center(support, &candidate_face));
    PlacementCandidate {
        spec,
        attached_face,
        anchor,
        support: PlacementSupport::Surface(hit.face.owner),
    }
}

fn oriented_grid_dimensions(dimensions: [u8; 3], rotation: GridRotation) -> [u8; 3] {
    let mut world_dimensions = [0; 3];
    for (local_axis, direction) in [Vec3::X, Vec3::Y, Vec3::Z].into_iter().enumerate() {
        let (world_axis, _) = cardinal_axis(rotation.quaternion() * direction);
        world_dimensions[world_axis] = dimensions[local_axis];
    }
    world_dimensions
}

/// Builds a cuboid candidate around a point in empty Garage space.
pub(crate) fn free_cuboid_candidate(
    point: Vec3,
    view_direction: Vec3,
    dimensions: [u8; 3],
    rotation: GridRotation,
    grid: PlacementGrid,
    bounds: PlacementBounds,
) -> PlacementCandidate {
    let world_dimensions = oriented_grid_dimensions(dimensions, rotation);
    let center_ticks = snap_global_center_ticks(
        snap_world_to_position_ticks(point),
        world_dimensions,
        grid,
        bounds,
    );
    let spec = CuboidSpec::new(
        dimensions,
        BuildPose::from_position_ticks(center_ticks, rotation),
    )
    .expect("authored placement dimensions are valid");
    let view_normal = cardinal_direction(-view_direction);
    PlacementCandidate {
        spec,
        attached_face: face_for_normal(rotation.quaternion().inverse() * -view_normal),
        anchor: None,
        support: PlacementSupport::Free,
    }
}

/// Builds a cylinder candidate around a point with its axis facing the view.
pub(crate) fn free_cylinder_candidate(
    point: Vec3,
    view_direction: Vec3,
    dimensions: CylinderDimensions,
    grid: PlacementGrid,
    bounds: PlacementBounds,
) -> CylinderPlacementCandidate {
    let axis = cardinal_direction(-view_direction);
    let axial_axis = cardinal_axis(axis).0;
    let mut approximate_dimensions = [1; 3];
    approximate_dimensions[axial_axis] = dimensions.axial_length_units();
    let center_ticks = snap_global_center_ticks(
        snap_world_to_position_ticks(point),
        approximate_dimensions,
        grid,
        bounds,
    );
    let spec = CylinderSpec::new(
        dimensions,
        BuildPose::from_position_ticks(center_ticks, rotation_y_to_normal(axis)),
    );
    CylinderPlacementCandidate {
        spec,
        attached_face: FaceKind::NegativeY,
        anchor: None,
        support: PlacementSupport::Free,
    }
}

#[cfg(test)]
pub(crate) fn cylinder_candidate_from_hit(
    graph: &ConstructionGraph,
    hit: SurfaceHit,
    dimensions: CylinderDimensions,
) -> Result<CylinderPlacementCandidate, PlacementError> {
    cylinder_candidate_from_hit_with_grid(
        graph,
        hit,
        dimensions,
        PlacementGrid::Centimetres25,
        PlacementBounds::Garage,
    )
}

pub(crate) fn cylinder_candidate_from_hit_with_grid(
    graph: &ConstructionGraph,
    hit: SurfaceHit,
    dimensions: CylinderDimensions,
    grid: PlacementGrid,
    bounds: PlacementBounds,
) -> Result<CylinderPlacementCandidate, PlacementError> {
    let supports = support_geometries_from_hit(graph, hit);
    let support = support_at_hit(&supports, hit.point).ok_or(PlacementError::CurvedSurface)?;
    if let FaceOwner::Part(part) = hit.face.owner
        && matches!(graph.part(part), Some(PartSpec::PipeJunction(_)))
        && (hit.point - support.center).dot(support.normal).abs() > CONTACT_EPSILON
    {
        // Only an arm's flat end continues a pipe; its walls take branches.
        return Err(PlacementError::CurvedSurface);
    }
    let support_center_ticks = snap_world_to_position_ticks(support.center);
    let axial_axis = cardinal_axis(support.normal).0;
    let mut approximate_dimensions = [1; 3];
    approximate_dimensions[axial_axis] = dimensions.axial_length_units();
    let mut center_ticks = snap_global_center_ticks(
        snap_world_to_position_ticks(hit.point),
        approximate_dimensions,
        grid,
        bounds,
    );
    let (axis, sign) = cardinal_axis(support.normal);
    center_ticks[axis] = support_center_ticks[axis]
        + sign * i32::from(dimensions.axial_length_units()) * POSITION_TICKS_PER_HALF_GRID_UNIT;
    if let FaceOwner::Part(part) = hit.face.owner
        && matches!(graph.part(part), Some(PartSpec::PipeJunction(_)))
    {
        // A junction face only takes a pipe on its channel axis.
        for lateral in (0..3).filter(|&lateral| lateral != axis) {
            center_ticks[lateral] = support_center_ticks[lateral];
        }
    }
    let rotation = rotation_y_to_normal(support.normal);
    let spec = CylinderSpec::new(
        dimensions,
        BuildPose::from_position_ticks(center_ticks, rotation),
    );
    let attached_face = FaceKind::NegativeY;
    let candidate_face = cylinder_face_geometry(spec, attached_face)
        .expect("negative-y is a cylinder connection face");
    Ok(CylinderPlacementCandidate {
        spec,
        attached_face,
        anchor: supports
            .iter()
            .find_map(|support| overlap_center(support, &candidate_face))
            .or_else(|| supporting_face_overlap(graph, support, &candidate_face)),
        support: PlacementSupport::Surface(hit.face.owner),
    })
}

fn support_geometries_from_hit(graph: &ConstructionGraph, hit: SurfaceHit) -> Vec<FaceGeometry> {
    if matches!(hit.face.owner, FaceOwner::Ground) {
        let mut center = hit.point;
        center.y = (center.y / POSITION_TICK_METERS).floor() * POSITION_TICK_METERS;
        return vec![FaceGeometry {
            center,
            normal: Vec3::Y,
            tangent_u: Vec3::X,
            tangent_v: Vec3::Z,
            profile: FaceProfile::Ground,
        }];
    }
    try_face_geometries_from_ref(hit.face, Some(graph))
}

fn support_at_hit(supports: &[FaceGeometry], point: Vec3) -> Option<&FaceGeometry> {
    supports
        .iter()
        .find(|support| {
            let offset = point - support.center;
            offset.dot(support.normal).abs() <= CONTACT_EPSILON
                && point_in_profile(
                    offset.dot(support.tangent_u),
                    offset.dot(support.tangent_v),
                    &support.profile,
                )
        })
        .or_else(|| supports.first())
}

fn supporting_face_overlap(
    graph: &ConstructionGraph,
    selected: &FaceGeometry,
    candidate: &FaceGeometry,
) -> Option<Vec3> {
    overlap_center(selected, candidate).or_else(|| {
        graph.parts().find_map(|(_, spec)| {
            ALL_FACES.into_iter().find_map(|face| {
                part_face_geometry(*spec, face)
                    .and_then(|support| overlap_center(&support, candidate))
            })
        })
    })
}

#[cfg(test)]
pub(crate) fn stage_cuboid(
    graph: &ConstructionGraph,
    candidate: PlacementCandidate,
) -> Result<ConstructionGraph, PlacementError> {
    stage_block_batch(graph, candidate, &[candidate.spec])
}

#[cfg(test)]
pub(crate) fn stage_block_batch(
    graph: &ConstructionGraph,
    start: PlacementCandidate,
    specs: &[CuboidSpec],
) -> Result<ConstructionGraph, PlacementError> {
    stage_connected_block_batch(graph, start, specs, None, None, PlacementBounds::Garage)
}

pub(crate) fn stage_controller_in_bounds(
    graph: &ConstructionGraph,
    start: PlacementCandidate,
    bounds: PlacementBounds,
) -> Result<ConstructionGraph, PlacementError> {
    stage_connected_part_batch(
        graph,
        start,
        &[start.spec],
        None,
        start.support.auto_weld_source(),
        FixedPartSpawn::Controller,
        bounds,
    )
}

/// Stages one inert engine, auto-welding it like an ordinary block.
#[cfg(test)]
pub(crate) fn stage_engine_from_source(
    graph: &ConstructionGraph,
    start: PlacementCandidate,
    source: FaceOwner,
    kind: EngineKind,
) -> Result<ConstructionGraph, PlacementError> {
    stage_engine_from_source_in_bounds(graph, start, source, kind, PlacementBounds::Garage)
}

#[cfg(test)]
pub(crate) fn stage_engine_from_source_in_bounds(
    graph: &ConstructionGraph,
    start: PlacementCandidate,
    source: FaceOwner,
    kind: EngineKind,
    bounds: PlacementBounds,
) -> Result<ConstructionGraph, PlacementError> {
    stage_connected_part_batch(
        graph,
        start,
        &[start.spec],
        None,
        Some(source),
        FixedPartSpawn::Engine(kind),
        bounds,
    )
}

pub(crate) fn stage_engine_in_bounds(
    graph: &ConstructionGraph,
    start: PlacementCandidate,
    kind: EngineKind,
    bounds: PlacementBounds,
) -> Result<ConstructionGraph, PlacementError> {
    stage_connected_part_batch(
        graph,
        start,
        &[start.spec],
        None,
        start.support.auto_weld_source(),
        FixedPartSpawn::Engine(kind),
        bounds,
    )
}

/// Builds the only valid transmission candidate for a hovered output face.
#[cfg(test)]
pub(crate) fn transmission_candidate_from_hit(
    graph: &ConstructionGraph,
    hit: SurfaceHit,
) -> Result<(PartId, PlacementCandidate), PlacementError> {
    transmission_candidate_from_hit_in_bounds(graph, hit, PlacementBounds::Garage)
}

pub(crate) fn transmission_candidate_from_hit_in_bounds(
    graph: &ConstructionGraph,
    hit: SurfaceHit,
    bounds: PlacementBounds,
) -> Result<(PartId, PlacementCandidate), PlacementError> {
    let FaceOwner::Part(parent) = hit.face.owner else {
        return Err(PlacementError::TransmissionOutputOnly);
    };
    if hit.face.face != FaceKind::PositiveZ {
        return Err(PlacementError::TransmissionOutputOnly);
    }
    let spec = graph
        .next_transmission_spec(parent)
        .map_err(|error| PlacementError::Graph(error.to_string()))?;
    validate_part_in_bounds(graph, PartSpec::Transmission(spec), bounds)?;
    Ok((
        parent,
        PlacementCandidate {
            spec: spec.cuboid(),
            attached_face: FaceKind::NegativeZ,
            anchor: Some(hit.point),
            support: PlacementSupport::Surface(hit.face.owner),
        },
    ))
}

/// Stages one graph-owned transmission, including its required weld and parent relation.
pub(crate) fn stage_transmission(
    graph: &ConstructionGraph,
    parent: PartId,
    candidate: PlacementCandidate,
) -> Result<ConstructionGraph, PlacementError> {
    let mut staged = graph.begin_edit();
    staged
        .apply(BuildCommand::AttachTransmission {
            parent,
            spec: TransmissionSpec::new(candidate.spec.pose),
        })
        .map_err(|error| PlacementError::Graph(error.to_string()))?;
    Ok(staged.finish())
}

pub(crate) fn stage_servo_in_bounds(
    graph: &ConstructionGraph,
    start: PlacementCandidate,
    bounds: PlacementBounds,
) -> Result<ConstructionGraph, PlacementError> {
    stage_connected_part_batch(
        graph,
        start,
        &[start.spec],
        None,
        start.support.auto_weld_source(),
        FixedPartSpawn::Servo,
        bounds,
    )
}

pub(crate) fn stage_seat_in_bounds(
    graph: &ConstructionGraph,
    start: PlacementCandidate,
    bounds: PlacementBounds,
) -> Result<ConstructionGraph, PlacementError> {
    stage_connected_part_batch(
        graph,
        start,
        &[start.spec],
        None,
        start.support.auto_weld_source(),
        FixedPartSpawn::Seat,
        bounds,
    )
}

pub(crate) fn stage_input_in_bounds(
    graph: &ConstructionGraph,
    start: PlacementCandidate,
    bounds: PlacementBounds,
) -> Result<ConstructionGraph, PlacementError> {
    stage_connected_part_batch(
        graph,
        start,
        &[start.spec],
        None,
        start.support.auto_weld_source(),
        FixedPartSpawn::Input,
        bounds,
    )
}

pub(crate) fn stage_dimension_link_in_bounds(
    graph: &ConstructionGraph,
    start: PlacementCandidate,
    id: DimensionLinkId,
    bounds: PlacementBounds,
) -> Result<ConstructionGraph, PlacementError> {
    stage_connected_part_batch(
        graph,
        start,
        &[start.spec],
        None,
        start.support.auto_weld_source(),
        FixedPartSpawn::DimensionLink(id),
        bounds,
    )
}

#[cfg(test)]
pub(crate) fn stage_block_batch_from_source(
    graph: &ConstructionGraph,
    start: PlacementCandidate,
    specs: &[CuboidSpec],
    source: FaceOwner,
) -> Result<ConstructionGraph, PlacementError> {
    stage_block_batch_from_source_in_bounds(graph, start, specs, source, PlacementBounds::Garage)
}

#[cfg(test)]
pub(crate) fn stage_block_batch_from_source_in_bounds(
    graph: &ConstructionGraph,
    start: PlacementCandidate,
    specs: &[CuboidSpec],
    source: FaceOwner,
    bounds: PlacementBounds,
) -> Result<ConstructionGraph, PlacementError> {
    stage_connected_block_batch(graph, start, specs, None, Some(source), bounds)
}

#[cfg(test)]
pub(crate) fn stage_block_batch_in_bounds(
    graph: &ConstructionGraph,
    start: PlacementCandidate,
    specs: &[CuboidSpec],
    bounds: PlacementBounds,
) -> Result<ConstructionGraph, PlacementError> {
    stage_connected_block_batch(
        graph,
        start,
        specs,
        None,
        start.support.auto_weld_source(),
        bounds,
    )
}

#[cfg(test)]
pub(crate) fn stage_bearing_block_batch(
    graph: &ConstructionGraph,
    start: PlacementCandidate,
    specs: &[CuboidSpec],
    source: FaceRef,
    anchor: Vec3,
    dimensions: BearingDimensions,
    rigid_targets: &[PartId],
) -> Result<ConstructionGraph, PlacementError> {
    stage_bearing_block_batch_in_bounds(
        graph,
        start,
        specs,
        source,
        anchor,
        dimensions,
        rigid_targets,
        PlacementBounds::Garage,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn stage_bearing_block_batch_in_bounds(
    graph: &ConstructionGraph,
    start: PlacementCandidate,
    specs: &[CuboidSpec],
    source: FaceRef,
    anchor: Vec3,
    dimensions: BearingDimensions,
    rigid_targets: &[PartId],
    bounds: PlacementBounds,
) -> Result<ConstructionGraph, PlacementError> {
    stage_connected_block_batch(
        graph,
        start,
        specs,
        Some(BearingAttachment::rotational(
            graph,
            source,
            anchor,
            dimensions,
            rigid_targets,
        )),
        None,
        bounds,
    )
}

/// Radial slack for a hit to count as lying on a cylinder wall. Walls are
/// drawn and picked as facets whose chords sit slightly inside the true radius.
const LAYER_WALL_TOLERANCE_METERS: f32 = 0.01;
/// Slack for a hit to count as lying on a flat face or end cap.
const LAYER_FACE_TOLERANCE_METERS: f32 = 2.0e-3;
/// Ray-to-pull alignment beyond which a drag's projected distance is
/// ill-conditioned, so each frame may move only one step.
const LAYER_DRAG_STABILITY: f32 = 0.25;

/// One part taking a layer on one of its own faces.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct LayerMember {
    pub(crate) part: PartId,
    pub(crate) spec: PartSpec,
    pub(crate) face: mechanic_core::LayerFace,
}

/// A surface a new material layer grows from.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct LayerTarget {
    pub(crate) part: PartId,
    pub(crate) spec: PartSpec,
    pub(crate) face: mechanic_core::LayerFace,
    /// Construction frame the part is authored in.
    pub(crate) frame: mechanic_core::ConstructionFrame,
    /// Picked surface point in world space.
    pub(crate) anchor: Vec3,
    /// World direction a thicker layer grows toward.
    pub(crate) normal: Vec3,
    /// Every part the layer covers, the picked part first. A block face
    /// brings the whole flat surface it continues.
    pub(crate) members: Vec<LayerMember>,
}

/// A block face as an axis-aligned rectangle in its construction frame.
#[derive(Clone, Copy)]
struct FaceRect {
    plane: f32,
    minimum: [f32; 2],
    maximum: [f32; 2],
}

/// Slack for block faces to count as one plane or as sharing an edge.
const LAYER_PLANE_TOLERANCE_METERS: f32 = 1.0e-4;

impl FaceRect {
    /// The slab `reach` metres thick on this face, along world `axis`.
    fn slab(self, axis: usize, reach: f32) -> (Vec3, Vec3) {
        let mut low = Vec3::ZERO;
        let mut high = Vec3::ZERO;
        low[axis] = self.plane.min(self.plane + reach);
        high[axis] = self.plane.max(self.plane + reach);
        for (index, tangent) in [(axis + 1) % 3, (axis + 2) % 3].into_iter().enumerate() {
            low[tangent] = self.minimum[index];
            high[tangent] = self.maximum[index];
        }
        (low, high)
    }

    /// Whether two coplanar faces share an edge, not merely a corner.
    fn touches_edge(self, other: Self) -> bool {
        let overlap = [0, 1].map(|index| {
            self.maximum[index].min(other.maximum[index])
                - self.minimum[index].max(other.minimum[index])
        });
        overlap
            .iter()
            .all(|&length| length >= -LAYER_PLANE_TOLERANCE_METERS)
            && overlap
                .iter()
                .any(|&length| length > LAYER_PLANE_TOLERANCE_METERS)
    }
}

/// Index of the cardinal axis a unit direction points along.
fn cardinal_axis_index(direction: Vec3) -> usize {
    if direction.x.abs() > 0.5 {
        0
    } else if direction.y.abs() > 0.5 {
        1
    } else {
        2
    }
}

/// A block taking a layer on whichever of its faces points along `normal`.
fn block_layer_member(part: PartId, spec: CuboidSpec, normal: Vec3) -> LayerMember {
    let local = spec.pose.rotation.quaternion().inverse() * normal;
    let axis = cardinal_axis_index(local);
    LayerMember {
        part,
        spec: PartSpec::Cuboid(spec),
        face: mechanic_core::LayerFace::Face(face_kind_on_axis(axis, local[axis] > 0.0)),
    }
}

/// A part's rigid body: everything welded or rigidly linked to it.
fn rigid_body(graph: &ConstructionGraph, part: PartId) -> HashSet<PartId> {
    let mut neighbours = std::collections::HashMap::<PartId, Vec<PartId>>::new();
    let links = graph
        .welds()
        .filter_map(|(_, weld)| match (weld.first.owner, weld.second.owner) {
            (FaceOwner::Part(first), FaceOwner::Part(second)) => Some((first, second)),
            _ => None,
        })
        .chain(
            graph
                .rigid_links()
                .map(|(_, link)| (link.first, link.second)),
        );
    for (first, second) in links {
        neighbours.entry(first).or_default().push(second);
        neighbours.entry(second).or_default().push(first);
    }
    let mut body = HashSet::from([part]);
    let mut pending = vec![part];
    while let Some(current) = pending.pop() {
        for &next in neighbours.get(&current).into_iter().flatten() {
            if body.insert(next) {
                pending.push(next);
            }
        }
    }
    body
}

/// Parts outside `excluded` that may reach the box from `minimum` to
/// `maximum` in `frame`, each with its transform into that frame. Parts in
/// other frames are always kept, since their bounds are not comparable.
fn nearby_obstacles(
    graph: &ConstructionGraph,
    frame: mechanic_core::ConstructionFrame,
    excluded: &HashSet<PartId>,
    minimum: Vec3,
    maximum: Vec3,
) -> Vec<(PartSpec, mechanic_core::ConstructionFrame)> {
    let into_frame = frame.inverse();
    graph
        .parts()
        .filter(|(other, _)| !excluded.contains(other))
        .filter_map(|(other, spec)| {
            let other_frame = graph.part_frame(other)?;
            if other_frame == frame {
                let (low, high) = part_world_bounds(*spec);
                if (low - maximum).cmpgt(Vec3::splat(CONTACT_EPSILON)).any()
                    || (minimum - high).cmpgt(Vec3::splat(CONTACT_EPSILON)).any()
                {
                    return None;
                }
            }
            Some((*spec, into_frame.compose(other_frame)))
        })
        .collect()
}

/// Every block face continuing the picked block's flat face: coplanar, facing
/// the same way, joined edge to edge, on the same rigid body and construction
/// frame, and not covered by another part.
fn flat_surface_members(
    graph: &ConstructionGraph,
    part: PartId,
    cuboid: CuboidSpec,
    frame: mechanic_core::ConstructionFrame,
    local_normal: Vec3,
) -> Vec<LayerMember> {
    let normal = cuboid.pose.rotation.quaternion() * local_normal;
    let axis = cardinal_axis_index(normal);
    let positive = normal[axis] > 0.0;
    let tangents = [(axis + 1) % 3, (axis + 2) % 3];
    let rect = |spec: CuboidSpec| {
        let (minimum, maximum) = part_world_bounds(PartSpec::Cuboid(spec));
        FaceRect {
            plane: if positive {
                maximum[axis]
            } else {
                minimum[axis]
            },
            minimum: tangents.map(|tangent| minimum[tangent]),
            maximum: tangents.map(|tangent| maximum[tangent]),
        }
    };
    let member = |part, spec| block_layer_member(part, spec, normal);

    let body = rigid_body(graph, part);
    let start = rect(cuboid);
    let mut candidates = vec![(part, cuboid, start)];
    candidates.extend(graph.parts().filter_map(|(other, spec)| {
        let PartSpec::Cuboid(other_cuboid) = *spec else {
            return None;
        };
        let face = rect(other_cuboid);
        (other != part
            && body.contains(&other)
            && graph.region_of(other).is_none()
            && graph.part_frame(other) == Some(frame)
            && (face.plane - start.plane).abs() <= LAYER_PLANE_TOLERANCE_METERS)
            .then_some((other, other_cuboid, face))
    }));
    if candidates.len() == 1 {
        return vec![member(part, cuboid)];
    }

    // Parts that could sit on the surface: anything near its footprint.
    let reach = if positive {
        MIN_LAYER_COVER_METERS
    } else {
        -MIN_LAYER_COVER_METERS
    };
    let (footprint_minimum, footprint_maximum) = candidates.iter().fold(
        (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)),
        |(minimum, maximum), (_, _, face)| {
            let (low, high) = face.slab(axis, reach);
            (minimum.min(low), maximum.max(high))
        },
    );
    let candidate_parts = candidates
        .iter()
        .map(|(candidate, _, _)| *candidate)
        .collect::<HashSet<_>>();
    let obstacles = nearby_obstacles(
        graph,
        frame,
        &candidate_parts,
        footprint_minimum,
        footprint_maximum,
    );
    let covered = |candidate: LayerMember| {
        candidate
            .spec
            .with_layer(
                candidate.face,
                MIN_LAYER_COVER_METERS,
                mechanic_core::ConstructionMaterial::Steel,
                mechanic_core::MaterialAppearance::BAKED,
            )
            .is_ok_and(|layered| {
                obstacles.iter().any(|&(obstacle, relative)| {
                    parts_overlap_with_frame(layered, obstacle, relative)
                })
            })
    };

    let mut reached = vec![false; candidates.len()];
    reached[0] = true;
    let mut members = vec![member(part, cuboid)];
    let mut frontier = vec![0];
    while let Some(index) = frontier.pop() {
        let face = candidates[index].2;
        for (next, &(other, other_cuboid, other_face)) in candidates.iter().enumerate() {
            if reached[next] || !face.touches_edge(other_face) {
                continue;
            }
            reached[next] = true;
            let candidate = member(other, other_cuboid);
            if !covered(candidate) {
                members.push(candidate);
                frontier.push(next);
            }
        }
    }
    members
}

/// Thickness probing whether a flat surface block is covered.
const MIN_LAYER_COVER_METERS: f32 = mechanic_core::MIN_LAYER_THICKNESS_METERS;

const fn face_kind_on_axis(axis: usize, positive: bool) -> FaceKind {
    match (axis, positive) {
        (0, true) => FaceKind::PositiveX,
        (0, false) => FaceKind::NegativeX,
        (1, true) => FaceKind::PositiveY,
        (1, false) => FaceKind::NegativeY,
        (_, true) => FaceKind::PositiveZ,
        (_, false) => FaceKind::NegativeZ,
    }
}

/// Resolves a hit on a cuboid face, or on a full cylinder's outer wall, bore,
/// or end cap, into a layer target. Rounded and chamfered surfaces lie off
/// every such surface and take no layer.
pub(crate) fn layer_target_from_hit(
    graph: &ConstructionGraph,
    hit: SurfaceHit,
) -> Result<LayerTarget, PlacementError> {
    let FaceOwner::Part(part) = hit.face.owner else {
        return Err(PlacementError::NotLayerSurface);
    };
    if graph.region_of(part).is_some() {
        return Err(PlacementError::NotLayerSurface);
    }
    let spec = graph
        .part(part)
        .copied()
        .ok_or(PlacementError::NotLayerSurface)?;
    let frame = graph
        .part_frame(part)
        .ok_or(PlacementError::NotLayerSurface)?;
    let pose = spec.pose();
    let local = pose.rotation.quaternion().inverse()
        * (frame.inverse().point(hit.point) - pose.translation());
    let (face, local_normal) = match spec {
        PartSpec::Cuboid(cuboid) => {
            let half = cuboid.size_meters() * 0.5;
            let gap = |axis: usize| (half[axis] - local[axis].abs()).abs();
            let axis = (0..3)
                .min_by(|&first, &second| gap(first).total_cmp(&gap(second)))
                .expect("a cuboid has three axes");
            if gap(axis) > LAYER_FACE_TOLERANCE_METERS {
                return Err(PlacementError::NotLayerSurface);
            }
            let positive = local[axis] > 0.0;
            let mut normal = Vec3::ZERO;
            normal[axis] = if positive { 1.0 } else { -1.0 };
            (
                mechanic_core::LayerFace::Face(face_kind_on_axis(axis, positive)),
                normal,
            )
        }
        PartSpec::Cylinder(cylinder) if cylinder.dimensions.sweep_angle_degrees() == 360 => {
            let radius = Vec2::new(local.x, local.z).length();
            let outer = cylinder.dimensions.outer_diameter() * 0.5;
            let inner = cylinder.dimensions.inner_diameter() * 0.5;
            let half_length = cylinder.dimensions.axial_length() * 0.5;
            let wall_tolerance = LAYER_WALL_TOLERANCE_METERS + radius * 0.02;
            let radial = Vec3::new(local.x, 0.0, local.z).normalize_or_zero();
            let cap_gap = (local.y.abs() - half_length).abs();
            let wall_gap = (radius - outer).abs();
            let bore_gap = if inner > 0.0 {
                (radius - inner).abs()
            } else {
                f32::INFINITY
            };
            let within_length = local.y.abs() <= half_length + LAYER_FACE_TOLERANCE_METERS;
            if cap_gap <= LAYER_FACE_TOLERANCE_METERS
                && cap_gap <= wall_gap.min(bore_gap)
                && radius <= outer + wall_tolerance
                && radius + wall_tolerance >= inner
            {
                let positive = local.y > 0.0;
                (
                    mechanic_core::LayerFace::Face(face_kind_on_axis(1, positive)),
                    Vec3::Y * if positive { 1.0 } else { -1.0 },
                )
            } else if within_length && wall_gap <= wall_tolerance && wall_gap <= bore_gap {
                (mechanic_core::LayerFace::OuterWall, radial)
            } else if within_length && bore_gap <= wall_tolerance {
                (mechanic_core::LayerFace::Bore, -radial)
            } else {
                return Err(PlacementError::NotLayerSurface);
            }
        }
        _ => return Err(PlacementError::NotLayerSurface),
    };
    let members = match spec {
        PartSpec::Cuboid(cuboid) => flat_surface_members(graph, part, cuboid, frame, local_normal),
        _ => vec![LayerMember { part, spec, face }],
    };
    Ok(LayerTarget {
        part,
        spec,
        face,
        frame,
        anchor: hit.point,
        normal: frame
            .vector(pose.rotation.quaternion() * local_normal)
            .normalize_or_zero(),
        members,
    })
}

/// One press-and-drag choosing a new layer's thickness along the surface
/// normal, committed only on release.
///
/// Like a Shape amount drag, pointer motion is measured on a plane through
/// the anchor that contains the normal and faces the camera, and accumulates
/// relative to the press, so dragging out thickens and dragging back thins.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct LayerDrag {
    pub(crate) target: LayerTarget,
    pub(crate) material: mechanic_core::ConstructionMaterial,
    pub(crate) appearance: mechanic_core::MaterialAppearance,
    /// Snapped thickness shown now and committed on release, in metres.
    pub(crate) thickness: f32,
    drag_plane_normal: Vec3,
    last_drag_value: f32,
    raw_thickness: f32,
}

impl LayerDrag {
    pub(crate) fn begin(
        target: LayerTarget,
        material: mechanic_core::ConstructionMaterial,
        appearance: mechanic_core::MaterialAppearance,
        thickness: f32,
        ray_origin: Vec3,
        ray_direction: Vec3,
    ) -> Self {
        let pull = target.normal;
        let anchor = target.anchor;
        let drag_plane_normal = (ray_direction - pull * ray_direction.dot(pull))
            .try_normalize()
            .unwrap_or_else(|| pull.any_orthonormal_vector());
        let projected = crate::shape_tool::project_onto_plane(
            ray_origin,
            ray_direction,
            anchor,
            drag_plane_normal,
        )
        .unwrap_or(anchor);
        Self {
            target,
            material,
            appearance,
            thickness,
            drag_plane_normal,
            last_drag_value: (projected - anchor).dot(pull),
            raw_thickness: thickness,
        }
    }

    /// Follows the pointer ray and snaps to the active placement step, never
    /// thinner than one step.
    pub(crate) fn update(&mut self, grid: PlacementGrid, ray_origin: Vec3, ray_direction: Vec3) {
        let step = grid.step_meters();
        let pull = self.target.normal;
        if let Some(projected) = crate::shape_tool::project_onto_plane(
            ray_origin,
            ray_direction,
            self.target.anchor,
            self.drag_plane_normal,
        ) {
            let drag_value = (projected - self.target.anchor).dot(pull);
            let mut delta = drag_value - self.last_drag_value;
            self.last_drag_value = drag_value;
            let alignment = ray_direction.normalize_or_zero().dot(pull).abs();
            if 1.0 - alignment * alignment < LAYER_DRAG_STABILITY {
                delta = delta.clamp(-step, step);
            }
            self.raw_thickness = (self.raw_thickness + delta).max(0.0);
        }
        self.thickness = ((self.raw_thickness / step).round() * step).max(step);
    }

    /// Drops pointer distance past the thickest layer that fit, so holding
    /// beyond a neighbour does not retry the refused layer every frame.
    pub(crate) const fn discard_rejected_excess(&mut self, accepted: f32) {
        self.raw_thickness = accepted;
        self.thickness = accepted;
    }
}

/// Every member of the target with a new layer, the picked part first.
pub(crate) fn layered_parts(
    target: &LayerTarget,
    thickness: f32,
    material: mechanic_core::ConstructionMaterial,
    appearance: mechanic_core::MaterialAppearance,
) -> Result<Vec<(PartId, PartSpec)>, PlacementError> {
    target
        .members
        .iter()
        .map(|member| {
            member
                .spec
                .with_layer(member.face, thickness, material, appearance)
                .map(|spec| (member.part, spec))
                .map_err(|error| PlacementError::Graph(error.to_string()))
        })
        .collect()
}

/// Checks layered parts against bounds and every part outside the layer,
/// comparing in the target's construction frame.
pub(crate) fn validate_layered_parts(
    graph: &ConstructionGraph,
    target: &LayerTarget,
    layered: &[(PartId, PartSpec)],
    bounds: PlacementBounds,
) -> Result<(), PlacementError> {
    let mut minimum = Vec3::splat(f32::INFINITY);
    let mut maximum = Vec3::splat(f32::NEG_INFINITY);
    for &(_, spec) in layered {
        let (low, high) = part_world_bounds(spec);
        minimum = minimum.min(low);
        maximum = maximum.max(high);
    }
    if target.frame == mechanic_core::ConstructionFrame::IDENTITY {
        validate_world_bounds(minimum, maximum, bounds)?;
    }
    let into_layer = target.frame.inverse();
    for (part, existing) in graph.parts() {
        if layered.iter().any(|&(member, _)| member == part) {
            continue;
        }
        let existing_frame = graph
            .part_frame(part)
            .expect("validated parts have construction frames");
        if existing_frame == target.frame {
            // Most parts are nowhere near the layer; skip them cheaply.
            let (low, high) = part_world_bounds(*existing);
            if (low - maximum).cmpgt(Vec3::splat(CONTACT_EPSILON)).any()
                || (minimum - high).cmpgt(Vec3::splat(CONTACT_EPSILON)).any()
            {
                continue;
            }
        }
        let relative = into_layer.compose(existing_frame);
        if layered
            .iter()
            .any(|&(_, spec)| parts_overlap_with_frame(spec, *existing, relative))
        {
            return Err(PlacementError::OverlapsPart(part));
        }
    }
    Ok(())
}

/// Adds a layer to every target member in place in one edit, keeping their
/// connections.
pub(crate) fn stage_layer(
    graph: &ConstructionGraph,
    target: &LayerTarget,
    thickness: f32,
    material: mechanic_core::ConstructionMaterial,
    appearance: mechanic_core::MaterialAppearance,
    bounds: PlacementBounds,
) -> Result<(ConstructionGraph, Vec<(PartId, PartSpec)>), PlacementError> {
    let layered = layered_parts(target, thickness, material, appearance)?;
    validate_layered_parts(graph, target, &layered, bounds)?;
    let mut staged = graph.begin_edit();
    staged
        .apply_batch(
            layered
                .iter()
                .map(|&(part, spec)| BuildCommand::SetLayers { part, spec }),
        )
        .map_err(|error| PlacementError::Graph(error.to_string()))?;
    Ok((staged.finish(), layered))
}

pub(crate) fn validate_cylinder_candidate_in_bounds(
    graph: &ConstructionGraph,
    candidate: CylinderPlacementCandidate,
    bounds: PlacementBounds,
) -> Result<(), PlacementError> {
    if candidate.support != PlacementSupport::Free && candidate.anchor.is_none() {
        return Err(PlacementError::NoFaceOverlap);
    }
    validate_part_in_bounds(graph, PartSpec::Cylinder(candidate.spec), bounds)
}

#[allow(dead_code)] // Retained as the focused straight-cylinder staging seam used by regression tests.
pub(crate) fn stage_cylinder_from_source(
    graph: &ConstructionGraph,
    candidate: CylinderPlacementCandidate,
    source: FaceOwner,
) -> Result<ConstructionGraph, PlacementError> {
    stage_connected_cylinder(
        graph,
        candidate,
        None,
        Some(source),
        PlacementBounds::Garage,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn stage_bearing_cylinder_in_bounds(
    graph: &ConstructionGraph,
    candidate: CylinderPlacementCandidate,
    source: FaceRef,
    anchor: Vec3,
    dimensions: BearingDimensions,
    rigid_targets: &[PartId],
    bounds: PlacementBounds,
) -> Result<ConstructionGraph, PlacementError> {
    stage_connected_cylinder(
        graph,
        candidate,
        Some(BearingAttachment::rotational(
            graph,
            source,
            anchor,
            dimensions,
            rigid_targets,
        )),
        None,
        bounds,
    )
}

fn stage_connected_cylinder(
    graph: &ConstructionGraph,
    candidate: CylinderPlacementCandidate,
    bearing: Option<BearingAttachment<'_>>,
    auto_weld_source: Option<FaceOwner>,
    bounds: PlacementBounds,
) -> Result<ConstructionGraph, PlacementError> {
    validate_cylinder_candidate_in_bounds(graph, candidate, bounds)?;
    let existing_parts = graph.parts().map(|(part, _)| part).collect::<Vec<_>>();
    let weld_scope =
        auto_weld_source.and_then(|source| bearing_connected_weld_scope(graph, source));
    let mut staged = graph.begin_edit();
    let BuildOutcome::Spawned(part) = staged
        .apply(BuildCommand::SpawnCylinder(candidate.spec))
        .map_err(|error| PlacementError::Graph(error.to_string()))?
    else {
        unreachable!()
    };
    let mut connections = Vec::new();
    if let Some(BearingAttachment {
        source,
        anchor,
        dimensions,
        kind,
        axis,
        rigid_targets,
    }) = bearing
    {
        connections.push(BuildCommand::AddBearing(
            BearingSpec::new(
                source,
                FaceRef::part(part, candidate.attached_face),
                anchor,
                axis,
            )
            .with_dimensions(dimensions)
            .with_kind(kind),
        ));
        connections.extend(rigid_targets.iter().copied().map(|target| {
            BuildCommand::RigidLink(RigidLinkSpec {
                first: target,
                second: part,
            })
        }));
    } else {
        if weld_scope.is_none()
            && bounds == PlacementBounds::Garage
            && let Some((first, second)) =
                touching_face_pair(&staged, FaceOwner::Part(part), FaceOwner::Ground)
        {
            connections.push(BuildCommand::Weld(WeldSpec { first, second }));
        }
        let mut tested_owners = HashSet::new();
        for other in existing_parts {
            if weld_scope
                .as_ref()
                .is_some_and(|members| !members.contains(&other))
                || !tested_owners.insert(connection_geometry_owner(&staged, other))
            {
                continue;
            }
            if let Some((first, second)) =
                touching_face_pair(&staged, FaceOwner::Part(part), FaceOwner::Part(other))
            {
                connections.push(BuildCommand::Weld(WeldSpec { first, second }));
            }
        }
    }
    staged
        .apply_batch(connections)
        .map_err(|error| PlacementError::Graph(error.to_string()))?;
    Ok(staged.finish())
}

/// What joins two consecutive legs of a pipe run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PipeNode {
    /// A 90-degree bend filling a square of `span` blocks.
    Bend { span: u8 },
}

impl PipeNode {
    /// Blocks of leg this node occupies, counted from its square's edge.
    pub(crate) const fn footprint_blocks(self, _outer_diameter: f32) -> u8 {
        match self {
            Self::Bend { span } => span,
        }
    }
}

/// Splits a pipe path into straight cylinders, block-span bends, and junctions.
///
/// `points` are the run start, each node's centre corner, and the run end.
/// Every straight left between node faces must be a whole number of blocks,
/// which holds when corners sit at the middle of their pipe channel.
pub(crate) fn pipe_run_pieces(
    points: &[Vec3],
    nodes: &[PipeNode],
    dimensions: CylinderDimensions,
    material: mechanic_core::ConstructionMaterial,
) -> Result<Vec<PipeRunPiece>, PlacementError> {
    if dimensions.sweep_angle_degrees() != 360 && !nodes.is_empty() {
        return Err(PlacementError::PipeRun(
            "partial-cylinder sectors support straight runs only".to_owned(),
        ));
    }
    let (directions, lengths) = pipe_path_segments(points, nodes)?;
    let fittings = nodes
        .iter()
        .enumerate()
        .map(|(index, &node)| {
            pipe_node_piece(
                node,
                points[index + 1],
                directions[index],
                directions[index + 1],
                dimensions,
                material,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut pieces = Vec::new();
    for segment in 0..directions.len() {
        append_pipe_segment(
            &mut pieces,
            points,
            &directions,
            &lengths,
            &fittings,
            dimensions,
            material,
            segment,
        )?;
    }
    if pieces.is_empty() {
        return Err(PlacementError::PipeRun(
            "pipe run contains no material".to_owned(),
        ));
    }
    for first in 0..pieces.len() {
        for second in first + 2..pieces.len() {
            if parts_overlap(pieces[first].spec, pieces[second].spec) {
                return Err(PlacementError::PipeRun(format!(
                    "pipe run intersects itself between pieces {} and {}",
                    first + 1,
                    second + 1
                )));
            }
        }
    }
    Ok(pieces)
}

/// Builds one node's fitting and the distance it trims from each adjacent leg.
fn pipe_node_piece(
    node: PipeNode,
    corner: Vec3,
    incoming: Vec3,
    outgoing: Vec3,
    dimensions: CylinderDimensions,
    material: mechanic_core::ConstructionMaterial,
) -> Result<(f32, PipeRunPiece), PlacementError> {
    let corner_ticks = snap_world_to_position_ticks(corner);
    match node {
        PipeNode::Bend { span } => {
            let bend = PipeBendDimensions::new(
                dimensions.outer_diameter(),
                dimensions.inner_diameter(),
                span,
            )
            .map_err(|error| PlacementError::PipeRun(error.to_string()))?;
            let rotation = rotation_xy_to_directions(incoming, outgoing).ok_or_else(|| {
                PlacementError::PipeRun("pipe turn has no cardinal orientation".to_owned())
            })?;
            Ok((
                bend.radius(),
                PipeRunPiece {
                    spec: PartSpec::PipeBend(
                        PipeBendSpec::new(
                            bend,
                            BuildPose::from_position_ticks(corner_ticks, rotation),
                        )
                        .with_material(material),
                    ),
                    inlet: FaceKind::NegativeX,
                    outlet: FaceKind::PositiveY,
                },
            ))
        }
    }
}

/// Unrotated cube face whose outward normal points along a cardinal direction.
pub(crate) fn face_toward(direction: Vec3) -> FaceKind {
    let absolute = direction.abs();
    if absolute.x >= absolute.y && absolute.x >= absolute.z {
        if direction.x >= 0.0 {
            FaceKind::PositiveX
        } else {
            FaceKind::NegativeX
        }
    } else if absolute.y >= absolute.z {
        if direction.y >= 0.0 {
            FaceKind::PositiveY
        } else {
            FaceKind::NegativeY
        }
    } else if direction.z >= 0.0 {
        FaceKind::PositiveZ
    } else {
        FaceKind::NegativeZ
    }
}

fn pipe_path_segments(
    points: &[Vec3],
    nodes: &[PipeNode],
) -> Result<(Vec<Vec3>, Vec<f32>), PlacementError> {
    if points.len() < 2 || nodes.len() + 2 != points.len() {
        return Err(PlacementError::PipeRun(
            "pipe run path and fitting counts do not match".to_owned(),
        ));
    }
    let mut directions = Vec::with_capacity(points.len() - 1);
    let mut lengths = Vec::with_capacity(points.len() - 1);
    for segment in points.windows(2) {
        let delta = segment[1] - segment[0];
        let length = delta.length();
        if length <= CONTACT_EPSILON {
            return Err(PlacementError::PipeRun(
                "pipe legs must have positive length".to_owned(),
            ));
        }
        let direction = delta / length;
        if !is_cardinal(direction) {
            return Err(PlacementError::PipeRun(
                "pipe legs must be grid-aligned".to_owned(),
            ));
        }
        directions.push(snap_cardinal(direction));
        lengths.push(length);
    }
    for (index, pair) in directions.windows(2).enumerate() {
        if pair[0].dot(pair[1]).abs() > CONTACT_EPSILON {
            return Err(PlacementError::PipeRun(format!(
                "bend {} must turn exactly 90°",
                index + 1
            )));
        }
    }
    Ok((directions, lengths))
}

#[allow(clippy::too_many_arguments)]
fn append_pipe_segment(
    pieces: &mut Vec<PipeRunPiece>,
    points: &[Vec3],
    directions: &[Vec3],
    lengths: &[f32],
    fittings: &[(f32, PipeRunPiece)],
    dimensions: CylinderDimensions,
    material: mechanic_core::ConstructionMaterial,
    segment: usize,
) -> Result<(), PlacementError> {
    let start_trim = segment
        .checked_sub(1)
        .and_then(|node| fittings.get(node))
        .map_or(0.0, |fitting| fitting.0);
    let end_trim = fittings.get(segment).map_or(0.0, |fitting| fitting.0);
    let residual = lengths[segment] - start_trim - end_trim;
    if residual < -CONTACT_EPSILON {
        let required = start_trim + end_trim;
        return Err(PlacementError::PipeRun(format!(
            "leg {} needs {:.2} m clearance for adjacent fittings",
            segment + 1,
            required
        )));
    }
    let residual_blocks = residual / GRID_UNIT_METERS;
    if residual > CONTACT_EPSILON && (residual_blocks - residual_blocks.round()).abs() > 1.0e-3 {
        return Err(PlacementError::PipeRun(format!(
            "leg {} straight must be a whole number of blocks",
            segment + 1
        )));
    }
    if residual > CONTACT_EPSILON {
        let start = points[segment] + directions[segment] * start_trim;
        let end = points[segment + 1] - directions[segment] * end_trim;
        let pose = pose_for_axis_segment(start, end, directions[segment])?;
        let cylinder = CylinderSpec::new(
            CylinderDimensions::new(
                dimensions.outer_diameter(),
                dimensions.inner_diameter(),
                residual,
            )
            .map_err(|error| PlacementError::PipeRun(error.to_string()))?
            .with_sweep_angle_degrees(dimensions.sweep_angle_degrees())
            .map_err(|error| PlacementError::PipeRun(error.to_string()))?,
            pose,
        )
        .with_material(material);
        pieces.push(PipeRunPiece {
            spec: PartSpec::Cylinder(cylinder),
            inlet: FaceKind::NegativeY,
            outlet: FaceKind::PositiveY,
        });
    }
    if let Some(&(_, fitting)) = fittings.get(segment) {
        pieces.push(fitting);
    }
    Ok(())
}

pub(crate) fn validate_pipe_run_in_bounds(
    graph: &ConstructionGraph,
    pieces: &[PipeRunPiece],
    bounds: PlacementBounds,
) -> Result<(), PlacementError> {
    for piece in pieces {
        validate_part_in_bounds(graph, piece.spec, bounds)?;
    }
    let existing = graph
        .parts()
        .filter(|(part, _)| graph.region_of(*part).is_none())
        .map(|(_, part)| match part {
            PartSpec::Cylinder(_) => mechanic_core::CYLINDER_COLLIDER_COUNT,
            PartSpec::PipeBend(_) => mechanic_core::PIPE_BEND_COLLIDER_COUNT,
            PartSpec::PipeJunction(junction) => junction.collider_count(),
            _ => 1,
        })
        .sum::<usize>()
        + graph
            .regions()
            .map(|(_, region)| region_pieces(region).len())
            .sum::<usize>();
    let required = existing
        + pieces
            .iter()
            .map(|piece| match piece.spec {
                PartSpec::Cylinder(_) => mechanic_core::CYLINDER_COLLIDER_COUNT,
                PartSpec::PipeBend(_) => mechanic_core::PIPE_BEND_COLLIDER_COUNT,
                PartSpec::PipeJunction(junction) => junction.collider_count(),
                _ => 0,
            })
            .sum::<usize>();
    if required > mechanic_core::MAX_COMPILED_COLLIDERS {
        return Err(PlacementError::PipeRun(format!(
            "pipe run needs {required} colliders; maximum is {}",
            mechanic_core::MAX_COMPILED_COLLIDERS
        )));
    }
    Ok(())
}

#[cfg(test)]
pub(crate) fn stage_pipe_run(
    graph: &ConstructionGraph,
    pieces: &[PipeRunPiece],
    attachment: PipeRunAttachment<'_>,
) -> Result<ConstructionGraph, PlacementError> {
    stage_pipe_run_in_bounds(graph, pieces, attachment, PlacementBounds::Garage)
}

#[allow(clippy::too_many_lines)] // Validate and connect every pipe piece in one atomic transaction.
pub(crate) fn stage_pipe_run_in_bounds(
    graph: &ConstructionGraph,
    pieces: &[PipeRunPiece],
    attachment: PipeRunAttachment<'_>,
    bounds: PlacementBounds,
) -> Result<ConstructionGraph, PlacementError> {
    let bearing = match attachment {
        PipeRunAttachment::Bearing {
            source,
            anchor,
            dimensions,
            rigid_targets,
        } => Some(BearingAttachment::rotational(
            graph,
            source,
            anchor,
            dimensions,
            rigid_targets,
        )),
        PipeRunAttachment::Linear(linear) => {
            validate_linear_attachment(graph, linear)?;
            Some(BearingAttachment::from(linear))
        }
        PipeRunAttachment::Free | PipeRunAttachment::AutoWeld { .. } => None,
    };
    validate_pipe_run_in_bounds(graph, pieces, bounds)?;
    let existing_parts = graph.parts().map(|(part, _)| part).collect::<Vec<_>>();
    let weld_scope = match attachment {
        PipeRunAttachment::AutoWeld { source } => bearing_connected_weld_scope(graph, source),
        PipeRunAttachment::Free
        | PipeRunAttachment::Bearing { .. }
        | PipeRunAttachment::Linear(_) => None,
    };
    let mut staged = graph.begin_edit();
    let mut spawned = Vec::with_capacity(pieces.len());
    for piece in pieces {
        let command = match piece.spec {
            PartSpec::Cylinder(spec) => BuildCommand::SpawnCylinder(spec),
            PartSpec::PipeBend(spec) => BuildCommand::SpawnPipeBend(spec),
            PartSpec::PipeJunction(spec) => BuildCommand::SpawnPipeJunction(spec),
            _ => unreachable!("pipe runs contain only straights, bends, and junctions"),
        };
        let BuildOutcome::Spawned(part) = staged
            .apply(command)
            .map_err(|error| PlacementError::Graph(error.to_string()))?
        else {
            unreachable!()
        };
        spawned.push(part);
    }
    let mut connections = pieces
        .windows(2)
        .enumerate()
        .map(|(index, pair)| {
            BuildCommand::Weld(WeldSpec {
                first: FaceRef::part(spawned[index], pair[0].outlet),
                second: FaceRef::part(spawned[index + 1], pair[1].inlet),
            })
        })
        .collect::<Vec<_>>();
    if let Some(BearingAttachment {
        source,
        anchor,
        dimensions,
        kind,
        axis,
        rigid_targets,
    }) = bearing
    {
        let target = FaceRef::part(spawned[0], pieces[0].inlet);
        connections.push(BuildCommand::AddBearing(
            BearingSpec::new(source, target, anchor, axis)
                .with_dimensions(dimensions)
                .with_kind(kind),
        ));
        connections.extend(rigid_targets.iter().copied().map(|target| {
            BuildCommand::RigidLink(RigidLinkSpec {
                first: target,
                second: spawned[0],
            })
        }));
    } else {
        for &part in &spawned {
            if weld_scope.is_none()
                && bounds == PlacementBounds::Garage
                && let Some((first, second)) =
                    touching_face_pair(&staged, FaceOwner::Part(part), FaceOwner::Ground)
            {
                connections.push(BuildCommand::Weld(WeldSpec { first, second }));
            }
            let mut tested_owners = HashSet::new();
            for &other in &existing_parts {
                if weld_scope
                    .as_ref()
                    .is_some_and(|members| !members.contains(&other))
                    || !tested_owners.insert(connection_geometry_owner(&staged, other))
                {
                    continue;
                }
                if let Some((first, second)) =
                    touching_face_pair(&staged, FaceOwner::Part(part), FaceOwner::Part(other))
                {
                    connections.push(BuildCommand::Weld(WeldSpec { first, second }));
                }
            }
        }
    }
    staged
        .apply_batch(connections)
        .map_err(|error| PlacementError::Graph(error.to_string()))?;
    Ok(staged.finish())
}

/// Rebuilds a junction with one more open face, keeping every weld on it.
fn open_pipe_junction_arm(
    graph: &ConstructionGraph,
    part: PartId,
    face: FaceKind,
) -> Result<(ConstructionGraph, PartId), PlacementError> {
    let Some(PartSpec::PipeJunction(junction)) = graph.part(part).copied() else {
        return Err(PlacementError::PipeRun(
            "pipe junction is no longer available".to_owned(),
        ));
    };
    ensure_pipe_part_replaceable(graph, part)?;
    let welds = welds_on_part(graph, part);
    let mut staged = graph.begin_edit();
    staged
        .apply(BuildCommand::Remove(part))
        .map_err(|error| PlacementError::Graph(error.to_string()))?;
    let BuildOutcome::Spawned(opened) = staged
        .apply(BuildCommand::SpawnPipeJunction(junction.with_arm(face)))
        .map_err(|error| PlacementError::Graph(error.to_string()))?
    else {
        unreachable!("spawning a junction reports its part")
    };
    staged
        .apply_batch(
            welds
                .into_iter()
                .map(|(own, other)| {
                    BuildCommand::Weld(WeldSpec {
                        first: FaceRef::part(opened, own),
                        second: other,
                    })
                })
                .collect::<Vec<_>>(),
        )
        .map_err(|error| PlacementError::Graph(error.to_string()))?;
    Ok((staged.finish(), opened))
}

/// Each weld on `part` as its own face kind and the face it holds.
fn welds_on_part(graph: &ConstructionGraph, part: PartId) -> Vec<(FaceKind, FaceRef)> {
    let owner = FaceOwner::Part(part);
    graph
        .welds()
        .filter_map(|(_, weld)| {
            if weld.first.owner == owner {
                Some((weld.first.face, weld.second))
            } else if weld.second.owner == owner {
                Some((weld.second.face, weld.first))
            } else {
                None
            }
        })
        .collect()
}

/// Pipe parts can only be swapped for new pieces when welds are all that hold them.
fn ensure_pipe_part_replaceable(
    graph: &ConstructionGraph,
    part: PartId,
) -> Result<(), PlacementError> {
    let refuse = |why: &str| Err(PlacementError::PipeRun(why.to_owned()));
    if graph.region_of(part).is_some()
        || graph.part_frame(part) != Some(mechanic_core::ConstructionFrame::IDENTITY)
    {
        return refuse("pipes inside shaped regions or moving frames cannot branch");
    }
    if graph.owner_has_shape_features(mechanic_core::SolidOwner::Part(part)) {
        return refuse("shaped pipes cannot branch");
    }
    let owner = FaceOwner::Part(part);
    if graph
        .bearings()
        .any(|(_, bearing)| bearing.source.owner == owner || bearing.target.owner == owner)
        || graph
            .rigid_links()
            .any(|(_, link)| link.first == part || link.second == part)
    {
        return refuse("pipes with bearings or rigid links cannot branch");
    }
    Ok(())
}

/// Where a branch leaves an existing pipe part.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PipeBranchSite {
    /// Cut a tee into this straight pipe.
    Split(PartId),
    /// Open another arm on this junction.
    Extend(PartId),
}

impl PipeBranchSite {
    /// Part the branch replaces.
    pub(crate) const fn part(self) -> PartId {
        match self {
            Self::Split(part) | Self::Extend(part) => part,
        }
    }
}

/// A junction planned where a new pipe branches off an existing one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct PipeBranch {
    pub(crate) site: PipeBranchSite,
    /// The junction once branched, with the new arm open.
    pub(crate) junction: PipeJunctionSpec,
}

/// Plans a branch where `hit` lands on the side of a straight pipe or a
/// junction, and the new pipe leaving it.
///
/// A straight pipe gets a tee on the channel cell nearest the hit. The new arm
/// takes the free direction facing `toward` most; each `turn` steps it on,
/// around the pipe for a tee, or to the next best facing free face.
pub(crate) fn pipe_branch_candidate(
    graph: &ConstructionGraph,
    hit: SurfaceHit,
    dimensions: CylinderDimensions,
    toward: Vec3,
    turn: u8,
) -> Result<(CylinderPlacementCandidate, PipeBranch), PlacementError> {
    let FaceOwner::Part(part) = hit.face.owner else {
        return Err(PlacementError::CurvedSurface);
    };
    let (site, base, free) = match graph.part(part).copied() {
        Some(PartSpec::Cylinder(cylinder)) => {
            let (tee, axis) = tee_on_pipe(cylinder, hit.point)?;
            let best = [
                Vec3::X,
                Vec3::NEG_X,
                Vec3::Y,
                Vec3::NEG_Y,
                Vec3::Z,
                Vec3::NEG_Z,
            ]
            .into_iter()
            .filter(|direction| direction.dot(axis).abs() < 0.5)
            .max_by(|left, right| left.dot(toward).total_cmp(&right.dot(toward)))
            .expect("three axes leave four perpendicular directions");
            let side = snap_cardinal(axis.cross(best));
            (
                PipeBranchSite::Split(part),
                tee,
                vec![best, side, -best, -side],
            )
        }
        Some(PartSpec::PipeJunction(junction)) => {
            let rotation = junction.pose.rotation.quaternion();
            let mut free = ALL_FACES
                .into_iter()
                .filter(|&face| !junction.arms.contains(face))
                .map(|face| snap_cardinal(rotation * face_normal(face)))
                .collect::<Vec<_>>();
            free.sort_by(|left, right| right.dot(toward).total_cmp(&left.dot(toward)));
            (PipeBranchSite::Extend(part), junction, free)
        }
        _ => return Err(PlacementError::CurvedSurface),
    };
    ensure_pipe_part_replaceable(graph, part)?;
    if free.is_empty() {
        return Err(PlacementError::PipeRun(
            "every face of this junction already has an arm".to_owned(),
        ));
    }
    let direction = free[usize::from(turn) % free.len()];
    let junction = base.with_arm(face_toward(
        base.pose.rotation.quaternion().inverse() * direction,
    ));
    let rotation = rotation_y_to_direction(direction).ok_or(PlacementError::CurvedSurface)?;
    // The branch keeps the junction's cross-section so it fits the new arm.
    let dimensions = CylinderDimensions::new(
        junction.dimensions.outer_diameter(),
        junction.dimensions.inner_diameter(),
        dimensions.axial_length(),
    )
    .map_err(|error| PlacementError::PipeRun(error.to_string()))?;
    let arm_end = junction.pose.translation() + direction * junction.dimensions.half_side();
    let spec = CylinderSpec::new(
        dimensions,
        BuildPose::from_position_ticks(
            snap_world_to_position_ticks(arm_end + direction * dimensions.axial_length() * 0.5),
            rotation,
        ),
    );
    Ok((
        CylinderPlacementCandidate {
            spec,
            attached_face: FaceKind::NegativeY,
            anchor: Some(arm_end),
            support: PlacementSupport::Surface(FaceOwner::Part(part)),
        },
        PipeBranch { site, junction },
    ))
}

/// The tee a straight pipe takes on the channel cell nearest `point`, with
/// its two axial arms open, and the pipe's axis.
fn tee_on_pipe(
    cylinder: CylinderSpec,
    point: Vec3,
) -> Result<(PipeJunctionSpec, Vec3), PlacementError> {
    if cylinder.dimensions.sweep_angle_degrees() != 360 {
        return Err(PlacementError::PipeRun(
            "only full pipes can branch".to_owned(),
        ));
    }
    let axis = snap_cardinal(cylinder.pose.rotation.quaternion() * Vec3::Y);
    let length = cylinder.dimensions.axial_length();
    let start = cylinder.pose.translation() - axis * length * 0.5;
    let offset = point - start;
    let radial = offset - axis * offset.dot(axis);
    if radial.length() < cylinder.dimensions.outer_diameter() * 0.25 {
        return Err(PlacementError::CurvedSurface);
    }
    let dimensions = PipeJunctionDimensions::new(
        cylinder.dimensions.outer_diameter(),
        cylinder.dimensions.inner_diameter(),
    )
    .map_err(|error| PlacementError::PipeRun(error.to_string()))?;
    let half = dimensions.half_side();
    let free_cells = ((length - 2.0 * half) / GRID_UNIT_METERS + 1.0e-3).floor();
    if free_cells < 0.0 {
        return Err(PlacementError::PipeRun(
            "pipe is too short for a tee".to_owned(),
        ));
    }
    let cell = ((offset.dot(axis) - half) / GRID_UNIT_METERS)
        .round()
        .clamp(0.0, free_cells);
    let center = start + axis * (half + cell * GRID_UNIT_METERS);
    let tee = PipeJunctionSpec::new(
        dimensions,
        PipeArms::single(face_toward(-axis)).with(face_toward(axis)),
        BuildPose::from_position_ticks(
            snap_world_to_position_ticks(center),
            GridRotation::default(),
        ),
    )
    .with_material(cylinder.material)
    .with_appearance(cylinder.appearance);
    Ok((tee, axis))
}

/// Makes a planned branch's junction, keeping the welds already on the part
/// it replaces, and returns the graph with the junction's part.
pub(crate) fn apply_pipe_branch(
    graph: &ConstructionGraph,
    branch: PipeBranch,
) -> Result<(ConstructionGraph, PartId), PlacementError> {
    match branch.site {
        PipeBranchSite::Split(pipe) => split_pipe_for_branch(graph, pipe, branch.junction),
        PipeBranchSite::Extend(junction) => {
            let Some(PartSpec::PipeJunction(current)) = graph.part(junction).copied() else {
                return Err(PlacementError::PipeRun(
                    "pipe junction is no longer available".to_owned(),
                ));
            };
            let face = ALL_FACES
                .into_iter()
                .find(|&face| branch.junction.arms.contains(face) && !current.arms.contains(face))
                .ok_or_else(|| {
                    PlacementError::PipeRun("the junction already has that arm".to_owned())
                })?;
            open_pipe_junction_arm(graph, junction, face)
        }
    }
}

/// Replaces a straight pipe with up to two shorter straights around its tee,
/// moving each end weld onto whichever piece now carries that end.
fn split_pipe_for_branch(
    graph: &ConstructionGraph,
    pipe: PartId,
    tee: PipeJunctionSpec,
) -> Result<(ConstructionGraph, PartId), PlacementError> {
    let Some(PartSpec::Cylinder(cylinder)) = graph.part(pipe).copied() else {
        return Err(PlacementError::PipeRun(
            "branched pipe is no longer available".to_owned(),
        ));
    };
    ensure_pipe_part_replaceable(graph, pipe)?;
    let axis = cylinder.pose.rotation.quaternion() * Vec3::Y;
    let length = cylinder.dimensions.axial_length();
    let start = cylinder.pose.translation() - axis * length * 0.5;
    let half = tee.dimensions.half_side();
    let along = (tee.pose.translation() - start).dot(axis);
    if along - half < -CONTACT_EPSILON || length - along - half < -CONTACT_EPSILON {
        return Err(PlacementError::PipeRun(
            "the tee no longer fits on this pipe".to_owned(),
        ));
    }
    let welds = welds_on_part(graph, pipe);
    if welds
        .iter()
        .any(|(face, _)| !matches!(face, FaceKind::NegativeY | FaceKind::PositiveY))
    {
        return Err(PlacementError::PipeRun(
            "pipes welded along their side cannot branch".to_owned(),
        ));
    }
    let graph_error = |error: mechanic_core::GraphError| PlacementError::Graph(error.to_string());
    let mut staged = graph.begin_edit();
    staged
        .apply(BuildCommand::Remove(pipe))
        .map_err(graph_error)?;
    let mut spawn_straight = |from: f32, to: f32| -> Result<Option<PartId>, PlacementError> {
        if to - from <= CONTACT_EPSILON {
            return Ok(None);
        }
        let dimensions = CylinderDimensions::new(
            cylinder.dimensions.outer_diameter(),
            cylinder.dimensions.inner_diameter(),
            to - from,
        )
        .map_err(|error| PlacementError::PipeRun(error.to_string()))?;
        let spec = CylinderSpec::new(
            dimensions,
            BuildPose::from_position_ticks(
                snap_world_to_position_ticks(start + axis * ((from + to) * 0.5)),
                cylinder.pose.rotation,
            ),
        )
        .with_material(cylinder.material)
        .with_appearance(cylinder.appearance);
        match staged
            .apply(BuildCommand::SpawnCylinder(spec))
            .map_err(graph_error)?
        {
            BuildOutcome::Spawned(part) => Ok(Some(part)),
            _ => unreachable!("spawning a cylinder reports its part"),
        }
    };
    let before = spawn_straight(0.0, along - half)?;
    let after = spawn_straight(along + half, length)?;
    let BuildOutcome::Spawned(junction) = staged
        .apply(BuildCommand::SpawnPipeJunction(tee))
        .map_err(graph_error)?
    else {
        unreachable!("spawning a junction reports its part")
    };
    let (back, ahead) = (face_toward(-axis), face_toward(axis));
    let start_face = before.map_or(FaceRef::part(junction, back), |part| {
        FaceRef::part(part, FaceKind::NegativeY)
    });
    let end_face = after.map_or(FaceRef::part(junction, ahead), |part| {
        FaceRef::part(part, FaceKind::PositiveY)
    });
    let mut connections = welds
        .into_iter()
        .map(|(face, other)| {
            BuildCommand::Weld(WeldSpec {
                first: if face == FaceKind::NegativeY {
                    start_face
                } else {
                    end_face
                },
                second: other,
            })
        })
        .collect::<Vec<_>>();
    if let Some(before) = before {
        connections.push(BuildCommand::Weld(WeldSpec {
            first: FaceRef::part(before, FaceKind::PositiveY),
            second: FaceRef::part(junction, back),
        }));
    }
    if let Some(after) = after {
        connections.push(BuildCommand::Weld(WeldSpec {
            first: FaceRef::part(junction, ahead),
            second: FaceRef::part(after, FaceKind::NegativeY),
        }));
    }
    staged.apply_batch(connections).map_err(graph_error)?;
    Ok((staged.finish(), junction))
}

fn is_cardinal(direction: Vec3) -> bool {
    let absolute = direction.abs();
    (absolute.max_element() - 1.0).abs() <= 1.0e-4
        && absolute.min_element() <= 1.0e-4
        && (absolute.x + absolute.y + absolute.z - 1.0).abs() <= 1.0e-4
}

fn pose_for_axis_segment(
    start: Vec3,
    end: Vec3,
    direction: Vec3,
) -> Result<BuildPose, PlacementError> {
    let rotation = rotation_y_to_direction(direction).ok_or_else(|| {
        PlacementError::PipeRun("pipe leg has no cardinal orientation".to_owned())
    })?;
    let center_ticks = ((start + end) * 0.5 / POSITION_TICK_METERS)
        .round()
        .as_ivec3();
    Ok(BuildPose::from_position_ticks(center_ticks, rotation))
}

fn rotation_y_to_direction(direction: Vec3) -> Option<GridRotation> {
    cardinal_rotations()
        .find(|rotation| (rotation.quaternion() * Vec3::Y).abs_diff_eq(direction, 1.0e-4))
}

fn rotation_xy_to_directions(incoming: Vec3, outgoing: Vec3) -> Option<GridRotation> {
    cardinal_rotations().find(|rotation| {
        (rotation.quaternion() * Vec3::X).abs_diff_eq(incoming, 1.0e-4)
            && (rotation.quaternion() * Vec3::Y).abs_diff_eq(outgoing, 1.0e-4)
    })
}

fn cardinal_rotations() -> impl Iterator<Item = GridRotation> {
    (0_u8..4).flat_map(|x| {
        (0_u8..4).flat_map(move |y| (0_u8..4).map(move |z| GridRotation::new(x, y, z)))
    })
}

fn stage_connected_block_batch(
    graph: &ConstructionGraph,
    start: PlacementCandidate,
    specs: &[CuboidSpec],
    bearing: Option<BearingAttachment<'_>>,
    auto_weld_source: Option<FaceOwner>,
    bounds: PlacementBounds,
) -> Result<ConstructionGraph, PlacementError> {
    stage_connected_part_batch(
        graph,
        start,
        specs,
        bearing,
        auto_weld_source,
        FixedPartSpawn::Cuboid,
        bounds,
    )
}

#[derive(Clone, Copy)]
struct BearingAttachment<'a> {
    source: FaceRef,
    anchor: Vec3,
    dimensions: BearingDimensions,
    kind: BearingKind,
    axis: Vec3,
    rigid_targets: &'a [PartId],
}

impl<'a> BearingAttachment<'a> {
    fn rotational(
        graph: &ConstructionGraph,
        source: FaceRef,
        anchor: Vec3,
        dimensions: BearingDimensions,
        rigid_targets: &'a [PartId],
    ) -> Self {
        Self {
            source,
            anchor,
            dimensions,
            kind: BearingKind::Rotational,
            axis: face_geometry_from_ref(source, Some(graph)).normal,
            rigid_targets,
        }
    }
}

/// A rail socket and any existing direct attachments on its occupied face.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct LinearAttachment<'a> {
    pub(crate) source: FaceRef,
    pub(crate) anchor: Vec3,
    pub(crate) rail: LinearBearing,
    pub(crate) axis: Vec3,
    pub(crate) rigid_targets: &'a [PartId],
}

impl<'a> From<LinearAttachment<'a>> for BearingAttachment<'a> {
    fn from(attachment: LinearAttachment<'a>) -> Self {
        Self {
            source: attachment.source,
            anchor: attachment.anchor,
            dimensions: BearingDimensions::default(),
            kind: BearingKind::Linear(attachment.rail),
            axis: attachment.axis,
            rigid_targets: attachment.rigid_targets,
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn stage_block_volume_in_bounds(
    graph: &ConstructionGraph,
    index: &PlacementSnapIndex,
    start: PlacementCandidate,
    volume: BlockVolume,
    bearing: Option<(FaceRef, Vec3, BearingDimensions, &[PartId])>,
    auto_weld_source: Option<FaceOwner>,
    bounds: PlacementBounds,
    publication_generation: u64,
) -> Result<BlockVolumePlacement, PlacementError> {
    stage_connected_block_volume_in_bounds(
        graph,
        index,
        start,
        volume,
        bearing.map(|(source, anchor, dimensions, targets)| {
            BearingAttachment::rotational(graph, source, anchor, dimensions, targets)
        }),
        auto_weld_source,
        bounds,
        publication_generation,
    )
}

fn validate_linear_attachment(
    graph: &ConstructionGraph,
    attachment: LinearAttachment<'_>,
) -> Result<(), PlacementError> {
    if !linear_mount_overlaps_face(
        graph,
        attachment.source,
        attachment.anchor,
        attachment.rail,
        attachment.axis,
    ) {
        return Err(PlacementError::BearingOutsideFace);
    }
    if graph.bearings().any(|(_, bearing)| {
        bearing.source == attachment.source
            && bearing.shared_anchor.abs_diff_eq(attachment.anchor, CONTACT_EPSILON)
            && bearing.axis.abs_diff_eq(attachment.axis, CONTACT_EPSILON)
            && matches!(bearing.kind, BearingKind::Linear(existing) if existing.face != attachment.rail.face)
    }) {
        return Err(PlacementError::Graph("the carriage already has attachments on another face".into()));
    }
    Ok(())
}

pub(crate) fn stage_linear_block_batch_in_bounds(
    graph: &ConstructionGraph,
    start: PlacementCandidate,
    specs: &[CuboidSpec],
    attachment: LinearAttachment<'_>,
    bounds: PlacementBounds,
) -> Result<ConstructionGraph, PlacementError> {
    validate_linear_attachment(graph, attachment)?;
    stage_connected_block_batch(graph, start, specs, Some(attachment.into()), None, bounds)
}

pub(crate) fn stage_linear_block_volume_in_bounds(
    graph: &ConstructionGraph,
    index: &PlacementSnapIndex,
    start: PlacementCandidate,
    volume: BlockVolume,
    attachment: LinearAttachment<'_>,
    bounds: PlacementBounds,
    publication_generation: u64,
) -> Result<BlockVolumePlacement, PlacementError> {
    validate_linear_attachment(graph, attachment)?;
    stage_connected_block_volume_in_bounds(
        graph,
        index,
        start,
        volume,
        Some(attachment.into()),
        None,
        bounds,
        publication_generation,
    )
}

pub(crate) fn stage_linear_cylinder_in_bounds(
    graph: &ConstructionGraph,
    candidate: CylinderPlacementCandidate,
    attachment: LinearAttachment<'_>,
    bounds: PlacementBounds,
) -> Result<ConstructionGraph, PlacementError> {
    validate_linear_attachment(graph, attachment)?;
    stage_connected_cylinder(graph, candidate, Some(attachment.into()), None, bounds)
}

/// Validates and commits a regular block volume without constructing any
/// all-pairs candidate sets.
#[allow(clippy::too_many_arguments)]
fn stage_connected_block_volume_in_bounds(
    graph: &ConstructionGraph,
    index: &PlacementSnapIndex,
    start: PlacementCandidate,
    volume: BlockVolume,
    bearing: Option<BearingAttachment<'_>>,
    auto_weld_source: Option<FaceOwner>,
    bounds: PlacementBounds,
    publication_generation: u64,
) -> Result<BlockVolumePlacement, PlacementError> {
    validate_block_volume_in_bounds(graph, index, start, volume, bounds)?;

    let weld_scope =
        auto_weld_source.and_then(|source| bearing_connected_weld_scope(graph, source));
    let mut staged = graph.begin_edit();
    staged.reserve_parts_and_welds(volume.count(), volume.count().saturating_mul(6));
    let new_parts = staged.spawn_cuboids(volume.specs());

    let mut connections = Vec::with_capacity(volume.count().saturating_mul(3));
    if let Some(BearingAttachment {
        source,
        anchor,
        dimensions,
        kind,
        axis,
        rigid_targets,
    }) = bearing
    {
        let first = new_parts[0];
        connections.push(BuildCommand::AddBearing(
            BearingSpec::new(
                source,
                FaceRef::part(first, start.attached_face),
                anchor,
                axis,
            )
            .with_dimensions(dimensions)
            .with_kind(kind),
        ));
        connections.extend(rigid_targets.iter().copied().map(|target| {
            BuildCommand::RigidLink(RigidLinkSpec {
                first: target,
                second: first,
            })
        }));
    }

    append_internal_volume_welds(volume, &new_parts, &mut connections);
    if bearing.is_none() {
        if weld_scope.is_none() && bounds == PlacementBounds::Garage {
            append_ground_volume_welds(volume, &new_parts, &mut connections);
        }
        append_existing_volume_welds(
            graph,
            &staged,
            index,
            volume,
            &new_parts,
            weld_scope.as_ref(),
            &mut connections,
        );
    }
    let weld_count = connections
        .iter()
        .filter(|command| matches!(command, BuildCommand::Weld(_)))
        .count();
    staged
        .apply_batch_discarding_outcomes(connections)
        .map_err(|error| PlacementError::Graph(error.to_string()))?;
    Ok(BlockVolumePlacement {
        graph: staged.finish(),
        new_parts,
        weld_count,
        bounds: volume.bounds(),
        publication_generation,
    })
}

fn append_internal_volume_welds(
    volume: BlockVolume,
    parts: &[PartId],
    connections: &mut Vec<BuildCommand>,
) {
    let counts = volume.dimensions();
    for x in 0..counts.x {
        for y in 0..counts.y {
            for z in 0..counts.z {
                let cell = UVec3::new(x, y, z);
                let first = volume.part_at_physical(parts, cell);
                for (axis, face) in [
                    (0, FaceKind::PositiveX),
                    (1, FaceKind::PositiveY),
                    (2, FaceKind::PositiveZ),
                ] {
                    let mut neighbour = cell;
                    neighbour[axis] += 1;
                    if neighbour[axis] >= counts[axis] {
                        continue;
                    }
                    let second = volume.part_at_physical(parts, neighbour);
                    let opposite = match face {
                        FaceKind::PositiveX => FaceKind::NegativeX,
                        FaceKind::PositiveY => FaceKind::NegativeY,
                        FaceKind::PositiveZ => FaceKind::NegativeZ,
                        _ => unreachable!(),
                    };
                    connections.push(BuildCommand::Weld(WeldSpec {
                        first: FaceRef::part(first, face),
                        second: FaceRef::part(second, opposite),
                    }));
                }
            }
        }
    }
}

fn append_ground_volume_welds(
    volume: BlockVolume,
    parts: &[PartId],
    connections: &mut Vec<BuildCommand>,
) {
    let (minimum, _) = volume.bounds();
    if minimum.y.abs() > CONTACT_EPSILON {
        return;
    }
    let counts = volume.dimensions();
    for x in 0..counts.x {
        for z in 0..counts.z {
            connections.push(BuildCommand::Weld(WeldSpec {
                first: FaceRef::part(
                    volume.part_at_physical(parts, UVec3::new(x, 0, z)),
                    FaceKind::NegativeY,
                ),
                second: FaceRef::ground(),
            }));
        }
    }
}

fn append_existing_volume_welds(
    graph: &ConstructionGraph,
    staged: &ConstructionGraph,
    index: &PlacementSnapIndex,
    volume: BlockVolume,
    parts: &[PartId],
    weld_scope: Option<&HashSet<PartId>>,
    connections: &mut Vec<BuildCommand>,
) {
    let (minimum, maximum) = volume.bounds();
    let mut tested_owners = HashSet::new();
    let simple_graph = graph.regions().next().is_none() && graph.shape_features().next().is_none();
    for target in index.nearby(minimum, maximum, CONTACT_EPSILON * 2.0) {
        if weld_scope.is_some_and(|members| !members.contains(&target.part)) {
            continue;
        }
        if simple_graph
            && target.frame == mechanic_core::ConstructionFrame::IDENTITY
            && let PartSpec::Cuboid(existing) = target.spec
            && let Some(cell) = direct_adjacent_unit_cell(volume, existing)
        {
            let part = volume.part_at_physical(parts, cell);
            if let Some((first, second)) =
                direct_block_face_pair(graph, part, volume.spec_at_physical(cell), target.part)
            {
                connections.push(BuildCommand::Weld(WeldSpec { first, second }));
            }
            continue;
        }
        let owner = connection_geometry_owner(graph, target.part);
        let (low, high) = volume_candidate_range(volume, target.minimum, target.maximum, true);
        for x in low.x..=high.x {
            for y in low.y..=high.y {
                for z in low.z..=high.z {
                    let cell = UVec3::new(x, y, z);
                    let part = volume.part_at_physical(parts, cell);
                    let (candidate_minimum, candidate_maximum) =
                        cuboid_world_bounds(volume.spec_at_physical(cell));
                    if !bounds_share_face(
                        candidate_minimum,
                        candidate_maximum,
                        target.minimum,
                        target.maximum,
                    ) || !tested_owners.insert((part, owner))
                    {
                        continue;
                    }
                    let direct = direct_block_face_pair(
                        graph,
                        part,
                        volume.spec_at_physical(cell),
                        target.part,
                    );
                    if let Some((first, second)) = direct.or_else(|| {
                        touching_face_pair(
                            staged,
                            FaceOwner::Part(part),
                            FaceOwner::Part(target.part),
                        )
                    }) {
                        connections.push(BuildCommand::Weld(WeldSpec { first, second }));
                    }
                }
            }
        }
    }
}

fn direct_adjacent_unit_cell(volume: BlockVolume, existing: CuboidSpec) -> Option<UVec3> {
    if existing.pose.rotation != GridRotation::default()
        || !existing
            .dimensions
            .iter()
            .all(|dimension| dimension.units() == volume.start().dimensions[0].units())
    {
        return None;
    }
    let block_ticks =
        i32::from(volume.start().dimensions[0].units()) * POSITION_TICKS_PER_GRID_UNIT;
    let first_center = volume.start().pose.translation_position_ticks()
        + volume.span.min(IVec3::ZERO) * block_ticks;
    let existing_center = existing.pose.translation_position_ticks();
    let counts = volume.dimensions();
    let mut cell = IVec3::ZERO;
    let mut adjacent_axes = 0;
    for axis in 0..3 {
        let delta = existing_center[axis] - first_center[axis];
        if delta == -block_ticks {
            cell[axis] = 0;
            adjacent_axes += 1;
        } else if delta == i32::try_from(counts[axis]).expect("volume count fits i32") * block_ticks
        {
            cell[axis] = i32::try_from(counts[axis]).expect("volume count fits i32") - 1;
            adjacent_axes += 1;
        } else if delta >= 0
            && delta % block_ticks == 0
            && delta / block_ticks < i32::try_from(counts[axis]).expect("volume count fits i32")
        {
            cell[axis] = delta / block_ticks;
        } else {
            return None;
        }
    }
    (adjacent_axes == 1).then_some(cell.as_uvec3())
}

fn direct_block_face_pair(
    graph: &ConstructionGraph,
    new_part: PartId,
    new_spec: CuboidSpec,
    existing_part: PartId,
) -> Option<(FaceRef, FaceRef)> {
    if graph.region_of(existing_part).is_some()
        || graph.owner_has_shape_features(mechanic_core::SolidOwner::Part(existing_part))
    {
        return None;
    }
    let PartSpec::Cuboid(existing) = graph.part(existing_part).copied()? else {
        return None;
    };
    if existing.pose.rotation != GridRotation::default() {
        return None;
    }
    let (new_minimum, new_maximum) = cuboid_world_bounds(new_spec);
    let (old_minimum, old_maximum) = cuboid_world_bounds(existing);
    for (axis, positive, negative) in [
        (0, FaceKind::PositiveX, FaceKind::NegativeX),
        (1, FaceKind::PositiveY, FaceKind::NegativeY),
        (2, FaceKind::PositiveZ, FaceKind::NegativeZ),
    ] {
        if (new_maximum[axis] - old_minimum[axis]).abs() <= CONTACT_EPSILON {
            return Some((
                FaceRef::part(new_part, positive),
                FaceRef::part(existing_part, negative),
            ));
        }
        if (old_maximum[axis] - new_minimum[axis]).abs() <= CONTACT_EPSILON {
            return Some((
                FaceRef::part(new_part, negative),
                FaceRef::part(existing_part, positive),
            ));
        }
    }
    None
}

fn bounds_share_face(
    first_minimum: Vec3,
    first_maximum: Vec3,
    second_minimum: Vec3,
    second_maximum: Vec3,
) -> bool {
    (0..3).any(|normal| {
        let touching = (first_maximum[normal] - second_minimum[normal]).abs() <= CONTACT_EPSILON
            || (second_maximum[normal] - first_minimum[normal]).abs() <= CONTACT_EPSILON;
        if !touching {
            return false;
        }
        let tangents = [(normal + 1) % 3, (normal + 2) % 3];
        tangents.into_iter().all(|axis| {
            first_maximum[axis].min(second_maximum[axis])
                - first_minimum[axis].max(second_minimum[axis])
                > CONTACT_EPSILON
        })
    })
}

#[derive(Clone, Copy)]
enum FixedPartSpawn {
    Cuboid,
    Controller,
    Engine(EngineKind),
    Servo,
    Seat,
    Input,
    DimensionLink(DimensionLinkId),
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)] // Placement, bearing attachment, and welding share one transaction.
fn stage_connected_part_batch(
    graph: &ConstructionGraph,
    start: PlacementCandidate,
    specs: &[CuboidSpec],
    bearing: Option<BearingAttachment<'_>>,
    auto_weld_source: Option<FaceOwner>,
    spawn: FixedPartSpawn,
    bounds: PlacementBounds,
) -> Result<ConstructionGraph, PlacementError> {
    validate_block_batch_in_bounds(graph, start, specs, bounds)?;
    for (index, spec) in specs.iter().enumerate() {
        for other in &specs[..index] {
            let (minimum, maximum) = cuboid_world_bounds(*spec);
            let (other_minimum, other_maximum) = cuboid_world_bounds(*other);
            if bounds_overlap_interior(minimum, maximum, other_minimum, other_maximum) {
                return Err(PlacementError::BlocksOverlap);
            }
        }
    }

    let existing_parts = graph.parts().map(|(part, _)| part).collect::<Vec<_>>();
    let weld_scope =
        auto_weld_source.and_then(|source| bearing_connected_weld_scope(graph, source));
    let mut staged = graph.begin_edit();
    let outcomes = staged
        .apply_batch(specs.iter().copied().map(|spec| match spawn {
            FixedPartSpawn::Cuboid => BuildCommand::Spawn(spec),
            FixedPartSpawn::Controller => {
                BuildCommand::SpawnController(ControllerSpec::new(spec.pose))
            }
            FixedPartSpawn::Engine(kind) => {
                BuildCommand::SpawnEngine(EngineSpec::new(kind, spec.pose))
            }
            FixedPartSpawn::Servo => BuildCommand::SpawnServo(ServoSpec::new(spec.pose)),
            FixedPartSpawn::Seat => BuildCommand::SpawnSeat(SeatSpec::new(spec.pose)),
            FixedPartSpawn::Input => BuildCommand::SpawnInput(InputSpec::new(spec.pose)),
            FixedPartSpawn::DimensionLink(id) => {
                BuildCommand::SpawnDimensionLink(DimensionLinkSpec::new(id, spec.pose))
            }
        }))
        .map_err(|error| PlacementError::Graph(error.to_string()))?;
    let new_parts = outcomes
        .into_iter()
        .map(|outcome| match outcome {
            BuildOutcome::Spawned(part) => part,
            _ => unreachable!("spawn batch contains only spawn commands"),
        })
        .collect::<Vec<_>>();

    let mut connections = Vec::new();
    if let Some(BearingAttachment {
        source,
        anchor,
        dimensions,
        kind,
        axis,
        rigid_targets,
    }) = bearing
    {
        let first = *new_parts
            .first()
            .expect("validated block batches are never empty");
        connections.push(BuildCommand::AddBearing(
            BearingSpec::new(
                source,
                FaceRef::part(first, start.attached_face),
                anchor,
                axis,
            )
            .with_dimensions(dimensions)
            .with_kind(kind),
        ));
        connections.extend(rigid_targets.iter().copied().map(|target| {
            BuildCommand::RigidLink(RigidLinkSpec {
                first: target,
                second: first,
            })
        }));
    }
    for (index, &part) in new_parts.iter().enumerate() {
        if bearing.is_none()
            && weld_scope.is_none()
            && bounds == PlacementBounds::Garage
            && let Some((first, second)) =
                touching_face_pair(&staged, FaceOwner::Part(part), FaceOwner::Ground)
        {
            connections.push(BuildCommand::Weld(WeldSpec { first, second }));
        }
        let mut tested_owners = HashSet::new();
        for &other in &existing_parts {
            if bearing.is_some()
                || weld_scope
                    .as_ref()
                    .is_some_and(|members| !members.contains(&other))
                || !tested_owners.insert(connection_geometry_owner(&staged, other))
            {
                continue;
            }
            if let Some((first, second)) =
                touching_face_pair(&staged, FaceOwner::Part(part), FaceOwner::Part(other))
            {
                connections.push(BuildCommand::Weld(WeldSpec { first, second }));
            }
        }
        for &other in &new_parts[..index] {
            if let Some((first, second)) =
                touching_face_pair(&staged, FaceOwner::Part(part), FaceOwner::Part(other))
            {
                connections.push(BuildCommand::Weld(WeldSpec { first, second }));
            }
        }
    }
    staged
        .apply_batch(connections)
        .map_err(|error| PlacementError::Graph(error.to_string()))?;
    Ok(staged.finish())
}

fn connection_geometry_owner(graph: &ConstructionGraph, part: PartId) -> mechanic_core::SolidOwner {
    let owner = graph.region_of(part).map_or(
        mechanic_core::SolidOwner::Part(part),
        mechanic_core::SolidOwner::Region,
    );
    if graph.owner_has_shape_features(owner) {
        owner
    } else {
        mechanic_core::SolidOwner::Part(part)
    }
}

fn bearing_connected_weld_scope(
    graph: &ConstructionGraph,
    source: FaceOwner,
) -> Option<HashSet<PartId>> {
    let FaceOwner::Part(seed) = source else {
        return None;
    };
    let members = rigid_body_parts(graph, seed)
        .into_iter()
        .collect::<HashSet<_>>();
    graph
        .bearings()
        .any(|(_, bearing)| {
            [bearing.source.owner, bearing.target.owner]
                .into_iter()
                .any(|owner| matches!(owner, FaceOwner::Part(part) if members.contains(&part)))
        })
        .then_some(members)
}

pub(crate) fn validate_block_batch_in_bounds(
    graph: &ConstructionGraph,
    start: PlacementCandidate,
    specs: &[CuboidSpec],
    bounds: PlacementBounds,
) -> Result<(), PlacementError> {
    validate_candidate_in_bounds(graph, start, bounds)?;
    if specs.is_empty() {
        return Err(PlacementError::EmptyBlockBatch);
    }
    for spec in specs {
        validate_spec_in_bounds(graph, *spec, bounds)?;
    }
    Ok(())
}

/// Preview counterpart to [`validate_block_batch_in_bounds`] that narrows exact
/// overlap tests through the placement index. The committed operation still
/// performs the full validation when it is applied.
pub(crate) fn validate_indexed_block_batch_in_bounds(
    index: &PlacementSnapIndex,
    start: PlacementCandidate,
    specs: &[CuboidSpec],
    bounds: PlacementBounds,
) -> Result<(), PlacementError> {
    if start.support != PlacementSupport::Free && start.anchor.is_none() {
        return Err(PlacementError::NoFaceOverlap);
    }
    if specs.is_empty() {
        return Err(PlacementError::EmptyBlockBatch);
    }
    for spec in specs {
        let part = PartSpec::Cuboid(*spec);
        let (minimum, maximum) = part_world_bounds(part);
        validate_world_bounds(minimum, maximum, bounds)?;
        for target in index.nearby(minimum, maximum, 0.0) {
            if parts_overlap_with_frame(part, target.spec, target.frame) {
                return Err(PlacementError::OverlapsPart(target.part));
            }
        }
    }
    Ok(())
}

/// Exact volume validation using the same spatial index as smart snapping.
pub(crate) fn validate_block_volume_in_bounds(
    graph: &ConstructionGraph,
    index: &PlacementSnapIndex,
    start: PlacementCandidate,
    volume: BlockVolume,
    bounds: PlacementBounds,
) -> Result<(), PlacementError> {
    if start.spec != volume.start() {
        return Err(PlacementError::Graph(
            "block volume does not begin at its placement candidate".to_owned(),
        ));
    }
    if start.support != PlacementSupport::Free && start.anchor.is_none() {
        return Err(PlacementError::NoFaceOverlap);
    }
    let (minimum, maximum) = volume.bounds();
    validate_world_bounds(minimum, maximum, bounds)?;

    for target in index.nearby(minimum, maximum, 0.0) {
        if !bounds_overlap_interior(minimum, maximum, target.minimum, target.maximum) {
            continue;
        }
        let Some(existing) = graph.part(target.part).copied() else {
            continue;
        };
        let (low, high) = volume_candidate_range(volume, target.minimum, target.maximum, false);
        for x in low.x..=high.x {
            for y in low.y..=high.y {
                for z in low.z..=high.z {
                    let candidate = volume.spec_at_physical(UVec3::new(x, y, z));
                    let (candidate_minimum, candidate_maximum) = cuboid_world_bounds(candidate);
                    if bounds_overlap_interior(
                        candidate_minimum,
                        candidate_maximum,
                        target.minimum,
                        target.maximum,
                    ) && parts_overlap_with_frame(
                        PartSpec::Cuboid(candidate),
                        existing,
                        target.frame,
                    ) {
                        return Err(PlacementError::OverlapsPart(target.part));
                    }
                }
            }
        }
    }
    Ok(())
}

fn volume_candidate_range(
    volume: BlockVolume,
    target_minimum: Vec3,
    target_maximum: Vec3,
    include_touching: bool,
) -> (UVec3, UVec3) {
    let (minimum, _) = volume.bounds();
    let block_size = f32::from(volume.start().dimensions[0].units()) * GRID_UNIT_METERS;
    let padding = i32::from(include_touching);
    let counts = volume.dimensions().as_ivec3();
    let low = (((target_minimum - minimum) / block_size).floor().as_ivec3()
        - IVec3::splat(padding))
    .clamp(IVec3::ZERO, counts - IVec3::ONE);
    let high = (((target_maximum - minimum) / block_size).ceil().as_ivec3()
        + IVec3::splat(padding))
    .clamp(IVec3::ZERO, counts - IVec3::ONE);
    (low.as_uvec3(), high.as_uvec3())
}

/// One plane of [`block_box_specs`], addressed by the endpoint the pointer is
/// aiming at rather than a span.
///
/// Drags themselves all work in spans now. This survives because a good many
/// tests are written against an endpoint, and routing it through the box keeps
/// the two from ever describing different geometry.
#[cfg(test)]
pub(crate) fn block_sheet_specs(
    start: CuboidSpec,
    endpoint_units: IVec3,
    plane: PlacementPlane,
) -> Result<Vec<CuboidSpec>, PlacementError> {
    let start_units = start.pose.translation_half_units();
    let block_units = i32::from(start.dimensions[0].units()) * 2;
    let mut span = IVec3::ZERO;
    for axis in plane.tangent_axes() {
        span[axis] = rounded_div(endpoint_units[axis] - start_units[axis], block_units);
    }
    block_box_specs(start, span)
}

/// Intersects a pointer ray with the plane through the dragged block's centre.
///
/// This deliberately leaves the point unsnapped. Block dragging subtracts two
/// such points before quantizing, so the press position rather than the snapped
/// block centre is the gesture's origin.
pub(crate) fn raycast_placement_plane_point(
    origin: Vec3,
    direction: Vec3,
    start: CuboidSpec,
    plane: PlacementPlane,
) -> Option<Vec3> {
    let axis = plane.normal_axis();
    let denominator = direction[axis];
    if !origin.is_finite() || !direction.is_finite() || denominator.abs() <= f32::EPSILON {
        return None;
    }
    let coordinate = start.pose.translation()[axis];
    let distance = (coordinate - origin[axis]) / denominator;
    if distance < 0.0 || !distance.is_finite() {
        return None;
    }
    Some(origin + direction * distance)
}

/// Every block in the solid cuboid spanning `span` blocks from `start`.
///
/// `span` counts blocks *beyond* the start block along each axis and may be
/// negative, so a zero span is the single starting block. This is what a drag
/// produces once Rotate has moved it into a third axis.
pub(crate) fn block_box_specs(
    start: CuboidSpec,
    span: IVec3,
) -> Result<Vec<CuboidSpec>, PlacementError> {
    Ok(BlockVolume::new(start, span)?.specs().collect())
}

/// The world-space bounds of the box a drag spans, without building its blocks.
pub(crate) fn block_box_bounds(start: CuboidSpec, span: IVec3) -> (Vec3, Vec3) {
    let block = f32::from(start.dimensions[0].units()) * GRID_UNIT_METERS;
    let centre = start.pose.translation();
    let reach = span.as_vec3() * block;
    let low = centre + reach.min(Vec3::ZERO) - Vec3::splat(block * 0.5);
    let high = centre + reach.max(Vec3::ZERO) + Vec3::splat(block * 0.5);
    (low, high)
}

/// Extends a drag's span by the pointer's motion within the active plane.
///
/// Only the plane's own two axes move; the third keeps whatever it already had.
/// That is what lets Rotate move the drag into a new plane without discarding the
/// extent already dragged, turning a rectangle into a box.
pub(crate) fn block_span_from_rays(
    start: CuboidSpec,
    plane: PlacementPlane,
    anchor_span: IVec3,
    press_origin: Vec3,
    press_direction: Vec3,
    current_origin: Vec3,
    current_direction: Vec3,
) -> Option<IVec3> {
    let press = raycast_placement_plane_point(press_origin, press_direction, start, plane)?;
    let current = raycast_placement_plane_point(current_origin, current_direction, start, plane)?;
    let steps = ((current - press) / BLOCK_SIZE_METERS).round().as_ivec3();
    let mut span = anchor_span;
    for axis in plane.tangent_axes() {
        span[axis] = anchor_span[axis].saturating_add(steps[axis]);
    }
    Some(span)
}

fn snap_world_to_position_ticks(position: Vec3) -> IVec3 {
    (position / POSITION_TICK_METERS).round().as_ivec3()
}

#[allow(clippy::cast_possible_truncation)]
fn rounded_position_tick(meters: f32) -> i32 {
    debug_assert!(meters.is_finite());
    (meters / POSITION_TICK_METERS).round() as i32
}

#[allow(clippy::cast_possible_truncation)]
fn rounded_position_tick_f64(meters: f64) -> i32 {
    debug_assert!(meters.is_finite());
    (meters / f64::from(POSITION_TICK_METERS)).round() as i32
}

fn snap_global_center_ticks(
    raw_local_ticks: IVec3,
    world_dimensions: [u8; 3],
    grid: PlacementGrid,
    bounds: PlacementBounds,
) -> IVec3 {
    let origin_ticks = placement_origin_ticks(bounds);
    let mut global = raw_local_ticks + origin_ticks;
    let step = grid.step_ticks();
    for axis in 0..3 {
        let plane_phase = if axis == 1 {
            0
        } else {
            POSITION_TICKS_PER_HALF_GRID_UNIT.rem_euclid(step)
        };
        let center_phase = (plane_phase
            + i32::from(world_dimensions[axis]) * POSITION_TICKS_PER_HALF_GRID_UNIT)
            .rem_euclid(step);
        global[axis] = quantise_with_phase(global[axis], step, center_phase);
    }
    global - origin_ticks
}

fn placement_origin_ticks(bounds: PlacementBounds) -> IVec3 {
    match bounds {
        PlacementBounds::Garage
        | PlacementBounds::GarageBuild
        | PlacementBounds::GarageBuildFrame { .. } => IVec3::ZERO,
        PlacementBounds::World { origin } => IVec3::new(
            rounded_position_tick_f64(origin.x),
            0,
            rounded_position_tick_f64(origin.y),
        ),
    }
}

fn quantise_with_phase(value: i32, step: i32, phase: i32) -> i32 {
    let shifted = value - phase;
    let quotient = shifted.div_euclid(step);
    let remainder = shifted.rem_euclid(step);
    phase
        + if remainder.saturating_mul(2) > step
            || (remainder.saturating_mul(2) == step && (shifted >= 0 || phase != 0))
        {
            quotient.saturating_add(1) * step
        } else {
            quotient * step
        }
}

#[cfg(test)]
fn rounded_div(value: i32, divisor: i32) -> i32 {
    let half = divisor / 2;
    if value >= 0 {
        value.saturating_add(half) / divisor
    } else {
        value.saturating_sub(half) / divisor
    }
}

#[cfg(test)]
pub(crate) fn begin_weld(
    graph: &mut ConstructionGraph,
    face: FaceRef,
) -> Result<(), PlacementError> {
    graph
        .apply(BuildCommand::BeginPending(
            mechanic_core::PendingOperation::Weld(face),
        ))
        .map_err(|error| PlacementError::Graph(error.to_string()))?;
    Ok(())
}

pub(crate) fn stage_weld_objects(
    graph: &ConstructionGraph,
    first: FaceOwner,
    second: FaceOwner,
) -> Result<ConstructionGraph, PlacementError> {
    if first == second || weld_body_owners(graph, first).contains(&second) {
        return Err(PlacementError::SameObject);
    }
    let Some((first_face, second_face)) = touching_weld_face_pair(graph, first, second) else {
        return Err(PlacementError::ObjectsDoNotTouch);
    };
    let mut staged = graph.begin_edit();
    staged
        .apply(BuildCommand::Weld(WeldSpec {
            first: first_face,
            second: second_face,
        }))
        .map_err(|error| PlacementError::Graph(error.to_string()))?;
    Ok(staged.finish())
}

fn touching_weld_face_pair(
    graph: &ConstructionGraph,
    first: FaceOwner,
    second: FaceOwner,
) -> Option<(FaceRef, FaceRef)> {
    let source = weld_body_owners(graph, first)
        .into_iter()
        .flat_map(|owner| owner_faces(graph, owner))
        .collect::<Vec<_>>();
    let destination = weld_body_owners(graph, second)
        .into_iter()
        .flat_map(|owner| owner_faces(graph, owner))
        .collect::<Vec<_>>();
    source.iter().find_map(|&first_face| {
        destination.iter().find_map(|&second_face| {
            overlap_center(
                &face_geometry_from_ref(first_face, Some(graph)),
                &face_geometry_from_ref(second_face, Some(graph)),
            )?;
            let mut mating = vec![second_face];
            mating.extend(
                destination
                    .iter()
                    .copied()
                    .filter(|face| *face != second_face),
            );
            graph.weld_contact_square(&source, &mating).ok()?;
            Some((first_face, second_face))
        })
    })
}

/// Every object that moves with `owner`, which is what the weld tool treats as
/// one selection.
fn weld_body_owners(graph: &ConstructionGraph, owner: FaceOwner) -> Vec<FaceOwner> {
    match owner {
        FaceOwner::Ground => vec![FaceOwner::Ground],
        FaceOwner::Part(part) => rigid_body_parts(graph, part)
            .into_iter()
            .map(FaceOwner::Part)
            .collect(),
    }
}

/// Bearings whose two sides have ended up in one rigid body, so the joint can
/// no longer turn. Welding into a loop is allowed; this reports the cost.
pub(crate) fn locked_bearings(graph: &ConstructionGraph) -> Vec<BearingId> {
    graph
        .bearings()
        .filter_map(|(id, spec)| bearing_is_locked(graph, spec).then_some(id))
        .collect()
}

fn bearing_is_locked(graph: &ConstructionGraph, spec: &BearingSpec) -> bool {
    let (FaceOwner::Part(source), FaceOwner::Part(target)) = (spec.source.owner, spec.target.owner)
    else {
        return false;
    };
    source == target || rigid_body_parts(graph, source).contains(&target)
}

/// How many bearings `after` locks that `before` left free.
pub(crate) fn newly_locked_bearings(
    before: &ConstructionGraph,
    after: &ConstructionGraph,
) -> usize {
    let was_locked = locked_bearings(before);
    locked_bearings(after)
        .into_iter()
        .filter(|id| !was_locked.contains(id))
        .count()
}

pub(crate) fn rigid_body_parts(graph: &ConstructionGraph, seed: PartId) -> Vec<PartId> {
    if graph.part(seed).is_none() {
        return Vec::new();
    }

    let mut neighbours = HashMap::<PartId, Vec<PartId>>::new();
    for (_, weld) in graph.welds() {
        if let (FaceOwner::Part(first), FaceOwner::Part(second)) =
            (weld.first.owner, weld.second.owner)
        {
            neighbours.entry(first).or_default().push(second);
            neighbours.entry(second).or_default().push(first);
        }
    }
    for (_, link) in graph.rigid_links() {
        neighbours.entry(link.first).or_default().push(link.second);
        neighbours.entry(link.second).or_default().push(link.first);
    }

    let mut members = HashSet::from([seed]);
    let mut pending = vec![seed];
    while let Some(part) = pending.pop() {
        if let Some(connected) = neighbours.get(&part) {
            for &candidate in connected {
                if members.insert(candidate) {
                    pending.push(candidate);
                }
            }
        }
    }

    graph
        .parts()
        .filter_map(|(part, _)| members.contains(&part).then_some(part))
        .collect()
}

#[cfg(test)]
pub(crate) fn bearing_anchor_from_hit(
    graph: &ConstructionGraph,
    hit: SurfaceHit,
) -> Result<Vec3, PlacementError> {
    bearing_anchor_from_hit_with_grid(
        graph,
        hit,
        PlacementGrid::Centimetres25,
        PlacementBounds::Garage,
    )
}

pub(crate) fn bearing_anchor_from_hit_with_grid(
    graph: &ConstructionGraph,
    hit: SurfaceHit,
    grid: PlacementGrid,
    bounds: PlacementBounds,
) -> Result<Vec3, PlacementError> {
    if matches!(hit.face.owner, FaceOwner::Ground) {
        return Err(PlacementError::BearingOnGround);
    }
    let face =
        try_face_geometry_from_ref(hit.face, Some(graph)).ok_or(PlacementError::CurvedSurface)?;
    let (normal_axis, _) = cardinal_axis(face.normal);
    let anchor_ticks = snap_global_center_ticks(
        snap_world_to_position_ticks(hit.point),
        [BLOCK_SIZE_UNITS; 3],
        grid,
        bounds,
    );
    let mut anchor = anchor_ticks.as_vec3() * POSITION_TICK_METERS;
    anchor[normal_axis] = face.center[normal_axis];
    let offset = anchor - face.center;
    let u = offset.dot(face.tangent_u);
    let v = offset.dot(face.tangent_v);
    let inside_face_extent = match face.profile {
        FaceProfile::Annulus { outer_radius, .. }
        | FaceProfile::AnnularSector { outer_radius, .. } => {
            u.mul_add(u, v * v) <= (outer_radius + CONTACT_EPSILON).powi(2)
        }
        _ => point_in_profile(u, v, &face.profile),
    };
    if !inside_face_extent {
        return Err(PlacementError::BearingOutsideFace);
    }
    Ok(anchor)
}

/// Rail undersides need coplanar material overlap, not full support containment.
pub(crate) fn linear_mount_overlaps_face(
    graph: &ConstructionGraph,
    source: FaceRef,
    anchor: Vec3,
    rail: LinearBearing,
    axis: Vec3,
) -> bool {
    if matches!(source.owner, FaceOwner::Ground)
        || !face_is_flat(graph, source)
        || !anchor.is_finite()
        || rail.rotation(axis).is_err()
    {
        return false;
    }
    let Some(face) = try_face_geometry_from_ref(source, Some(graph)) else {
        return false;
    };
    let mount = FaceGeometry {
        center: anchor,
        normal: rail.mount_normal,
        tangent_u: axis,
        tangent_v: axis.cross(rail.mount_normal),
        profile: FaceProfile::Rectangle {
            half_u: rail.dimensions.length() * 0.5,
            half_v: rail.dimensions.width() * 0.5,
        },
    };
    faces_share_plane_and_normal(&mount, &face) && profiles_overlap(&mount, &face)
}

/// Finds a surviving coplanar support when the original mounting part is deleted.
/// The rail may overhang the replacement; only its underside must overlap.
pub(crate) fn linear_support_face_excluding(
    graph: &ConstructionGraph,
    selected_face: FaceRef,
    anchor: Vec3,
    rail: LinearBearing,
    axis: Vec3,
    excluded_parts: &HashSet<PartId>,
) -> Option<FaceRef> {
    let selected = try_face_geometry_from_ref(selected_face, Some(graph))?;
    graph
        .parts()
        .filter(|(part, _)| !excluded_parts.contains(part))
        .find_map(|(part, _)| {
            ALL_FACES.into_iter().find_map(|face| {
                let candidate = FaceRef::part(part, face);
                let geometry = try_face_geometry_from_ref(candidate, Some(graph))?;
                (faces_share_plane_and_normal(&selected, &geometry)
                    && linear_mount_overlaps_face(graph, candidate, anchor, rail, axis))
                .then_some(candidate)
            })
        })
}

pub(crate) fn linear_carriage_face(
    anchor: Vec3,
    rail: LinearBearing,
    axis: Vec3,
) -> Result<FaceGeometry, PlacementError> {
    let rotation = rail
        .rotation(axis)
        .map_err(|error| PlacementError::Graph(error.to_string()))?;
    let normal = snap_cardinal(rotation * rail.face.normal());
    let size = rail.face.size(rail.dimensions);
    Ok(FaceGeometry {
        center: anchor + rotation * rail.face.origin(rail.dimensions),
        normal,
        tangent_u: axis,
        tangent_v: axis.cross(normal),
        profile: FaceProfile::Rectangle {
            half_u: size.x * 0.5,
            half_v: size.y * 0.5,
        },
    })
}

fn linear_lattice_point(face: &FaceGeometry, point: Vec3) -> Vec3 {
    let delta = point - face.center;
    let pitch = LinearBearingDimensions::ATTACHMENT_PITCH;
    face.center
        + face.tangent_u * ((delta.dot(face.tangent_u) / pitch).round() * pitch)
        + face.tangent_v * ((delta.dot(face.tangent_v) / pitch).round() * pitch)
}

pub(crate) fn linear_block_candidate(
    anchor: Vec3,
    rail: LinearBearing,
    axis: Vec3,
    hit_point: Vec3,
    dimensions: [u8; 3],
    rotation: GridRotation,
) -> Result<PlacementCandidate, PlacementError> {
    let surface = linear_carriage_face(anchor, rail, axis)?;
    let point = linear_lattice_point(&surface, hit_point);
    let world_dimensions = oriented_grid_dimensions(dimensions, rotation);
    let normal_axis = cardinal_axis(surface.normal).0;
    let center = point + surface.normal * (f32::from(world_dimensions[normal_axis]) * 0.125);
    let spec = CuboidSpec::new(
        dimensions,
        BuildPose::from_position_ticks(snap_world_to_position_ticks(center), rotation),
    )
    .map_err(|error| PlacementError::Graph(error.to_string()))?;
    let attached_face = face_for_normal(rotation.quaternion().inverse() * -surface.normal);
    let candidate_face = face_geometry(spec, attached_face);
    Ok(PlacementCandidate {
        spec,
        attached_face,
        anchor: overlap_center(&surface, &candidate_face),
        support: PlacementSupport::Bearing,
    })
}

pub(crate) fn linear_cylinder_candidate(
    anchor: Vec3,
    rail: LinearBearing,
    axis: Vec3,
    hit_point: Vec3,
    dimensions: CylinderDimensions,
    quarter_turns: u8,
) -> Result<CylinderPlacementCandidate, PlacementError> {
    let surface = linear_carriage_face(anchor, rail, axis)?;
    let point = linear_lattice_point(&surface, hit_point);
    let frame = rotation_y_to_normal(surface.normal).quaternion()
        * GridRotation::new(0, quarter_turns % 4, 0).quaternion();
    let rotation = rotation_xy_to_directions(frame * Vec3::X, surface.normal)
        .expect("cardinal carriage faces have a cardinal cylinder frame");
    let center = point + surface.normal * (dimensions.axial_length() * 0.5);
    let spec = CylinderSpec::new(
        dimensions,
        BuildPose::from_position_ticks(snap_world_to_position_ticks(center), rotation),
    );
    let attached_face = FaceKind::NegativeY;
    let candidate_face = cylinder_face_geometry(spec, attached_face).expect("cylinder end is flat");
    Ok(CylinderPlacementCandidate {
        spec,
        attached_face,
        anchor: overlap_center(&surface, &candidate_face),
        support: PlacementSupport::Bearing,
    })
}

pub(crate) fn bearing_attachment_candidate(
    graph: &ConstructionGraph,
    source: FaceRef,
    anchor: Vec3,
) -> PlacementCandidate {
    candidate_from_hit(
        graph,
        SurfaceHit {
            distance: 0.0,
            point: anchor,
            face: source,
        },
    )
}

pub(crate) fn bearing_support_face(
    graph: &ConstructionGraph,
    selected_face: FaceRef,
    anchor: Vec3,
    dimensions: BearingDimensions,
) -> Option<FaceRef> {
    bearing_support_face_excluding(graph, selected_face, anchor, dimensions, &HashSet::new())
}

pub(crate) fn bearing_support_face_excluding(
    graph: &ConstructionGraph,
    selected_face: FaceRef,
    anchor: Vec3,
    dimensions: BearingDimensions,
    excluded_parts: &HashSet<PartId>,
) -> Option<FaceRef> {
    let selected = face_geometry_from_ref(selected_face, Some(graph));
    let mut fallback = None;
    for (part, spec) in graph.parts() {
        if excluded_parts.contains(&part) {
            continue;
        }
        for face_kind in ALL_FACES {
            let face_ref = FaceRef::part(part, face_kind);
            let Some(face) = part_face_geometry(*spec, face_kind) else {
                continue;
            };
            if !faces_share_plane_and_normal(&selected, &face)
                || !bearing_ring_overlaps_face(anchor, dimensions, &face)
            {
                continue;
            }
            if bearing_ring_contains_face_center(anchor, dimensions, &face) {
                return Some(face_ref);
            }
            fallback.get_or_insert(face_ref);
        }
    }
    fallback
}

pub(crate) fn bearing_overlaps_candidate(
    graph: &ConstructionGraph,
    source: FaceRef,
    anchor: Vec3,
    dimensions: BearingDimensions,
    candidate: PlacementCandidate,
) -> bool {
    let source_face = face_geometry_from_ref(source, Some(graph));
    let target_face = face_geometry(candidate.spec, candidate.attached_face);
    if source_face.normal.dot(target_face.normal) > -1.0 + CONTACT_EPSILON
        || (source_face.center - target_face.center)
            .dot(source_face.normal)
            .abs()
            > CONTACT_EPSILON
    {
        return false;
    }
    bearing_ring_overlaps_face(anchor, dimensions, &target_face)
}

pub(crate) fn bearing_overlaps_cylinder_candidate(
    graph: &ConstructionGraph,
    source: FaceRef,
    anchor: Vec3,
    dimensions: BearingDimensions,
    candidate: CylinderPlacementCandidate,
) -> bool {
    let source_face = face_geometry_from_ref(source, Some(graph));
    let target_face = cylinder_face_geometry(candidate.spec, candidate.attached_face)
        .expect("cylinder attachment face is flat");
    source_face.normal.dot(target_face.normal) <= -1.0 + CONTACT_EPSILON
        && (source_face.center - target_face.center)
            .dot(source_face.normal)
            .abs()
            <= CONTACT_EPSILON
        && bearing_ring_overlaps_face(anchor, dimensions, &target_face)
}

/// Moves a cylinder laterally so its attachment-face centre starts on the
/// bearing axis. The axial position and orientation already come from the
/// supporting face and remain unchanged.
pub(crate) fn center_cylinder_candidate_on_bearing(
    mut candidate: CylinderPlacementCandidate,
    anchor: Vec3,
) -> CylinderPlacementCandidate {
    let attachment = cylinder_face_geometry(candidate.spec, candidate.attached_face)
        .expect("cylinder attachment face is flat");
    let translation_ticks = candidate.spec.pose.translation_position_ticks()
        + snap_world_to_position_ticks(anchor - attachment.center);
    candidate.spec.pose =
        BuildPose::from_position_ticks(translation_ticks, candidate.spec.pose.rotation);
    candidate.anchor = Some(anchor);
    candidate.support = PlacementSupport::Bearing;
    candidate
}

#[cfg(test)]
pub(crate) fn stage_bearing_attachment(
    graph: &ConstructionGraph,
    candidate: PlacementCandidate,
    source: FaceRef,
    anchor: Vec3,
    dimensions: BearingDimensions,
) -> Result<ConstructionGraph, PlacementError> {
    stage_bearing_attachment_in_bounds(
        graph,
        candidate,
        source,
        anchor,
        dimensions,
        PlacementBounds::Garage,
    )
}

pub(crate) fn stage_bearing_attachment_in_bounds(
    graph: &ConstructionGraph,
    candidate: PlacementCandidate,
    source: FaceRef,
    anchor: Vec3,
    dimensions: BearingDimensions,
    bounds: PlacementBounds,
) -> Result<ConstructionGraph, PlacementError> {
    stage_bearing_block_batch_in_bounds(
        graph,
        candidate,
        &[candidate.spec],
        source,
        anchor,
        dimensions,
        &[],
        bounds,
    )
}

fn bearing_ring_overlaps_face(
    anchor: Vec3,
    dimensions: BearingDimensions,
    face: &FaceGeometry,
) -> bool {
    profiles_overlap(
        &FaceGeometry {
            center: anchor,
            normal: face.normal,
            tangent_u: face.tangent_u,
            tangent_v: face.tangent_v,
            profile: FaceProfile::Annulus {
                inner_radius: dimensions.inner_diameter() * 0.5,
                outer_radius: dimensions.outer_diameter() * 0.5,
            },
        },
        face,
    )
}

fn bearing_ring_contains_face_center(
    anchor: Vec3,
    dimensions: BearingDimensions,
    face: &FaceGeometry,
) -> bool {
    let offset = anchor - face.center;
    if offset.dot(face.normal).abs() > CONTACT_EPSILON {
        return false;
    }
    let radial = offset - face.normal * offset.dot(face.normal);
    point_in_profile(
        radial.dot(face.tangent_u),
        radial.dot(face.tangent_v),
        &FaceProfile::Annulus {
            inner_radius: dimensions.inner_diameter() * 0.5,
            outer_radius: dimensions.outer_diameter() * 0.5,
        },
    )
}

fn faces_share_plane_and_normal(first: &FaceGeometry, second: &FaceGeometry) -> bool {
    first.normal.dot(second.normal) > 1.0 - CONTACT_EPSILON
        && (first.center - second.center).dot(first.normal).abs() <= CONTACT_EPSILON
}

pub(crate) fn face_geometry_from_ref(
    face: FaceRef,
    graph: Option<&ConstructionGraph>,
) -> FaceGeometry {
    try_face_geometry_from_ref(face, graph).expect("face reference must expose flat geometry")
}

pub(crate) fn try_face_geometry_from_ref(
    face: FaceRef,
    graph: Option<&ConstructionGraph>,
) -> Option<FaceGeometry> {
    try_face_geometries_from_ref(face, graph).into_iter().next()
}

fn try_face_geometries_from_ref(
    face: FaceRef,
    graph: Option<&ConstructionGraph>,
) -> Vec<FaceGeometry> {
    match face.owner {
        FaceOwner::Ground => vec![FaceGeometry {
            center: Vec3::ZERO,
            normal: Vec3::Y,
            tangent_u: Vec3::X,
            tangent_v: Vec3::Z,
            profile: FaceProfile::Ground,
        }],
        FaceOwner::Part(part) => {
            let graph = graph.expect("live face references have a graph");
            let spec = graph
                .part(part)
                .copied()
                .expect("live face references have a part");
            let owner = graph.region_of(part).map_or(
                mechanic_core::SolidOwner::Part(part),
                mechanic_core::SolidOwner::Region,
            );
            let patch = face.patch.or_else(|| {
                graph
                    .owner_has_shape_features(owner)
                    .then(|| primitive_surface_patch(spec, face.face))
            });
            let Some(patch) = patch else {
                let frame = graph
                    .part_frame(part)
                    .expect("live parts have construction frames");
                return part_face_geometry(spec, face.face)
                    .map(|mut geometry| {
                        geometry.center = frame.point(geometry.center);
                        geometry.normal = frame.vector(geometry.normal);
                        geometry.tangent_u = frame.vector(geometry.tangent_u);
                        geometry.tangent_v = frame.vector(geometry.tangent_v);
                        geometry
                    })
                    .into_iter()
                    .collect();
            };
            let Ok(solid) = graph.evaluated_solid_shared(owner) else {
                return Vec::new();
            };
            solid
                .surfaces
                .iter()
                .filter(|surface| surface.key == patch)
                .filter_map(|surface| evaluated_surface_geometry(&solid, surface))
                .collect()
        }
    }
}

pub(crate) const fn primitive_surface_patch(
    spec: PartSpec,
    face: FaceKind,
) -> mechanic_core::SurfacePatchKey {
    let local = match spec {
        PartSpec::Cylinder(_) => match face {
            FaceKind::NegativeY => 0,
            FaceKind::PositiveY => 1,
            _ => u32::MAX,
        },
        PartSpec::PipeBend(_) => match face {
            FaceKind::NegativeX => 0,
            FaceKind::PositiveY => 1,
            _ => u32::MAX,
        },
        _ => match face {
            FaceKind::NegativeX => 0,
            FaceKind::PositiveX => 1,
            FaceKind::NegativeY => 2,
            FaceKind::PositiveY => 3,
            FaceKind::NegativeZ => 4,
            FaceKind::PositiveZ => 5,
        },
    };
    mechanic_core::SurfacePatchKey {
        source: mechanic_core::TopologySource::Base,
        local,
    }
}

fn evaluated_surface_geometry(
    solid: &mechanic_core::EvaluatedSolid,
    surface: &mechanic_core::SurfacePatch,
) -> Option<FaceGeometry> {
    let mut points = Vec::new();
    let mut edge = surface.half_edge;
    loop {
        let half_edge = *solid.half_edges.get(edge as usize)?;
        points.push(solid.vertices.get(half_edge.origin as usize)?.position);
        edge = half_edge.next;
        if edge == surface.half_edge {
            break;
        }
    }
    if points.len() < 3 {
        return None;
    }
    let count = f32::from(u16::try_from(points.len()).ok()?);
    let center = points.iter().copied().sum::<Vec3>() / count;
    let normal = surface.normal.normalize_or_zero();
    let tangent_u = normal.any_orthonormal_vector();
    let tangent_v = normal.cross(tangent_u);
    let vertices = points
        .into_iter()
        .map(|point| {
            let offset = point - center;
            Vec2::new(offset.dot(tangent_u), offset.dot(tangent_v))
        })
        .collect();
    Some(FaceGeometry {
        center,
        normal,
        tangent_u,
        tangent_v,
        profile: FaceProfile::Polygon { vertices },
    })
}

pub(crate) fn face_geometry(spec: CuboidSpec, face: FaceKind) -> FaceGeometry {
    let rotation = spec.pose.rotation.quaternion();
    let size = spec.size_meters();
    let (normal, tangent_u, tangent_v, normal_extent, half_u, half_v) = match face {
        FaceKind::PositiveX => (Vec3::X, Vec3::Y, Vec3::Z, size.x, size.y, size.z),
        FaceKind::NegativeX => (-Vec3::X, Vec3::Y, Vec3::Z, size.x, size.y, size.z),
        FaceKind::PositiveY => (Vec3::Y, Vec3::X, Vec3::Z, size.y, size.x, size.z),
        FaceKind::NegativeY => (-Vec3::Y, Vec3::X, Vec3::Z, size.y, size.x, size.z),
        FaceKind::PositiveZ => (Vec3::Z, Vec3::X, Vec3::Y, size.z, size.x, size.y),
        FaceKind::NegativeZ => (-Vec3::Z, Vec3::X, Vec3::Y, size.z, size.x, size.y),
    };
    let normal = snap_cardinal(rotation * normal);
    FaceGeometry {
        center: spec.pose.translation() + normal * normal_extent * 0.5,
        normal,
        tangent_u: snap_cardinal(rotation * tangent_u),
        tangent_v: snap_cardinal(rotation * tangent_v),
        profile: FaceProfile::Rectangle {
            half_u: half_u * 0.5,
            half_v: half_v * 0.5,
        },
    }
}

fn cylinder_face_geometry(spec: CylinderSpec, face: FaceKind) -> Option<FaceGeometry> {
    if !matches!(face, FaceKind::PositiveY | FaceKind::NegativeY) {
        return None;
    }
    let rotation = spec.pose.rotation.quaternion();
    let local_normal = if face == FaceKind::PositiveY {
        Vec3::Y
    } else {
        Vec3::NEG_Y
    };
    let normal = snap_cardinal(rotation * local_normal);
    let profile = if spec.dimensions.sweep_angle_degrees() == 360 {
        FaceProfile::Annulus {
            inner_radius: spec.dimensions.inner_diameter() * 0.5,
            outer_radius: spec.dimensions.outer_diameter() * 0.5,
        }
    } else {
        FaceProfile::AnnularSector {
            inner_radius: spec.dimensions.inner_diameter() * 0.5,
            outer_radius: spec.dimensions.outer_diameter() * 0.5,
            half_angle: spec.dimensions.sweep_angle_radians() * 0.5,
        }
    };
    Some(FaceGeometry {
        center: spec.pose.translation() + normal * spec.dimensions.axial_length() * 0.5,
        normal,
        tangent_u: snap_cardinal(rotation * Vec3::X),
        tangent_v: snap_cardinal(rotation * Vec3::Z),
        profile,
    })
}

fn pipe_bend_face_geometry(spec: PipeBendSpec, face: FaceKind) -> Option<FaceGeometry> {
    let radius = spec.dimensions.radius();
    let (local_center, local_normal, local_u, local_v) = match face {
        FaceKind::NegativeX => (Vec3::new(-radius, 0.0, 0.0), Vec3::NEG_X, Vec3::Y, Vec3::Z),
        FaceKind::PositiveY => (Vec3::new(0.0, radius, 0.0), Vec3::Y, Vec3::X, Vec3::Z),
        _ => return None,
    };
    let rotation = spec.pose.rotation.quaternion();
    Some(FaceGeometry {
        center: spec.pose.translation() + rotation * local_center,
        normal: snap_cardinal(rotation * local_normal),
        tangent_u: snap_cardinal(rotation * local_u),
        tangent_v: snap_cardinal(rotation * local_v),
        profile: FaceProfile::Annulus {
            inner_radius: spec.dimensions.inner_diameter() * 0.5,
            outer_radius: spec.dimensions.outer_diameter() * 0.5,
        },
    })
}

fn part_face_geometry(spec: PartSpec, face: FaceKind) -> Option<FaceGeometry> {
    match spec {
        PartSpec::Cuboid(spec) => Some(face_geometry(spec, face)),
        PartSpec::Controller(spec) => Some(face_geometry(spec.cuboid(), face)),
        PartSpec::Engine(spec) => Some(face_geometry(spec.cuboid(), face)),
        PartSpec::Transmission(spec) => Some(face_geometry(spec.cuboid(), face)),
        PartSpec::Servo(spec) => Some(face_geometry(spec.cuboid(), face)),
        PartSpec::Seat(spec) => Some(face_geometry(spec.cuboid(), face)),
        PartSpec::Input(spec) => Some(face_geometry(spec.cuboid(), face)),
        PartSpec::DimensionLink(spec) => Some(face_geometry(spec.cuboid(), face)),
        PartSpec::Cylinder(spec) => cylinder_face_geometry(spec, face),
        PartSpec::PipeBend(spec) => pipe_bend_face_geometry(spec, face),
        PartSpec::PipeJunction(spec) => pipe_junction_face_geometry(spec, face),
    }
}

/// Open arm end of a junction; closed faces are not connection faces.
fn pipe_junction_face_geometry(spec: PipeJunctionSpec, face: FaceKind) -> Option<FaceGeometry> {
    if !spec.arms.contains(face) {
        return None;
    }
    let rotation = spec.pose.rotation.quaternion();
    let (tangent_u, tangent_v) = match face {
        FaceKind::PositiveX | FaceKind::NegativeX => (Vec3::Y, Vec3::Z),
        FaceKind::PositiveY | FaceKind::NegativeY => (Vec3::X, Vec3::Z),
        FaceKind::PositiveZ | FaceKind::NegativeZ => (Vec3::X, Vec3::Y),
    };
    let normal = snap_cardinal(rotation * face_normal(face));
    Some(FaceGeometry {
        center: spec.pose.translation() + normal * spec.dimensions.half_side(),
        normal,
        tangent_u: snap_cardinal(rotation * tangent_u),
        tangent_v: snap_cardinal(rotation * tangent_v),
        profile: FaceProfile::Annulus {
            inner_radius: spec.dimensions.inner_diameter() * 0.5,
            outer_radius: spec.dimensions.outer_diameter() * 0.5,
        },
    })
}

/// Whether something can be mounted on this face.
///
/// Only genuinely flat surfaces take a new part: a shaped face is no longer an
/// axis-aligned rectangle, so a grid-aligned block could not sit flush on it.
/// Flattening the face back onto the grid makes it placeable again.
pub(crate) fn face_is_flat(graph: &ConstructionGraph, face: FaceRef) -> bool {
    let FaceOwner::Part(part) = face.owner else {
        return true;
    };
    if face
        .patch
        .is_some_and(|patch| matches!(patch.source, mechanic_core::TopologySource::Feature(_)))
    {
        return false;
    }
    let Some(id) = graph.region_of(part) else {
        if face.patch.is_none() {
            return true;
        }
        let Some(spec) = graph.part(part).copied() else {
            return false;
        };
        let normal = face_normal(face.face);
        return match spec {
            PartSpec::Cuboid(_) => true,
            PartSpec::PipeJunction(spec) => spec.arms.faces().any(|arm| {
                normal.dot(spec.pose.rotation.quaternion() * face_normal(arm))
                    >= 1.0 - CONTACT_EPSILON
            }),
            PartSpec::Cylinder(spec) => {
                normal.dot(spec.pose.rotation.quaternion() * Vec3::Y).abs() >= 1.0 - CONTACT_EPSILON
            }
            PartSpec::PipeBend(spec) => {
                let rotation = spec.pose.rotation.quaternion();
                [rotation * Vec3::NEG_X, rotation * Vec3::Y]
                    .into_iter()
                    .any(|end| normal.dot(end) >= 1.0 - CONTACT_EPSILON)
            }
            PartSpec::Controller(_)
            | PartSpec::Engine(_)
            | PartSpec::Transmission(_)
            | PartSpec::Servo(_)
            | PartSpec::Seat(_)
            | PartSpec::Input(_)
            | PartSpec::DimensionLink(_) => false,
        };
    };
    let Some(region) = graph.region(id) else {
        return true;
    };
    let Some(spec) = graph.part(part).and_then(|spec| spec.as_cuboid()) else {
        return true;
    };
    // The face is named in the part's own frame, so rotate it into the world
    // before asking the region, whose cage is world-aligned.
    let normal = spec.pose.rotation.quaternion() * face_normal(face.face);
    let (axis, sign) = cardinal_axis(normal);
    region.face_is_flat(axis, sign > 0)
}

/// Outward normal of one local face.
const fn face_normal(face: FaceKind) -> Vec3 {
    match face {
        FaceKind::PositiveX => Vec3::X,
        FaceKind::NegativeX => Vec3::NEG_X,
        FaceKind::PositiveY => Vec3::Y,
        FaceKind::NegativeY => Vec3::NEG_Y,
        FaceKind::PositiveZ => Vec3::Z,
        FaceKind::NegativeZ => Vec3::NEG_Z,
    }
}

fn validate_candidate_in_bounds(
    graph: &ConstructionGraph,
    candidate: PlacementCandidate,
    bounds: PlacementBounds,
) -> Result<(), PlacementError> {
    validate_spec_in_bounds(graph, candidate.spec, bounds)?;
    if candidate.support != PlacementSupport::Free && candidate.anchor.is_none() {
        return Err(PlacementError::NoFaceOverlap);
    }
    Ok(())
}

fn validate_spec_in_bounds(
    graph: &ConstructionGraph,
    spec: CuboidSpec,
    bounds: PlacementBounds,
) -> Result<(), PlacementError> {
    validate_part_in_bounds(graph, PartSpec::Cuboid(spec), bounds)
}

#[cfg(test)]
fn validate_part(graph: &ConstructionGraph, spec: PartSpec) -> Result<(), PlacementError> {
    validate_part_in_bounds(graph, spec, PlacementBounds::Garage)
}

fn validate_part_in_bounds(
    graph: &ConstructionGraph,
    spec: PartSpec,
    bounds: PlacementBounds,
) -> Result<(), PlacementError> {
    let (minimum, maximum) = part_world_bounds(spec);
    validate_world_bounds(minimum, maximum, bounds)?;
    for (part, existing) in graph.parts() {
        let frame = graph
            .part_frame(part)
            .expect("validated parts have construction frames");
        if parts_overlap_with_frame(spec, *existing, frame) {
            return Err(PlacementError::OverlapsPart(part));
        }
    }
    Ok(())
}

pub(crate) fn validate_world_bounds(
    minimum: Vec3,
    maximum: Vec3,
    bounds: PlacementBounds,
) -> Result<(), PlacementError> {
    let outside = match bounds {
        PlacementBounds::GarageBuildFrame { to_garage } => {
            for x in [minimum.x, maximum.x] {
                for y in [minimum.y, maximum.y] {
                    for z in [minimum.z, maximum.z] {
                        let point = to_garage.point(Vec3::new(x, y, z));
                        validate_world_bounds(point, point, PlacementBounds::GarageBuild)?;
                    }
                }
            }
            false
        }
        PlacementBounds::Garage => {
            minimum.x < -GROUND_HALF_SIZE - CONTACT_EPSILON
                || maximum.x > GROUND_HALF_SIZE + CONTACT_EPSILON
                || minimum.z < -GROUND_HALF_SIZE - CONTACT_EPSILON
                || maximum.z > GROUND_HALF_SIZE + CONTACT_EPSILON
                || minimum.y < -CONTACT_EPSILON
        }
        PlacementBounds::GarageBuild => {
            minimum.x < -GROUND_HALF_SIZE - CONTACT_EPSILON
                || maximum.x > GROUND_HALF_SIZE + CONTACT_EPSILON
                || minimum.z < -GROUND_HALF_SIZE - CONTACT_EPSILON
                || maximum.z > GROUND_HALF_SIZE + CONTACT_EPSILON
                || minimum.y < crate::garage::BUILD_MIN_Y - CONTACT_EPSILON
                || maximum.y > crate::garage::BUILD_MAX_Y + CONTACT_EPSILON
        }
        PlacementBounds::World { origin } => {
            f64::from(minimum.x) + origin.x < -WORLD_HALF_EXTENT_METERS
                || f64::from(maximum.x) + origin.x > WORLD_HALF_EXTENT_METERS
                || f64::from(minimum.z) + origin.y < -WORLD_HALF_EXTENT_METERS
                || f64::from(maximum.z) + origin.y > WORLD_HALF_EXTENT_METERS
        }
    };
    if outside {
        return Err(PlacementError::OutsidePlatform);
    }
    Ok(())
}

fn touching_face_pair(
    graph: &ConstructionGraph,
    first: FaceOwner,
    second: FaceOwner,
) -> Option<(FaceRef, FaceRef)> {
    owner_faces(graph, first)
        .into_iter()
        .find_map(|first_face| {
            owner_faces(graph, second)
                .into_iter()
                .find_map(|second_face| {
                    overlap_center(
                        &face_geometry_from_ref(first_face, Some(graph)),
                        &face_geometry_from_ref(second_face, Some(graph)),
                    )
                    .map(|_| (first_face, second_face))
                })
        })
}

fn owner_faces(graph: &ConstructionGraph, owner: FaceOwner) -> Vec<FaceRef> {
    match owner {
        FaceOwner::Ground => vec![FaceRef::ground()],
        FaceOwner::Part(part) => match graph.part(part).copied() {
            Some(
                PartSpec::Cuboid(_)
                | PartSpec::Controller(_)
                | PartSpec::Engine(_)
                | PartSpec::Transmission(_)
                | PartSpec::Servo(_)
                | PartSpec::Seat(_)
                | PartSpec::Input(_)
                | PartSpec::DimensionLink(_),
            ) => ALL_FACES
                .into_iter()
                .map(|face| FaceRef::part(part, face))
                .collect(),
            Some(PartSpec::Cylinder(_)) => [FaceKind::PositiveY, FaceKind::NegativeY]
                .into_iter()
                .map(|face| FaceRef::part(part, face))
                .collect(),
            Some(PartSpec::PipeJunction(junction)) => junction
                .arms
                .faces()
                .map(|face| FaceRef::part(part, face))
                .collect(),
            Some(PartSpec::PipeBend(_)) => [FaceKind::NegativeX, FaceKind::PositiveY]
                .into_iter()
                .map(|face| FaceRef::part(part, face))
                .collect(),
            None => Vec::new(),
        },
    }
}

fn raycast_ground(origin: Vec3, direction: Vec3) -> Option<SurfaceHit> {
    raycast_horizontal_surface(origin, direction, 0.0)
}

fn raycast_horizontal_surface(origin: Vec3, direction: Vec3, height: f32) -> Option<SurfaceHit> {
    // The platform is a build surface from above, not a wall that hides the
    // construction when the camera is underneath it.
    if direction.y >= -f32::EPSILON {
        return None;
    }
    let distance = (height - origin.y) / direction.y;
    let point = origin + direction * distance;
    (distance >= 0.0 && point.x.abs() <= GROUND_HALF_SIZE && point.z.abs() <= GROUND_HALF_SIZE)
        .then_some(SurfaceHit {
            distance,
            point,
            face: FaceRef::ground(),
        })
}

fn raycast_cuboid(
    origin: Vec3,
    direction: Vec3,
    part: PartId,
    spec: CuboidSpec,
) -> Option<SurfaceHit> {
    let hit = raycast_oriented_cuboid(
        origin,
        direction,
        spec.pose.translation(),
        spec.pose.rotation.quaternion(),
        spec.size_meters() * 0.5,
    )?;
    Some(SurfaceHit {
        distance: hit.distance,
        point: hit.point,
        face: FaceRef::part(part, face_for_normal(hit.local_normal)),
    })
}

/// Raycasts one shaped region against the same pieces its colliders and its
/// mesh come from, so the cursor lands where the surface actually is.
///
/// The reported face is still the grid face the surface came from, not the
/// tilted plane the ray met. Placement, welding, and face snapping therefore go
/// on working in grid coordinates: the grid stays the grid, and only the hit
/// test gets truthful.
pub(crate) fn raycast_region(
    origin: Vec3,
    direction: Vec3,
    part: PartId,
    region: &ShapeRegion,
) -> Option<SurfaceHit> {
    let direction = direction.normalize();
    let inverse_rotation = Quat::IDENTITY;
    let mut best: Option<SurfaceHit> = None;
    for piece in region_pieces(region) {
        let hit =
            match piece {
                PartPiece::Cuboid {
                    center,
                    half_extents,
                    rotation,
                    ..
                } => raycast_oriented_cuboid(origin, direction, center, rotation, half_extents)
                    .map(|hit| SurfaceHit {
                        distance: hit.distance,
                        point: hit.point,
                        face: FaceRef::part(
                            part,
                            face_for_normal(inverse_rotation * (rotation * hit.local_normal)),
                        ),
                    }),
                PartPiece::Convex(convex) => raycast_convex_piece(origin, direction, &convex).map(
                    |(distance, point, normal)| SurfaceHit {
                        distance,
                        point,
                        face: FaceRef::part(part, face_for_normal(inverse_rotation * normal)),
                    },
                ),
            };
        let Some(hit) = hit else {
            continue;
        };
        if best.is_none_or(|best| hit.distance < best.distance) {
            best = Some(hit);
        }
    }
    best
}

fn raycast_evaluated_solid(
    origin: Vec3,
    direction: Vec3,
    part: PartId,
    solid: &mechanic_core::EvaluatedSolid,
    frame: mechanic_core::ConstructionFrame,
) -> Option<SurfaceHit> {
    solid
        .surfaces
        .iter()
        .filter_map(|surface| {
            let (distance, point) = raycast_evaluated_surface(origin, direction, solid, surface)?;
            let placement_surface = if surface.smoothing_group == 0 {
                surface
            } else {
                // A rounded facet is not itself a mounting plane. Route the hit
                // to the nearest retained planar base patch so a block can still
                // bridge the recess wherever its face has real flat contact.
                solid
                    .surfaces
                    .iter()
                    .filter(|candidate| candidate.smoothing_group == 0)
                    .filter(|candidate| {
                        matches!(candidate.key.source, mechanic_core::TopologySource::Base)
                    })
                    .max_by(|left, right| {
                        left.normal
                            .dot(surface.normal)
                            .total_cmp(&right.normal.dot(surface.normal))
                    })?
            };
            Some(SurfaceHit {
                distance,
                point,
                face: FaceRef::patch(
                    part,
                    face_for_normal(frame.inverse().vector(placement_surface.normal)),
                    placement_surface.key,
                ),
            })
        })
        .min_by(|left, right| {
            left.distance
                .partial_cmp(&right.distance)
                .unwrap_or(Ordering::Equal)
        })
}

pub(crate) fn raycast_evaluated_surface(
    origin: Vec3,
    direction: Vec3,
    solid: &mechanic_core::EvaluatedSolid,
    surface: &mechanic_core::SurfacePatch,
) -> Option<(f32, Vec3)> {
    let first_edge = solid.half_edges.get(surface.half_edge as usize)?;
    let plane_point = solid.vertices.get(first_edge.origin as usize)?.position;
    let denominator = direction.dot(surface.normal);
    if denominator.abs() <= f32::EPSILON {
        return None;
    }
    let distance = (plane_point - origin).dot(surface.normal) / denominator;
    if distance < 0.0 || !distance.is_finite() {
        return None;
    }
    let point = origin + direction * distance;
    let mut edge = surface.half_edge;
    loop {
        let half_edge = solid.half_edges.get(edge as usize)?;
        let next = solid.half_edges.get(half_edge.next as usize)?;
        let start = solid.vertices.get(half_edge.origin as usize)?.position;
        let end = solid.vertices.get(next.origin as usize)?.position;
        if surface.normal.dot((end - start).cross(point - start)) < -CONTACT_EPSILON {
            return None;
        }
        edge = half_edge.next;
        if edge == surface.half_edge {
            break;
        }
    }
    Some((distance, point))
}

/// Slab-clips a ray against a convex piece, returning where it enters.
fn raycast_convex_piece(
    origin: Vec3,
    direction: Vec3,
    piece: &ConvexPiece,
) -> Option<(f32, Vec3, Vec3)> {
    let mut near = f32::NEG_INFINITY;
    let mut far = f32::INFINITY;
    let mut entry_normal = Vec3::Y;
    for face in &piece.faces {
        let denominator = direction.dot(face.normal);
        let distance = face.offset - origin.dot(face.normal);
        if denominator.abs() <= f32::EPSILON {
            // Parallel to this plane: outside it means the ray misses entirely.
            if distance < 0.0 {
                return None;
            }
            continue;
        }
        let crossing = distance / denominator;
        if denominator < 0.0 {
            if crossing > near {
                near = crossing;
                entry_normal = face.normal;
            }
        } else {
            far = far.min(crossing);
        }
        if near > far {
            return None;
        }
    }
    if !near.is_finite() || near < 0.0 {
        return None;
    }
    Some((near, origin + direction * near, entry_normal))
}

/// The convex pieces one region's cage describes.
pub(crate) fn region_pieces(region: &ShapeRegion) -> Vec<PartPiece> {
    let grid = region.grid();
    mechanic_core::decompose(&grid, &|cell, corner| region.corner_steps(cell, corner))
}

fn raycast_part(origin: Vec3, direction: Vec3, part: PartId, spec: PartSpec) -> Option<SurfaceHit> {
    match spec {
        PartSpec::Cuboid(spec) => raycast_cuboid(origin, direction, part, spec),
        PartSpec::Controller(spec) => raycast_cuboid(origin, direction, part, spec.cuboid()),
        PartSpec::Engine(spec) => raycast_cuboid(origin, direction, part, spec.cuboid()),
        PartSpec::Transmission(spec) => raycast_cuboid(origin, direction, part, spec.cuboid()),
        PartSpec::Servo(spec) => raycast_cuboid(origin, direction, part, spec.cuboid()),
        PartSpec::Seat(spec) => raycast_cuboid(origin, direction, part, spec.cuboid()),
        PartSpec::Input(spec) => raycast_cuboid(origin, direction, part, spec.cuboid()),
        PartSpec::DimensionLink(spec) => raycast_cuboid(origin, direction, part, spec.cuboid()),
        PartSpec::Cylinder(spec) => raycast_cylinder(origin, direction, part, spec),
        PartSpec::PipeBend(spec) => raycast_pipe_bend(origin, direction, part, spec),
        PartSpec::PipeJunction(spec) => raycast_pipe_junction(origin, direction, part, spec),
    }
}

/// Hits a junction's arms as capped pipes. A hit on an open arm's flat end
/// reports that arm; a hit on a wall reports the cube face nearest the hit,
/// which only branching uses.
fn raycast_pipe_junction(
    origin: Vec3,
    direction: Vec3,
    part: PartId,
    spec: PipeJunctionSpec,
) -> Option<SurfaceHit> {
    let direction = direction.normalize();
    let inverse = spec.pose.rotation.quaternion().inverse();
    let local_origin = inverse * (origin - spec.pose.translation());
    let local_direction = inverse * direction;
    let radius = spec.dimensions.outer_diameter() * 0.5;
    let reach = spec.dimensions.half_side();
    let mut nearest: Option<(f32, Option<FaceKind>)> = None;
    let mut offer = |distance: f32, end: Option<FaceKind>| {
        if distance >= 0.0 && nearest.is_none_or(|(best, _)| distance < best) {
            nearest = Some((distance, end));
        }
    };
    for arm in spec.arms.faces() {
        let axis = face_normal(arm);
        let along_origin = local_origin.dot(axis);
        let along_direction = local_direction.dot(axis);
        let lateral_origin = local_origin - axis * along_origin;
        let lateral_direction = local_direction - axis * along_direction;
        let a = lateral_direction.length_squared();
        let b = 2.0 * lateral_origin.dot(lateral_direction);
        let c = lateral_origin.length_squared() - radius * radius;
        let discriminant = b * b - 4.0 * a * c;
        if a > 1.0e-12 && discriminant >= 0.0 {
            for distance in [
                (-b - discriminant.sqrt()) / (2.0 * a),
                (-b + discriminant.sqrt()) / (2.0 * a),
            ] {
                if (0.0..=reach).contains(&(along_origin + along_direction * distance)) {
                    offer(distance, None);
                }
            }
        }
        if along_direction.abs() > 1.0e-12 {
            let distance = (reach - along_origin) / along_direction;
            if (lateral_origin + lateral_direction * distance).length_squared() <= radius * radius {
                offer(distance, Some(arm));
            }
        }
    }
    // The ball at the centre.
    let b = local_origin.dot(local_direction);
    let discriminant = b * b - (local_origin.length_squared() - radius * radius);
    if discriminant >= 0.0 {
        offer(-b - discriminant.sqrt(), None);
    }
    let (distance, end) = nearest?;
    let point = origin + direction * distance;
    let face = end.unwrap_or_else(|| face_toward(inverse * (point - spec.pose.translation())));
    Some(SurfaceHit {
        distance,
        point,
        face: FaceRef::part(part, face),
    })
}

fn raycast_pipe_bend(
    origin: Vec3,
    direction: Vec3,
    part: PartId,
    spec: PipeBendSpec,
) -> Option<SurfaceHit> {
    let direction = direction.normalize();
    let rotation = spec.pose.rotation.quaternion();
    let inverse = rotation.inverse();
    let local_origin = inverse * (origin - spec.pose.translation());
    let local_direction = inverse * direction;
    let outer = spec.dimensions.outer_diameter() * 0.5;
    let inner = spec.dimensions.inner_diameter() * 0.5;
    let radius = spec.dimensions.radius();
    let mut candidates = Vec::new();
    for (center, normal, face) in [
        (
            Vec3::new(-radius, 0.0, 0.0),
            Vec3::NEG_X,
            FaceKind::NegativeX,
        ),
        (Vec3::new(0.0, radius, 0.0), Vec3::Y, FaceKind::PositiveY),
    ] {
        let denominator = local_direction.dot(normal);
        if denominator.abs() <= f32::EPSILON {
            continue;
        }
        let distance = (center - local_origin).dot(normal) / denominator;
        if distance < 0.0 {
            continue;
        }
        let offset = local_origin + local_direction * distance - center;
        let radial_squared = offset.length_squared();
        if radial_squared >= inner * inner - CONTACT_EPSILON
            && radial_squared <= outer * outer + CONTACT_EPSILON
        {
            candidates.push((distance, face));
        }
    }
    for collider in pipe_bend_collision_boxes(spec) {
        if let Some(hit) = raycast_oriented_cuboid(
            origin,
            direction,
            collider.center,
            collider.rotation,
            collider.half,
        ) {
            candidates.push((hit.distance, FaceKind::PositiveZ));
        }
    }
    let (distance, face) = candidates
        .into_iter()
        .min_by(|left, right| left.0.total_cmp(&right.0))?;
    Some(SurfaceHit {
        distance,
        point: origin + direction * distance,
        face: FaceRef::part(part, face),
    })
}

fn raycast_cylinder(
    origin: Vec3,
    direction: Vec3,
    part: PartId,
    spec: CylinderSpec,
) -> Option<SurfaceHit> {
    let direction = direction.normalize();
    let rotation = spec.pose.rotation.quaternion();
    let inverse = rotation.inverse();
    let local_origin = inverse * (origin - spec.pose.translation());
    let local_direction = inverse * direction;
    let outer = spec.dimensions.outer_diameter() * 0.5;
    let inner = spec.dimensions.inner_diameter() * 0.5;
    let half_length = spec.dimensions.axial_length() * 0.5;
    let mut candidates = Vec::with_capacity(6);

    if local_direction.y.abs() > f32::EPSILON {
        for (y, face) in [
            (half_length, FaceKind::PositiveY),
            (-half_length, FaceKind::NegativeY),
        ] {
            let distance = (y - local_origin.y) / local_direction.y;
            if distance >= 0.0 {
                let point = local_origin + local_direction * distance;
                let radial_squared = point.x.mul_add(point.x, point.z * point.z);
                if radial_squared <= outer * outer + CONTACT_EPSILON
                    && radial_squared >= inner * inner - CONTACT_EPSILON
                    && point_in_cylinder_sweep(point.x, point.z, spec.dimensions)
                {
                    candidates.push((distance, face));
                }
            }
        }
    }
    for radius in [outer, inner] {
        if radius <= 0.0 {
            continue;
        }
        let a = local_direction
            .x
            .mul_add(local_direction.x, local_direction.z * local_direction.z);
        if a <= f32::EPSILON {
            continue;
        }
        let b = 2.0
            * local_origin
                .x
                .mul_add(local_direction.x, local_origin.z * local_direction.z);
        let c = local_origin
            .x
            .mul_add(local_origin.x, local_origin.z * local_origin.z)
            - radius * radius;
        let discriminant = b.mul_add(b, -4.0 * a * c);
        if discriminant < 0.0 {
            continue;
        }
        let root = discriminant.sqrt();
        for distance in [(-b - root) / (2.0 * a), (-b + root) / (2.0 * a)] {
            if distance >= 0.0 {
                let y = local_origin.y + local_direction.y * distance;
                let point = local_origin + local_direction * distance;
                if y.abs() <= half_length + CONTACT_EPSILON
                    && point_in_cylinder_sweep(point.x, point.z, spec.dimensions)
                {
                    candidates.push((distance, FaceKind::PositiveX));
                }
            }
        }
    }
    if spec.dimensions.sweep_angle_degrees() < 360 {
        let half_sweep = spec.dimensions.sweep_angle_radians() * 0.5;
        for (angle, outward) in [(-half_sweep, -1.0_f32), (half_sweep, 1.0_f32)] {
            let radial = Vec3::new(angle.cos(), 0.0, angle.sin());
            let angular = Vec3::new(-angle.sin(), 0.0, angle.cos()) * outward;
            let denominator = local_direction.dot(angular);
            if denominator.abs() <= f32::EPSILON {
                continue;
            }
            let distance = -local_origin.dot(angular) / denominator;
            if distance < 0.0 {
                continue;
            }
            let point = local_origin + local_direction * distance;
            let radius = point.dot(radial);
            if point.y.abs() <= half_length + CONTACT_EPSILON
                && radius >= inner - CONTACT_EPSILON
                && radius <= outer + CONTACT_EPSILON
            {
                candidates.push((distance, FaceKind::PositiveX));
            }
        }
    }
    let (distance, face) = candidates
        .into_iter()
        .min_by(|left, right| left.0.partial_cmp(&right.0).unwrap_or(Ordering::Equal))?;
    Some(SurfaceHit {
        distance,
        point: origin + direction * distance,
        face: FaceRef::part(part, face),
    })
}

fn point_in_cylinder_sweep(x: f32, z: f32, dimensions: CylinderDimensions) -> bool {
    dimensions.sweep_angle_degrees() == 360
        || z.atan2(x).abs() <= dimensions.sweep_angle_radians() * 0.5 + CONTACT_EPSILON
}

pub(crate) fn raycast_oriented_cuboid(
    origin: Vec3,
    direction: Vec3,
    center: Vec3,
    rotation: Quat,
    half_extents: Vec3,
) -> Option<OrientedCuboidHit> {
    if !origin.is_finite()
        || !direction.is_finite()
        || direction.length_squared() < f32::EPSILON
        || !center.is_finite()
        || !rotation.is_finite()
        || !half_extents.is_finite()
        || half_extents.cmple(Vec3::ZERO).any()
    {
        return None;
    }
    let direction = direction.normalize();
    let inverse_rotation = rotation.inverse();
    let local_origin = inverse_rotation * (origin - center);
    let local_direction = inverse_rotation * direction;
    let mut near = f32::NEG_INFINITY;
    let mut far = f32::INFINITY;
    let mut hit_axis = 0;
    let mut hit_sign = -1.0;

    for axis in 0..3 {
        if local_direction[axis].abs() <= f32::EPSILON {
            if local_origin[axis] < -half_extents[axis] || local_origin[axis] > half_extents[axis] {
                return None;
            }
            continue;
        }
        let inverse = local_direction[axis].recip();
        let first = (-half_extents[axis] - local_origin[axis]) * inverse;
        let second = (half_extents[axis] - local_origin[axis]) * inverse;
        let axis_near = first.min(second);
        let axis_far = first.max(second);
        if axis_near > near {
            near = axis_near;
            hit_axis = axis;
            hit_sign = if first < second { -1.0 } else { 1.0 };
        }
        far = far.min(axis_far);
        if near > far {
            return None;
        }
    }
    if far < 0.0 {
        return None;
    }
    let distance = near.max(0.0);
    let local_normal = Vec3::from_array(match hit_axis {
        0 => [hit_sign, 0.0, 0.0],
        1 => [0.0, hit_sign, 0.0],
        _ => [0.0, 0.0, hit_sign],
    });
    Some(OrientedCuboidHit {
        distance,
        point: origin + direction * distance,
        local_normal,
    })
}

fn overlap_center(first: &FaceGeometry, second: &FaceGeometry) -> Option<Vec3> {
    if first.normal.dot(second.normal) > -1.0 + CONTACT_EPSILON
        || (first.center - second.center).dot(first.normal).abs() > CONTACT_EPSILON
    {
        return None;
    }
    profiles_overlap(first, second).then_some((first.center + second.center) * 0.5)
}

fn point_in_profile(u: f32, v: f32, profile: &FaceProfile) -> bool {
    match profile {
        FaceProfile::Rectangle { half_u, half_v } => {
            u.abs() <= *half_u + CONTACT_EPSILON && v.abs() <= *half_v + CONTACT_EPSILON
        }
        FaceProfile::Annulus {
            inner_radius,
            outer_radius,
        } => {
            let squared = u.mul_add(u, v * v);
            squared >= (*inner_radius - CONTACT_EPSILON).max(0.0).powi(2)
                && squared <= (*outer_radius + CONTACT_EPSILON).powi(2)
        }
        FaceProfile::AnnularSector {
            inner_radius,
            outer_radius,
            half_angle,
        } => {
            let squared = u.mul_add(u, v * v);
            squared >= (*inner_radius - CONTACT_EPSILON).max(0.0).powi(2)
                && squared <= (*outer_radius + CONTACT_EPSILON).powi(2)
                && v.atan2(u).abs() <= *half_angle + CONTACT_EPSILON
        }
        FaceProfile::Polygon { vertices } => point_in_convex_polygon(Vec2::new(u, v), vertices),
        FaceProfile::Ground => true,
    }
}

fn profiles_overlap(first: &FaceGeometry, second: &FaceGeometry) -> bool {
    match (&first.profile, &second.profile) {
        (FaceProfile::Ground, _) | (_, FaceProfile::Ground) => true,
        (FaceProfile::Rectangle { half_u, half_v }, FaceProfile::Rectangle { .. }) => {
            positive_rect_overlap(first, second, first.tangent_u, *half_u)
                && positive_rect_overlap(first, second, first.tangent_v, *half_v)
        }
        (FaceProfile::Annulus { .. }, FaceProfile::Rectangle { .. }) => {
            annulus_rectangle_overlap(first, second)
        }
        (FaceProfile::Rectangle { .. }, FaceProfile::Annulus { .. }) => {
            annulus_rectangle_overlap(second, first)
        }
        (FaceProfile::Annulus { .. }, FaceProfile::Annulus { .. }) => {
            let FaceProfile::Annulus {
                inner_radius: inner_a,
                outer_radius: outer_a,
            } = &first.profile
            else {
                unreachable!()
            };
            let FaceProfile::Annulus {
                inner_radius: inner_b,
                outer_radius: outer_b,
            } = &second.profile
            else {
                unreachable!()
            };
            let offset = second.center - first.center;
            let distance =
                Vec2::new(offset.dot(first.tangent_u), offset.dot(first.tangent_v)).length();
            distance < *outer_a + *outer_b - CONTACT_EPSILON
                && distance + *outer_a > *inner_b + CONTACT_EPSILON
                && distance + *outer_b > *inner_a + CONTACT_EPSILON
        }
        (FaceProfile::AnnularSector { .. } | FaceProfile::Polygon { .. }, _)
        | (_, FaceProfile::AnnularSector { .. } | FaceProfile::Polygon { .. }) => {
            sector_profiles_overlap(first, second)
        }
    }
}

fn sector_profiles_overlap(first: &FaceGeometry, second: &FaceGeometry) -> bool {
    let first_cells = profile_cells(first, first.center, first.tangent_u, first.tangent_v);
    let second_cells = profile_cells(second, first.center, first.tangent_u, first.tangent_v);
    first_cells.iter().any(|first| {
        second_cells
            .iter()
            .any(|second| convex_polygons_overlap(first, second))
    })
}

fn profile_cells(
    face: &FaceGeometry,
    origin: Vec3,
    plane_u: Vec3,
    plane_v: Vec3,
) -> Vec<Vec<Vec2>> {
    let project = |point: Vec3| {
        let offset = point - origin;
        Vec2::new(offset.dot(plane_u), offset.dot(plane_v))
    };
    match &face.profile {
        FaceProfile::Rectangle { half_u, half_v } => vec![vec![
            project(face.center - face.tangent_u * *half_u - face.tangent_v * *half_v),
            project(face.center + face.tangent_u * *half_u - face.tangent_v * *half_v),
            project(face.center + face.tangent_u * *half_u + face.tangent_v * *half_v),
            project(face.center - face.tangent_u * *half_u + face.tangent_v * *half_v),
        ]],
        FaceProfile::Annulus {
            inner_radius,
            outer_radius,
        } => annular_profile_cells(
            face,
            origin,
            plane_u,
            plane_v,
            *inner_radius,
            *outer_radius,
            std::f32::consts::PI,
        ),
        FaceProfile::AnnularSector {
            inner_radius,
            outer_radius,
            half_angle,
        } => annular_profile_cells(
            face,
            origin,
            plane_u,
            plane_v,
            *inner_radius,
            *outer_radius,
            *half_angle,
        ),
        FaceProfile::Polygon { vertices } => vec![
            vertices
                .iter()
                .map(|vertex| {
                    project(face.center + face.tangent_u * vertex.x + face.tangent_v * vertex.y)
                })
                .collect(),
        ],
        FaceProfile::Ground => Vec::new(),
    }
}

#[allow(clippy::too_many_arguments)]
fn annular_profile_cells(
    face: &FaceGeometry,
    origin: Vec3,
    plane_u: Vec3,
    plane_v: Vec3,
    inner_radius: f32,
    outer_radius: f32,
    half_angle: f32,
) -> Vec<Vec<Vec2>> {
    let sweep = half_angle * 2.0;
    let segment_count = (1_u16..=24)
        .find(|&count| (f32::from(count) * (std::f32::consts::PI / 12.0) - sweep).abs() < 1.0e-4)
        .expect("annular profiles use 15-degree increments");
    let project = |point: Vec3| {
        let offset = point - origin;
        Vec2::new(offset.dot(plane_u), offset.dot(plane_v))
    };
    (0..segment_count)
        .map(|segment| {
            let first_angle = -half_angle + sweep * f32::from(segment) / f32::from(segment_count);
            let second_angle =
                -half_angle + sweep * f32::from(segment + 1) / f32::from(segment_count);
            let radial = |angle: f32| face.tangent_u * angle.cos() + face.tangent_v * angle.sin();
            let outer_first = project(face.center + radial(first_angle) * outer_radius);
            let outer_second = project(face.center + radial(second_angle) * outer_radius);
            if inner_radius == 0.0 {
                vec![project(face.center), outer_first, outer_second]
            } else {
                vec![
                    project(face.center + radial(first_angle) * inner_radius),
                    outer_first,
                    outer_second,
                    project(face.center + radial(second_angle) * inner_radius),
                ]
            }
        })
        .collect()
}

fn point_in_convex_polygon(point: Vec2, vertices: &[Vec2]) -> bool {
    if vertices.len() < 3 {
        return false;
    }
    let mut sign = 0.0_f32;
    for index in 0..vertices.len() {
        let edge = vertices[(index + 1) % vertices.len()] - vertices[index];
        let cross = edge.perp_dot(point - vertices[index]);
        if cross.abs() <= CONTACT_EPSILON {
            continue;
        }
        if sign == 0.0 {
            sign = cross.signum();
        } else if sign * cross < 0.0 {
            return false;
        }
    }
    true
}

fn convex_polygons_overlap(first: &[Vec2], second: &[Vec2]) -> bool {
    first
        .iter()
        .zip(first.iter().cycle().skip(1))
        .chain(second.iter().zip(second.iter().cycle().skip(1)))
        .all(|(start, end)| {
            let edge = *end - *start;
            let axis = Vec2::new(-edge.y, edge.x).normalize();
            let project = |polygon: &[Vec2]| {
                polygon.iter().fold(
                    (f32::INFINITY, f32::NEG_INFINITY),
                    |(minimum, maximum), point| {
                        let value = point.dot(axis);
                        (minimum.min(value), maximum.max(value))
                    },
                )
            };
            let (first_minimum, first_maximum) = project(first);
            let (second_minimum, second_maximum) = project(second);
            first_maximum.min(second_maximum) - first_minimum.max(second_minimum) > CONTACT_EPSILON
        })
}

fn positive_rect_overlap(
    first: &FaceGeometry,
    second: &FaceGeometry,
    axis: Vec3,
    first_half: f32,
) -> bool {
    let FaceProfile::Rectangle { half_u, half_v } = &second.profile else {
        unreachable!()
    };
    let second_half =
        second.tangent_u.dot(axis).abs() * *half_u + second.tangent_v.dot(axis).abs() * *half_v;
    first_half + second_half - (second.center - first.center).dot(axis).abs() > CONTACT_EPSILON
}

fn annulus_rectangle_overlap(annulus: &FaceGeometry, rectangle: &FaceGeometry) -> bool {
    let FaceProfile::Annulus {
        inner_radius,
        outer_radius,
    } = &annulus.profile
    else {
        unreachable!()
    };
    let FaceProfile::Rectangle { half_u, half_v } = &rectangle.profile else {
        unreachable!()
    };
    let offset = annulus.center - rectangle.center;
    let center_u = offset.dot(rectangle.tangent_u).abs();
    let center_v = offset.dot(rectangle.tangent_v).abs();
    let nearest_u = (center_u - *half_u).max(0.0);
    let nearest_v = (center_v - *half_v).max(0.0);
    let nearest_squared = nearest_u.mul_add(nearest_u, nearest_v * nearest_v);
    let farthest_u = center_u + *half_u;
    let farthest_v = center_v + *half_v;
    let farthest_squared = farthest_u.mul_add(farthest_u, farthest_v * farthest_v);
    nearest_squared < (*outer_radius - CONTACT_EPSILON).max(0.0).powi(2)
        && farthest_squared > (*inner_radius + CONTACT_EPSILON).powi(2)
}

fn rotation_y_to_normal(normal: Vec3) -> GridRotation {
    let (axis, sign) = cardinal_axis(normal);
    match (axis, sign) {
        (0, 1) => GridRotation::new(0, 0, 3),
        (0, _) => GridRotation::new(0, 0, 1),
        (1, 1) => GridRotation::default(),
        (1, _) => GridRotation::new(2, 0, 0),
        (2, 1) => GridRotation::new(1, 0, 0),
        _ => GridRotation::new(3, 0, 0),
    }
}

fn cuboid_world_bounds(spec: CuboidSpec) -> (Vec3, Vec3) {
    let rotation = Mat3::from_quat(spec.pose.rotation.quaternion());
    let half = spec.size_meters() * 0.5;
    let world_half = Vec3::new(
        rotation.x_axis.x.abs() * half.x
            + rotation.y_axis.x.abs() * half.y
            + rotation.z_axis.x.abs() * half.z,
        rotation.x_axis.y.abs() * half.x
            + rotation.y_axis.y.abs() * half.y
            + rotation.z_axis.y.abs() * half.z,
        rotation.x_axis.z.abs() * half.x
            + rotation.y_axis.z.abs() * half.y
            + rotation.z_axis.z.abs() * half.z,
    );
    let center = spec.pose.translation();
    (center - world_half, center + world_half)
}

/// Axis-aligned authored bounds after applying the part's construction frame.
pub(crate) fn composed_part_world_bounds(
    graph: &ConstructionGraph,
    part: PartId,
) -> Option<(Vec3, Vec3)> {
    let frame = graph.part_frame(part)?;
    let (minimum, maximum) = part_world_bounds(*graph.part(part)?);
    Some(transformed_bounds(
        frame.translation(),
        frame.rotation(),
        minimum,
        maximum,
    ))
}

pub(crate) fn part_world_bounds(spec: PartSpec) -> (Vec3, Vec3) {
    match spec {
        PartSpec::Cuboid(spec) => cuboid_world_bounds(spec),
        PartSpec::Controller(spec) => cuboid_world_bounds(spec.cuboid()),
        PartSpec::Engine(spec) => cuboid_world_bounds(spec.cuboid()),
        PartSpec::Transmission(spec) => cuboid_world_bounds(spec.cuboid()),
        PartSpec::Servo(spec) => cuboid_world_bounds(spec.cuboid()),
        PartSpec::Seat(spec) => cuboid_world_bounds(spec.cuboid()),
        PartSpec::Input(spec) => cuboid_world_bounds(spec.cuboid()),
        PartSpec::DimensionLink(spec) => cuboid_world_bounds(spec.cuboid()),
        PartSpec::Cylinder(spec) => {
            let rotation = Mat3::from_quat(spec.pose.rotation.quaternion());
            let (local_minimum, local_maximum) = cylinder_local_bounds(spec.dimensions);
            let mut world_minimum = Vec3::splat(f32::INFINITY);
            let mut world_maximum = Vec3::splat(f32::NEG_INFINITY);
            for x in [local_minimum.x, local_maximum.x] {
                for y in [local_minimum.y, local_maximum.y] {
                    for z in [local_minimum.z, local_maximum.z] {
                        let point = spec.pose.translation() + rotation * Vec3::new(x, y, z);
                        world_minimum = world_minimum.min(point);
                        world_maximum = world_maximum.max(point);
                    }
                }
            }
            (world_minimum, world_maximum)
        }
        PartSpec::PipeJunction(spec) => transformed_bounds(
            spec.pose.translation(),
            spec.pose.rotation.quaternion(),
            Vec3::splat(-spec.dimensions.half_side()),
            Vec3::splat(spec.dimensions.half_side()),
        ),
        PartSpec::PipeBend(spec) => {
            let outer = spec.dimensions.outer_diameter() * 0.5;
            let radius = spec.dimensions.radius();
            let local_minimum = Vec3::new(-radius, -outer, -outer);
            let local_maximum = Vec3::new(outer, radius, outer);
            transformed_bounds(
                spec.pose.translation(),
                spec.pose.rotation.quaternion(),
                local_minimum,
                local_maximum,
            )
        }
    }
}

fn transformed_bounds(
    translation: Vec3,
    rotation: Quat,
    local_minimum: Vec3,
    local_maximum: Vec3,
) -> (Vec3, Vec3) {
    let mut world_minimum = Vec3::splat(f32::INFINITY);
    let mut world_maximum = Vec3::splat(f32::NEG_INFINITY);
    for x in [local_minimum.x, local_maximum.x] {
        for y in [local_minimum.y, local_maximum.y] {
            for z in [local_minimum.z, local_maximum.z] {
                let point = translation + rotation * Vec3::new(x, y, z);
                world_minimum = world_minimum.min(point);
                world_maximum = world_maximum.max(point);
            }
        }
    }
    (world_minimum, world_maximum)
}

fn cylinder_local_bounds(dimensions: CylinderDimensions) -> (Vec3, Vec3) {
    let outer = dimensions.outer_diameter() * 0.5;
    let inner = dimensions.inner_diameter() * 0.5;
    let half_length = dimensions.axial_length() * 0.5;
    if dimensions.sweep_angle_degrees() == 360 {
        return (
            Vec3::new(-outer, -half_length, -outer),
            Vec3::new(outer, half_length, outer),
        );
    }

    let half_sweep = dimensions.sweep_angle_radians() * 0.5;
    let mut minimum = Vec3::new(f32::INFINITY, -half_length, f32::INFINITY);
    let mut maximum = Vec3::new(f32::NEG_INFINITY, half_length, f32::NEG_INFINITY);
    for angle in [
        -half_sweep,
        half_sweep,
        -std::f32::consts::FRAC_PI_2,
        0.0,
        std::f32::consts::FRAC_PI_2,
    ] {
        if angle.abs() > half_sweep + CONTACT_EPSILON {
            continue;
        }
        for radius in [inner, outer] {
            let point = Vec3::new(radius * angle.cos(), 0.0, radius * angle.sin());
            minimum = minimum.min(point);
            maximum = maximum.max(point);
        }
    }
    (minimum, maximum)
}

#[derive(Clone, Copy)]
struct CollisionBox {
    center: Vec3,
    rotation: Quat,
    half: Vec3,
}

pub(crate) fn parts_overlap(first: PartSpec, second: PartSpec) -> bool {
    parts_overlap_with_frame(first, second, mechanic_core::ConstructionFrame::IDENTITY)
}

/// Placement candidates use the current tool-view grid; committed parts may
/// belong to another rigid frame in that same view.
fn parts_overlap_with_frame(
    candidate: PartSpec,
    target: PartSpec,
    frame: mechanic_core::ConstructionFrame,
) -> bool {
    let candidate_boxes = part_collision_boxes(candidate);
    let mut target_boxes = part_collision_boxes(target);
    if frame != mechanic_core::ConstructionFrame::IDENTITY {
        for shape in &mut target_boxes {
            shape.center = frame.point(shape.center);
            shape.rotation = frame.rotation() * shape.rotation;
        }
    }
    let (first_minimum, first_maximum) = collision_boxes_bounds(&candidate_boxes);
    let (second_minimum, second_maximum) = collision_boxes_bounds(&target_boxes);
    if (first_minimum - second_maximum)
        .cmpgt(Vec3::splat(CONTACT_EPSILON))
        .any()
        || (second_minimum - first_maximum)
            .cmpgt(Vec3::splat(CONTACT_EPSILON))
            .any()
    {
        return false;
    }
    candidate_boxes.into_iter().any(|candidate| {
        target_boxes
            .iter()
            .copied()
            .any(|target| boxes_overlap(candidate, target))
    })
}

/// Bounds the actual collision boxes, including the conservative wall boxes
/// outside a round pipe's ideal radius. Authored bounds alone can miss those.
fn collision_boxes_bounds(boxes: &[CollisionBox]) -> (Vec3, Vec3) {
    boxes.iter().fold(
        (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)),
        |(minimum, maximum), shape| {
            let extent = (shape.rotation * Vec3::X).abs() * shape.half.x
                + (shape.rotation * Vec3::Y).abs() * shape.half.y
                + (shape.rotation * Vec3::Z).abs() * shape.half.z;
            (
                minimum.min(shape.center - extent),
                maximum.max(shape.center + extent),
            )
        },
    )
}

fn part_collision_boxes(spec: PartSpec) -> Vec<CollisionBox> {
    match spec {
        PartSpec::Controller(spec) => part_collision_boxes(PartSpec::Cuboid(spec.cuboid())),
        PartSpec::Engine(spec) => part_collision_boxes(PartSpec::Cuboid(spec.cuboid())),
        PartSpec::Transmission(spec) => part_collision_boxes(PartSpec::Cuboid(spec.cuboid())),
        PartSpec::Servo(spec) => part_collision_boxes(PartSpec::Cuboid(spec.cuboid())),
        PartSpec::Seat(spec) => part_collision_boxes(PartSpec::Cuboid(spec.cuboid())),
        PartSpec::Input(spec) => part_collision_boxes(PartSpec::Cuboid(spec.cuboid())),
        PartSpec::DimensionLink(spec) => part_collision_boxes(PartSpec::Cuboid(spec.cuboid())),
        PartSpec::Cuboid(spec) => vec![CollisionBox {
            center: spec.pose.translation(),
            rotation: spec.pose.rotation.quaternion(),
            half: spec.size_meters() * 0.5,
        }],
        PartSpec::Cylinder(spec) => {
            let outer = spec.dimensions.outer_diameter() * 0.5;
            let inner = spec.dimensions.inner_diameter() * 0.5;
            let radial_half = (outer - inner) * 0.5;
            let center_radius = (outer + inner) * 0.5;
            let sweep = spec.dimensions.sweep_angle_radians();
            let segment_angle = sweep / 16.0;
            let tangent_half = outer * (segment_angle * 0.5).tan();
            let start_angle = if spec.dimensions.sweep_angle_degrees() == 360 {
                -segment_angle * 0.5
            } else {
                -sweep * 0.5
            };
            let rotation = spec.pose.rotation.quaternion();
            (0_u16..16)
                .map(|segment| {
                    let angle = start_angle + segment_angle * (f32::from(segment) + 0.5);
                    let radial = Vec3::new(angle.cos(), 0.0, angle.sin());
                    CollisionBox {
                        center: spec.pose.translation() + rotation * (radial * center_radius),
                        rotation: rotation * Quat::from_rotation_y(-angle),
                        half: Vec3::new(
                            radial_half,
                            spec.dimensions.axial_length() * 0.5,
                            tangent_half,
                        ),
                    }
                })
                .collect()
        }
        PartSpec::PipeBend(spec) => pipe_bend_collision_boxes(spec),
        PartSpec::PipeJunction(spec) => {
            let rotation = spec.pose.rotation.quaternion();
            mechanic_core::pipe_junction_wall_boxes(spec)
                .into_iter()
                .map(|wall| CollisionBox {
                    center: spec.pose.translation() + rotation * wall.center,
                    rotation: rotation * wall.rotation,
                    half: wall.half_extents,
                })
                .collect()
        }
    }
}

fn pipe_bend_collision_boxes(spec: PipeBendSpec) -> Vec<CollisionBox> {
    let outer = spec.dimensions.outer_diameter() * 0.5;
    let inner = spec.dimensions.inner_diameter() * 0.5;
    let half_radial = (outer - inner) * 0.5;
    let cross_radius = (outer + inner) * 0.5;
    let bend_radius = spec.dimensions.radius();
    let bend_step = std::f32::consts::FRAC_PI_2 / 12.0;
    let cross_step = std::f32::consts::TAU / 16.0;
    let part_rotation = spec.pose.rotation.quaternion();
    let mut boxes = Vec::with_capacity(12 * 16);
    for bend_slice in 0_u16..12 {
        let theta = -std::f32::consts::FRAC_PI_2 + bend_step * (f32::from(bend_slice) + 0.5);
        let radial = Vec3::new(theta.cos(), theta.sin(), 0.0);
        let tangent = Vec3::new(-theta.sin(), theta.cos(), 0.0);
        for sector in 0_u16..16 {
            let phi = cross_step * (f32::from(sector) + 0.5);
            let normal = radial * phi.cos() + Vec3::Z * phi.sin();
            let cross_tangent = -radial * phi.sin() + Vec3::Z * phi.cos();
            let mut center = Vec3::new(-bend_radius, bend_radius, 0.0)
                + radial * (bend_radius + cross_radius * phi.cos())
                + Vec3::Z * (cross_radius * phi.sin());
            let mut half = Vec3::new(
                half_radial,
                (bend_radius + outer) * (bend_step * 0.5).tan(),
                outer * (cross_step * 0.5).tan(),
            );
            // Tight bends reach their end planes from inner slices too, so
            // every box stays behind both caps. Creased bends leave slivers at
            // the crease that no shortening pulls back; they hold no material.
            let mut protrudes = false;
            for (plane_center, outward) in [
                (Vec3::new(-bend_radius, 0.0, 0.0), Vec3::NEG_X),
                (Vec3::new(0.0, bend_radius, 0.0), Vec3::Y),
            ] {
                protrudes |= trim_pipe_bend_box_to_end_plane(
                    &mut center,
                    &mut half,
                    [normal, tangent, cross_tangent],
                    plane_center,
                    outward,
                );
            }
            if protrudes {
                continue;
            }
            boxes.push(CollisionBox {
                center: spec.pose.translation() + part_rotation * center,
                rotation: part_rotation
                    * Quat::from_mat3(&Mat3::from_cols(normal, tangent, cross_tangent)),
                half,
            });
        }
    }
    boxes
}

/// Keeps the conservative bend tessellation behind its two exact tangent caps.
/// The box is shortened along whichever of its axes faces the cap most
/// directly; the bend's material never crosses either cap plane. Returns
/// whether the box still crosses the cap afterwards.
fn trim_pipe_bend_box_to_end_plane(
    center: &mut Vec3,
    half: &mut Vec3,
    axes: [Vec3; 3],
    plane_center: Vec3,
    outward: Vec3,
) -> bool {
    let projections = axes.map(|axis| axis.dot(outward));
    let reach = |center: Vec3, half: Vec3| {
        (center - plane_center).dot(outward)
            + half.x * projections[0].abs()
            + half.y * projections[1].abs()
            + half.z * projections[2].abs()
    };
    let protrusion = reach(*center, *half) + CONTACT_EPSILON;
    if protrusion <= 0.0 {
        return false;
    }
    let index = (0..3)
        .max_by(|&left, &right| projections[left].abs().total_cmp(&projections[right].abs()))
        .expect("a box has three axes");
    let trim = (protrusion / projections[index].abs()).min(half[index] * 2.0);
    *center -= axes[index] * projections[index].signum() * trim * 0.5;
    half[index] -= trim * 0.5;
    reach(*center, *half) > 0.0
}

fn boxes_overlap(first: CollisionBox, second: CollisionBox) -> bool {
    let first_axes = [
        first.rotation * Vec3::X,
        first.rotation * Vec3::Y,
        first.rotation * Vec3::Z,
    ];
    let second_axes = [
        second.rotation * Vec3::X,
        second.rotation * Vec3::Y,
        second.rotation * Vec3::Z,
    ];
    let offset = second.center - first.center;
    let separates = |axis: Vec3| {
        if axis.length_squared() <= 1.0e-10 {
            return false;
        }
        let axis = axis.normalize();
        let radius = |axes: [Vec3; 3], half: Vec3| {
            axes[0].dot(axis).abs() * half.x
                + axes[1].dot(axis).abs() * half.y
                + axes[2].dot(axis).abs() * half.z
        };
        offset.dot(axis).abs()
            >= radius(first_axes, first.half) + radius(second_axes, second.half) - CONTACT_EPSILON
    };
    if first_axes.into_iter().chain(second_axes).any(separates) {
        return false;
    }
    for first_axis in first_axes {
        for second_axis in second_axes {
            if separates(first_axis.cross(second_axis)) {
                return false;
            }
        }
    }
    true
}

fn bounds_overlap_interior(
    first_minimum: Vec3,
    first_maximum: Vec3,
    second_minimum: Vec3,
    second_maximum: Vec3,
) -> bool {
    (first_minimum.x < second_maximum.x - CONTACT_EPSILON
        && first_maximum.x > second_minimum.x + CONTACT_EPSILON)
        && (first_minimum.y < second_maximum.y - CONTACT_EPSILON
            && first_maximum.y > second_minimum.y + CONTACT_EPSILON)
        && (first_minimum.z < second_maximum.z - CONTACT_EPSILON
            && first_maximum.z > second_minimum.z + CONTACT_EPSILON)
}

fn cardinal_axis(normal: Vec3) -> (usize, i32) {
    let absolute = normal.abs();
    let axis = if absolute.x > absolute.y && absolute.x > absolute.z {
        0
    } else if absolute.y > absolute.z {
        1
    } else {
        2
    };
    (axis, if normal[axis] >= 0.0 { 1 } else { -1 })
}

fn cardinal_direction(direction: Vec3) -> Vec3 {
    let (axis, sign) = cardinal_axis(direction);
    let mut cardinal = Vec3::ZERO;
    cardinal[axis] = if sign > 0 { 1.0 } else { -1.0 };
    cardinal
}

fn face_for_normal(normal: Vec3) -> FaceKind {
    let (axis, sign) = cardinal_axis(normal);
    match (axis, sign) {
        (0, 1) => FaceKind::PositiveX,
        (0, _) => FaceKind::NegativeX,
        (1, 1) => FaceKind::PositiveY,
        (1, _) => FaceKind::NegativeY,
        (2, 1) => FaceKind::PositiveZ,
        _ => FaceKind::NegativeZ,
    }
}

fn snap_cardinal(vector: Vec3) -> Vec3 {
    Vec3::new(vector.x.round(), vector.y.round(), vector.z.round())
}

pub(crate) fn suspension_block_candidate(
    socket: crate::PlacedBearing,
) -> Result<PlacementCandidate, PlacementError> {
    let BearingKind::Suspension(spec) = socket.kind else {
        return Err(PlacementError::BearingOutsideFace);
    };
    let center = socket.anchor + socket.axis * (spec.initial_length() + 0.125);
    Ok(PlacementCandidate {
        spec: CuboidSpec::new(
            [1; 3],
            BuildPose::from_position_ticks(
                snap_world_to_position_ticks(center),
                GridRotation::default(),
            ),
        )
        .map_err(|e| PlacementError::Graph(e.to_string()))?,
        attached_face: face_for_normal(-socket.axis),
        anchor: Some(socket.anchor + socket.axis * spec.initial_length()),
        support: PlacementSupport::Bearing,
    })
}
pub(crate) fn suspension_cylinder_candidate(
    socket: crate::PlacedBearing,
    dimensions: CylinderDimensions,
) -> Result<CylinderPlacementCandidate, PlacementError> {
    let BearingKind::Suspension(spec) = socket.kind else {
        return Err(PlacementError::BearingOutsideFace);
    };
    let center =
        socket.anchor + socket.axis * (spec.initial_length() + dimensions.axial_length() / 2.0);
    Ok(CylinderPlacementCandidate {
        spec: CylinderSpec::new(
            dimensions,
            BuildPose::from_position_ticks(
                snap_world_to_position_ticks(center),
                rotation_y_to_normal(socket.axis),
            ),
        ),
        attached_face: FaceKind::NegativeY,
        anchor: Some(socket.anchor + socket.axis * spec.initial_length()),
        support: PlacementSupport::Bearing,
    })
}
fn suspension_attachment(
    socket: crate::PlacedBearing,
    targets: &[PartId],
) -> BearingAttachment<'_> {
    BearingAttachment {
        source: socket.source,
        anchor: socket.anchor,
        dimensions: socket.dimensions,
        kind: socket.kind,
        axis: socket.axis,
        rigid_targets: targets,
    }
}
pub(crate) fn stage_suspension_block(
    graph: &ConstructionGraph,
    socket: crate::PlacedBearing,
    candidate: PlacementCandidate,
    targets: &[PartId],
    bounds: PlacementBounds,
) -> Result<ConstructionGraph, PlacementError> {
    stage_connected_block_batch(
        graph,
        candidate,
        &[candidate.spec],
        Some(suspension_attachment(socket, targets)),
        None,
        bounds,
    )
}
pub(crate) fn stage_suspension_cylinder(
    graph: &ConstructionGraph,
    socket: crate::PlacedBearing,
    candidate: CylinderPlacementCandidate,
    targets: &[PartId],
    bounds: PlacementBounds,
) -> Result<ConstructionGraph, PlacementError> {
    stage_connected_cylinder(
        graph,
        candidate,
        Some(suspension_attachment(socket, targets)),
        None,
        bounds,
    )
}

#[cfg(test)]
mod tests;
