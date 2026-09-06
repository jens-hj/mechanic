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
    PartPiece, PartSpec, PipeBendDimensions, PipeBendSpec, RigidLinkSpec, SeatSpec, ServoSpec,
    ShapeRegion, TransmissionSpec, WeldSpec,
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
    World {
        origin: DVec2,
    },
}

impl PlacementBounds {
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
                .evaluated_solid(mechanic_core::SolidOwner::Region(id))
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
            .evaluated_solid(mechanic_core::SolidOwner::Part(part))
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

pub(crate) fn pipe_run_pieces(
    points: &[Vec3],
    bend_radii: &[f32],
    dimensions: CylinderDimensions,
    material: mechanic_core::ConstructionMaterial,
) -> Result<Vec<PipeRunPiece>, PlacementError> {
    if dimensions.sweep_angle_degrees() != 360 && !bend_radii.is_empty() {
        return Err(PlacementError::PipeRun(
            "partial-cylinder sectors support straight runs only".to_owned(),
        ));
    }
    let (directions, lengths) = pipe_path_segments(points, bend_radii)?;
    let bend_dimensions = bend_radii
        .iter()
        .copied()
        .map(|radius| {
            PipeBendDimensions::new(
                dimensions.outer_diameter(),
                dimensions.inner_diameter(),
                radius,
            )
            .map_err(|error| PlacementError::PipeRun(error.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut pieces = Vec::new();
    for segment in 0..directions.len() {
        append_pipe_segment(
            &mut pieces,
            points,
            &directions,
            &lengths,
            bend_radii,
            &bend_dimensions,
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

fn pipe_path_segments(
    points: &[Vec3],
    bend_radii: &[f32],
) -> Result<(Vec<Vec3>, Vec<f32>), PlacementError> {
    if points.len() < 2 || bend_radii.len() + 2 != points.len() {
        return Err(PlacementError::PipeRun(
            "pipe run path and bend counts do not match".to_owned(),
        ));
    }
    let mut directions = Vec::with_capacity(points.len() - 1);
    let mut lengths = Vec::with_capacity(points.len() - 1);
    for segment in points.windows(2) {
        let delta = segment[1] - segment[0];
        let length = delta.length();
        if length < GRID_UNIT_METERS - CONTACT_EPSILON
            || (length / GRID_UNIT_METERS - (length / GRID_UNIT_METERS).round()).abs() > 1.0e-4
        {
            return Err(PlacementError::PipeRun(
                "pipe legs must be positive whole-block lengths".to_owned(),
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
    for (corner, pair) in directions.windows(2).enumerate() {
        if pair[0].dot(pair[1]).abs() > CONTACT_EPSILON {
            return Err(PlacementError::PipeRun(format!(
                "bend {} must turn exactly 90°",
                corner + 1
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
    bend_radii: &[f32],
    bend_dimensions: &[PipeBendDimensions],
    dimensions: CylinderDimensions,
    material: mechanic_core::ConstructionMaterial,
    segment: usize,
) -> Result<(), PlacementError> {
    let start_trim = segment
        .checked_sub(1)
        .and_then(|bend| bend_radii.get(bend))
        .copied()
        .unwrap_or(0.0);
    let end_trim = bend_radii.get(segment).copied().unwrap_or(0.0);
    let residual = lengths[segment] - start_trim - end_trim;
    if residual < -CONTACT_EPSILON {
        let required = start_trim + end_trim;
        return Err(PlacementError::PipeRun(format!(
            "leg {} needs {:.2} m clearance for adjacent bends",
            segment + 1,
            required
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
    if let Some(&bend_dimensions) = bend_dimensions.get(segment) {
        let rotation = rotation_xy_to_directions(directions[segment], directions[segment + 1])
            .ok_or_else(|| {
                PlacementError::PipeRun("pipe turn has no cardinal orientation".to_owned())
            })?;
        let corner_ticks = snap_world_to_position_ticks(points[segment + 1]);
        pieces.push(PipeRunPiece {
            spec: PartSpec::PipeBend(
                PipeBendSpec::new(
                    bend_dimensions,
                    BuildPose::from_position_ticks(corner_ticks, rotation),
                )
                .with_material(material),
            ),
            inlet: FaceKind::NegativeX,
            outlet: FaceKind::PositiveY,
        });
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
            _ => unreachable!("pipe runs contain only straights and bends"),
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
        PlacementBounds::Garage | PlacementBounds::GarageBuild => IVec3::ZERO,
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
            let Ok(solid) = graph.evaluated_solid(owner) else {
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
    }
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
    }
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
    part_collision_boxes(first).into_iter().any(|first| {
        part_collision_boxes(second)
            .into_iter()
            .any(|second| boxes_overlap(first, second))
    })
}

/// Placement candidates use the current tool-view grid; committed parts may
/// belong to another rigid frame in that same view.
fn parts_overlap_with_frame(
    candidate: PartSpec,
    target: PartSpec,
    frame: mechanic_core::ConstructionFrame,
) -> bool {
    if frame == mechanic_core::ConstructionFrame::IDENTITY {
        return parts_overlap(candidate, target);
    }
    let target_boxes = part_collision_boxes(target)
        .into_iter()
        .map(|shape| CollisionBox {
            center: frame.point(shape.center),
            rotation: frame.rotation() * shape.rotation,
            ..shape
        })
        .collect::<Vec<_>>();
    part_collision_boxes(candidate)
        .into_iter()
        .any(|candidate| {
            target_boxes
                .iter()
                .copied()
                .any(|target| boxes_overlap(candidate, target))
        })
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
            if bend_slice == 0 {
                trim_pipe_bend_box_to_end_plane(
                    &mut center,
                    &mut half,
                    [normal, tangent, cross_tangent],
                    Vec3::new(-bend_radius, 0.0, 0.0),
                    Vec3::NEG_X,
                );
            } else if bend_slice == 11 {
                trim_pipe_bend_box_to_end_plane(
                    &mut center,
                    &mut half,
                    [normal, tangent, cross_tangent],
                    Vec3::new(0.0, bend_radius, 0.0),
                    Vec3::Y,
                );
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
fn trim_pipe_bend_box_to_end_plane(
    center: &mut Vec3,
    half: &mut Vec3,
    axes: [Vec3; 3],
    plane_center: Vec3,
    outward: Vec3,
) {
    let [normal, tangent, cross_tangent] = axes;
    let tangent_projection = tangent.dot(outward);
    let extent = half.x * normal.dot(outward).abs()
        + half.y * tangent_projection.abs()
        + half.z * cross_tangent.dot(outward).abs();
    let protrusion = (*center - plane_center).dot(outward) + extent + CONTACT_EPSILON;
    if protrusion <= 0.0 {
        return;
    }
    let trim = (protrusion / tangent_projection.abs()).min(half.y * 2.0);
    *center -= tangent * tangent_projection.signum() * trim * 0.5;
    half.y -= trim * 0.5;
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

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use bevy::{
        math::DVec2,
        prelude::{IVec3, Quat, Vec3},
    };
    use mechanic_core::{
        BearingDimensions, BearingSpec, BuildCommand, BuildOutcome, BuildPose, ConstructionGraph,
        ConstructionMaterial, CuboidSpec, CylinderDimensions, CylinderSpec, DimensionLinkId,
        EdgeChainRef, EdgeTreatment, EngineKind, EngineSpec, FaceKind, FaceOwner, FaceRef,
        GridRotation, POSITION_TICKS_PER_GRID_UNIT, PartId, PartSpec, PendingOperation,
        PipeBendDimensions, PipeBendSpec, RigidLinkSpec, ShapeFeature, SolidOwner, WeldSpec,
    };

    use super::{
        AxisGuide, BLOCK_SIZE_METERS, BlockVolume, GuideKind, PipeRunAttachment, PipeRunPiece,
        PlacementBounds, PlacementCandidate, PlacementError, PlacementGrid, PlacementPlane,
        PlacementSnapIndex, PlacementSupport, SurfaceHit, bearing_anchor_from_hit,
        bearing_attachment_candidate, bearing_overlaps_candidate, bearing_ring_overlaps_face,
        bearing_support_face, begin_weld, block_box_bounds, block_box_specs, block_sheet_specs,
        block_span_from_rays, candidate_from_hit, center_cylinder_candidate_on_bearing,
        cuboid_candidate_from_hit, cylinder_candidate_from_hit, face_geometry_from_ref,
        face_is_flat, free_cuboid_candidate, free_cylinder_candidate, locked_bearings,
        newly_locked_bearings, oriented_cuboid_candidate_from_hit,
        oriented_cuboid_candidate_from_hit_with_grid, pipe_run_pieces, raycast_construction,
        raycast_construction_for_annulus, raycast_construction_with_ground,
        raycast_placement_plane_point, raycast_sources, render_free_smart_guides, rigid_body_parts,
        smart_snap_block_span, smart_snap_cuboid_candidate, smart_snap_free_cuboid_candidate,
        stage_bearing_attachment, stage_bearing_block_batch, stage_block_batch,
        stage_block_batch_from_source, stage_block_batch_from_source_in_bounds,
        stage_block_batch_in_bounds, stage_block_volume_in_bounds, stage_controller_in_bounds,
        stage_cuboid, stage_cylinder_from_source, stage_dimension_link_in_bounds,
        stage_engine_from_source, stage_engine_in_bounds, stage_input_in_bounds, stage_pipe_run,
        stage_pipe_run_in_bounds, stage_seat_in_bounds, stage_servo_in_bounds, stage_transmission,
        stage_weld_objects, transmission_candidate_from_hit, validate_block_batch_in_bounds,
        validate_indexed_block_batch_in_bounds, validate_part,
    };

    #[test]
    fn linear_candidates_are_flush_and_lattice_snapped_on_every_face_orientation() {
        use mechanic_core::{CarriageFace, LinearBearing, LinearBearingDimensions};
        for length in [0.25, 1.0, 8.0] {
            for normal in [
                Vec3::X,
                Vec3::Y,
                Vec3::Z,
                Vec3::NEG_X,
                Vec3::NEG_Y,
                Vec3::NEG_Z,
            ] {
                for axis in [
                    Vec3::X,
                    Vec3::Y,
                    Vec3::Z,
                    Vec3::NEG_X,
                    Vec3::NEG_Y,
                    Vec3::NEG_Z,
                ] {
                    if normal.dot(axis) != 0.0 {
                        continue;
                    }
                    for face in [
                        CarriageFace::Top,
                        CarriageFace::PositiveSide,
                        CarriageFace::NegativeSide,
                    ] {
                        let rail = LinearBearing {
                            dimensions: LinearBearingDimensions::new(length, 0.1).unwrap(),
                            mount_normal: normal,
                            face,
                        };
                        let surface = super::linear_carriage_face(Vec3::ZERO, rail, axis).unwrap();
                        let hit =
                            surface.center + surface.tangent_u * 0.029 + surface.tangent_v * 0.009;
                        let candidate = super::linear_block_candidate(
                            Vec3::ZERO,
                            rail,
                            axis,
                            hit,
                            [1; 3],
                            GridRotation::default(),
                        )
                        .unwrap();
                        let block_face =
                            super::face_geometry(candidate.spec, candidate.attached_face);
                        assert!(
                            (block_face.center - surface.center)
                                .dot(surface.normal)
                                .abs()
                                < 1.0e-6
                        );
                        assert!((block_face.normal + surface.normal).length() < 1.0e-6);
                        assert!(candidate.anchor.is_some());
                        assert!(
                            ((block_face.center - surface.center).dot(axis) - 0.025).abs() < 1.0e-6
                        );
                        let cylinder = super::linear_cylinder_candidate(
                            Vec3::ZERO,
                            rail,
                            axis,
                            hit,
                            CylinderDimensions::new(0.25, 0.0, 0.25).unwrap(),
                            1,
                        )
                        .unwrap();
                        let cylinder_face =
                            super::cylinder_face_geometry(cylinder.spec, cylinder.attached_face)
                                .unwrap();
                        assert!(
                            (cylinder_face.center - surface.center)
                                .dot(surface.normal)
                                .abs()
                                < 1.0e-6
                        );
                        assert!(cylinder.anchor.is_some());
                    }
                }
            }
        }
    }

    #[test]
    fn linear_deleted_support_migrates_to_overlapping_coplanar_survivor() {
        use mechanic_core::{CarriageFace, LinearBearing, LinearBearingDimensions};
        let mut graph = ConstructionGraph::new();
        let original = spawn_cube(&mut graph, IVec3::ZERO, 1);
        let survivor = spawn_cube(&mut graph, IVec3::X, 1);
        spawn_cube(&mut graph, IVec3::new(-1, 1, 0), 1);
        spawn_cube(&mut graph, IVec3::new(8, 0, 0), 1);
        let selected = FaceRef::part(original, FaceKind::PositiveY);
        let anchor = Vec3::new(0.0, 0.125, 0.0);
        let rail = LinearBearing {
            dimensions: LinearBearingDimensions::default(),
            mount_normal: Vec3::Y,
            face: CarriageFace::Top,
        };
        let mut deleted = std::collections::HashSet::from([original]);
        let replacement =
            super::linear_support_face_excluding(&graph, selected, anchor, rail, Vec3::X, &deleted);
        assert_eq!(
            replacement,
            Some(FaceRef::part(survivor, FaceKind::PositiveY))
        );
        // The anchor lies outside the survivor, but the long rail still overlaps it.
        assert!(
            anchor.x
                < super::face_geometry_from_ref(replacement.unwrap(), Some(&graph))
                    .center
                    .x
                    - 0.125
        );
        deleted.insert(survivor);
        assert_eq!(
            super::linear_support_face_excluding(&graph, selected, anchor, rail, Vec3::X, &deleted),
            None,
            "raised or distant faces must not rescue an unsupported rail"
        );
    }

    #[test]
    fn linear_rail_overhang_and_flush_side_attachment_use_real_support_overlap() {
        use mechanic_core::{CarriageFace, LinearBearing, LinearBearingDimensions};
        for (face, side) in [
            (CarriageFace::PositiveSide, 1.0),
            (CarriageFace::NegativeSide, -1.0),
        ] {
            let mut graph = ConstructionGraph::new();
            let base = spawn_cube(&mut graph, IVec3::ZERO, 1);
            let source = FaceRef::part(base, FaceKind::PositiveY);
            let rail = LinearBearing {
                dimensions: LinearBearingDimensions::default(),
                mount_normal: Vec3::Y,
                face,
            };
            let anchor = Vec3::new(0.0, 0.125, side * 0.1);
            assert!(super::linear_mount_overlaps_face(
                &graph,
                source,
                anchor,
                rail,
                Vec3::X
            ));
            assert!(!super::linear_mount_overlaps_face(
                &graph,
                source,
                anchor + Vec3::Z,
                rail,
                Vec3::X
            ));
            let surface = super::linear_carriage_face(anchor, rail, Vec3::X).unwrap();
            let candidate = super::linear_block_candidate(
                anchor,
                rail,
                Vec3::X,
                surface.center,
                [1; 3],
                GridRotation::default(),
            )
            .unwrap();
            let staged = super::stage_linear_block_batch_in_bounds(
                &graph,
                candidate,
                &[candidate.spec],
                super::LinearAttachment {
                    source,
                    anchor,
                    rail,
                    axis: Vec3::X,
                    rigid_targets: &[],
                },
                PlacementBounds::Garage,
            )
            .unwrap();
            assert_eq!(staged.compile().unwrap().bearings.len(), 1);
        }
    }

    #[test]
    fn linear_block_and_cylinder_direct_attachments_share_one_moving_compound() {
        use mechanic_core::{CarriageFace, LinearBearing, LinearBearingDimensions};
        let mut graph = ConstructionGraph::new();
        let base = spawn_cube(&mut graph, IVec3::ZERO, 1);
        let source = FaceRef::part(base, FaceKind::PositiveY);
        let rail = LinearBearing {
            dimensions: LinearBearingDimensions::default(),
            mount_normal: Vec3::Y,
            face: CarriageFace::Top,
        };
        let anchor = Vec3::new(0.0, 0.125, 0.0);
        let surface = super::linear_carriage_face(anchor, rail, Vec3::X).unwrap();
        let block = super::linear_block_candidate(
            anchor,
            rail,
            Vec3::X,
            surface.center - Vec3::Z * 0.125,
            [1; 3],
            GridRotation::default(),
        )
        .unwrap();
        let attachment = super::LinearAttachment {
            source,
            anchor,
            rail,
            axis: Vec3::X,
            rigid_targets: &[],
        };
        let mut index = PlacementSnapIndex::default();
        index.rebuild(&graph);
        let placed = super::stage_linear_block_volume_in_bounds(
            &graph,
            &index,
            block,
            BlockVolume::new(block.spec, IVec3::ZERO).unwrap(),
            attachment,
            PlacementBounds::Garage,
            7,
        )
        .unwrap();
        assert_eq!(placed.publication_generation, 7);
        let graph = placed.graph;
        let targets = placed.new_parts;
        let cylinder = super::linear_cylinder_candidate(
            anchor,
            rail,
            Vec3::X,
            surface.center + Vec3::Z * 0.125,
            CylinderDimensions::new(0.25, 0.0, 0.25).unwrap(),
            0,
        )
        .unwrap();
        let graph = super::stage_linear_cylinder_in_bounds(
            &graph,
            cylinder,
            super::LinearAttachment {
                rigid_targets: &targets,
                ..attachment
            },
            PlacementBounds::Garage,
        )
        .unwrap();
        let compiled = graph.compile().unwrap();
        assert_eq!(compiled.compounds.len(), 2);
        assert_eq!(compiled.bearings.len(), 1);
        assert!(matches!(
            compiled.bearings[0].kind,
            mechanic_core::BearingKind::Linear(_)
        ));
        let side_rail = LinearBearing {
            face: CarriageFace::PositiveSide,
            ..rail
        };
        let side_surface = super::linear_carriage_face(anchor, side_rail, Vec3::X).unwrap();
        let side = super::linear_block_candidate(
            anchor,
            side_rail,
            Vec3::X,
            side_surface.center,
            [1; 3],
            GridRotation::default(),
        )
        .unwrap();
        assert!(
            super::stage_linear_block_batch_in_bounds(
                &graph,
                side,
                &[side.spec],
                super::LinearAttachment {
                    rail: side_rail,
                    rigid_targets: &targets,
                    ..attachment
                },
                PlacementBounds::Garage
            )
            .is_err()
        );
    }

    fn spawn_cube(graph: &mut ConstructionGraph, units: IVec3, size: u8) -> mechanic_core::PartId {
        let spec =
            CuboidSpec::new([size; 3], BuildPose::new(units, GridRotation::default())).unwrap();
        let Ok(BuildOutcome::Spawned(part)) = graph.apply(BuildCommand::Spawn(spec)) else {
            panic!("cube must spawn");
        };
        part
    }

    fn ground_volume_candidate(start_ticks: IVec3) -> PlacementCandidate {
        PlacementCandidate {
            spec: CuboidSpec::new(
                [1; 3],
                BuildPose::from_position_ticks(start_ticks, GridRotation::default()),
            )
            .unwrap(),
            attached_face: FaceKind::NegativeY,
            anchor: Some(start_ticks.as_vec3() * mechanic_core::POSITION_TICK_METERS),
            support: PlacementSupport::Surface(FaceOwner::Ground),
        }
    }

    fn spawn_cylinder(
        graph: &mut ConstructionGraph,
        dimensions: CylinderDimensions,
        pose: BuildPose,
    ) -> mechanic_core::PartId {
        let BuildOutcome::Spawned(part) = graph
            .apply(BuildCommand::SpawnCylinder(CylinderSpec::new(
                dimensions, pose,
            )))
            .unwrap()
        else {
            unreachable!()
        };
        part
    }

    #[test]
    fn fixed_global_grid_quantises_all_three_modifier_modes() {
        let graph = ConstructionGraph::new();
        let candidate = |point: Vec3, grid| {
            oriented_cuboid_candidate_from_hit_with_grid(
                &graph,
                SurfaceHit {
                    distance: 1.0,
                    point,
                    face: FaceRef::ground(),
                },
                [1, 1, 1],
                GridRotation::default(),
                grid,
                PlacementBounds::Garage,
            )
        };

        assert_eq!(
            candidate(Vec3::new(0.11, 0.0, -0.11), PlacementGrid::Centimetres25)
                .spec
                .pose
                .translation_position_ticks(),
            IVec3::new(0, 50, 0)
        );
        assert_eq!(
            candidate(Vec3::new(0.038, 0.0, -0.038), PlacementGrid::Centimetres5)
                .spec
                .pose
                .translation_position_ticks(),
            IVec3::new(20, 50, -20)
        );
        assert_eq!(
            candidate(Vec3::new(0.006, 0.0, -0.006), PlacementGrid::Centimetres1)
                .spec
                .pose
                .translation_position_ticks(),
            IVec3::new(4, 50, -4)
        );
    }

    #[test]
    fn global_grid_uses_absolute_world_origin_and_object_parity() {
        let graph = ConstructionGraph::new();
        let candidate = oriented_cuboid_candidate_from_hit_with_grid(
            &graph,
            SurfaceHit {
                distance: 1.0,
                point: Vec3::new(0.013, 0.0, 0.013),
                face: FaceRef::ground(),
            },
            [2, 1, 2],
            GridRotation::default(),
            PlacementGrid::Centimetres1,
            PlacementBounds::World {
                origin: DVec2::new(10.0, -20.0),
            },
        );
        let global = candidate.spec.pose.translation() + Vec3::new(10.0, 0.0, -20.0);
        assert!(((global.x - 0.005) / 0.01).fract().abs() < 1.0e-3);
        assert!(((global.z - 0.005) / 0.01).fract().abs() < 1.0e-3);
        assert!((global.y - 0.125).abs() < 1.0e-6);
    }

    #[test]
    fn smart_snap_combines_center_and_edge_guides_and_falls_back_when_invalid() {
        let mut graph = ConstructionGraph::new();
        graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new(
                    [1, 1, 1],
                    BuildPose::from_position_ticks(IVec3::new(6, 50, 400), GridRotation::default()),
                )
                .unwrap(),
            ))
            .unwrap();
        let hit = SurfaceHit {
            distance: 1.0,
            point: Vec3::new(0.014, 0.0, 0.74),
            face: FaceRef::ground(),
        };
        let gridded = oriented_cuboid_candidate_from_hit_with_grid(
            &graph,
            hit,
            [1, 1, 1],
            GridRotation::default(),
            PlacementGrid::Centimetres25,
            PlacementBounds::Garage,
        );
        let mut index = PlacementSnapIndex::default();
        index.rebuild(&graph);
        let (snapped_candidate, active_guides) = smart_snap_cuboid_candidate(
            &graph,
            &index,
            hit,
            gridded,
            PlacementGrid::Centimetres25,
            1.0,
            |_| true,
        );
        assert_eq!(active_guides.len(), 2);
        assert_eq!(
            snapped_candidate.spec.pose.translation_position_ticks(),
            IVec3::new(6, 50, 300)
        );

        let (fallback, rejected_guides) = smart_snap_cuboid_candidate(
            &graph,
            &index,
            hit,
            gridded,
            PlacementGrid::Centimetres25,
            1.0,
            |_| false,
        );
        assert!(rejected_guides.is_empty());
        assert_eq!(fallback.spec.pose, gridded.spec.pose);
    }

    #[test]
    fn dense_aligned_parts_do_not_multiply_smart_snap_validation() {
        let mut graph = ConstructionGraph::new();
        graph
            .apply_batch((-3..=3).flat_map(|x| {
                (-3..=3).map(move |z| {
                    BuildCommand::Spawn(
                        CuboidSpec::new(
                            [1; 3],
                            BuildPose::from_position_ticks(
                                IVec3::new(x * 100, 50, z * 100),
                                GridRotation::default(),
                            ),
                        )
                        .unwrap(),
                    )
                })
            }))
            .unwrap();
        let hit = SurfaceHit {
            distance: 1.0,
            point: Vec3::new(0.0, 0.0, 0.0),
            face: FaceRef::ground(),
        };
        let gridded = oriented_cuboid_candidate_from_hit_with_grid(
            &graph,
            hit,
            [1; 3],
            GridRotation::default(),
            PlacementGrid::Centimetres25,
            PlacementBounds::Garage,
        );
        let mut index = PlacementSnapIndex::default();
        index.rebuild(&graph);
        let mut validation_count = 0;

        let (fallback, guides) = smart_snap_cuboid_candidate(
            &graph,
            &index,
            hit,
            gridded,
            PlacementGrid::Centimetres25,
            1.0,
            |_| {
                validation_count += 1;
                false
            },
        );

        assert_eq!(fallback.spec.pose, gridded.spec.pose);
        assert!(guides.is_empty());
        assert!(
            validation_count <= 25,
            "equivalent guides caused {validation_count} duplicate validations"
        );
    }

    #[test]
    fn framed_overlap_validation_agrees_across_snap_batch_and_volume_paths() {
        let mut graph = ConstructionGraph::new();
        let BuildOutcome::Spawned(target) = graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new([8, 1, 1], BuildPose::default()).unwrap(),
            ))
            .unwrap()
        else {
            unreachable!()
        };
        graph
            .reframe_parts(
                [target],
                mechanic_core::ConstructionFrame::new(
                    Vec3::new(2.0, 1.0, 3.0),
                    Quat::from_rotation_y(std::f32::consts::FRAC_PI_4),
                )
                .unwrap(),
            )
            .unwrap();
        let mut index = PlacementSnapIndex::default();
        index.rebuild(&graph);
        let bounds = PlacementBounds::World {
            origin: DVec2::ZERO,
        };
        // Inside the rotated rod, at its obsolete raw origin, and inside its
        // AABB but outside its actual oriented volume, respectively.
        for (units, overlaps) in [
            (IVec3::new(8, 4, 12), true),
            (IVec3::ZERO, false),
            (IVec3::new(10, 4, 14), false),
        ] {
            let spec =
                CuboidSpec::new([1; 3], BuildPose::new(units, GridRotation::default())).unwrap();
            let start = PlacementCandidate {
                spec,
                attached_face: FaceKind::NegativeY,
                anchor: None,
                support: PlacementSupport::Free,
            };
            let volume = BlockVolume::new(spec, IVec3::ZERO).unwrap();
            if units != IVec3::ZERO {
                let (minimum, maximum) = super::part_world_bounds(PartSpec::Cuboid(spec));
                assert!(
                    index
                        .nearby(minimum, maximum, 0.0)
                        .iter()
                        .any(|row| row.part == target)
                );
            }
            assert_eq!(index.overlaps(PartSpec::Cuboid(spec)), overlaps);
            let expected = if overlaps {
                Err(PlacementError::OverlapsPart(target))
            } else {
                Ok(())
            };
            assert_eq!(
                validate_indexed_block_batch_in_bounds(&index, start, &[spec], bounds),
                expected
            );
            assert_eq!(
                validate_block_batch_in_bounds(&graph, start, &[spec], bounds),
                expected
            );
            assert_eq!(
                super::validate_block_volume_in_bounds(&graph, &index, start, volume, bounds),
                expected
            );
        }
    }

    #[test]
    fn framed_overlap_keeps_cylinder_bores_empty_and_identity_behavior_exact() {
        let mut graph = ConstructionGraph::new();
        let target = spawn_cylinder(
            &mut graph,
            CylinderDimensions::new(2.0, 1.0, 0.5).unwrap(),
            BuildPose::default(),
        );
        let frame = mechanic_core::ConstructionFrame::new(
            Vec3::new(2.0, 1.0, 3.0),
            Quat::from_rotation_y(0.6),
        )
        .unwrap();
        graph.reframe_parts([target], frame).unwrap();
        let mut index = PlacementSnapIndex::default();
        index.rebuild(&graph);
        for (x, overlaps) in [(8, false), (11, true)] {
            let candidate = PartSpec::Cuboid(
                CuboidSpec::new(
                    [1; 3],
                    BuildPose::new(IVec3::new(x, 4, 12), GridRotation::default()),
                )
                .unwrap(),
            );
            assert_eq!(index.overlaps(candidate), overlaps);
            assert_eq!(
                super::parts_overlap_with_frame(
                    candidate,
                    *graph.part(target).unwrap(),
                    mechanic_core::ConstructionFrame::IDENTITY
                ),
                super::parts_overlap(candidate, *graph.part(target).unwrap())
            );
        }
    }

    #[test]
    fn indexed_preview_validation_rejects_only_nearby_overlaps() {
        let mut graph = ConstructionGraph::new();
        let existing = graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new([1; 3], BuildPose::default()).unwrap(),
            ))
            .unwrap();
        let BuildOutcome::Spawned(existing) = existing else {
            panic!("spawn returned a different outcome");
        };
        let mut index = PlacementSnapIndex::default();
        index.rebuild(&graph);
        let candidate = |x| PlacementCandidate {
            spec: CuboidSpec::new(
                [1; 3],
                BuildPose::new(IVec3::new(x, 0, 0), GridRotation::default()),
            )
            .unwrap(),
            attached_face: FaceKind::NegativeY,
            anchor: None,
            support: PlacementSupport::Free,
        };

        assert!(matches!(
            validate_indexed_block_batch_in_bounds(
                &index,
                candidate(0),
                &[candidate(0).spec],
                PlacementBounds::World {
                    origin: DVec2::ZERO,
                },
            ),
            Err(PlacementError::OverlapsPart(part)) if part == existing
        ));
        assert!(
            validate_indexed_block_batch_in_bounds(
                &index,
                candidate(1_000),
                &[candidate(1_000).spec],
                PlacementBounds::World {
                    origin: DVec2::ZERO,
                },
            )
            .is_ok()
        );
    }

    #[test]
    fn free_smart_snap_can_align_all_three_axes() {
        let mut graph = ConstructionGraph::new();
        graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new(
                    [1; 3],
                    BuildPose::from_position_ticks(IVec3::new(6, 56, 406), GridRotation::default()),
                )
                .unwrap(),
            ))
            .unwrap();
        let mut index = PlacementSnapIndex::default();
        index.rebuild(&graph);
        let spec = CuboidSpec::new(
            [1; 3],
            BuildPose::from_position_ticks(IVec3::new(0, 50, 300), GridRotation::default()),
        )
        .unwrap();
        let gridded = PlacementCandidate {
            spec,
            attached_face: FaceKind::NegativeY,
            anchor: None,
            support: PlacementSupport::Free,
        };

        let (snapped, guides) = smart_snap_free_cuboid_candidate(
            &index,
            gridded,
            PlacementGrid::Centimetres25,
            1.0,
            |_| true,
        );

        assert_eq!(
            snapped.spec.pose.translation_position_ticks(),
            IVec3::new(6, 56, 306)
        );
        assert_eq!(guides.len(), 2);
        assert!(guides.iter().all(|guide| {
            let delta = (guide.to - guide.from).abs();
            [delta.x, delta.y, delta.z]
                .into_iter()
                .filter(|component| *component > f32::EPSILON)
                .count()
                == 1
        }));
    }

    #[test]
    fn free_smart_snap_guides_never_connect_centers_diagonally() {
        let mut graph = ConstructionGraph::new();
        let part = spawn_cube(&mut graph, IVec3::ZERO, 1);
        let diagonal = AxisGuide {
            delta: 0.01,
            coordinate: 1.0,
            kind: GuideKind::Center,
            part,
            target_center: Vec3::new(1.0, 2.0, 3.0),
        };
        let choices = [vec![diagonal], Vec::new(), Vec::new()];

        assert!(
            render_free_smart_guides(
                [Some(diagonal), None, None],
                &choices,
                Vec3::new(1.0, 0.0, 0.0),
            )
            .is_empty()
        );

        let cardinal = AxisGuide {
            target_center: Vec3::new(1.0, 0.0, 3.0),
            ..diagonal
        };
        let choices = [vec![cardinal], Vec::new(), Vec::new()];
        let guides = render_free_smart_guides(
            [Some(cardinal), None, None],
            &choices,
            Vec3::new(1.0, 0.0, 0.0),
        );

        assert_eq!(guides.len(), 1);
        assert_eq!(guides[0].from, Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(guides[0].to, Vec3::new(1.0, 0.0, 3.0));
    }

    #[test]
    fn single_axis_smart_snap_preserves_the_other_grid_axis() {
        let mut graph = ConstructionGraph::new();
        let support = graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new(
                    [1, 1, 1],
                    BuildPose::from_position_ticks(
                        IVec3::new(40, 50, 280),
                        GridRotation::default(),
                    ),
                )
                .unwrap(),
            ))
            .unwrap();
        let BuildOutcome::Spawned(support) = support else {
            unreachable!();
        };
        let mut target_graph = ConstructionGraph::new();
        target_graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new(
                    [1, 1, 1],
                    BuildPose::from_position_ticks(IVec3::new(6, 50, 480), GridRotation::default()),
                )
                .unwrap(),
            ))
            .unwrap();
        let hit = SurfaceHit {
            distance: 1.0,
            point: Vec3::new(0.014, 0.25, 0.63),
            face: FaceRef::part(support, FaceKind::PositiveY),
        };
        let gridded = oriented_cuboid_candidate_from_hit_with_grid(
            &graph,
            hit,
            [1, 1, 1],
            GridRotation::default(),
            PlacementGrid::Centimetres25,
            PlacementBounds::Garage,
        );
        let mut index = PlacementSnapIndex::default();
        index.rebuild(&target_graph);

        let (snapped, guides) = smart_snap_cuboid_candidate(
            &graph,
            &index,
            hit,
            gridded,
            PlacementGrid::Centimetres25,
            1.0,
            |_| true,
        );

        assert_eq!(
            snapped.spec.pose.translation_position_ticks(),
            IVec3::new(6, 150, 300)
        );
        assert_eq!(guides.len(), 1);
        assert_eq!(guides[0].axis, 0);
        assert!((guides[0].from.x - guides[0].to.x).abs() < f32::EPSILON);
        assert!((guides[0].from.y - guides[0].to.y).abs() < f32::EPSILON);
        assert!((guides[0].from.z - guides[0].to.z).abs() > f32::EPSILON);
    }

    #[test]
    fn smart_snap_is_stable_within_a_grid_cell_and_shows_coincident_guides() {
        let mut graph = ConstructionGraph::new();
        for ticks in [
            IVec3::new(8, 50, 100),
            IVec3::new(8, 50, 500),
            IVec3::new(-8, 50, 700),
        ] {
            graph
                .apply(BuildCommand::Spawn(
                    CuboidSpec::new(
                        [1, 1, 1],
                        BuildPose::from_position_ticks(ticks, GridRotation::default()),
                    )
                    .unwrap(),
                ))
                .unwrap();
        }
        let mut index = PlacementSnapIndex::default();
        index.rebuild(&graph);
        let snap = |point| {
            let hit = SurfaceHit {
                distance: 1.0,
                point,
                face: FaceRef::ground(),
            };
            let gridded = oriented_cuboid_candidate_from_hit_with_grid(
                &graph,
                hit,
                [1, 1, 1],
                GridRotation::default(),
                PlacementGrid::Centimetres25,
                PlacementBounds::Garage,
            );
            smart_snap_cuboid_candidate(
                &graph,
                &index,
                hit,
                gridded,
                PlacementGrid::Centimetres25,
                1.0,
                |_| true,
            )
        };

        let (right_candidate, right_guides) = snap(Vec3::new(0.024, 0.0, 0.63));
        let (left_candidate, left_guides) = snap(Vec3::new(-0.024, 0.0, 0.63));

        assert_eq!(right_candidate.spec.pose, left_candidate.spec.pose);
        assert_eq!(
            right_candidate.spec.pose.translation_position_ticks(),
            IVec3::new(8, 50, 300)
        );
        assert_eq!(right_guides, left_guides);
        assert_eq!(right_guides.len(), 2);
        assert!(right_guides.iter().all(|guide| guide.axis == 0));
    }

    #[test]
    fn block_drag_snaps_its_other_corner_on_two_axes_and_keeps_whole_blocks() {
        let mut graph = ConstructionGraph::new();
        for ticks in [
            IVec3::new(200, 0, 400),
            IVec3::new(200, 0, -100),
            IVec3::new(400, 0, 300),
        ] {
            graph
                .apply(BuildCommand::Spawn(
                    CuboidSpec::new(
                        [1, 1, 1],
                        BuildPose::from_position_ticks(ticks, GridRotation::default()),
                    )
                    .unwrap(),
                ))
                .unwrap();
        }
        let mut index = PlacementSnapIndex::default();
        index.rebuild(&graph);
        let start = CuboidSpec::new([1, 1, 1], BuildPose::default()).unwrap();
        let snap = |pointer| {
            smart_snap_block_span(
                &index,
                start,
                PlacementPlane::Xz,
                IVec3::new(1, 0, 2),
                pointer,
                1.0,
                |_| true,
            )
        };

        let (left, left_guides) = snap(Vec3::new(0.49, 0.0, 0.74));
        let (right, right_guides) = snap(Vec3::new(0.51, 0.0, 0.76));

        assert_eq!(left, IVec3::new(2, 0, 3));
        assert_eq!(right, left);
        assert_eq!(right_guides, left_guides);
        assert!(left_guides.iter().any(|guide| guide.axis == 0));
        assert!(left_guides.iter().any(|guide| guide.axis == 2));
        assert!(
            left_guides.iter().filter(|guide| guide.axis == 0).count() >= 2,
            "every committed object sharing the endpoint alignment may show a guide"
        );

        let specs = block_box_specs(start, left).unwrap();
        assert!(specs.iter().all(|spec| {
            spec.pose
                .translation_position_ticks()
                .to_array()
                .into_iter()
                .all(|ticks| ticks % POSITION_TICKS_PER_GRID_UNIT == 0)
        }));
    }

    #[test]
    fn invalid_block_endpoint_guide_falls_back_independently_per_axis() {
        let mut graph = ConstructionGraph::new();
        for ticks in [IVec3::new(200, 0, 400), IVec3::new(400, 0, 300)] {
            graph
                .apply(BuildCommand::Spawn(
                    CuboidSpec::new(
                        [1, 1, 1],
                        BuildPose::from_position_ticks(ticks, GridRotation::default()),
                    )
                    .unwrap(),
                ))
                .unwrap();
        }
        let mut index = PlacementSnapIndex::default();
        index.rebuild(&graph);
        let start = CuboidSpec::new([1, 1, 1], BuildPose::default()).unwrap();

        let (span, guides) = smart_snap_block_span(
            &index,
            start,
            PlacementPlane::Xz,
            IVec3::new(1, 0, 2),
            Vec3::new(0.50, 0.0, 0.75),
            1.0,
            |candidate| candidate.x != 2,
        );

        assert_eq!(span, IVec3::new(1, 0, 3));
        assert!(guides.iter().any(|guide| guide.axis == 2));
    }

    #[test]
    fn transmission_preview_accepts_only_the_current_positive_z_output() {
        let mut graph = ConstructionGraph::new();
        let BuildOutcome::Spawned(engine) = graph
            .apply(BuildCommand::SpawnEngine(EngineSpec::new(
                EngineKind::Gas,
                BuildPose::new(IVec3::new(0, 1, 0), GridRotation::default()),
            )))
            .unwrap()
        else {
            unreachable!()
        };
        let output = FaceRef::part(engine, FaceKind::PositiveZ);
        let output_geometry = face_geometry_from_ref(output, Some(&graph));
        let hit = SurfaceHit {
            distance: 0.0,
            point: output_geometry.center,
            face: output,
        };
        let (parent, candidate) = transmission_candidate_from_hit(&graph, hit).unwrap();
        let staged = stage_transmission(&graph, parent, candidate).unwrap();
        assert_eq!(staged.engine_transmission_depth(engine), Some(1));
        assert!(matches!(
            transmission_candidate_from_hit(
                &graph,
                SurfaceHit {
                    face: FaceRef::part(engine, FaceKind::PositiveX),
                    ..hit
                }
            ),
            Err(PlacementError::TransmissionOutputOnly)
        ));
        assert!(transmission_candidate_from_hit(&staged, hit).is_err());
    }

    #[test]
    fn rays_pass_through_cylinder_bores_but_hit_annular_material() {
        let mut graph = ConstructionGraph::new();
        let cylinder = spawn_cylinder(
            &mut graph,
            CylinderDimensions::new(1.0, 0.5, 1.0).unwrap(),
            BuildPose::new(IVec3::new(0, 8, 0), GridRotation::default()),
        );
        let cube = spawn_cube(&mut graph, IVec3::new(0, 4, 0), 1);
        let through_bore =
            raycast_construction(&graph, Vec3::new(0.0, 5.0, 0.0), Vec3::NEG_Y).unwrap();
        assert_eq!(through_bore.face.owner, FaceOwner::Part(cube));
        let annulus = raycast_construction(&graph, Vec3::new(0.4, 5.0, 0.0), Vec3::NEG_Y).unwrap();
        assert_eq!(annulus.face.owner, FaceOwner::Part(cylinder));
    }

    #[test]
    fn cylinder_sector_raycast_hits_retained_caps_and_cut_walls_only() {
        let mut graph = ConstructionGraph::new();
        let dimensions = CylinderDimensions::new(1.0, 0.0, 1.0)
            .unwrap()
            .with_sweep_angle_degrees(90)
            .unwrap();
        let cylinder = spawn_cylinder(
            &mut graph,
            dimensions,
            BuildPose::new(IVec3::new(0, 8, 0), GridRotation::default()),
        );
        let cube = spawn_cube(&mut graph, IVec3::new(0, 4, 0), 4);

        let retained = raycast_construction(&graph, Vec3::new(0.3, 5.0, 0.0), Vec3::NEG_Y).unwrap();
        assert_eq!(retained.face.owner, FaceOwner::Part(cylinder));
        assert_eq!(retained.face.face, FaceKind::PositiveY);

        let missing = raycast_construction(&graph, Vec3::new(-0.3, 5.0, 0.0), Vec3::NEG_Y).unwrap();
        assert_eq!(missing.face.owner, FaceOwner::Part(cube));

        let cut_wall = raycast_construction(&graph, Vec3::new(0.3, 2.0, 2.0), Vec3::NEG_Z).unwrap();
        assert_eq!(cut_wall.face.owner, FaceOwner::Part(cylinder));
        assert_eq!(cut_wall.face.face, FaceKind::PositiveX);
    }

    #[test]
    fn annular_placement_stops_at_a_bore_only_when_material_cannot_pass() {
        let mut graph = ConstructionGraph::new();
        let cylinder = spawn_cylinder(
            &mut graph,
            CylinderDimensions::new(1.0, 0.5, 1.0).unwrap(),
            BuildPose::new(IVec3::new(0, 8, 0), GridRotation::default()),
        );
        let cube = spawn_cube(&mut graph, IVec3::new(0, 4, 0), 1);
        let origin = Vec3::new(0.0, 5.0, 0.0);

        let fitting =
            raycast_construction_for_annulus(&graph, origin, Vec3::NEG_Y, 0.0, 0.4).unwrap();
        assert_eq!(fitting.face.owner, FaceOwner::Part(cube));

        let obstructed =
            raycast_construction_for_annulus(&graph, origin, Vec3::NEG_Y, 0.0, 0.6).unwrap();
        assert_eq!(obstructed.face.owner, FaceOwner::Part(cylinder));
        assert_eq!(obstructed.face.face, FaceKind::PositiveY);
        let candidate = cylinder_candidate_from_hit(
            &graph,
            obstructed,
            CylinderDimensions::new(0.6, 0.0, 0.25).unwrap(),
        )
        .unwrap();
        let staged = stage_cylinder_from_source(&graph, candidate, obstructed.face.owner).unwrap();
        assert_eq!(staged.weld_count(), 1);

        let surrounding_sleeve =
            raycast_construction_for_annulus(&graph, origin, Vec3::NEG_Y, 1.1, 1.2).unwrap();
        assert_eq!(surrounding_sleeve.face.owner, FaceOwner::Part(cube));
    }

    #[test]
    fn bearing_can_center_over_a_bore_when_its_ring_has_support() {
        let mut graph = ConstructionGraph::new();
        let cylinder = spawn_cylinder(
            &mut graph,
            CylinderDimensions::new(1.0, 0.5, 1.0).unwrap(),
            BuildPose::new(IVec3::new(0, 8, 0), GridRotation::default()),
        );
        let source = FaceRef::part(cylinder, FaceKind::PositiveY);
        let hit = SurfaceHit {
            distance: 2.5,
            point: Vec3::new(0.0, 2.5, 0.0),
            face: source,
        };

        let anchor = bearing_anchor_from_hit(&graph, hit).unwrap();
        assert_eq!(anchor, hit.point);
        assert_eq!(
            bearing_support_face(
                &graph,
                source,
                anchor,
                BearingDimensions::new(0.6, 0.2).unwrap(),
            ),
            Some(source)
        );
        assert!(
            bearing_support_face(
                &graph,
                source,
                anchor,
                BearingDimensions::new(0.4, 0.2).unwrap(),
            )
            .is_none()
        );
    }

    #[test]
    fn small_blocks_can_occupy_a_large_cylinder_bore() {
        let mut graph = ConstructionGraph::new();
        spawn_cylinder(
            &mut graph,
            CylinderDimensions::new(1.0, 0.6, 1.0).unwrap(),
            BuildPose::new(IVec3::new(0, 2, 0), GridRotation::default()),
        );
        let cube = CuboidSpec::new(
            [1; 3],
            BuildPose::new(IVec3::new(0, 2, 0), GridRotation::default()),
        )
        .unwrap();
        let candidate = PlacementCandidate {
            spec: cube,
            attached_face: FaceKind::NegativeY,
            anchor: Some(Vec3::ZERO),
            support: PlacementSupport::Surface(FaceOwner::Ground),
        };
        assert!(stage_cuboid(&graph, candidate).is_ok());
    }

    #[test]
    fn blocks_can_occupy_the_missing_side_of_a_cylinder_sector() {
        let mut graph = ConstructionGraph::new();
        spawn_cylinder(
            &mut graph,
            CylinderDimensions::new(1.0, 0.0, 1.0)
                .unwrap()
                .with_sweep_angle_degrees(90)
                .unwrap(),
            BuildPose::new(IVec3::new(0, 4, 0), GridRotation::default()),
        );
        let cube = CuboidSpec::new(
            [1; 3],
            BuildPose::new(IVec3::new(-1, 4, 0), GridRotation::default()),
        )
        .unwrap();
        let candidate = PlacementCandidate {
            spec: cube,
            attached_face: FaceKind::NegativeY,
            anchor: Some(Vec3::ZERO),
            support: PlacementSupport::Surface(FaceOwner::Ground),
        };

        assert!(stage_cuboid(&graph, candidate).is_ok());
    }

    #[test]
    fn cylinders_place_along_all_six_flat_face_normals() {
        let cases = [
            Vec3::X,
            Vec3::NEG_X,
            Vec3::Y,
            Vec3::NEG_Y,
            Vec3::Z,
            Vec3::NEG_Z,
        ];
        for outward in cases {
            let mut graph = ConstructionGraph::new();
            let support = spawn_cube(&mut graph, IVec3::new(0, 16, 0), 4);
            let hit = super::raycast_cuboid(
                Vec3::new(0.0, 4.0, 0.0) + outward * 5.0,
                -outward,
                support,
                graph.part(support).copied().unwrap().as_cuboid().unwrap(),
            )
            .unwrap();
            let candidate = cylinder_candidate_from_hit(
                &graph,
                hit,
                CylinderDimensions::new(0.25, 0.0, 0.5).unwrap(),
            )
            .unwrap();
            let axis = candidate.spec.pose.rotation.quaternion() * Vec3::Y;
            assert!(axis.abs_diff_eq(outward, 1.0e-6));
            assert!(stage_cylinder_from_source(&graph, candidate, hit.face.owner).is_ok());
        }
    }

    #[test]
    fn thin_annular_cylinder_places_on_a_coplanar_block_sheet() {
        let mut graph = ConstructionGraph::new();
        let mut center = None;
        for x in -2..=2 {
            for z in -2..=2 {
                let spec = CuboidSpec::new(
                    [1; 3],
                    BuildPose::from_half_grid(IVec3::new(x * 2, 1, z * 2), GridRotation::default()),
                )
                .unwrap();
                let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(spec)).unwrap()
                else {
                    unreachable!()
                };
                if x == 0 && z == 0 {
                    center = Some(part);
                }
            }
        }
        let center = center.unwrap();
        let hit = SurfaceHit {
            distance: 1.0,
            point: Vec3::new(0.0, 0.25, 0.0),
            face: FaceRef::part(center, FaceKind::PositiveY),
        };
        let candidate = cylinder_candidate_from_hit(
            &graph,
            hit,
            CylinderDimensions::new(0.75, 0.70, 0.25).unwrap(),
        )
        .unwrap();

        assert!(candidate.anchor.is_some());
        assert!(stage_cylinder_from_source(&graph, candidate, hit.face.owner).is_ok());
    }

    #[test]
    fn bearing_anchor_rejects_a_curved_cylinder_wall() {
        let mut graph = ConstructionGraph::new();
        let cylinder = spawn_cylinder(
            &mut graph,
            CylinderDimensions::new(0.5, 0.25, 0.5).unwrap(),
            BuildPose::new(IVec3::new(0, 2, 0), GridRotation::default()),
        );
        let curved_hit = SurfaceHit {
            distance: 1.0,
            point: Vec3::new(0.25, 0.5, 0.0),
            face: FaceRef::part(cylinder, FaceKind::PositiveX),
        };

        assert_eq!(
            bearing_anchor_from_hit(&graph, curved_hit),
            Err(PlacementError::CurvedSurface)
        );
    }

    #[test]
    fn filtered_raycast_reaches_an_accepted_part_behind_an_excluded_one() {
        let mut graph = ConstructionGraph::new();
        let near = spawn_cube(&mut graph, IVec3::ZERO, 4);
        let far = spawn_cube(&mut graph, IVec3::new(0, 0, -8), 4);
        let origin = Vec3::new(0.0, 0.0, 4.0);
        assert_eq!(
            raycast_construction_with_ground(&graph, origin, Vec3::NEG_Z, None)
                .unwrap()
                .face
                .owner,
            FaceOwner::Part(near)
        );
        let hit = super::raycast_construction_filtered_with_ground(
            &graph,
            origin,
            Vec3::NEG_Z,
            None,
            |part| part == far,
        )
        .unwrap();
        assert_eq!(hit.face.owner, FaceOwner::Part(far));
        assert!((hit.distance - 5.5).abs() < 1.0e-5);
    }

    #[test]
    fn filtered_annulus_raycast_excludes_bore_obstructions_too() {
        let mut graph = ConstructionGraph::new();
        let far = spawn_cube(&mut graph, IVec3::ZERO, 4);
        let cylinder = spawn_cylinder(
            &mut graph,
            CylinderDimensions::new(2.0, 1.0, 0.5).unwrap(),
            BuildPose::new(IVec3::new(0, 8, 0), GridRotation::default()),
        );
        let origin = Vec3::Y * 4.0;
        let unfiltered = super::raycast_construction_for_annulus_with_ground(
            &graph,
            origin,
            Vec3::NEG_Y,
            0.5,
            1.5,
            None,
        )
        .unwrap();
        assert_eq!(unfiltered.face.owner, FaceOwner::Part(cylinder));
        let hit = super::raycast_construction_for_annulus_filtered_with_ground(
            &graph,
            origin,
            Vec3::NEG_Y,
            0.5,
            1.5,
            None,
            |part| part == far,
        )
        .unwrap();
        assert_eq!(hit.face.owner, FaceOwner::Part(far));
        assert!((hit.distance - 3.5).abs() < 1.0e-5);
    }

    #[test]
    fn reframed_primitive_picking_faces_and_bounds_follow_the_authored_frame() {
        let mut graph = ConstructionGraph::new();
        let part = spawn_cube(&mut graph, IVec3::new(0, 4, 0), 4);
        let origin = Vec3::new(0.0, 4.0, 0.0);
        let hit = raycast_construction_with_ground(&graph, origin, Vec3::NEG_Y, None).unwrap();
        let face = face_geometry_from_ref(hit.face, Some(&graph));
        let frame = mechanic_core::ConstructionFrame::new(
            Vec3::new(4.0, 3.0, 2.0),
            Quat::from_rotation_z(0.47) * Quat::from_rotation_y(0.31),
        )
        .unwrap();
        graph.reframe_parts([part], frame).unwrap();
        let reframed = raycast_construction_with_ground(
            &graph,
            frame.point(origin),
            frame.vector(Vec3::NEG_Y),
            None,
        )
        .unwrap();
        assert_eq!(reframed.face, hit.face);
        assert!(reframed.point.distance(frame.point(hit.point)) < 1.0e-5);
        assert!((reframed.distance - hit.distance).abs() < 1.0e-5);
        let reframed_face = face_geometry_from_ref(reframed.face, Some(&graph));
        assert!(reframed_face.center.distance(frame.point(face.center)) < 1.0e-5);
        assert!(reframed_face.normal.distance(frame.vector(face.normal)) < 1.0e-5);
        assert!(
            reframed_face
                .tangent_u
                .distance(frame.vector(face.tangent_u))
                < 1.0e-5
        );
        assert!(
            reframed_face
                .tangent_v
                .distance(frame.vector(face.tangent_v))
                < 1.0e-5
        );
        let (minimum, maximum) = super::composed_part_world_bounds(&graph, part).unwrap();
        assert!(minimum.cmple(reframed.point + Vec3::splat(1.0e-5)).all());
        assert!(maximum.cmpge(reframed.point - Vec3::splat(1.0e-5)).all());
        assert!(((minimum + maximum) * 0.5).distance(frame.point(Vec3::Y)) < 1.0e-5);
    }

    #[test]
    fn reframed_annulus_obstruction_uses_the_cylinder_frame() {
        let mut graph = ConstructionGraph::new();
        let part = spawn_cylinder(
            &mut graph,
            CylinderDimensions::new(1.0, 0.5, 1.0).unwrap(),
            BuildPose::new(IVec3::new(0, 8, 0), GridRotation::default()),
        );
        let origin = Vec3::new(0.0, 5.0, 0.0);
        let hit = super::raycast_construction_for_annulus_with_ground(
            &graph,
            origin,
            Vec3::NEG_Y,
            0.0,
            0.6,
            None,
        )
        .unwrap();
        let frame = mechanic_core::ConstructionFrame::new(
            Vec3::new(2.0, 3.0, 4.0),
            Quat::from_rotation_z(0.6),
        )
        .unwrap();
        graph.reframe_parts([part], frame).unwrap();
        let reframed = super::raycast_construction_for_annulus_with_ground(
            &graph,
            frame.point(origin),
            frame.vector(Vec3::NEG_Y),
            0.0,
            0.6,
            None,
        )
        .unwrap();
        assert_eq!(reframed.face, hit.face);
        assert!(reframed.point.distance(frame.point(hit.point)) < 1.0e-5);
        assert!((reframed.distance - hit.distance).abs() < 1.0e-5);
    }

    #[test]
    fn reframed_evaluated_solid_picking_and_faces_apply_the_frame_once() {
        let mut graph = ConstructionGraph::new();
        let part = spawn_cube(&mut graph, IVec3::new(0, 4, 0), 4);
        let owner = SolidOwner::Part(part);
        let edge = graph.evaluated_solid(owner).unwrap().logical_edges[0].key;
        graph
            .apply(BuildCommand::AddShapeFeature(ShapeFeature::new(
                [EdgeChainRef { owner, edge }],
                EdgeTreatment::Fillet,
                20,
            )))
            .unwrap();
        let origin = Vec3::new(0.0, 4.0, 0.0);
        let hit = raycast_construction_with_ground(&graph, origin, Vec3::NEG_Y, None).unwrap();
        let face = face_geometry_from_ref(hit.face, Some(&graph));
        let frame = mechanic_core::ConstructionFrame::new(
            Vec3::new(3.0, 2.0, 1.0),
            Quat::from_rotation_z(0.9),
        )
        .unwrap();
        graph.reframe_parts([part], frame).unwrap();
        let reframed = raycast_construction_with_ground(
            &graph,
            frame.point(origin),
            frame.vector(Vec3::NEG_Y),
            None,
        )
        .unwrap();
        assert_eq!(reframed.face, hit.face);
        assert!(reframed.point.distance(frame.point(hit.point)) < 1.0e-5);
        let reframed_face = face_geometry_from_ref(reframed.face, Some(&graph));
        assert!(reframed_face.center.distance(frame.point(face.center)) < 1.0e-5);
        assert!(reframed_face.normal.distance(frame.vector(face.normal)) < 1.0e-5);
    }

    #[test]
    fn raycast_selects_nearest_cuboid_face_before_ground() {
        let mut graph = ConstructionGraph::new();
        spawn_cube(&mut graph, IVec3::new(0, 2, 0), 4);
        let hit = raycast_construction(&graph, Vec3::new(0.0, 4.0, 0.0), Vec3::NEG_Y)
            .expect("cube is under ray");
        assert_eq!(hit.face.face, FaceKind::PositiveY);
        assert!(matches!(hit.face.owner, FaceOwner::Part(_)));
        assert!((hit.point.y - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn raycast_from_below_ignores_the_floor_and_reaches_the_underside() {
        let mut graph = ConstructionGraph::new();
        let part = spawn_cube(&mut graph, IVec3::new(0, 4, 0), 1);

        let hit = raycast_construction(&graph, Vec3::new(0.0, -1.0, 0.0), Vec3::Y)
            .expect("the ray reaches the elevated block");

        assert_eq!(hit.face.owner, FaceOwner::Part(part));
        assert_eq!(hit.face.face, FaceKind::NegativeY);
        assert!((hit.point.y - 0.875).abs() < 1.0e-6);
    }

    #[test]
    fn fixed_quarter_metre_blocks_place_flush_on_ground_and_faces() {
        let graph = ConstructionGraph::new();
        let ground_hit = SurfaceHit {
            distance: 1.0,
            point: Vec3::ZERO,
            face: mechanic_core::FaceRef::ground(),
        };
        let candidate = candidate_from_hit(&graph, ground_hit);
        assert_eq!(
            candidate
                .spec
                .dimensions
                .map(mechanic_core::GridDimension::units),
            [1; 3]
        );
        assert!((candidate.spec.pose.translation().y - BLOCK_SIZE_METERS * 0.5).abs() < 1.0e-6);
        let graph = stage_cuboid(&graph, candidate).unwrap();

        let top = raycast_construction(&graph, Vec3::new(0.0, 5.0, 0.0), Vec3::NEG_Y)
            .expect("placed block is under ray");
        let attached = candidate_from_hit(&graph, top);
        assert!((attached.spec.pose.translation().y - 0.375).abs() < 1.0e-6);
        assert!(stage_cuboid(&graph, attached).is_ok());
    }

    #[test]
    fn gas_engine_places_flush_with_its_authored_footprint_and_stays_semantic() {
        let graph = ConstructionGraph::new();
        let ground_hit = SurfaceHit {
            distance: 1.0,
            point: Vec3::ZERO,
            face: FaceRef::ground(),
        };
        let candidate = cuboid_candidate_from_hit(&graph, ground_hit, EngineKind::Gas.grid_units());

        assert_eq!(
            candidate.spec.pose.translation(),
            Vec3::new(0.125, 0.25, 0.0)
        );
        assert_eq!(candidate.spec.size_meters(), Vec3::new(0.5, 0.5, 0.75));

        let graph = stage_engine_from_source(&graph, candidate, FaceOwner::Ground, EngineKind::Gas)
            .unwrap();
        let (_, part) = graph.parts().next().expect("the engine was staged");
        assert!(matches!(
            part,
            PartSpec::Engine(engine) if engine.kind == EngineKind::Gas
        ));
        assert_eq!(graph.welds().count(), 1);
    }

    #[test]
    fn electric_engine_spans_two_by_two_ground_cells_without_a_half_block_offset() {
        let graph = ConstructionGraph::new();
        let candidate = cuboid_candidate_from_hit(
            &graph,
            SurfaceHit {
                distance: 1.0,
                point: Vec3::ZERO,
                face: FaceRef::ground(),
            },
            EngineKind::Electric.grid_units(),
        );

        assert_eq!(
            candidate.spec.pose.translation_half_units(),
            IVec3::new(1, 2, 1)
        );
        assert_eq!(candidate.spec.size_meters(), Vec3::splat(0.5));
        assert_eq!(
            super::cuboid_world_bounds(candidate.spec),
            (Vec3::new(-0.125, 0.0, -0.125), Vec3::new(0.375, 0.5, 0.375))
        );
    }

    #[test]
    fn quarter_turn_rotates_an_authored_footprint_and_survives_staging() {
        let graph = ConstructionGraph::new();
        let hit = SurfaceHit {
            distance: 1.0,
            point: Vec3::ZERO,
            face: FaceRef::ground(),
        };
        let candidate = oriented_cuboid_candidate_from_hit(
            &graph,
            hit,
            EngineKind::Gas.grid_units(),
            GridRotation::new(0, 1, 0),
        );

        assert_eq!(candidate.spec.pose.rotation.quarter_turns_xyz(), [0, 1, 0]);
        assert_eq!(
            candidate.spec.pose.translation_half_units(),
            IVec3::new(0, 2, 1)
        );
        assert_eq!(candidate.attached_face, FaceKind::NegativeY);
        let (minimum, maximum) = super::cuboid_world_bounds(candidate.spec);
        assert!((maximum.x - minimum.x - 0.75).abs() < 1.0e-6);
        assert!((maximum.z - minimum.z - 0.50).abs() < 1.0e-6);

        let staged =
            stage_engine_from_source(&graph, candidate, FaceOwner::Ground, EngineKind::Gas)
                .unwrap();
        let (_, PartSpec::Engine(engine)) = staged.parts().next().unwrap() else {
            panic!("the staged part must remain an engine")
        };
        assert_eq!(engine.pose.rotation.quarter_turns_xyz(), [0, 1, 0]);
    }

    #[test]
    fn every_authored_orientation_attaches_flush_from_every_world_face() {
        for outward in [
            Vec3::X,
            Vec3::NEG_X,
            Vec3::Y,
            Vec3::NEG_Y,
            Vec3::Z,
            Vec3::NEG_Z,
        ] {
            for x in 0..4 {
                for y in 0..4 {
                    for z in 0..4 {
                        let mut graph = ConstructionGraph::new();
                        let support = spawn_cube(&mut graph, IVec3::new(0, 16, 0), 4);
                        let hit = super::raycast_cuboid(
                            Vec3::new(0.0, 4.0, 0.0) + outward * 5.0,
                            -outward,
                            support,
                            graph.part(support).copied().unwrap().as_cuboid().unwrap(),
                        )
                        .expect("ray reaches requested support face");
                        let candidate = oriented_cuboid_candidate_from_hit(
                            &graph,
                            hit,
                            EngineKind::Gas.grid_units(),
                            GridRotation::new(x, y, z),
                        );
                        let attached =
                            super::face_geometry(candidate.spec, candidate.attached_face);

                        assert!(attached.normal.abs_diff_eq(-outward, 1.0e-6));
                        assert!(
                            stage_engine_from_source(
                                &graph,
                                candidate,
                                FaceOwner::Part(support),
                                EngineKind::Gas,
                            )
                            .is_ok()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn tipped_authored_part_uses_its_rotated_height_and_footprint() {
        let graph = ConstructionGraph::new();
        let candidate = oriented_cuboid_candidate_from_hit(
            &graph,
            SurfaceHit {
                distance: 1.0,
                point: Vec3::ZERO,
                face: FaceRef::ground(),
            },
            EngineKind::Gas.grid_units(),
            GridRotation::new(1, 0, 0),
        );

        assert_eq!(candidate.spec.pose.rotation.quarter_turns_xyz(), [1, 0, 0]);
        let (minimum, maximum) = super::cuboid_world_bounds(candidate.spec);
        assert!((maximum.x - minimum.x - 0.5).abs() < 1.0e-6);
        assert!((maximum.y - minimum.y - 0.75).abs() < 1.0e-6);
        assert!((maximum.z - minimum.z - 0.5).abs() < 1.0e-6);
        assert!(minimum.y.abs() < 1.0e-6);
        assert_eq!(candidate.attached_face, FaceKind::PositiveZ);
    }

    #[test]
    fn placement_works_from_all_six_cuboid_faces() {
        let cases = [
            (Vec3::X, FaceKind::PositiveX),
            (Vec3::NEG_X, FaceKind::NegativeX),
            (Vec3::Y, FaceKind::PositiveY),
            (Vec3::NEG_Y, FaceKind::NegativeY),
            (Vec3::Z, FaceKind::PositiveZ),
            (Vec3::NEG_Z, FaceKind::NegativeZ),
        ];
        for (outward, expected_face) in cases {
            let mut graph = ConstructionGraph::new();
            let part = spawn_cube(&mut graph, IVec3::new(0, 16, 0), 4);
            let hit = super::raycast_cuboid(
                Vec3::new(0.0, 4.0, 0.0) + outward * 5.0,
                -outward,
                part,
                graph.part(part).copied().unwrap().as_cuboid().unwrap(),
            )
            .expect("ray reaches requested face");
            assert_eq!(hit.face.face, expected_face);
            let candidate = candidate_from_hit(&graph, hit);
            assert!(stage_cuboid(&graph, candidate).is_ok());
        }
    }

    #[test]
    fn side_placement_preserves_the_supporting_quarter_block_lattice() {
        for face in [
            FaceKind::PositiveX,
            FaceKind::NegativeX,
            FaceKind::PositiveZ,
            FaceKind::NegativeZ,
        ] {
            let graph = ConstructionGraph::new();
            let support = candidate_from_hit(
                &graph,
                SurfaceHit {
                    distance: 1.0,
                    point: Vec3::ZERO,
                    face: FaceRef::ground(),
                },
            );
            let graph = stage_cuboid(&graph, support).unwrap();
            let part = graph.parts().next().unwrap().0;
            let source = FaceRef::part(part, face);
            let source_face = super::face_geometry_from_ref(source, Some(&graph));

            let candidate = candidate_from_hit(
                &graph,
                SurfaceHit {
                    distance: 1.0,
                    point: source_face.center,
                    face: source,
                },
            );

            assert_eq!(
                candidate.spec.pose.translation_half_units().y,
                support.spec.pose.translation_half_units().y
            );
            assert!(stage_cuboid(&graph, candidate).is_ok());

            let bearing_candidate =
                bearing_attachment_candidate(&graph, source, source_face.center);
            assert_eq!(
                bearing_candidate.spec.pose.translation_half_units().y,
                support.spec.pose.translation_half_units().y
            );
            let attached = stage_bearing_attachment(
                &graph,
                bearing_candidate,
                source,
                source_face.center,
                BearingDimensions::default(),
            )
            .unwrap();
            assert_eq!(attached.bearing_count(), 1);
            assert_eq!(attached.weld_count(), 1);
        }
    }

    #[test]
    fn placement_rejects_cubes_extending_beyond_platform() {
        let graph = ConstructionGraph::new();
        let hit = SurfaceHit {
            distance: 1.0,
            point: Vec3::new(super::GROUND_HALF_SIZE, 0.0, 0.0),
            face: mechanic_core::FaceRef::ground(),
        };
        let candidate = candidate_from_hit(&graph, hit);
        assert!(matches!(
            stage_cuboid(&graph, candidate),
            Err(PlacementError::OutsidePlatform)
        ));
    }

    #[test]
    fn world_terrain_placement_is_not_limited_to_the_garage_platform() {
        let graph = ConstructionGraph::new();
        let terrain_hit = SurfaceHit {
            distance: 4.0,
            point: Vec3::new(24.15, 13.337, -31.20),
            face: FaceRef::ground(),
        };
        let candidate = candidate_from_hit(&graph, terrain_hit);

        let bottom = candidate.spec.pose.translation().y - BLOCK_SIZE_METERS * 0.5;
        assert!(bottom <= terrain_hit.point.y);
        assert!(terrain_hit.point.y - bottom < 0.025);
        assert!(matches!(
            validate_block_batch_in_bounds(
                &graph,
                candidate,
                &[candidate.spec],
                PlacementBounds::Garage,
            ),
            Err(PlacementError::OutsidePlatform)
        ));
        assert!(
            validate_block_batch_in_bounds(
                &graph,
                candidate,
                &[candidate.spec],
                PlacementBounds::World {
                    origin: DVec2::ZERO,
                },
            )
            .is_ok()
        );
    }

    #[test]
    fn world_terrain_placement_does_not_create_a_flat_garage_ground_weld() {
        let graph = ConstructionGraph::new();
        let terrain_hit = SurfaceHit {
            distance: 4.0,
            point: Vec3::new(24.15, 0.0, -31.20),
            face: FaceRef::ground(),
        };
        let candidate = candidate_from_hit(&graph, terrain_hit);
        let staged = stage_block_batch_from_source_in_bounds(
            &graph,
            candidate,
            &[candidate.spec],
            FaceOwner::Ground,
            PlacementBounds::World {
                origin: DVec2::ZERO,
            },
        )
        .unwrap();

        assert_eq!(staged.weld_count(), 0);
        assert!(!staged.compile().unwrap().compounds[0].is_static);
    }

    #[test]
    fn isolated_free_block_remains_unwelded() {
        let graph = ConstructionGraph::new();
        let candidate = free_cuboid_candidate(
            Vec3::new(0.0, crate::garage::BUILD_MIN_Y + 1.0, 0.0),
            Vec3::NEG_Z,
            [1; 3],
            GridRotation::default(),
            PlacementGrid::Centimetres25,
            PlacementBounds::GarageBuild,
        );
        let staged = stage_block_batch_in_bounds(
            &graph,
            candidate,
            &[candidate.spec],
            PlacementBounds::GarageBuild,
        )
        .unwrap();

        assert_eq!(staged.weld_count(), 0);
        assert!(!staged.compile().unwrap().compounds[0].is_static);
    }

    #[test]
    fn free_candidates_snap_globally_and_face_the_view_cardinally() {
        let cuboid = free_cuboid_candidate(
            Vec3::new(0.18, 6.18, -0.18),
            Vec3::new(0.9, 0.1, 0.2),
            [1, 2, 3],
            GridRotation::new(0, 1, 0),
            PlacementGrid::Centimetres25,
            PlacementBounds::GarageBuild,
        );
        let ticks = cuboid.spec.pose.translation_position_ticks();
        let world_dimensions =
            super::oriented_grid_dimensions([1, 2, 3], cuboid.spec.pose.rotation);
        assert_eq!(
            ticks,
            super::snap_global_center_ticks(
                super::snap_world_to_position_ticks(Vec3::new(0.18, 6.18, -0.18)),
                world_dimensions,
                PlacementGrid::Centimetres25,
                PlacementBounds::GarageBuild,
            )
        );
        assert_eq!(cuboid.spec.pose.rotation, GridRotation::new(0, 1, 0));
        assert_eq!(cuboid.support, PlacementSupport::Free);

        let cylinder = free_cylinder_candidate(
            Vec3::new(0.18, 6.18, -0.18),
            Vec3::new(0.9, 0.1, 0.2),
            CylinderDimensions::default(),
            PlacementGrid::Centimetres25,
            PlacementBounds::GarageBuild,
        );
        let axis = cylinder.spec.pose.rotation.quaternion() * Vec3::Y;
        assert!(axis.abs_diff_eq(Vec3::NEG_X, 1.0e-5), "axis was {axis:?}");
        assert_eq!(cylinder.support, PlacementSupport::Free);
    }

    #[test]
    #[allow(clippy::too_many_lines)] // One table exercises every standalone spawn path.
    fn every_standalone_tool_stages_an_isolated_free_part() {
        let point = Vec3::new(0.0, 7.5, 0.0);
        let direction = Vec3::NEG_Z;
        let bounds = PlacementBounds::GarageBuild;
        let grid = PlacementGrid::Centimetres25;
        let rotation = GridRotation::default();
        let cases = [
            (
                free_cuboid_candidate(
                    point,
                    direction,
                    mechanic_core::ControllerSpec::GRID_UNITS,
                    rotation,
                    grid,
                    bounds,
                ),
                0_u8,
            ),
            (
                free_cuboid_candidate(
                    point,
                    direction,
                    EngineKind::Gas.grid_units(),
                    rotation,
                    grid,
                    bounds,
                ),
                1,
            ),
            (
                free_cuboid_candidate(
                    point,
                    direction,
                    EngineKind::Electric.grid_units(),
                    rotation,
                    grid,
                    bounds,
                ),
                2,
            ),
            (
                free_cuboid_candidate(
                    point,
                    direction,
                    mechanic_core::ServoSpec::GRID_UNITS,
                    rotation,
                    grid,
                    bounds,
                ),
                3,
            ),
            (
                free_cuboid_candidate(
                    point,
                    direction,
                    mechanic_core::SeatSpec::GRID_UNITS,
                    rotation,
                    grid,
                    bounds,
                ),
                4,
            ),
            (
                free_cuboid_candidate(
                    point,
                    direction,
                    mechanic_core::InputSpec::GRID_UNITS,
                    rotation,
                    grid,
                    bounds,
                ),
                5,
            ),
            (
                free_cuboid_candidate(
                    point,
                    direction,
                    mechanic_core::DimensionLinkSpec::GRID_UNITS,
                    rotation,
                    grid,
                    bounds,
                ),
                6,
            ),
        ];
        for (candidate, kind) in cases {
            let graph = ConstructionGraph::new();
            let staged = match kind {
                0 => stage_controller_in_bounds(&graph, candidate, bounds),
                1 => stage_engine_in_bounds(&graph, candidate, EngineKind::Gas, bounds),
                2 => stage_engine_in_bounds(&graph, candidate, EngineKind::Electric, bounds),
                3 => stage_servo_in_bounds(&graph, candidate, bounds),
                4 => stage_seat_in_bounds(&graph, candidate, bounds),
                5 => stage_input_in_bounds(&graph, candidate, bounds),
                6 => stage_dimension_link_in_bounds(&graph, candidate, DimensionLinkId(1), bounds),
                _ => unreachable!(),
            }
            .unwrap();
            assert_eq!(staged.part_count(), 1);
            assert_eq!(staged.weld_count(), 0);
            assert!(!staged.compile().unwrap().compounds[0].is_static);
        }

        let graph = ConstructionGraph::new();
        let candidate = free_cuboid_candidate(point, direction, [1; 3], rotation, grid, bounds);
        let block =
            stage_block_batch_in_bounds(&graph, candidate, &[candidate.spec], bounds).unwrap();
        assert_eq!(block.weld_count(), 0);

        let cylinder = free_cylinder_candidate(
            point,
            direction,
            CylinderDimensions::default(),
            grid,
            bounds,
        );
        let pipe = [PipeRunPiece {
            spec: PartSpec::Cylinder(cylinder.spec),
            inlet: FaceKind::NegativeY,
            outlet: FaceKind::PositiveY,
        }];
        let staged =
            stage_pipe_run_in_bounds(&graph, &pipe, PipeRunAttachment::Free, bounds).unwrap();
        assert_eq!(staged.part_count(), 1);
        assert_eq!(staged.weld_count(), 0);
    }

    #[test]
    fn multiple_dimension_links_with_distinct_ids_can_coexist() {
        let bounds = PlacementBounds::GarageBuild;
        let grid = PlacementGrid::Centimetres25;
        let rotation = GridRotation::default();
        let base = free_cuboid_candidate(
            Vec3::new(0.0, 7.5, 0.0),
            Vec3::NEG_Z,
            [2, 1, 1],
            rotation,
            grid,
            bounds,
        );
        let graph =
            stage_block_batch_in_bounds(&ConstructionGraph::new(), base, &[base.spec], bounds)
                .unwrap();
        let first = free_cuboid_candidate(
            Vec3::new(-0.5, 7.5, 0.0),
            Vec3::NEG_Z,
            mechanic_core::DimensionLinkSpec::GRID_UNITS,
            rotation,
            grid,
            bounds,
        );
        let graph =
            stage_dimension_link_in_bounds(&graph, first, DimensionLinkId(11), bounds).unwrap();
        let second = free_cuboid_candidate(
            Vec3::new(0.5, 7.5, 0.0),
            Vec3::NEG_Z,
            mechanic_core::DimensionLinkSpec::GRID_UNITS,
            rotation,
            grid,
            bounds,
        );
        let graph =
            stage_dimension_link_in_bounds(&graph, second, DimensionLinkId(12), bounds).unwrap();

        let mut ids = graph
            .parts()
            .filter_map(|(part, _)| graph.dimension_link_id(part))
            .collect::<Vec<_>>();
        ids.sort_unstable_by_key(|id| id.0);
        assert_eq!(ids, vec![DimensionLinkId(11), DimensionLinkId(12)]);
        assert_eq!(graph.part_count(), 3);
        assert_eq!(graph.weld_count(), 2);
        assert_eq!(graph.compile().unwrap().compounds.len(), 1);
    }

    #[test]
    fn free_parts_auto_weld_on_contact_and_reject_overlap_or_bounds_escape() {
        let bounds = PlacementBounds::GarageBuild;
        let first = free_cuboid_candidate(
            Vec3::new(0.0, 6.0, 0.0),
            Vec3::NEG_Z,
            [1; 3],
            GridRotation::default(),
            PlacementGrid::Centimetres25,
            bounds,
        );
        let graph =
            stage_block_batch_in_bounds(&ConstructionGraph::new(), first, &[first.spec], bounds)
                .unwrap();
        assert!(matches!(
            stage_block_batch_in_bounds(&graph, first, &[first.spec], bounds),
            Err(PlacementError::OverlapsPart(_))
        ));

        let touching = free_cuboid_candidate(
            first.spec.pose.translation() + Vec3::X * BLOCK_SIZE_METERS,
            Vec3::NEG_Z,
            [1; 3],
            GridRotation::default(),
            PlacementGrid::Centimetres25,
            bounds,
        );
        let welded =
            stage_block_batch_in_bounds(&graph, touching, &[touching.spec], bounds).unwrap();
        assert_eq!(welded.weld_count(), 1);

        let outside = free_cuboid_candidate(
            Vec3::new(super::GROUND_HALF_SIZE, 6.0, 0.0),
            Vec3::NEG_Z,
            [1; 3],
            GridRotation::default(),
            PlacementGrid::Centimetres25,
            bounds,
        );
        assert_eq!(
            validate_block_batch_in_bounds(
                &ConstructionGraph::new(),
                outside,
                &[outside.spec],
                bounds,
            ),
            Err(PlacementError::OutsidePlatform)
        );
    }

    #[test]
    fn world_raycast_uses_the_supplied_terrain_surface() {
        let graph = ConstructionGraph::new();
        let terrain_hit = SurfaceHit {
            distance: 6.65,
            point: Vec3::new(24.0, 13.35, -31.0),
            face: FaceRef::ground(),
        };

        let hit = raycast_construction_with_ground(
            &graph,
            Vec3::new(24.0, 20.0, -31.0),
            Vec3::NEG_Y,
            Some(terrain_hit),
        )
        .expect("terrain is the world build surface");

        assert!(matches!(hit.face.owner, FaceOwner::Ground));
        assert!(hit.point.abs_diff_eq(terrain_hit.point, 1.0e-6));
    }

    #[test]
    fn cylinder_slice_platform_bounds_ignore_the_omitted_sector() {
        let graph = ConstructionGraph::new();
        let pose = BuildPose::new(IVec3::new(-40, 2, 0), GridRotation::default());
        let slice = CylinderDimensions::new(1.0, 0.0, 1.0)
            .unwrap()
            .with_sweep_angle_degrees(90)
            .unwrap();
        assert!(validate_part(&graph, PartSpec::Cylinder(CylinderSpec::new(slice, pose))).is_ok());

        let full = CylinderDimensions::new(1.0, 0.0, 1.0).unwrap();
        assert!(matches!(
            validate_part(&graph, PartSpec::Cylinder(CylinderSpec::new(full, pose))),
            Err(PlacementError::OutsidePlatform)
        ));
    }

    #[test]
    fn single_block_automatically_welds_to_touching_block() {
        let mut graph = ConstructionGraph::new();
        spawn_cube(&mut graph, IVec3::new(0, 2, 0), 4);
        let hit = raycast_construction(&graph, Vec3::new(0.0, 5.0, 0.0), Vec3::NEG_Y)
            .expect("support block is under ray");
        let candidate = candidate_from_hit(&graph, hit);

        let graph = stage_cuboid(&graph, candidate).unwrap();

        assert_eq!(graph.part_count(), 2);
        assert_eq!(graph.weld_count(), 1);
        assert_eq!(graph.compile().unwrap().compounds.len(), 1);
    }

    #[test]
    fn single_block_placed_on_ground_is_automatically_welded() {
        let graph = ConstructionGraph::new();
        let candidate = candidate_from_hit(
            &graph,
            SurfaceHit {
                distance: 1.0,
                point: Vec3::ZERO,
                face: FaceRef::ground(),
            },
        );

        let graph = stage_cuboid(&graph, candidate).unwrap();

        assert_eq!(graph.part_count(), 1);
        assert_eq!(graph.weld_count(), 1);
        assert!(graph.compile().unwrap().compounds[0].is_static);
    }

    #[test]
    fn dragged_sheet_is_face_connected_and_welded() {
        let graph = ConstructionGraph::new();
        let start = candidate_from_hit(
            &graph,
            SurfaceHit {
                distance: 1.0,
                point: Vec3::ZERO,
                face: FaceRef::ground(),
            },
        );
        let endpoint = start.spec.pose.translation_half_units() + IVec3::new(4, 0, 2);
        let specs = block_sheet_specs(start.spec, endpoint, PlacementPlane::Xz).unwrap();

        let graph = stage_block_batch(&graph, start, &specs).unwrap();

        assert_eq!(graph.part_count(), 6);
        assert_eq!(graph.weld_count(), 13);
        let compiled = graph.compile().unwrap();
        assert_eq!(compiled.compounds.len(), 1);
        assert!(compiled.compounds[0].is_static);
    }

    #[test]
    fn fast_volume_path_keeps_4096_blocks_individual_with_exact_welds() {
        let graph = ConstructionGraph::new();
        let start = ground_volume_candidate(IVec3::new(-3_150, 50, -3_150));
        let volume = BlockVolume::new(start.spec, IVec3::new(63, 0, 63)).unwrap();
        let index = PlacementSnapIndex::default();

        let started = Instant::now();
        let placed = stage_block_volume_in_bounds(
            &graph,
            &index,
            start,
            volume,
            None,
            Some(FaceOwner::Ground),
            PlacementBounds::Garage,
            17,
        )
        .unwrap();
        let elapsed = started.elapsed();

        assert_eq!(placed.new_parts.len(), 4_096);
        assert_eq!(placed.graph.part_count(), 4_096);
        assert_eq!(placed.weld_count, 4_032 + 4_032 + 4_096);
        assert_eq!(placed.graph.weld_count(), placed.weld_count);
        assert_eq!(placed.publication_generation, 17);
        assert_eq!(
            placed.graph.part(placed.new_parts[0]),
            Some(&start.spec.into())
        );
        assert!(
            elapsed.as_secs_f64() < 1.0 / 60.0,
            "bulk staging regressed to {elapsed:?}"
        );
    }

    #[test]
    fn fast_volume_path_places_a_16_cubed_solid() {
        let graph = ConstructionGraph::new();
        let start = ground_volume_candidate(IVec3::new(-750, 50, -750));
        let placed = stage_block_volume_in_bounds(
            &graph,
            &PlacementSnapIndex::default(),
            start,
            BlockVolume::new(start.spec, IVec3::splat(15)).unwrap(),
            None,
            Some(FaceOwner::Ground),
            PlacementBounds::Garage,
            3,
        )
        .unwrap();

        assert_eq!(placed.graph.part_count(), 4_096);
        assert_eq!(placed.weld_count, 3 * 15 * 16 * 16 + 16 * 16);
    }

    #[test]
    fn fast_volume_path_welds_only_the_adjacent_boundary() {
        let empty = ConstructionGraph::new();
        let bottom_start = ground_volume_candidate(IVec3::new(-3_150, 50, -3_150));
        let sheet = BlockVolume::new(bottom_start.spec, IVec3::new(63, 0, 63)).unwrap();
        let placed = stage_block_volume_in_bounds(
            &empty,
            &PlacementSnapIndex::default(),
            bottom_start,
            sheet,
            None,
            Some(FaceOwner::Ground),
            PlacementBounds::Garage,
            1,
        )
        .unwrap();
        let mut index = PlacementSnapIndex::default();
        index.rebuild(&placed.graph);
        let top_start = PlacementCandidate {
            spec: CuboidSpec::new(
                [1; 3],
                BuildPose::from_position_ticks(
                    IVec3::new(-3_150, 150, -3_150),
                    GridRotation::default(),
                ),
            )
            .unwrap(),
            attached_face: FaceKind::NegativeY,
            anchor: None,
            support: PlacementSupport::Free,
        };

        let started = Instant::now();
        let top = stage_block_volume_in_bounds(
            &placed.graph,
            &index,
            top_start,
            BlockVolume::new(top_start.spec, IVec3::new(63, 0, 63)).unwrap(),
            None,
            None,
            PlacementBounds::Garage,
            2,
        )
        .unwrap();
        let elapsed = started.elapsed();

        assert_eq!(top.new_parts.len(), 4_096);
        assert_eq!(top.weld_count, 4_032 + 4_032 + 4_096);
        assert_eq!(
            top.graph.weld_count(),
            placed.graph.weld_count() + top.weld_count
        );
        assert!(
            elapsed.as_secs_f64() < 1.0 / 60.0,
            "adjacent bulk staging regressed to {elapsed:?}"
        );
    }

    #[test]
    fn negative_volume_span_preserves_start_material_and_appearance() {
        let appearance = mechanic_core::MaterialAppearance::new(
            mechanic_core::MaterialColor::Dye(
                mechanic_core::MaterialDye::new([12, 34, 56], 1.0).unwrap(),
            ),
            mechanic_core::MaterialFinish::Painted,
        );
        let start = CuboidSpec::new(
            [1; 3],
            BuildPose::from_position_ticks(IVec3::new(200, 250, 300), GridRotation::default()),
        )
        .unwrap()
        .with_material(ConstructionMaterial::Copper)
        .with_appearance(appearance);
        let volume = BlockVolume::new(start, IVec3::new(-3, -3, -3)).unwrap();
        let specs = volume.specs().collect::<Vec<_>>();

        assert_eq!(specs.len(), 64);
        assert_eq!(specs[0], start);
        assert!(specs.iter().all(|spec| {
            spec.material == ConstructionMaterial::Copper && spec.appearance == appearance
        }));
        assert_eq!(volume.bounds().0, Vec3::new(-0.375, -0.25, -0.125));
    }

    #[test]
    fn a_box_drag_places_a_solid_cuboid() {
        let start = CuboidSpec::new([1; 3], BuildPose::default()).unwrap();
        let specs = block_box_specs(start, IVec3::new(2, 1, 3)).unwrap();
        assert_eq!(
            specs.len(),
            3 * 2 * 4,
            "span counts blocks beyond the start"
        );

        // Every cell of the cuboid is filled exactly once: solid, no gaps and
        // no duplicates, which is what a region will later be able to claim.
        let mut centres = specs
            .iter()
            .map(|spec| spec.pose.translation_half_units().to_array())
            .collect::<Vec<_>>();
        centres.sort_unstable();
        let unique = {
            let mut copy = centres.clone();
            copy.dedup();
            copy
        };
        assert_eq!(centres, unique, "a box drag must not stack blocks");
    }

    #[test]
    fn a_zero_span_box_drag_is_the_single_starting_block() {
        let start = CuboidSpec::new([1; 3], BuildPose::default()).unwrap();
        let specs = block_box_specs(start, IVec3::ZERO).unwrap();
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].pose.translation_half_units(), IVec3::ZERO);
    }

    #[test]
    fn box_drag_bounds_cover_the_full_selected_blocks() {
        let start = CuboidSpec::new([1; 3], BuildPose::default()).unwrap();

        assert_eq!(
            block_box_bounds(start, IVec3::new(2, 0, -1)),
            (
                Vec3::new(-0.125, -0.125, -0.375),
                Vec3::new(0.625, 0.125, 0.125),
            )
        );
    }

    #[test]
    fn a_box_drag_keeps_the_selected_material_and_respects_the_cap() {
        let start = CuboidSpec::new([1; 3], BuildPose::default())
            .unwrap()
            .with_material(ConstructionMaterial::Wood);
        let specs = block_box_specs(start, IVec3::new(3, 3, 3)).unwrap();
        assert!(
            specs
                .iter()
                .all(|spec| spec.material == ConstructionMaterial::Wood)
        );
        assert!(matches!(
            block_box_specs(start, IVec3::new(31, 31, 31)),
            Err(PlacementError::TooManyBlocks { .. })
        ));
    }

    #[test]
    fn rotating_the_plane_keeps_the_extent_and_extends_the_third_axis() {
        // This is what makes a big cuboid easy: drag a rectangle, press Rotate, and
        // carry on into the axis the first plane could not reach.
        let start = CuboidSpec::new([1; 3], BuildPose::default()).unwrap();
        let down = Vec3::NEG_Y;
        let press_origin = Vec3::new(0.0, 4.0, 0.0);

        // A rectangle in XZ.
        let flat = block_span_from_rays(
            start,
            PlacementPlane::Xz,
            IVec3::ZERO,
            press_origin,
            down,
            press_origin + Vec3::new(BLOCK_SIZE_METERS * 3.0, 0.0, BLOCK_SIZE_METERS * 2.0),
            down,
        )
        .expect("the XZ plane is reachable from above");
        assert_eq!(flat, IVec3::new(3, 0, 2));

        // Rotate moves into a plane containing Y; the frozen span carries over and
        // only the new plane's axes move.
        let horizontal = Vec3::new(1.0, 0.0, 0.0);
        let side_origin = Vec3::new(-4.0, 0.0, 0.0);
        let boxed = block_span_from_rays(
            start,
            PlacementPlane::Yz,
            flat,
            side_origin,
            horizontal,
            side_origin + Vec3::new(0.0, BLOCK_SIZE_METERS * 4.0, 0.0),
            horizontal,
        )
        .expect("the YZ plane is reachable from the side");
        assert_eq!(
            boxed.x, 3,
            "the axis the new plane does not own must keep its extent"
        );
        assert_eq!(boxed.y, 4, "the new axis grows from the rotation onward");
    }

    #[test]
    fn dragged_sheet_keeps_one_selected_material_for_every_block() {
        let start = CuboidSpec::new([1; 3], BuildPose::default())
            .unwrap()
            .with_material(ConstructionMaterial::Wood);
        let specs = block_sheet_specs(start, IVec3::new(4, 0, 4), PlacementPlane::Xz).unwrap();
        assert!(
            specs
                .iter()
                .all(|spec| spec.material == ConstructionMaterial::Wood)
        );
    }

    #[test]
    fn invalid_drag_batch_preserves_graph() {
        let graph = ConstructionGraph::new();
        let start = candidate_from_hit(
            &graph,
            SurfaceHit {
                distance: 1.0,
                point: Vec3::ZERO,
                face: FaceRef::ground(),
            },
        );
        let endpoint = start.spec.pose.translation_half_units() + IVec3::new(96, 0, 0);
        let specs = block_sheet_specs(start.spec, endpoint, PlacementPlane::Xz).unwrap();

        assert!(matches!(
            stage_block_batch(&graph, start, &specs),
            Err(PlacementError::OutsidePlatform)
        ));
        assert_eq!(graph.part_count(), 0);
        assert_eq!(graph.weld_count(), 0);
    }

    #[test]
    fn drag_plane_projection_and_cycle_are_deterministic() {
        let graph = ConstructionGraph::new();
        let start = candidate_from_hit(
            &graph,
            SurfaceHit {
                distance: 1.0,
                point: Vec3::ZERO,
                face: FaceRef::ground(),
            },
        );
        let point = raycast_placement_plane_point(
            Vec3::new(2.0, 5.0, 3.0),
            Vec3::NEG_Y,
            start.spec,
            PlacementPlane::Xz,
        )
        .unwrap();

        // The plane runs through the dragged block's centre, and the point is
        // left unsnapped for the span arithmetic to quantize.
        assert!(point.abs_diff_eq(Vec3::new(2.0, BLOCK_SIZE_METERS * 0.5, 3.0), 1.0e-6));
        assert_eq!(PlacementPlane::Xz.cycle(), PlacementPlane::Xy);
        assert_eq!(PlacementPlane::Xy.cycle(), PlacementPlane::Yz);
        assert_eq!(PlacementPlane::Yz.cycle(), PlacementPlane::Xz);
    }

    #[test]
    fn weld_selects_two_objects_without_spawning_a_part() {
        let mut graph = ConstructionGraph::new();
        let left = spawn_cube(&mut graph, IVec3::new(0, 2, 0), 4);
        let right = spawn_cube(&mut graph, IVec3::new(4, 2, 0), 4);
        begin_weld(&mut graph, FaceRef::part(left, FaceKind::PositiveY)).unwrap();
        assert!(matches!(graph.pending(), Some(PendingOperation::Weld(_))));

        let graph =
            stage_weld_objects(&graph, FaceOwner::Part(left), FaceOwner::Part(right)).unwrap();

        assert_eq!(graph.part_count(), 2);
        assert_eq!(graph.weld_count(), 1);
        assert!(graph.pending().is_none());
        let compiled = graph.compile().unwrap();
        assert_eq!(compiled.compounds.len(), 1);
        assert_eq!(compiled.compounds[0].source_parts.len(), 2);
    }

    #[test]
    fn weld_to_ground_resolves_contact_across_the_selected_rigid_body() {
        let mut graph = ConstructionGraph::new();
        let parts = [IVec3::new(0, 1, 0), IVec3::new(0, 3, 0)].map(|center| {
            let spec = CuboidSpec::new(
                [1; 3],
                BuildPose::from_half_grid(center, GridRotation::default()),
            )
            .unwrap();
            let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(spec)).unwrap()
            else {
                unreachable!()
            };
            part
        });
        let [bottom, top] = parts;
        graph
            .apply(BuildCommand::Weld(WeldSpec {
                first: FaceRef::part(bottom, FaceKind::PositiveY),
                second: FaceRef::part(top, FaceKind::NegativeY),
            }))
            .unwrap();

        let grounded = stage_weld_objects(&graph, FaceOwner::Part(top), FaceOwner::Ground).unwrap();

        assert_eq!(grounded.weld_count(), 2);
        let compiled = grounded.compile().unwrap();
        assert_eq!(compiled.compounds.len(), 1);
        assert!(compiled.compounds[0].is_static);
    }

    #[test]
    fn weld_resolves_contact_across_both_rigid_bodies() {
        let mut graph = ConstructionGraph::new();
        let left = spawn_cube(&mut graph, IVec3::new(0, 2, 0), 4);
        let middle = spawn_cube(&mut graph, IVec3::new(4, 2, 0), 4);
        let right = spawn_cube(&mut graph, IVec3::new(8, 2, 0), 4);
        graph
            .apply(BuildCommand::Weld(WeldSpec {
                first: FaceRef::part(left, FaceKind::PositiveX),
                second: FaceRef::part(middle, FaceKind::NegativeX),
            }))
            .unwrap();

        // `left` does not touch `right`, but the body it belongs to does, and
        // the body is what the tool highlights and claims to weld.
        let staged =
            stage_weld_objects(&graph, FaceOwner::Part(left), FaceOwner::Part(right)).unwrap();

        assert_eq!(staged.weld_count(), 2);
        let compiled = staged.compile().unwrap();
        assert_eq!(compiled.compounds.len(), 1);
        assert_eq!(compiled.compounds[0].source_parts.len(), 3);
    }

    #[test]
    fn weld_refuses_two_parts_of_one_rigid_body() {
        let mut graph = ConstructionGraph::new();
        let left = spawn_cube(&mut graph, IVec3::new(0, 2, 0), 4);
        let right = spawn_cube(&mut graph, IVec3::new(4, 2, 0), 4);
        graph
            .apply(BuildCommand::Weld(WeldSpec {
                first: FaceRef::part(left, FaceKind::PositiveX),
                second: FaceRef::part(right, FaceKind::NegativeX),
            }))
            .unwrap();

        assert!(matches!(
            stage_weld_objects(&graph, FaceOwner::Part(left), FaceOwner::Part(right)),
            Err(PlacementError::SameObject)
        ));
        assert_eq!(graph.weld_count(), 1);
    }

    /// Base spanning two cells with two one-cell blocks bearing-mounted on top,
    /// side by side and touching one another.
    fn twin_bearing_rig() -> (ConstructionGraph, PartId, PartId, PartId) {
        let mut graph = ConstructionGraph::new();
        let base = spawn_cube(&mut graph, IVec3::ZERO, 2);
        let mounted = [-1, 1].map(|x| {
            let spec = CuboidSpec::new(
                [1; 3],
                BuildPose::from_half_grid(IVec3::new(x, 3, -1), GridRotation::default()),
            )
            .unwrap();
            let Ok(BuildOutcome::Spawned(part)) = graph.apply(BuildCommand::Spawn(spec)) else {
                panic!("block must spawn");
            };
            graph
                .apply(BuildCommand::AddBearing(BearingSpec::new(
                    FaceRef::part(base, FaceKind::PositiveY),
                    FaceRef::part(part, FaceKind::NegativeY),
                    Vec3::new(
                        f32::from(i8::try_from(x).expect("small")) * 0.125,
                        0.25,
                        -0.125,
                    ),
                    Vec3::Y,
                )))
                .unwrap();
            part
        });
        let [first, second] = mounted;
        (graph, base, first, second)
    }

    #[test]
    fn welding_across_a_bearing_is_allowed_and_reports_the_lockup() {
        let (graph, base, mounted, _) = twin_bearing_rig();
        assert!(locked_bearings(&graph).is_empty());

        let staged =
            stage_weld_objects(&graph, FaceOwner::Part(base), FaceOwner::Part(mounted)).unwrap();

        assert_eq!(newly_locked_bearings(&graph, &staged), 1);
        // Allowed, so it has to survive compilation rather than fail later.
        staged.compile().unwrap();
    }

    #[test]
    fn a_loop_that_leaves_every_bearing_free_reports_no_lockup() {
        let (graph, _, first, second) = twin_bearing_rig();

        // Both blocks turn on their own bearing; joining them to each other
        // closes a loop through the base without locking either joint.
        let staged =
            stage_weld_objects(&graph, FaceOwner::Part(first), FaceOwner::Part(second)).unwrap();

        assert_eq!(staged.weld_count(), 1);
        assert_eq!(newly_locked_bearings(&graph, &staged), 0);
        staged.compile().unwrap();
    }

    #[test]
    fn weld_rejects_same_or_separated_objects_without_mutation() {
        let mut graph = ConstructionGraph::new();
        let left = spawn_cube(&mut graph, IVec3::new(0, 2, 0), 4);
        let far = spawn_cube(&mut graph, IVec3::new(8, 2, 0), 4);
        assert!(matches!(
            stage_weld_objects(&graph, FaceOwner::Part(left), FaceOwner::Part(left)),
            Err(PlacementError::SameObject)
        ));
        assert!(matches!(
            stage_weld_objects(&graph, FaceOwner::Part(left), FaceOwner::Part(far)),
            Err(PlacementError::ObjectsDoNotTouch)
        ));
        assert_eq!(graph.weld_count(), 0);
    }

    #[test]
    fn bearing_anchor_snaps_without_mutating_the_graph() {
        let mut graph = ConstructionGraph::new();
        let base = spawn_cube(&mut graph, IVec3::new(0, 2, 0), 4);
        let hit = SurfaceHit {
            distance: 1.0,
            point: Vec3::new(0.18, 1.0, -0.18),
            face: FaceRef::part(base, FaceKind::PositiveY),
        };
        let anchor = bearing_anchor_from_hit(&graph, hit).unwrap();
        assert_eq!(anchor, Vec3::new(0.25, 1.0, -0.25));

        assert_eq!(graph.part_count(), 1);
        assert_eq!(graph.bearing_count(), 0);
        assert!(graph.pending().is_none());
    }

    #[test]
    fn bearing_anchor_uses_the_same_grid_phase_as_blocks_and_cylinders() {
        let graph = ConstructionGraph::new();
        let block = candidate_from_hit(
            &graph,
            SurfaceHit {
                distance: 1.0,
                point: Vec3::ZERO,
                face: FaceRef::ground(),
            },
        );
        let graph = stage_cuboid(&graph, block).unwrap();
        let part = graph.parts().next().unwrap().0;
        let source = FaceRef::part(part, FaceKind::PositiveX);
        let face = face_geometry_from_ref(source, Some(&graph));
        let hit = SurfaceHit {
            distance: 1.0,
            point: face.center + Vec3::new(0.0, 0.02, 0.10),
            face: source,
        };

        for grid in [
            PlacementGrid::Centimetres25,
            PlacementGrid::Centimetres5,
            PlacementGrid::Centimetres1,
        ] {
            let anchor = super::bearing_anchor_from_hit_with_grid(
                &graph,
                hit,
                grid,
                PlacementBounds::Garage,
            )
            .unwrap();
            let block = oriented_cuboid_candidate_from_hit_with_grid(
                &graph,
                hit,
                [1; 3],
                GridRotation::default(),
                grid,
                PlacementBounds::Garage,
            );
            let cylinder = super::cylinder_candidate_from_hit_with_grid(
                &graph,
                hit,
                CylinderDimensions::new(0.5, 0.0, 0.25).unwrap(),
                grid,
                PlacementBounds::Garage,
            )
            .unwrap();

            for axis in [1, 2] {
                assert!((anchor[axis] - block.spec.pose.translation()[axis]).abs() < 1.0e-6);
                assert!((anchor[axis] - cylinder.spec.pose.translation()[axis]).abs() < 1.0e-6);
            }
        }
    }

    #[test]
    fn bearing_second_click_attaches_a_cuboid_without_collider_geometry_for_connector() {
        let mut graph = ConstructionGraph::new();
        let base = spawn_cube(&mut graph, IVec3::new(0, 2, 0), 4);
        let source = FaceRef::part(base, FaceKind::PositiveY);
        let hit = SurfaceHit {
            distance: 1.0,
            point: Vec3::new(0.0, 1.0, 0.0),
            face: source,
        };
        let anchor = bearing_anchor_from_hit(&graph, hit).unwrap();
        let candidate = bearing_attachment_candidate(&graph, source, anchor);

        let dimensions = BearingDimensions::new(0.75, 0.25).unwrap();
        let graph =
            stage_bearing_attachment(&graph, candidate, source, anchor, dimensions).unwrap();

        assert_eq!(graph.part_count(), 2);
        assert_eq!(graph.bearing_count(), 1);
        assert_eq!(graph.bearings().next().unwrap().1.shared_anchor, anchor);
        assert_eq!(graph.bearings().next().unwrap().1.dimensions, dimensions);
        assert!(graph.pending().is_none());
    }

    #[test]
    fn bearing_attachment_centres_a_cylinder_before_it_is_dragged() {
        let mut graph = ConstructionGraph::new();
        let base = spawn_cube(&mut graph, IVec3::new(0, 2, 0), 4);
        let source = FaceRef::part(base, FaceKind::PositiveY);
        let anchor = Vec3::new(0.25, 1.0, -0.25);
        let candidate = cylinder_candidate_from_hit(
            &graph,
            SurfaceHit {
                distance: 1.0,
                point: Vec3::new(0.50, 1.0, 0.25),
                face: source,
            },
            CylinderDimensions::default(),
        )
        .unwrap();

        let centered = center_cylinder_candidate_on_bearing(candidate, anchor);
        let face = super::cylinder_face_geometry(centered.spec, centered.attached_face).unwrap();

        assert!(face.center.abs_diff_eq(anchor, 1.0e-5));
        assert_eq!(centered.anchor, Some(anchor));
        assert_eq!(centered.support, PlacementSupport::Bearing);
    }

    #[test]
    fn oversized_bearing_attaches_to_any_block_face_overlapped_by_its_ring() {
        let mut graph = ConstructionGraph::new();
        let base = spawn_cube(&mut graph, IVec3::new(0, 2, 0), 4);
        let source = FaceRef::part(base, FaceKind::PositiveY);
        let anchor = Vec3::Y;
        let dimensions = BearingDimensions::new(0.80, 0.10).unwrap();
        let candidate = candidate_from_hit(
            &graph,
            SurfaceHit {
                distance: 1.0,
                point: Vec3::new(0.36, 1.0, 0.0),
                face: source,
            },
        );

        assert!(
            !super::face_geometry(candidate.spec, candidate.attached_face)
                .center
                .abs_diff_eq(anchor, 1.0e-5)
        );
        assert!(bearing_overlaps_candidate(
            &graph, source, anchor, dimensions, candidate,
        ));
        let attached =
            stage_bearing_attachment(&graph, candidate, source, anchor, dimensions).unwrap();

        assert_eq!(attached.bearing_count(), 1);
        assert_eq!(attached.weld_count(), 0);
        assert_eq!(attached.bearings().next().unwrap().1.dimensions, dimensions);
    }

    #[test]
    fn bearing_overhang_claims_a_block_placed_on_an_adjacent_support_face() {
        let mut graph = ConstructionGraph::new();
        let source_part = spawn_cube(&mut graph, IVec3::new(0, 2, 0), 4);
        let adjacent_part = spawn_cube(&mut graph, IVec3::new(4, 2, 0), 4);
        let source = FaceRef::part(source_part, FaceKind::PositiveY);
        let adjacent_face = FaceRef::part(adjacent_part, FaceKind::PositiveY);
        let anchor = Vec3::Y;
        let dimensions = BearingDimensions::new(2.40, 0.10).unwrap();
        let candidate = candidate_from_hit(
            &graph,
            SurfaceHit {
                distance: 1.0,
                point: Vec3::new(1.0, 1.0, 0.0),
                face: adjacent_face,
            },
        );

        assert!(bearing_overlaps_candidate(
            &graph, source, anchor, dimensions, candidate,
        ));
        let attached =
            stage_bearing_attachment(&graph, candidate, source, anchor, dimensions).unwrap();

        assert_eq!(attached.bearing_count(), 1);
        assert_eq!(attached.weld_count(), 0);
        assert_eq!(attached.compile().unwrap().compounds.len(), 3);
    }

    #[test]
    fn block_face_entirely_inside_bearing_hole_is_not_covered() {
        let mut graph = ConstructionGraph::new();
        let base = spawn_cube(&mut graph, IVec3::new(0, 2, 0), 4);
        let source = FaceRef::part(base, FaceKind::PositiveY);
        let candidate = bearing_attachment_candidate(&graph, source, Vec3::Y);

        assert!(!bearing_overlaps_candidate(
            &graph,
            source,
            Vec3::Y,
            BearingDimensions::new(1.0, 0.50).unwrap(),
            candidate,
        ));
    }

    #[test]
    fn large_hollow_bearing_uses_a_ring_block_instead_of_the_center_block() {
        let mut graph = ConstructionGraph::new();
        let mut center = None;
        for x in -1..=1 {
            for z in -1..=1 {
                let spec = CuboidSpec::new(
                    [1, 1, 1],
                    BuildPose::from_half_grid(IVec3::new(x * 2, 1, z * 2), GridRotation::default()),
                )
                .unwrap();
                let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(spec)).unwrap()
                else {
                    unreachable!()
                };
                if x == 0 && z == 0 {
                    center = Some(part);
                }
            }
        }
        let center = center.unwrap();
        let selected = FaceRef::part(center, FaceKind::PositiveY);
        let dimensions = BearingDimensions::new(0.75, 0.40).unwrap();

        let support =
            bearing_support_face(&graph, selected, Vec3::new(0.0, 0.25, 0.0), dimensions).unwrap();

        assert_ne!(support.owner, FaceOwner::Part(center));
        assert!(bearing_ring_overlaps_face(
            Vec3::new(0.0, 0.25, 0.0),
            dimensions,
            &face_geometry_from_ref(support, Some(&graph)),
        ));
    }

    #[test]
    fn placement_from_bearing_body_welds_only_to_the_clicked_rigid_group() {
        let mut graph = ConstructionGraph::new();
        let base = spawn_cube(&mut graph, IVec3::ZERO, 1);
        let attached = spawn_cube(&mut graph, IVec3::new(0, 1, 0), 1);
        let sibling = spawn_cube(&mut graph, IVec3::new(-1, 1, 0), 1);
        let neighbour = spawn_cube(&mut graph, IVec3::new(1, 2, 0), 1);
        let dimensions = BearingDimensions::new(0.80, 0.10).unwrap();
        for target in [attached, sibling] {
            graph
                .apply(BuildCommand::AddBearing(
                    BearingSpec::new(
                        FaceRef::part(base, FaceKind::PositiveY),
                        FaceRef::part(target, FaceKind::NegativeY),
                        Vec3::new(0.0, 0.125, 0.0),
                        Vec3::Y,
                    )
                    .with_dimensions(dimensions),
                ))
                .unwrap();
        }
        graph
            .apply(BuildCommand::RigidLink(RigidLinkSpec {
                first: attached,
                second: sibling,
            }))
            .unwrap();
        let source = FaceRef::part(attached, FaceKind::PositiveY);
        let source_face = face_geometry_from_ref(source, Some(&graph));
        let candidate = candidate_from_hit(
            &graph,
            SurfaceHit {
                distance: 0.0,
                point: source_face.center,
                face: source,
            },
        );

        let staged =
            stage_block_batch_from_source(&graph, candidate, &[candidate.spec], source.owner)
                .unwrap();
        let placed = staged
            .parts()
            .find_map(|(part, _)| graph.part(part).is_none().then_some(part))
            .unwrap();
        let attached_group = rigid_body_parts(&staged, attached);

        assert!(attached_group.contains(&placed));
        assert!(attached_group.contains(&sibling));
        assert!(!attached_group.contains(&neighbour));
        assert!(!attached_group.contains(&base));
        assert_eq!(staged.bearing_count(), 2);
        assert_eq!(staged.compile().unwrap().bearings.len(), 1);
    }

    #[test]
    fn one_bearing_groups_multiple_direct_attachments_into_one_rotor() {
        let mut graph = ConstructionGraph::new();
        let support = spawn_cube(&mut graph, IVec3::new(0, 2, 0), 4);
        let source = FaceRef::part(support, FaceKind::PositiveY);
        let dimensions = BearingDimensions::new(0.80, 0.10).unwrap();
        let candidates = [0.0, 0.25].map(|x| {
            candidate_from_hit(
                &graph,
                SurfaceHit {
                    distance: 0.0,
                    point: Vec3::new(x, 1.0, 0.0),
                    face: source,
                },
            )
        });

        for candidate in candidates {
            let rigid_targets = graph
                .bearings()
                .filter_map(|(_, bearing)| match bearing.target.owner {
                    FaceOwner::Part(part) => Some(part),
                    FaceOwner::Ground => None,
                })
                .collect::<Vec<_>>();
            graph = stage_bearing_block_batch(
                &graph,
                candidate,
                &[candidate.spec],
                source,
                Vec3::Y,
                dimensions,
                &rigid_targets,
            )
            .unwrap();
        }

        let targets = graph
            .parts()
            .filter_map(|(part, _)| (part != support).then_some(part))
            .collect::<Vec<_>>();
        assert_eq!(targets.len(), 2);
        assert_eq!(graph.bearing_count(), 2);
        assert_eq!(graph.weld_count(), 0);
        assert_eq!(graph.rigid_link_count(), 1);
        assert_eq!(rigid_body_parts(&graph, targets[0]), targets);
        let compiled = graph.compile().unwrap();
        assert_eq!(compiled.compounds.len(), 2);
        assert_eq!(compiled.bearings.len(), 1);
    }

    #[test]
    fn bearing_drag_attaches_one_internally_welded_sheet() {
        let mut graph = ConstructionGraph::new();
        let base = spawn_cube(&mut graph, IVec3::new(0, 2, 0), 4);
        let source = FaceRef::part(base, FaceKind::PositiveY);
        let anchor = Vec3::Y;
        let candidate = bearing_attachment_candidate(&graph, source, anchor);
        let mut endpoint = candidate.spec.pose.translation_half_units();
        endpoint.x += 2;
        let specs = block_sheet_specs(candidate.spec, endpoint, PlacementPlane::Xz).unwrap();

        let graph = stage_bearing_block_batch(
            &graph,
            candidate,
            &specs,
            source,
            anchor,
            BearingDimensions::default(),
            &[],
        )
        .unwrap();

        assert_eq!(graph.part_count(), 3);
        assert_eq!(graph.bearing_count(), 1);
        assert_eq!(graph.weld_count(), 1);
        assert_eq!(graph.compile().unwrap().compounds.len(), 2);
    }

    #[test]
    fn bearing_centres_and_attaches_on_a_quarter_metre_block() {
        let graph = ConstructionGraph::new();
        let block = candidate_from_hit(
            &graph,
            SurfaceHit {
                distance: 1.0,
                point: Vec3::ZERO,
                face: FaceRef::ground(),
            },
        );
        let graph = stage_cuboid(&graph, block).unwrap();
        let base = graph.parts().next().unwrap().0;
        let source = FaceRef::part(base, FaceKind::PositiveY);
        let hit = SurfaceHit {
            distance: 1.0,
            point: Vec3::new(0.1, 0.25, -0.1),
            face: source,
        };

        let anchor = bearing_anchor_from_hit(&graph, hit).unwrap();
        assert!(anchor.abs_diff_eq(Vec3::new(0.0, 0.25, 0.0), 1.0e-6));
        let candidate = bearing_attachment_candidate(&graph, source, anchor);
        let graph = stage_bearing_attachment(
            &graph,
            candidate,
            source,
            anchor,
            BearingDimensions::default(),
        )
        .unwrap();

        assert_eq!(graph.part_count(), 2);
        assert_eq!(graph.bearing_count(), 1);
    }

    #[test]
    fn bearing_rejects_ground_but_allows_visual_overhang_at_face_edges() {
        let mut graph = ConstructionGraph::new();
        let ground_hit = SurfaceHit {
            distance: 1.0,
            point: Vec3::ZERO,
            face: FaceRef::ground(),
        };
        assert!(matches!(
            bearing_anchor_from_hit(&graph, ground_hit),
            Err(PlacementError::BearingOnGround)
        ));

        let part = spawn_cube(&mut graph, IVec3::new(0, 2, 0), 2);
        let edge_hit = SurfaceHit {
            distance: 1.0,
            point: Vec3::new(0.25, 0.5, 0.25),
            face: FaceRef::part(part, FaceKind::PositiveY),
        };
        let anchor = bearing_anchor_from_hit(&graph, edge_hit).unwrap();
        assert_eq!(anchor, Vec3::new(0.25, 0.75, 0.25));
        let candidate = bearing_attachment_candidate(&graph, edge_hit.face, anchor);
        let dimensions = BearingDimensions::new(8.0, 0.10).unwrap();
        let attached =
            stage_bearing_attachment(&graph, candidate, edge_hit.face, anchor, dimensions).unwrap();
        assert_eq!(attached.bearings().next().unwrap().1.dimensions, dimensions);
        assert_eq!(graph.part_count(), 1);
    }

    #[test]
    fn rejected_overlap_does_not_mutate_source_graph() {
        let graph = ConstructionGraph::new();
        let base_hit = SurfaceHit {
            distance: 1.0,
            point: Vec3::ZERO,
            face: FaceRef::ground(),
        };
        let candidate = candidate_from_hit(&graph, base_hit);
        let graph = stage_cuboid(&graph, candidate).unwrap();
        assert!(matches!(
            stage_cuboid(&graph, candidate),
            Err(PlacementError::OverlapsPart(_))
        ));
        assert_eq!(graph.part_count(), 1);
    }

    #[test]
    fn remove_cascades_through_incident_bearing() {
        let mut graph = ConstructionGraph::new();
        let base = spawn_cube(&mut graph, IVec3::new(0, 2, 0), 4);
        let source = FaceRef::part(base, FaceKind::PositiveY);
        let anchor = Vec3::new(0.0, 1.0, 0.0);
        let candidate = bearing_attachment_candidate(&graph, source, anchor);
        let graph = stage_bearing_attachment(
            &graph,
            candidate,
            source,
            anchor,
            BearingDimensions::default(),
        )
        .unwrap();
        let top = raycast_construction(&graph, Vec3::new(0.0, 5.0, 0.0), Vec3::NEG_Y).unwrap();
        let upper = match top.face.owner {
            FaceOwner::Part(part) => part,
            FaceOwner::Ground => panic!("top ray must hit attached part"),
        };
        let mut graph = graph;
        graph.apply(BuildCommand::Remove(upper)).unwrap();
        assert_eq!(graph.part_count(), 1);
        assert_eq!(graph.bearing_count(), 0);
    }

    #[test]
    fn a_ray_meets_a_shaped_face_where_the_surface_actually_is() {
        // A wedge's sloped face sits well inside the block's old box, so a hit
        // that still lands on the box would put the cursor in mid-air.
        let spec = CuboidSpec::new(
            [1, 1, 1],
            BuildPose::from_half_grid(IVec3::new(1, 1, 1), GridRotation::default()),
        )
        .unwrap();
        let mut graph = ConstructionGraph::new();
        graph.apply(BuildCommand::Spawn(spec)).unwrap();
        let region = mechanic_core::ShapeRegion::new(
            IVec3::ZERO,
            IVec3::ONE,
            mechanic_core::ConstructionMaterial::Steel,
        )
        .unwrap();
        let BuildOutcome::RegionAdded(id) = graph.apply(BuildCommand::AddRegion(region)).unwrap()
        else {
            panic!("wrong outcome")
        };
        let cell = i16::try_from(mechanic_core::STEPS_PER_CELL).unwrap();
        graph
            .apply(BuildCommand::SetRegionVertices {
                region: id,
                vertices: vec![([0, 1, 1], [0, -cell, 0]), ([1, 1, 1], [0, -cell, 0])],
            })
            .expect("collapsing an edge makes a wedge");

        // Straight down onto the sloped half of the top face.
        let origin = Vec3::new(0.125, 2.0, 0.1875);
        let hit =
            raycast_construction(&graph, origin, Vec3::NEG_Y).expect("the wedge is under the ray");
        assert!(
            matches!(hit.face.owner, FaceOwner::Part(_)),
            "the ray should meet the block, not the ground"
        );
        assert!(
            hit.point.y < 0.25 - 1.0e-3,
            "the slope is below the old box top; hit at y={}",
            hit.point.y
        );
        assert!(
            hit.point.y > 0.0,
            "the hit should still be on the wedge, not through it"
        );
    }

    #[test]
    fn raycast_tests_a_multi_block_region_once() {
        fn spawn_at(graph: &mut ConstructionGraph, half_grid: IVec3) -> PartId {
            let spec = CuboidSpec::new(
                [1; 3],
                BuildPose::from_half_grid(half_grid, GridRotation::default()),
            )
            .unwrap();
            let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(spec)).unwrap()
            else {
                unreachable!()
            };
            part
        }

        let mut graph = ConstructionGraph::new();
        let first = spawn_at(&mut graph, IVec3::new(1, 1, 1));
        let second = spawn_at(&mut graph, IVec3::new(3, 1, 1));
        graph
            .apply(BuildCommand::Weld(WeldSpec {
                first: FaceRef::part(first, FaceKind::PositiveX),
                second: FaceRef::part(second, FaceKind::NegativeX),
            }))
            .unwrap();
        let region = mechanic_core::ShapeRegion::new(
            IVec3::ZERO,
            IVec3::new(2, 1, 1),
            ConstructionMaterial::Steel,
        )
        .unwrap();
        graph.apply(BuildCommand::AddRegion(region)).unwrap();
        spawn_at(&mut graph, IVec3::new(9, 1, 1));

        let sources = raycast_sources(&graph, |_| true).collect::<Vec<_>>();
        assert_eq!(sources.len(), 2, "one region and one standalone part");
        let accepted = super::raycast_construction_filtered_with_ground(
            &graph,
            Vec3::new(0.125, 2.0, 0.125),
            Vec3::NEG_Y,
            None,
            |part| part == second,
        )
        .unwrap();
        assert_eq!(accepted.face.owner, FaceOwner::Part(second));
        assert_eq!(
            sources
                .iter()
                .filter(|(_, _, region)| region.is_some())
                .count(),
            1,
            "all region members share one raycast source"
        );

        let origin = Vec3::new(0.375, 2.0, 0.125);
        let hit = raycast_construction_with_ground(&graph, origin, Vec3::NEG_Y, None).unwrap();
        let frame = mechanic_core::ConstructionFrame::new(
            Vec3::new(3.0, 4.0, 2.0),
            Quat::from_rotation_z(0.43),
        )
        .unwrap();
        graph.reframe_parts([first, second], frame).unwrap();
        let reframed = raycast_construction_with_ground(
            &graph,
            frame.point(origin),
            frame.vector(Vec3::NEG_Y),
            None,
        )
        .unwrap();
        assert_eq!(reframed.face, hit.face);
        assert!(reframed.point.distance(frame.point(hit.point)) < 1.0e-5);
    }

    #[test]
    fn featured_region_flat_patch_accepts_placement_across_member_blocks() {
        let mut graph = ConstructionGraph::new();
        let spawn_at = |graph: &mut ConstructionGraph, center| {
            let spec = CuboidSpec::new(
                [1; 3],
                BuildPose::from_half_grid(center, GridRotation::default()),
            )
            .unwrap();
            let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(spec)).unwrap()
            else {
                unreachable!()
            };
            part
        };
        let first = spawn_at(&mut graph, IVec3::new(1, 1, 1));
        let second = spawn_at(&mut graph, IVec3::new(3, 1, 1));
        graph
            .apply(BuildCommand::Weld(WeldSpec {
                first: FaceRef::part(first, FaceKind::PositiveX),
                second: FaceRef::part(second, FaceKind::NegativeX),
            }))
            .unwrap();
        let region = mechanic_core::ShapeRegion::new(
            IVec3::ZERO,
            IVec3::new(2, 1, 1),
            ConstructionMaterial::Steel,
        )
        .unwrap();
        let BuildOutcome::RegionAdded(region) =
            graph.apply(BuildCommand::AddRegion(region)).unwrap()
        else {
            unreachable!()
        };
        let owner = SolidOwner::Region(region);
        let edge = graph.evaluated_solid(owner).unwrap().logical_edges[0].key;
        graph
            .apply(BuildCommand::AddShapeFeature(ShapeFeature::new(
                [EdgeChainRef { owner, edge }],
                EdgeTreatment::Fillet,
                20,
            )))
            .unwrap();

        let hit = raycast_construction(&graph, Vec3::new(0.375, 2.0, 0.125), Vec3::NEG_Y)
            .expect("the retained top patch is under the ray");
        let candidate = candidate_from_hit(&graph, hit);

        assert!(
            hit.face.patch.is_some(),
            "evaluated hits retain patch identity"
        );
        assert!(
            candidate.anchor.is_some(),
            "the patch must not collapse to the representative member's 25 cm face"
        );
        validate_block_batch_in_bounds(
            &graph,
            candidate,
            &[candidate.spec],
            PlacementBounds::Garage,
        )
        .expect("a block may be placed on the second member's flat patch");
        let staged = stage_block_batch_in_bounds(
            &graph,
            candidate,
            &[candidate.spec],
            PlacementBounds::Garage,
        )
        .expect("placement commits on the evaluated patch");
        assert_eq!(staged.part_count(), 3);
        assert_eq!(staged.weld_count(), 2);
    }

    #[test]
    fn fillet_hit_places_against_the_nearest_retained_flat_patch() {
        let mut graph = ConstructionGraph::new();
        let BuildOutcome::Spawned(part) = graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new(
                    [4; 3],
                    BuildPose::new(IVec3::new(1, 1, 1), GridRotation::default()),
                )
                .unwrap(),
            ))
            .unwrap()
        else {
            unreachable!()
        };
        let owner = SolidOwner::Part(part);
        let base = graph.evaluated_solid(owner).unwrap();
        let logical = &base.logical_edges[0];
        let half_edge = base.half_edges[logical.half_edges[0] as usize];
        let twin = base.half_edges[half_edge.twin as usize];
        let start = base.vertices[half_edge.origin as usize].position;
        let end = base.vertices[base.half_edges[half_edge.next as usize].origin as usize].position;
        let outward = (base.surfaces[half_edge.face as usize].normal
            + base.surfaces[twin.face as usize].normal)
            .normalize();
        graph
            .apply(BuildCommand::AddShapeFeature(ShapeFeature::new(
                [EdgeChainRef {
                    owner,
                    edge: logical.key,
                }],
                EdgeTreatment::Fillet,
                20,
            )))
            .unwrap();

        let edge_midpoint = (start + end) * 0.5;
        let hit = raycast_construction(&graph, edge_midpoint + outward * 2.0, -outward)
            .expect("the ray crosses the rounded edge");
        let candidate = candidate_from_hit(&graph, hit);

        assert!(
            matches!(
                hit.face.patch.map(|patch| patch.source),
                Some(mechanic_core::TopologySource::Base)
            ),
            "a rounded facet routes placement to an adjacent base plane"
        );
        assert!(candidate.anchor.is_some());
    }

    #[test]
    fn block_sheet_only_needs_one_block_on_a_promoted_filleted_region() {
        let mut graph = ConstructionGraph::new();
        let mut members = Vec::new();
        for y in 0..4 {
            for x in 0..4 {
                let spec = CuboidSpec::new(
                    [1; 3],
                    BuildPose::from_half_grid(
                        IVec3::new(1 + x * 2, 1 + y * 2, 1),
                        GridRotation::default(),
                    ),
                )
                .unwrap();
                let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(spec)).unwrap()
                else {
                    unreachable!()
                };
                if let Some(&previous) = members.last() {
                    graph
                        .apply(BuildCommand::RigidLink(RigidLinkSpec {
                            first: previous,
                            second: part,
                        }))
                        .unwrap();
                }
                members.push(part);
            }
        }
        let region = mechanic_core::ShapeRegion::new(
            IVec3::ZERO,
            IVec3::new(4, 4, 1),
            ConstructionMaterial::Steel,
        )
        .unwrap();
        let BuildOutcome::RegionAdded(region) =
            graph.apply(BuildCommand::AddRegion(region)).unwrap()
        else {
            unreachable!()
        };
        let owner = SolidOwner::Region(region);
        let base = graph.evaluated_solid(owner).unwrap();
        let edge = base
            .logical_edges
            .iter()
            .find(|logical| {
                let half_edge = base.half_edges[logical.half_edges[0] as usize];
                let twin = base.half_edges[half_edge.twin as usize];
                let normals = [
                    base.surfaces[half_edge.face as usize].normal,
                    base.surfaces[twin.face as usize].normal,
                ];
                normals.contains(&Vec3::X) && normals.contains(&Vec3::Y)
            })
            .expect("the cuboid has a positive-x/positive-y edge")
            .key;
        graph
            .apply(BuildCommand::AddShapeFeature(ShapeFeature::new(
                [EdgeChainRef { owner, edge }],
                EdgeTreatment::Fillet,
                120,
            )))
            .unwrap();

        let hit = raycast_construction(&graph, Vec3::new(0.625, 2.0, 0.125), Vec3::NEG_Y)
            .expect("the retained top patch is under the ray");
        let start = candidate_from_hit(&graph, hit);
        let specs = block_box_specs(start.spec, IVec3::X).unwrap();

        assert!(start.anchor.is_some());
        let staged = stage_block_batch_in_bounds(&graph, start, &specs, PlacementBounds::Garage)
            .expect("one supported block keeps the connected sheet placeable");
        assert_eq!(staged.part_count(), 18);
    }

    #[test]
    fn an_unshaped_part_still_reports_its_grid_face() {
        let spec = CuboidSpec::new(
            [1, 1, 1],
            BuildPose::from_half_grid(IVec3::new(1, 1, 1), GridRotation::default()),
        )
        .unwrap();
        let mut graph = ConstructionGraph::new();
        graph.apply(BuildCommand::Spawn(spec)).unwrap();
        let hit = raycast_construction(&graph, Vec3::new(0.125, 2.0, 0.125), Vec3::NEG_Y)
            .expect("the block is under the ray");
        assert_eq!(hit.face.face, FaceKind::PositiveY);
        assert!((hit.point.y - 0.25).abs() < 1.0e-4);
    }

    /// One block claimed as a region, with its top +z edge optionally collapsed.
    fn block_with_region(shaped: bool) -> (ConstructionGraph, FaceRef) {
        let spec = CuboidSpec::new(
            [1, 1, 1],
            BuildPose::from_half_grid(IVec3::new(1, 1, 1), GridRotation::default()),
        )
        .unwrap();
        let mut graph = ConstructionGraph::new();
        let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(spec)).unwrap() else {
            panic!("wrong spawn outcome")
        };
        let region = mechanic_core::ShapeRegion::new(
            IVec3::ZERO,
            IVec3::ONE,
            mechanic_core::ConstructionMaterial::Steel,
        )
        .unwrap();
        let BuildOutcome::RegionAdded(id) = graph.apply(BuildCommand::AddRegion(region)).unwrap()
        else {
            panic!("wrong outcome")
        };
        if shaped {
            let cell = i16::try_from(mechanic_core::STEPS_PER_CELL).unwrap();
            graph
                .apply(BuildCommand::SetRegionVertices {
                    region: id,
                    vertices: vec![([0, 1, 1], [0, -cell, 0]), ([1, 1, 1], [0, -cell, 0])],
                })
                .unwrap();
        }
        (graph, FaceRef::part(part, FaceKind::PositiveY))
    }

    #[test]
    fn placement_is_refused_on_a_shaped_face() {
        // The top face has been sloped, so nothing can sit flush on it.
        let (graph, top) = block_with_region(true);
        assert!(!face_is_flat(&graph, top));
    }

    #[test]
    fn flattening_a_shaped_face_makes_it_placeable_again() {
        // Bringing those corners back onto the grid is how a mounting surface
        // is made where the shaping had removed one.
        let (mut graph, top) = block_with_region(true);
        assert!(!face_is_flat(&graph, top));
        let id = graph.regions().next().unwrap().0;
        graph
            .apply(BuildCommand::SetRegionVertices {
                region: id,
                vertices: vec![([0, 1, 1], [0, 0, 0]), ([1, 1, 1], [0, 0, 0])],
            })
            .unwrap();
        assert!(face_is_flat(&graph, top));
    }

    #[test]
    fn an_unshaped_face_is_always_placeable() {
        let (graph, top) = block_with_region(false);
        assert!(face_is_flat(&graph, top));
        assert!(
            face_is_flat(&graph, FaceRef::ground()),
            "the ground is always flat"
        );
    }

    #[test]
    fn shaping_one_face_leaves_the_others_placeable() {
        // Only the face that moved loses its mounting surface.
        let (graph, _) = block_with_region(true);
        let part = graph.parts().next().unwrap().0;
        assert!(
            face_is_flat(&graph, FaceRef::part(part, FaceKind::NegativeY)),
            "the untouched underside must still take a block"
        );
    }

    #[test]
    fn pipe_run_preserves_fine_grid_lateral_offsets() {
        let start = Vec3::new(0.05, 0.0, 0.10);
        let pieces = pipe_run_pieces(
            &[start, start + Vec3::Y * 0.25],
            &[],
            CylinderDimensions::default(),
            ConstructionMaterial::Steel,
        )
        .unwrap();
        let PartSpec::Cylinder(pipe) = pieces[0].spec else {
            panic!("a straight run produces a cylinder")
        };

        assert!(
            pipe.pose
                .translation()
                .abs_diff_eq(Vec3::new(0.05, 0.125, 0.10), 1.0e-5)
        );
    }

    #[test]
    fn pipe_run_trims_straights_to_bend_tangencies() {
        let dimensions = CylinderDimensions::new(0.25, 0.10, 0.25).unwrap();
        let pieces = pipe_run_pieces(
            &[
                Vec3::ZERO,
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
            ],
            &[0.25],
            dimensions,
            ConstructionMaterial::Steel,
        )
        .unwrap();
        assert_eq!(pieces.len(), 3);
        assert!(matches!(pieces[1].spec, PartSpec::PipeBend(_)));
        for piece in [pieces[0], pieces[2]] {
            let PartSpec::Cylinder(cylinder) = piece.spec else {
                panic!("the ends remain straight")
            };
            assert!((cylinder.dimensions.axial_length() - 0.75).abs() < 1.0e-5);
        }
    }

    #[test]
    fn pipe_run_omits_zero_length_straights_and_keeps_per_corner_radii() {
        let dimensions = CylinderDimensions::new(0.25, 0.10, 0.25).unwrap();
        let one = pipe_run_pieces(
            &[Vec3::ZERO, Vec3::X * 0.25, Vec3::new(0.25, 0.25, 0.0)],
            &[0.25],
            dimensions,
            ConstructionMaterial::Steel,
        )
        .unwrap();
        assert_eq!(one.len(), 1);
        assert!(matches!(one[0].spec, PartSpec::PipeBend(_)));

        let multiple = pipe_run_pieces(
            &[
                Vec3::ZERO,
                Vec3::X,
                Vec3::new(1.0, 1.5, 0.0),
                Vec3::new(2.0, 1.5, 0.0),
            ],
            &[0.25, 0.50],
            dimensions,
            ConstructionMaterial::Aluminium,
        )
        .unwrap();
        let radii = multiple
            .iter()
            .filter_map(|piece| piece.spec.as_pipe_bend())
            .map(|bend| bend.dimensions.radius())
            .collect::<Vec<_>>();
        assert_eq!(radii, vec![0.25, 0.50]);
    }

    #[test]
    fn pipe_run_rejects_insufficient_between_bend_clearance() {
        let dimensions = CylinderDimensions::new(0.25, 0.10, 0.25).unwrap();
        let error = pipe_run_pieces(
            &[
                Vec3::ZERO,
                Vec3::X,
                Vec3::new(1.0, 0.5, 0.0),
                Vec3::new(2.0, 0.5, 0.0),
            ],
            &[0.50, 0.50],
            dimensions,
            ConstructionMaterial::Steel,
        )
        .unwrap_err();
        assert!(error.to_string().contains("1.00 m clearance"));
    }

    #[test]
    fn pipe_bend_end_raycasts_report_the_flat_caps() {
        let mut graph = ConstructionGraph::new();
        let dimensions = PipeBendDimensions::new(0.20, 0.10, 0.25).unwrap();
        let BuildOutcome::Spawned(part) = graph
            .apply(BuildCommand::SpawnPipeBend(PipeBendSpec::new(
                dimensions,
                BuildPose::default(),
            )))
            .unwrap()
        else {
            panic!("pipe bend must spawn")
        };

        for (origin, direction, face) in [
            (
                Vec3::new(0.075, dimensions.radius() + 1.0, 0.0),
                Vec3::NEG_Y,
                FaceKind::PositiveY,
            ),
            (
                Vec3::new(-dimensions.radius() - 1.0, 0.0, 0.075),
                Vec3::X,
                FaceKind::NegativeX,
            ),
        ] {
            let hit = raycast_construction(&graph, origin, direction)
                .expect("the pipe bend cap is visible");
            assert_eq!(hit.face, FaceRef::part(part, face));
        }
    }

    #[test]
    fn sub_block_pipe_can_turn_immediately_after_an_existing_bend() {
        let mut graph = ConstructionGraph::new();
        let bend_dimensions = PipeBendDimensions::new(0.20, 0.10, 0.25).unwrap();
        let BuildOutcome::Spawned(source) = graph
            .apply(BuildCommand::SpawnPipeBend(PipeBendSpec::new(
                bend_dimensions,
                BuildPose::default(),
            )))
            .unwrap()
        else {
            panic!("source bend must spawn")
        };
        let start = Vec3::Y * bend_dimensions.radius();
        let corner = start + Vec3::Y * 0.25;
        let pieces = pipe_run_pieces(
            &[start, corner, corner + Vec3::X * 0.25],
            &[0.25],
            CylinderDimensions::new(0.20, 0.10, 0.25).unwrap(),
            ConstructionMaterial::Steel,
        )
        .unwrap();

        let staged = stage_pipe_run(
            &graph,
            &pieces,
            PipeRunAttachment::AutoWeld {
                source: FaceOwner::Part(source),
            },
        )
        .expect("the new bend only touches the source at its inlet cap");

        assert_eq!(pieces.len(), 1);
        assert_eq!(staged.part_count(), 2);
        assert_eq!(staged.weld_count(), 1);
        assert!(staged.compile().is_ok());
    }

    #[test]
    fn linear_bent_pipe_joins_existing_carriage_attachments_as_one_compound() {
        use mechanic_core::{CarriageFace, LinearBearing, LinearBearingDimensions};
        let mut graph = ConstructionGraph::new();
        let base = spawn_cube(&mut graph, IVec3::ZERO, 1);
        let source = FaceRef::part(base, FaceKind::PositiveY);
        let anchor = Vec3::new(0.0, 0.125, 0.0);
        let rail = LinearBearing {
            dimensions: LinearBearingDimensions::default(),
            mount_normal: Vec3::Y,
            face: CarriageFace::Top,
        };
        let surface = super::linear_carriage_face(anchor, rail, Vec3::X).unwrap();
        let attachment = super::LinearAttachment {
            source,
            anchor,
            rail,
            axis: Vec3::X,
            rigid_targets: &[],
        };
        let block = super::linear_block_candidate(
            anchor,
            rail,
            Vec3::X,
            surface.center - Vec3::Z * 0.125,
            [1; 3],
            GridRotation::default(),
        )
        .unwrap();
        let graph = super::stage_linear_block_batch_in_bounds(
            &graph,
            block,
            &[block.spec],
            attachment,
            PlacementBounds::Garage,
        )
        .unwrap();
        let targets = graph
            .parts()
            .filter_map(|(part, _)| (part != base).then_some(part))
            .collect::<Vec<_>>();
        let start = surface.center + Vec3::Z * 0.125;
        let corner = start + Vec3::Y;
        let pieces = pipe_run_pieces(
            &[start, corner, corner + Vec3::X],
            &[0.25],
            CylinderDimensions::new(0.20, 0.10, 0.25).unwrap(),
            ConstructionMaterial::Steel,
        )
        .unwrap();
        let staged = super::stage_pipe_run_in_bounds(
            &graph,
            &pieces,
            PipeRunAttachment::Linear(super::LinearAttachment {
                rigid_targets: &targets,
                ..attachment
            }),
            PlacementBounds::Garage,
        )
        .unwrap();
        assert!(
            pieces
                .iter()
                .any(|piece| matches!(piece.spec, PartSpec::PipeBend(_)))
        );
        assert_eq!(staged.part_count(), graph.part_count() + pieces.len());
        assert_eq!(staged.weld_count(), pieces.len() - 1);
        let compiled = staged.compile().unwrap();
        assert_eq!(compiled.compounds.len(), 2);
        assert_eq!(compiled.bearings.len(), 1);
        assert!(matches!(
            compiled.bearings[0].kind,
            mechanic_core::BearingKind::Linear(_)
        ));
        assert_eq!(
            graph.part_count(),
            2,
            "staging must preserve its input graph"
        );
    }

    #[test]
    fn staged_pipe_run_spawns_and_welds_every_piece_atomically() {
        let graph = ConstructionGraph::new();
        let dimensions = CylinderDimensions::new(0.25, 0.10, 0.25).unwrap();
        let pieces = pipe_run_pieces(
            &[Vec3::ZERO, Vec3::Y, Vec3::new(1.0, 1.0, 0.0)],
            &[0.25],
            dimensions,
            ConstructionMaterial::Steel,
        )
        .unwrap();
        let staged = stage_pipe_run(
            &graph,
            &pieces,
            PipeRunAttachment::AutoWeld {
                source: FaceOwner::Ground,
            },
        )
        .unwrap();
        assert_eq!(staged.part_count(), pieces.len());
        assert_eq!(staged.weld_count(), pieces.len());
        assert_eq!(
            graph.part_count(),
            0,
            "staging leaves the source graph untouched"
        );
        assert!(staged.compile().is_ok());
        assert!(pieces.iter().any(|piece| {
            piece
                .spec
                .as_pipe_bend()
                .is_some_and(|bend| (bend.dimensions.radius() - 0.25).abs() < 1.0e-5)
        }));
    }
}
