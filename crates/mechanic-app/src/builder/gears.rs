//! Gears: fitting the tool's settings to the part under the cursor, cutting
//! teeth into it, and finding the parts it then meshes with.

use super::spiral::{SpiralTarget, spiral_target_from_hit, validate_spiral};
use super::{PlacementBounds, PlacementError, SurfaceHit, Vec3};
use bevy::prelude::Vec2;
use mechanic_core::{
    BuildCommand, ConstructionFrame, ConstructionGraph, CuboidSpec, CylinderDimensions,
    CylinderSpec, FaceKind, FaceOwner, GEAR_PARALLEL_COSINE, GEAR_PERPENDICULAR_COSINE, GearEnd,
    GearError, GearKind, GearLinkKind, GearLinkSpec, GearMesh, GearSpec, MAX_GEAR_MODULE_TICKS,
    MAX_GEAR_TEETH, MIN_CYLINDER_DIAMETER_GAP, MIN_GEAR_MODULE_TICKS, MIN_GEAR_TEETH,
    POSITION_TICK_METERS, PartId, PartSpec, RackSpec, SpiralEnd, mesh,
};

/// The cylinder teeth go on: the same target the Spiral tool cuts into.
pub(crate) type GearTarget = SpiralTarget;

/// Finds the cylinder under the cursor, as the Spiral tool does.
pub(crate) fn gear_target_from_hit(
    graph: &ConstructionGraph,
    hit: SurfaceHit,
) -> Result<GearTarget, PlacementError> {
    spiral_target_from_hit(graph, hit)
}

/// A cuboid face rack teeth go along.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RackTarget {
    pub(crate) part: PartId,
    pub(crate) spec: CuboidSpec,
    /// Construction frame the part is authored in.
    pub(crate) frame: ConstructionFrame,
    pub(crate) face: FaceKind,
}

pub(crate) fn rack_target_from_hit(
    graph: &ConstructionGraph,
    hit: SurfaceHit,
) -> Result<RackTarget, PlacementError> {
    let not_a_block = || PlacementError::Graph("point at a block's face".to_owned());
    let FaceOwner::Part(part) = hit.face.owner else {
        return Err(not_a_block());
    };
    let Some(PartSpec::Cuboid(spec)) = graph.part(part).copied() else {
        return Err(not_a_block());
    };
    let frame = graph.part_frame(part).ok_or_else(not_a_block)?;
    Ok(RackTarget {
        part,
        spec,
        frame,
        face: hit.face.face,
    })
}

/// What the Gear tool cuts. Lengths are position ticks of 2.5 mm.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GearSettings {
    pub(crate) module_ticks: u8,
    pub(crate) teeth: u16,
    pub(crate) kind: GearKind,
    /// Whether rack teeth run across a face's shorter side instead of its longer one.
    pub(crate) across: bool,
    /// Whether the player set the tooth count, so cylinders stop sizing it.
    pub(crate) fitted: bool,
}

impl Default for GearSettings {
    fn default() -> Self {
        Self {
            module_ticks: 4,
            teeth: 24,
            kind: GearKind::Spur,
            across: false,
            fitted: false,
        }
    }
}

/// One adjustable number of the tool.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GearDimension {
    Module,
    Teeth,
}

pub(crate) const fn kind_label(kind: GearKind) -> &'static str {
    match kind {
        GearKind::Spur => "spur",
        GearKind::Internal => "ring",
        GearKind::Bevel { .. } => "bevel",
    }
}

impl GearSettings {
    /// Steps one number up or down within its range. Teeth step by one, or
    /// by six when `coarse`; setting them stops cylinders sizing them.
    pub(crate) fn adjusted(
        mut self,
        dimension: GearDimension,
        direction: i8,
        coarse: bool,
    ) -> Self {
        match dimension {
            GearDimension::Module => {
                self.module_ticks =
                    u8::try_from((i16::from(self.module_ticks) + i16::from(direction)).clamp(
                        i16::from(MIN_GEAR_MODULE_TICKS),
                        i16::from(MAX_GEAR_MODULE_TICKS),
                    ))
                    .expect("clamped into u8 range");
            }
            GearDimension::Teeth => {
                let step = i32::from(direction) * if coarse { 6 } else { 1 };
                self.teeth = u16::try_from(
                    (i32::from(self.teeth) + step)
                        .clamp(i32::from(MIN_GEAR_TEETH), i32::from(MAX_GEAR_TEETH)),
                )
                .expect("clamped into u16 range");
                self.fitted = true;
            }
        }
        self
    }

    /// Spur, ring, bevel, and round again.
    #[must_use]
    pub(crate) fn cycled(mut self) -> Self {
        self.kind = match self.kind {
            GearKind::Spur => GearKind::Internal,
            GearKind::Internal => GearKind::Bevel {
                cone_angle_degrees: 45,
                large_end: SpiralEnd::PositiveY,
            },
            GearKind::Bevel { .. } => GearKind::Spur,
        };
        self
    }

    /// Sizes the tooth count to a cylinder, as many teeth as its diameter
    /// holds at the current module, until the player sets the count.
    pub(crate) fn fitted_to(mut self, cylinder: CylinderSpec) -> Self {
        if self.fitted {
            return self;
        }
        let suited = self.suited_to(cylinder);
        let envelope = if suited.kind == GearKind::Internal {
            cylinder.dimensions.inner_diameter()
        } else {
            cylinder.dimensions.outer_diameter()
        };
        self.teeth = GearSpec::teeth_for_tip_diameter(self.module_ticks, suited.kind, envelope);
        self
    }

    /// The settings as they go onto one cylinder: a solid cylinder has no bore
    /// for a ring, so it takes external teeth instead.
    pub(crate) fn suited_to(mut self, cylinder: CylinderSpec) -> Self {
        if self.kind == GearKind::Internal && cylinder.dimensions.inner_diameter() <= 0.0 {
            self.kind = GearKind::Spur;
        }
        self
    }

    /// Reads the settings back off a part that carries teeth.
    pub(crate) fn picked_from(self, spec: PartSpec) -> Option<Self> {
        match spec {
            PartSpec::Cylinder(cylinder) => cylinder.gear().map(|gear| Self {
                module_ticks: gear.module_ticks(),
                teeth: gear.teeth(),
                kind: gear.kind(),
                fitted: true,
                ..self
            }),
            PartSpec::Cuboid(cuboid) => cuboid.rack().map(|rack| Self {
                module_ticks: rack.module_ticks(),
                fitted: true,
                ..self
            }),
            _ => None,
        }
    }

    pub(crate) fn gear(self) -> Result<GearSpec, PlacementError> {
        GearSpec::new(self.module_ticks, self.teeth, self.kind).map_err(gear_error)
    }

    pub(crate) fn summary(self) -> String {
        let module = f32::from(self.module_ticks) * mechanic_core::POSITION_TICK_METERS;
        format!(
            "{}-tooth {} gear, module {:.2} cm, pitch diameter {:.1} cm",
            self.teeth,
            kind_label(self.kind),
            module * 100.0,
            module * f32::from(self.teeth) * 100.0
        )
    }

    /// What a rack cut says: its direction on the first block, how many
    /// blocks it runs across, and its tooth size.
    pub(crate) fn rack_summary(self, cuts: &[(RackTarget, CuboidSpec)]) -> String {
        let module = f32::from(self.module_ticks) * mechanic_core::POSITION_TICK_METERS;
        let along = cuts
            .first()
            .and_then(|(_, spec)| spec.rack())
            .map_or("the face", |rack| match rack.along() {
                mechanic_core::Axis::X => "X",
                mechanic_core::Axis::Y => "Y",
                mechanic_core::Axis::Z => "Z",
            });
        let blocks = match cuts.len() {
            0 | 1 => String::new(),
            count => format!(" across {count} blocks"),
        };
        format!(
            "rack along {along}{blocks}, module {:.2} cm, tooth pitch {:.2} cm",
            module * 100.0,
            module * core::f32::consts::PI * 100.0
        )
    }
}

fn gear_error(error: GearError) -> PlacementError {
    PlacementError::Graph(error.to_string())
}

fn graph_error(error: mechanic_core::GraphError) -> PlacementError {
    PlacementError::Graph(error.to_string())
}

// The same plain cylinder with other diameters.
fn resized(cylinder: CylinderSpec, outer: f32, inner: f32) -> Result<CylinderSpec, PlacementError> {
    let dimensions =
        CylinderDimensions::new(outer, inner.max(0.0), cylinder.dimensions.axial_length())
            .map_err(|error| PlacementError::Graph(error.to_string()))?;
    Ok(CylinderSpec::new(dimensions, cylinder.pose)
        .with_material(cylinder.material)
        .with_appearance(cylinder.appearance))
}

/// The target with the settings' teeth cut in. External teeth set the outer
/// diameter to their tips and a ring's teeth its bore, so the tooth count
/// decides the cylinder's size; the bore or rim keeps at least its minimum wall.
pub(crate) fn toothed(
    settings: GearSettings,
    target: &GearTarget,
) -> Result<CylinderSpec, PlacementError> {
    let cylinder = target.spec;
    if !cylinder.layers().is_empty() {
        return Err(gear_error(GearError::Layered));
    }
    if cylinder.spiral().is_some() {
        return Err(gear_error(GearError::Spiralled));
    }
    if cylinder.dimensions.sweep_angle_degrees() != mechanic_core::MAX_CYLINDER_SWEEP_DEGREES {
        return Err(gear_error(GearError::PartialSector));
    }
    let gear = settings.suited_to(cylinder).gear()?;
    let (outer, inner) = (
        cylinder.dimensions.outer_diameter(),
        cylinder.dimensions.inner_diameter(),
    );
    let (outer, inner) = if gear.is_internal() {
        let inner = gear.tip_diameter();
        (outer.max(inner + 2.0 * MIN_CYLINDER_DIAMETER_GAP), inner)
    } else {
        let outer = gear.tip_diameter();
        (outer, inner.min(outer - MIN_CYLINDER_DIAMETER_GAP))
    };
    resized(cylinder, outer, inner)?
        .with_gear(gear)
        .map_err(gear_error)
}

/// The target without its teeth, keeping the envelope they reached.
pub(crate) fn untoothed(target: &GearTarget) -> CylinderSpec {
    target.spec.without_gear()
}

/// Whether the target may become `spec`: a grown envelope must stay inside the
/// build space and clear of every other part.
pub(crate) fn validate_gear(
    graph: &ConstructionGraph,
    target: &GearTarget,
    spec: CylinderSpec,
    bounds: PlacementBounds,
) -> Result<(), PlacementError> {
    validate_spiral(graph, target, spec, bounds)
}

/// The axis rack teeth run along: the face's longer side, or its shorter one
/// when the settings say across.
pub(crate) fn rack_along(settings: GearSettings, target: &RackTarget) -> mechanic_core::Axis {
    let (first, second) = match target.face {
        FaceKind::PositiveX | FaceKind::NegativeX => {
            (mechanic_core::Axis::Y, mechanic_core::Axis::Z)
        }
        FaceKind::PositiveY | FaceKind::NegativeY => {
            (mechanic_core::Axis::X, mechanic_core::Axis::Z)
        }
        FaceKind::PositiveZ | FaceKind::NegativeZ => {
            (mechanic_core::Axis::X, mechanic_core::Axis::Y)
        }
    };
    let size = target.spec.size_meters();
    let longer_first = size[first.index()] >= size[second.index()];
    if longer_first == settings.across {
        second
    } else {
        first
    }
}

/// Where a pointer ray meets the plane of a target's face, in the world.
pub(crate) fn face_plane_hit(target: &RackTarget, origin: Vec3, direction: Vec3) -> Option<Vec3> {
    let rotation = target.spec.pose.rotation.quaternion();
    let outward = target.face.axis().unit() * target.face.sign();
    let normal = target.frame.vector(rotation * outward);
    let point = target.frame.point(
        target.spec.pose.translation()
            + rotation * outward * (target.spec.size_meters()[target.face.axis().index()] * 0.5),
    );
    let approach = direction.dot(normal);
    if approach.abs() < 1.0e-4 {
        return None;
    }
    let distance = (point - origin).dot(normal) / approach;
    (distance >= 0.0).then(|| origin + direction * distance)
}

/// A block face's plane in its construction frame: the frame axis it faces
/// along, how far out it lies, and the two frame axes across it.
#[derive(Clone, Copy, Debug)]
struct FacePlane {
    normal: Vec3,
    offset: f32,
    tangents: [Vec3; 2],
}

/// The frame axis a rotated block axis lies on.
fn cardinal(direction: Vec3) -> Vec3 {
    let axis = if direction.x.abs() >= direction.y.abs() && direction.x.abs() >= direction.z.abs() {
        Vec3::X
    } else if direction.y.abs() >= direction.z.abs() {
        Vec3::Y
    } else {
        Vec3::Z
    };
    axis * direction.dot(axis).signum()
}

/// The block axis a frame direction lies along.
fn block_axis(spec: CuboidSpec, direction: Vec3) -> mechanic_core::Axis {
    let local = spec.pose.rotation.quaternion().inverse() * direction;
    if local.x.abs() >= local.y.abs() && local.x.abs() >= local.z.abs() {
        mechanic_core::Axis::X
    } else if local.y.abs() >= local.z.abs() {
        mechanic_core::Axis::Y
    } else {
        mechanic_core::Axis::Z
    }
}

impl FacePlane {
    fn of(spec: CuboidSpec, face: FaceKind) -> Self {
        let normal = cardinal(spec.pose.rotation.quaternion() * (face.axis().unit() * face.sign()));
        let offset =
            spec.pose.translation().dot(normal) + spec.size_meters()[face.axis().index()] * 0.5;
        let mut tangents = [Vec3::X, Vec3::Y, Vec3::Z]
            .into_iter()
            .filter(|axis| axis.dot(normal).abs() < 0.5);
        Self {
            normal,
            offset,
            tangents: [
                tangents.next().expect("two axes cross a face"),
                tangents.next().expect("two axes cross a face"),
            ],
        }
    }

    /// A frame point's position across the plane.
    fn across(self, point: Vec3) -> Vec2 {
        Vec2::new(point.dot(self.tangents[0]), point.dot(self.tangents[1]))
    }

    /// The face of a block lying in this plane, if one does.
    fn face_of(self, spec: CuboidSpec) -> Option<FaceKind> {
        [
            FaceKind::PositiveX,
            FaceKind::NegativeX,
            FaceKind::PositiveY,
            FaceKind::NegativeY,
            FaceKind::PositiveZ,
            FaceKind::NegativeZ,
        ]
        .into_iter()
        .find(|&face| {
            let plane = Self::of(spec, face);
            plane.normal.dot(self.normal) > 0.5
                && (plane.offset - self.offset).abs() < POSITION_TICK_METERS * 0.5
        })
    }

    /// The corners of a block's outline across the plane, low then high.
    fn rect(self, spec: CuboidSpec) -> (Vec2, Vec2) {
        let rotation = spec.pose.rotation.quaternion();
        let half = spec.size_meters() * 0.5;
        let center = self.across(spec.pose.translation());
        let extent = [
            mechanic_core::Axis::X,
            mechanic_core::Axis::Y,
            mechanic_core::Axis::Z,
        ]
        .into_iter()
        .map(|axis| self.across(rotation * axis.unit()).abs() * half[axis.index()])
        .sum::<Vec2>();
        (center - extent, center + extent)
    }
}

/// The blocks a rack drag from `start` covers while the pointer stands over
/// `far`, a point in the world: every block welded to the start block whose
/// face lies in the start face's plane and reaches into the rectangle
/// between the start face and the pointer. The start block is always one.
pub(crate) fn rack_run(
    graph: &ConstructionGraph,
    start: &RackTarget,
    far: Vec3,
) -> Vec<RackTarget> {
    let plane = FacePlane::of(start.spec, start.face);
    let far = plane.across(start.frame.inverse().point(far));
    let (low, high) = plane.rect(start.spec);
    let (low, high) = (low.min(far), high.max(far));
    let frame = graph.part_frame_id(start.part);
    graph
        .rigid_group(start.part)
        .into_iter()
        .filter_map(|part| {
            if part == start.part {
                return Some(*start);
            }
            if graph.part_frame_id(part) != frame {
                return None;
            }
            let Some(PartSpec::Cuboid(spec)) = graph.part(part).copied() else {
                return None;
            };
            let face = plane.face_of(spec)?;
            let (a, b) = plane.rect(spec);
            // A block only touching the rectangle's edge stays out of it.
            let inset = POSITION_TICK_METERS * 0.5;
            (a.x < high.x - inset
                && b.x > low.x + inset
                && a.y < high.y - inset
                && b.y > low.y + inset)
                .then_some(RackTarget {
                    part,
                    spec,
                    frame: start.frame,
                    face,
                })
        })
        .collect()
}

/// Rack teeth for a run of blocks, all on one line: along the run's longer
/// side, or its shorter one with Rotate, and along the first block's own
/// longer side when they are the same.
pub(crate) fn rack_cut(
    settings: GearSettings,
    run: &[RackTarget],
) -> Result<Vec<(RackTarget, CuboidSpec)>, PlacementError> {
    let start = run
        .first()
        .ok_or_else(|| PlacementError::Graph("point at a block's face".to_owned()))?;
    let plane = FacePlane::of(start.spec, start.face);
    let (low, high) = run.iter().map(|target| plane.rect(target.spec)).fold(
        (Vec2::INFINITY, Vec2::NEG_INFINITY),
        |(low, high), (a, b)| (low.min(a), high.max(b)),
    );
    let extent = high - low;
    let along = if run.len() == 1 || (extent.x - extent.y).abs() < POSITION_TICK_METERS {
        start.spec.pose.rotation.quaternion() * rack_along(settings, start).unit()
    } else if (extent.x > extent.y) == settings.across {
        plane.tangents[1]
    } else {
        plane.tangents[0]
    };
    run.iter()
        .map(|target| {
            let rack = RackSpec::new(
                settings.module_ticks,
                target.face,
                block_axis(target.spec, along),
            )
            .map_err(gear_error)?;
            let spec = target
                .spec
                .without_rack()
                .with_rack(rack)
                .map_err(gear_error)?;
            Ok((*target, spec))
        })
        .collect()
}

/// The target without its rack teeth.
pub(crate) fn unracked(target: &RackTarget) -> CuboidSpec {
    target.spec.without_rack()
}

/// Whether a part can take part in a mesh by itself: teeth, a rack, or a thread.
pub(crate) fn is_meshable(spec: PartSpec) -> bool {
    match spec {
        PartSpec::Cylinder(cylinder) => cylinder.gear().is_some() || cylinder.spiral().is_some(),
        PartSpec::Cuboid(cuboid) => cuboid.rack().is_some(),
        _ => false,
    }
}

/// Every meshable part `part` could mesh with as built and does not yet.
pub(crate) fn meshes_admitted(graph: &ConstructionGraph, part: PartId) -> Vec<(PartId, GearMesh)> {
    let mut admitted: Vec<(PartId, GearMesh)> = Vec::new();
    let meshed_racks = |gear: PartId| {
        graph
            .part_gear_links(gear)
            .filter_map(move |(_, link)| link.other(gear))
    };
    for (other, spec) in graph.parts() {
        if other == part || !is_meshable(*spec) {
            continue;
        }
        let Ok(mesh) = graph.validate_gear_link(GearLinkSpec {
            first: part,
            second: other,
        }) else {
            continue;
        };
        // A rack running over several blocks is one rack: a pinion over the
        // joint between two of them meshes with the rack once.
        let already = if is_rack(graph, part) {
            meshed_racks(other).any(|rack| same_rack(graph, part, rack))
        } else {
            meshed_racks(part)
                .chain(admitted.iter().map(|(rack, _)| *rack))
                .any(|rack| same_rack(graph, other, rack))
        };
        if !already {
            admitted.push((other, mesh));
        }
    }
    admitted
}

fn is_rack(graph: &ConstructionGraph, part: PartId) -> bool {
    matches!(graph.part(part), Some(PartSpec::Cuboid(cuboid)) if cuboid.rack().is_some())
}

/// Whether two racks are one: cut into blocks of one rigid body, in one
/// pitch plane.
fn same_rack(graph: &ConstructionGraph, a: PartId, b: PartId) -> bool {
    let plane = |part: PartId| {
        let spec = graph.part(part)?;
        let frame = graph.part_frame(part)?;
        match GearEnd::of_part(*spec, frame) {
            GearEnd::Rack {
                pitch_center,
                normal,
                ..
            } => Some((pitch_center, normal)),
            _ => None,
        }
    };
    let (Some((center_a, normal_a)), Some((center_b, normal_b))) = (plane(a), plane(b)) else {
        return false;
    };
    normal_a.dot(normal_b) > GEAR_PARALLEL_COSINE
        && (center_a - center_b).dot(normal_a).abs() < POSITION_TICK_METERS
        && graph.rigid_group(a).contains(&b)
}

/// Where a gear stands: the axis through its centre and half its face width,
/// in world space at the rest pose.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct GearSeat {
    pub(crate) center: Vec3,
    pub(crate) axis: Vec3,
    pub(crate) half_length: f32,
}

impl GearSeat {
    pub(crate) fn of_cylinder(spec: CylinderSpec, frame: ConstructionFrame) -> Self {
        let rotation = spec.pose.rotation.quaternion();
        Self {
            center: frame.point(spec.pose.translation()),
            axis: frame.vector(rotation * Vec3::Y),
            half_length: spec.dimensions.axial_length() * 0.5,
        }
    }

    fn end(self, gear: GearSpec) -> GearEnd {
        GearEnd::Gear {
            spec: gear,
            center: self.center,
            axis: self.axis,
            half_length: self.half_length,
        }
    }

    /// The module and pitch radius that put a gear of `kind` here tangent to
    /// `partner`, for the pairings that can be fitted: a spur or ring gear on
    /// a parallel axis, a rack the axis runs across, a worm the axis crosses.
    fn fit(self, kind: GearKind, partner: GearEnd) -> Option<(u8, f32)> {
        let internal = kind == GearKind::Internal;
        match partner {
            GearEnd::Gear {
                spec, center, axis, ..
            } => {
                if matches!(spec.kind(), GearKind::Bevel { .. })
                    || self.axis.dot(axis).abs() < GEAR_PARALLEL_COSINE
                {
                    return None;
                }
                let offset = center - self.center;
                let distance = (offset - self.axis * offset.dot(self.axis)).length();
                let radius = match (internal, spec.is_internal()) {
                    (false, false) => distance - spec.pitch_radius(),
                    (false, true) => spec.pitch_radius() - distance,
                    (true, false) => distance + spec.pitch_radius(),
                    (true, true) => return None,
                };
                Some((spec.module_ticks(), radius))
            }
            GearEnd::Rack {
                spec,
                pitch_center,
                normal,
                across,
                ..
            } => {
                if internal || self.axis.dot(across).abs() < GEAR_PARALLEL_COSINE {
                    return None;
                }
                Some((
                    spec.module_ticks(),
                    (self.center - pitch_center).dot(normal),
                ))
            }
            GearEnd::Screw {
                center,
                axis,
                pitch_radius,
                pitch,
                ..
            } => {
                if internal || self.axis.dot(axis).abs() > GEAR_PERPENDICULAR_COSINE {
                    return None;
                }
                let cross = self.axis.cross(axis).normalize_or_zero();
                let distance = (center - self.center).dot(cross).abs();
                Some((module_ticks_for_pitch(pitch), distance - pitch_radius))
            }
            GearEnd::Plain { .. } => None,
        }
    }
}

/// The module, in ticks, whose teeth are `pitch` apart along the pitch circle.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "rounded and clamped into the module range"
)]
fn module_ticks_for_pitch(pitch: f32) -> u8 {
    let ticks = pitch / (core::f32::consts::PI * POSITION_TICK_METERS);
    (ticks.round().max(0.0) as u8).clamp(MIN_GEAR_MODULE_TICKS, MAX_GEAR_MODULE_TICKS)
}

/// A toothed part a gear at a seat can reach, and the settings that reach it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Reach {
    pub(crate) part: PartId,
    /// The partner as the mesh sees it.
    pub(crate) partner: GearEnd,
    /// The settings with the module and tooth count that put the pitch
    /// surfaces tangent, verified to mesh.
    pub(crate) fitted: GearSettings,
    /// The pitch radius that puts them exactly tangent.
    pub(crate) radius: f32,
}

impl Reach {
    /// How far the pitch surface of a gear with `settings` misses the partner,
    /// in metres: positive when the gear is too small.
    fn miss(self, settings: GearSettings) -> f32 {
        let module = f32::from(settings.module_ticks) * POSITION_TICK_METERS;
        self.radius - module * f32::from(settings.teeth) * 0.5
    }
}

/// One partner chosen from those a gear reaches, and its place among them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ReachChoice {
    pub(crate) reach: Reach,
    /// Which of the admitted partners this is, from 0.
    pub(crate) index: usize,
    /// How many partners were admitted.
    pub(crate) count: usize,
}

/// The `choice`th of the reaches that `admits`, round and round: the first
/// again after the last. Nothing when none is admitted.
pub(crate) fn choose_reach(
    reaches: Vec<Reach>,
    choice: usize,
    admits: impl FnMut(&Reach) -> bool,
) -> Option<ReachChoice> {
    let admitted = reaches.into_iter().filter(admits).collect::<Vec<_>>();
    let count = admitted.len();
    let index = choice.checked_rem(count)?;
    Some(ReachChoice {
        reach: admitted[index],
        index,
        count,
    })
}

/// Every meshable part a gear of these settings at `seat` can reach, other
/// than `except`, each with the tooth count and module that reach it: the
/// one the smallest gear reaches first, or with a hand-set count the one it
/// comes nearest to meshing. Empty when no part admits a mesh on this axis,
/// or for a bevel gear.
pub(crate) fn reaches(
    graph: &ConstructionGraph,
    settings: GearSettings,
    seat: GearSeat,
    except: Option<PartId>,
) -> Vec<Reach> {
    if matches!(settings.kind, GearKind::Bevel { .. }) {
        return Vec::new();
    }
    let mut found = Vec::new();
    for (other, spec) in graph.parts() {
        if Some(other) == except || !is_meshable(*spec) {
            continue;
        }
        let Some(frame) = graph.part_frame(other) else {
            continue;
        };
        let partner = GearEnd::of_part(*spec, frame);
        let Some((module_ticks, radius)) = seat.fit(settings.kind, partner) else {
            continue;
        };
        let module = f32::from(module_ticks) * POSITION_TICK_METERS;
        let tip = if settings.kind == GearKind::Internal {
            2.0 * radius - 2.0 * module
        } else {
            2.0 * radius + 2.0 * module
        };
        let fitted = GearSettings {
            module_ticks,
            teeth: GearSpec::teeth_for_tip_diameter(module_ticks, settings.kind, tip),
            ..settings
        };
        let Ok(gear) = fitted.gear() else {
            continue;
        };
        if mesh(seat.end(gear), partner).is_err() {
            continue;
        }
        found.push(Reach {
            part: other,
            partner,
            fitted,
            radius,
        });
    }
    let rank = |reach: &Reach| {
        if settings.fitted {
            reach.miss(settings).abs()
        } else {
            reach.radius
        }
    };
    found.sort_by(|a, b| rank(a).total_cmp(&rank(b)));
    found
}

/// What the status line says about a reach for a gear of `settings` at
/// `seat`: the mesh they make, or what keeps them from it.
pub(crate) fn reach_line(
    graph: &ConstructionGraph,
    settings: GearSettings,
    seat: GearSeat,
    reach: Reach,
) -> String {
    let meshed = settings
        .gear()
        .ok()
        .and_then(|gear| mesh(seat.end(gear), reach.partner).ok());
    if let Some(meshed) = meshed {
        return format!("meshes {}", partner_label(graph, reach.part, true, meshed));
    }
    let name = partner_name(graph, reach.part);
    if settings.module_ticks != reach.fitted.module_ticks {
        return format!(
            "{name} needs module {:.2} cm",
            f32::from(reach.fitted.module_ticks) * POSITION_TICK_METERS * 100.0
        );
    }
    let miss = reach.miss(settings);
    format!(
        "pitch circle {:.0} mm {} {name}",
        miss.abs() * 1000.0,
        if miss > 0.0 { "short of" } else { "past" }
    )
}

/// What a part is called as a mesh partner.
fn partner_name(graph: &ConstructionGraph, part: PartId) -> String {
    match graph.part(part).copied() {
        Some(PartSpec::Cylinder(cylinder)) => match (cylinder.gear(), cylinder.spiral()) {
            (Some(gear), _) => format!("{}-tooth {}", gear.teeth(), kind_label(gear.kind())),
            (None, Some(_)) => "worm".to_owned(),
            (None, None) => "part".to_owned(),
        },
        Some(PartSpec::Cuboid(cuboid)) if cuboid.rack().is_some() => "rack".to_owned(),
        _ => "nut".to_owned(),
    }
}

/// One line naming a mesh from `part`'s side of `link`.
pub(crate) fn mesh_label(
    graph: &ConstructionGraph,
    link: GearLinkSpec,
    part: PartId,
    mesh: GearMesh,
) -> String {
    let other = link.other(part).unwrap_or(link.second);
    partner_label(graph, other, link.first == part, mesh)
}

/// One line naming `other` and the ratio seen from the side that was `first`
/// when `mesh` was resolved.
fn partner_label(graph: &ConstructionGraph, other: PartId, first: bool, mesh: GearMesh) -> String {
    let partner = partner_name(graph, other);
    let ratio = match mesh.kind {
        GearLinkKind::Gears | GearLinkKind::Worm => mesh.ratio().map(|ratio| {
            // The mesh's ratio is its first side's turns per turn of its
            // second; this part is the first side unless the sides were swapped.
            let mine = if first == mesh.swapped {
                1.0 / ratio
            } else {
                ratio
            };
            if mine >= 1.0 {
                format!(" ({mine:.2}:1)")
            } else {
                format!(" (1:{:.2})", 1.0 / mine)
            }
        }),
        GearLinkKind::Rack | GearLinkKind::Screw => None,
    };
    format!("{partner}{}", ratio.unwrap_or_default())
}

/// What `part` meshes with, for the status line.
pub(crate) fn mesh_summary(graph: &ConstructionGraph, part: PartId) -> String {
    let partners = graph
        .part_gear_links(part)
        .filter_map(|(_, link)| {
            let mesh = graph.gear_mesh(*link).ok()?;
            Some(mesh_label(graph, *link, part, mesh))
        })
        .collect::<Vec<_>>();
    if partners.is_empty() {
        "no mesh".to_owned()
    } else {
        format!("meshes {}", partners.join(", "))
    }
}

/// The graph with every mesh `part` admits added, and who it met.
fn auto_meshed(
    mut staged: mechanic_core::ConstructionGraphEdit,
    part: PartId,
) -> Result<(ConstructionGraph, Vec<(PartId, GearMesh)>), PlacementError> {
    let partners = if graph_is_meshable(&staged, part) {
        meshes_admitted(&staged, part)
    } else {
        Vec::new()
    };
    for (other, _) in &partners {
        staged
            .apply(BuildCommand::AddGearLink(GearLinkSpec {
                first: part,
                second: *other,
            }))
            .map_err(graph_error)?;
    }
    Ok((staged.finish(), partners))
}

fn graph_is_meshable(graph: &ConstructionGraph, part: PartId) -> bool {
    graph.part(part).is_some_and(|spec| is_meshable(*spec))
}

/// Meshes a part that just arrived with everything its teeth reach.
pub(crate) fn stage_meshes_admitted(
    graph: &ConstructionGraph,
    part: PartId,
) -> Result<(ConstructionGraph, Vec<(PartId, GearMesh)>), PlacementError> {
    auto_meshed(graph.begin_edit(), part)
}

/// Replaces the target's teeth in one edit, keeping its connections, and
/// meshes it with everything its new teeth reach.
pub(crate) fn stage_gear(
    graph: &ConstructionGraph,
    target: &GearTarget,
    spec: CylinderSpec,
    bounds: PlacementBounds,
) -> Result<(ConstructionGraph, Vec<(PartId, GearMesh)>), PlacementError> {
    validate_gear(graph, target, spec, bounds)?;
    let mut staged = graph.begin_edit();
    staged
        .apply(BuildCommand::SetGear {
            part: target.part,
            spec,
        })
        .map_err(graph_error)?;
    auto_meshed(staged, target.part)
}

/// Replaces the target's rack teeth in one edit and meshes it with every
/// pinion over its face.
pub(crate) fn stage_rack(
    graph: &ConstructionGraph,
    target: &RackTarget,
    spec: CuboidSpec,
) -> Result<(ConstructionGraph, Vec<(PartId, GearMesh)>), PlacementError> {
    stage_rack_run(graph, &[(*target, spec)])
}

/// Cuts a run of racks in one edit and meshes each with every pinion over
/// it, once per pinion.
pub(crate) fn stage_rack_run(
    graph: &ConstructionGraph,
    cuts: &[(RackTarget, CuboidSpec)],
) -> Result<(ConstructionGraph, Vec<(PartId, GearMesh)>), PlacementError> {
    let mut staged = graph.begin_edit();
    for (target, spec) in cuts {
        staged
            .apply(BuildCommand::SetRack {
                part: target.part,
                spec: *spec,
            })
            .map_err(graph_error)?;
    }
    let mut partners = Vec::new();
    for (target, _) in cuts {
        if !graph_is_meshable(&staged, target.part) {
            continue;
        }
        let admitted = meshes_admitted(&staged, target.part);
        for (other, _) in &admitted {
            staged
                .apply(BuildCommand::AddGearLink(GearLinkSpec {
                    first: target.part,
                    second: *other,
                }))
                .map_err(graph_error)?;
        }
        partners.extend(admitted);
    }
    Ok((staged.finish(), partners))
}

/// Meshes two parts the player dragged between.
pub(crate) fn stage_mesh(
    graph: &ConstructionGraph,
    first: PartId,
    second: PartId,
) -> Result<(ConstructionGraph, GearMesh), PlacementError> {
    let mesh = graph
        .validate_gear_link(GearLinkSpec { first, second })
        .map_err(graph_error)?;
    let mut staged = graph.begin_edit();
    staged
        .apply(BuildCommand::AddGearLink(GearLinkSpec { first, second }))
        .map_err(graph_error)?;
    Ok((staged.finish(), mesh))
}

/// Breaks every mesh `part` is in; how many there were.
pub(crate) fn stage_unmesh(
    graph: &ConstructionGraph,
    part: PartId,
) -> Result<(ConstructionGraph, usize), PlacementError> {
    let links = graph
        .part_gear_links(part)
        .map(|(id, _)| id)
        .collect::<Vec<_>>();
    let mut staged = graph.begin_edit();
    for link in &links {
        staged
            .apply(BuildCommand::RemoveGearLink(*link))
            .map_err(graph_error)?;
    }
    Ok((staged.finish(), links.len()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::math::IVec3;
    use mechanic_core::{BuildOutcome, BuildPose, GridRotation};

    // A plain 30 cm cylinder standing in an empty build space.
    fn shaft(outer: f32, inner: f32) -> (ConstructionGraph, GearTarget) {
        let mut graph = ConstructionGraph::new();
        let spec = CylinderSpec::new(
            CylinderDimensions::new(outer, inner, 0.25).unwrap(),
            BuildPose::from_position_ticks(IVec3::Y * 800, GridRotation::default()),
        );
        let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::SpawnCylinder(spec)).unwrap()
        else {
            unreachable!()
        };
        let frame = graph.part_frame(part).unwrap();
        (
            graph,
            GearTarget {
                part,
                spec,
                frame,
                near_end: SpiralEnd::NegativeY,
            },
        )
    }

    #[test]
    fn settings_size_their_teeth_to_the_first_cylinder_and_the_teeth_size_the_cylinder() {
        let (graph, target) = shaft(0.3, 0.0);
        let settings = GearSettings::default().fitted_to(target.spec);
        assert_eq!(settings.teeth, 28, "30 cm across at module 1 cm");
        let spec = toothed(settings, &target).unwrap();
        assert!((spec.dimensions.outer_diameter() - 0.30).abs() < 1.0e-6);
        assert_eq!(spec.gear().unwrap().teeth(), 28);
        let bigger = settings.adjusted(GearDimension::Teeth, 1, true);
        let spec = toothed(bigger, &target).unwrap();
        assert!((spec.dimensions.outer_diameter() - 0.36).abs() < 1.0e-6);
        assert_eq!(
            validate_gear(&graph, &target, spec, PlacementBounds::default()),
            Ok(())
        );
        // A ring on a solid cylinder falls back to external teeth.
        let ring = settings.cycled();
        assert_eq!(ring.kind, GearKind::Internal);
        assert_eq!(
            toothed(ring, &target).unwrap().gear().unwrap().kind(),
            GearKind::Spur
        );
        let (_, hollow) = shaft(0.4, 0.3);
        let ring = GearSettings::default().cycled().fitted_to(hollow.spec);
        assert_eq!(ring.teeth, 32, "a 30 cm bore at module 1 cm");
        let spec = toothed(ring, &hollow).unwrap();
        assert_eq!(spec.gear().unwrap().kind(), GearKind::Internal);
        assert!((spec.dimensions.inner_diameter() - 0.30).abs() < 1.0e-6);
    }

    #[test]
    fn cutting_teeth_meshes_with_a_tangent_partner_and_reports_it() {
        let (mut graph, target) = shaft(0.26, 0.0);
        let settings = GearSettings::default();
        let (staged, partners) = stage_gear(
            &graph,
            &target,
            toothed(settings, &target).unwrap(),
            PlacementBounds::default(),
        )
        .unwrap();
        assert!(partners.is_empty());
        graph = staged;
        // A 36-tooth wheel whose pitch circle touches the pinion's.
        let spec = CylinderSpec::new(
            CylinderDimensions::new(0.38, 0.0, 0.25).unwrap(),
            BuildPose::from_position_ticks(IVec3::new(120, 800, 0), GridRotation::default()),
        );
        let BuildOutcome::Spawned(wheel) = graph.apply(BuildCommand::SpawnCylinder(spec)).unwrap()
        else {
            unreachable!()
        };
        let wheel_target = GearTarget {
            part: wheel,
            spec,
            frame: graph.part_frame(wheel).unwrap(),
            near_end: SpiralEnd::NegativeY,
        };
        let wheel_settings = settings.fitted_to(spec);
        assert_eq!(wheel_settings.teeth, 36);
        let (staged, partners) = stage_gear(
            &graph,
            &wheel_target,
            toothed(wheel_settings, &wheel_target).unwrap(),
            PlacementBounds::default(),
        )
        .unwrap();
        assert_eq!(partners.len(), 1);
        assert_eq!(partners[0].0, target.part);
        assert_eq!(staged.gear_links().count(), 1);
        assert_eq!(
            mesh_summary(&staged, wheel),
            "meshes 24-tooth spur (1:1.50)"
        );
        assert_eq!(
            mesh_summary(&staged, target.part),
            "meshes 36-tooth spur (1.50:1)"
        );
        let (unmeshed, count) = stage_unmesh(&staged, wheel).unwrap();
        assert_eq!(count, 1);
        assert_eq!(unmeshed.gear_links().count(), 0);
        let (remeshed, _) = stage_mesh(&unmeshed, wheel, target.part).unwrap();
        assert_eq!(remeshed.gear_links().count(), 1);
    }

    // A toothed cylinder `outer` across the tips, centred `x` ticks along X.
    fn gear_at(outer: f32, teeth: u16, x: i32) -> PartSpec {
        let spec = CylinderSpec::new(
            CylinderDimensions::new(outer, 0.0, 0.25).unwrap(),
            BuildPose::from_position_ticks(IVec3::new(x, 800, 0), GridRotation::default()),
        );
        PartSpec::Cylinder(
            spec.with_gear(GearSpec::new(4, teeth, GearKind::Spur).unwrap())
                .unwrap(),
        )
    }

    #[test]
    fn a_gear_reaches_into_a_partners_tooth_band_but_not_into_a_plain_part() {
        let pinion = gear_at(0.26, 24, 0);
        // Pitch circles tangent at 30 cm; the tips overlap by two modules.
        let wheel = gear_at(0.38, 36, 120);
        assert!(!super::super::bounds::parts_overlap(wheel, pinion));
        assert!(!super::super::bounds::parts_overlap(pinion, wheel));
        // The same envelope without teeth is a part the tips must clear.
        let PartSpec::Cylinder(plain) = wheel else {
            unreachable!()
        };
        let plain = PartSpec::Cylinder(plain.without_gear());
        assert!(super::super::bounds::parts_overlap(plain, pinion));
        // Pitch circles 2.5 mm into each other still mesh, and still clear.
        let close = gear_at(0.38, 36, 119);
        assert!(!super::super::bounds::parts_overlap(close, pinion));
        // Root circles touching is where the exemption ends.
        let jammed = gear_at(0.38, 36, 108);
        assert!(super::super::bounds::parts_overlap(jammed, pinion));
    }

    #[test]
    fn settings_reach_the_nearest_toothed_neighbour_and_say_how_they_miss_it() {
        let mut graph = ConstructionGraph::new();
        let BuildOutcome::Spawned(pinion) = graph
            .apply(BuildCommand::SpawnCylinder(
                gear_at(0.26, 24, 0).as_cylinder().unwrap(),
            ))
            .unwrap()
        else {
            unreachable!()
        };
        // A seat 31 cm from the pinion's axis: 62 modules of centre distance
        // leave 38 teeth for the gear that sits there.
        let seat = GearSeat {
            center: Vec3::new(0.31, 2.0, 0.0),
            axis: Vec3::Y,
            half_length: 0.125,
        };
        let settings = GearSettings::default();
        let reached = reaches(&graph, settings, seat, None);
        assert_eq!(reached.len(), 1, "the pinion is in reach");
        let reached = reached[0];
        assert_eq!(reached.part, pinion);
        assert_eq!(reached.fitted.teeth, 38);
        assert_eq!(reached.fitted.module_ticks, 4);
        assert!(!reached.fitted.fitted, "reaching does not pin the count");
        assert_eq!(
            reach_line(&graph, reached.fitted, seat, reached),
            "meshes 24-tooth spur (1:1.58)"
        );
        // The default 24 teeth stop 7 cm short of the pinion.
        assert_eq!(
            reach_line(&graph, settings, seat, reached),
            "pitch circle 70 mm short of 24-tooth spur"
        );
        // A coarser module cannot mesh whatever its count.
        let coarse = settings.adjusted(GearDimension::Module, 2, false);
        assert_eq!(
            reach_line(&graph, coarse, seat, reached),
            "24-tooth spur needs module 1.00 cm"
        );
        // Nothing to reach across the axis.
        let crossed = GearSeat {
            axis: Vec3::X,
            ..seat
        };
        assert_eq!(reaches(&graph, settings, crossed, None), Vec::new());
    }

    #[test]
    fn reaches_put_the_smallest_fitted_gear_first_and_a_choice_wraps_round() {
        let mut graph = ConstructionGraph::new();
        let BuildOutcome::Spawned(near) = graph
            .apply(BuildCommand::SpawnCylinder(
                gear_at(0.26, 24, 0).as_cylinder().unwrap(),
            ))
            .unwrap()
        else {
            unreachable!()
        };
        // A 72-tooth wheel 70 cm the other way: a seat 31 cm from the pinion
        // fits it with a 68-tooth gear, and the pinion with 38.
        let BuildOutcome::Spawned(far) = graph
            .apply(BuildCommand::SpawnCylinder(
                gear_at(0.74, 72, -156).as_cylinder().unwrap(),
            ))
            .unwrap()
        else {
            unreachable!()
        };
        let seat = GearSeat {
            center: Vec3::new(0.31, 2.0, 0.0),
            axis: Vec3::Y,
            half_length: 0.125,
        };
        // Even with the count left over from a big gear, the small fit leads.
        let settings = GearSettings {
            teeth: 68,
            ..GearSettings::default()
        };
        let found = reaches(&graph, settings, seat, None);
        assert_eq!(
            found
                .iter()
                .map(|reach| (reach.part, reach.fitted.teeth))
                .collect::<Vec<_>>(),
            vec![(near, 38), (far, 68)]
        );
        // A hand-set count ranks by what it misses instead.
        let by_hand = GearSettings {
            fitted: true,
            ..settings
        };
        assert_eq!(
            reaches(&graph, by_hand, seat, None)
                .iter()
                .map(|reach| reach.part)
                .collect::<Vec<_>>(),
            vec![far, near]
        );
        let choice = |choice: usize| choose_reach(found.clone(), choice, |_| true).unwrap();
        assert_eq!(
            (choice(0).reach.part, choice(0).index, choice(0).count),
            (near, 0, 2)
        );
        assert_eq!(choice(1).reach.part, far);
        assert_eq!(choice(2).reach.part, near, "the choice wraps round");
        let only_near = choose_reach(found.clone(), 1, |reach| reach.part == near).unwrap();
        assert_eq!((only_near.reach.part, only_near.count), (near, 1));
        assert_eq!(choose_reach(found, 0, |_| false), None);
    }
}
