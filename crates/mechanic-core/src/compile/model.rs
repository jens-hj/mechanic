//! The compiled creation: bodies, colliders, bearings, drives, and the errors compilation reports.

use super::drives::{
    CoordinateActuation, coordinate_drive, resolve_coordinate_actuation, resolve_coordinate_drives,
};
use crate::{
    BearingId, ConstructionGraph, DriveLimits, DriveTarget, EngineKind, MaterialProperties, PartId,
};
use bevy_math::{Mat3, Quat, Vec3, Vec4};
use std::collections::BTreeMap;
use std::ops::Range;
use thiserror::Error;

/// Number of cuboid colliders used for each cylinder.
pub const CYLINDER_COLLIDER_COUNT: usize = 16;

/// Number of cuboid colliders used for each pipe bend: sixteen annular
/// sectors over each of twelve centreline slices.
pub const PIPE_BEND_COLLIDER_COUNT: usize = 16 * 12;

/// Largest number of collider rows one compiled creation may produce.
///
/// Shaping multiplies collider rows on the cells it touches, so a creation that
/// would swamp the solver must fail loudly at compile time rather than quietly
/// sinking the tick rate.
pub const MAX_COMPILED_COLLIDERS: usize = 131_072;

/// Aggregate mass properties expressed in the compiled root frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MassProperties {
    /// Total mass in kilograms, retained for diagnostics even for static bodies.
    pub mass: f32,
    /// Inverse mass used by the solver; zero for static compounds.
    pub inverse_mass: f32,
    /// Centre of mass in build-world coordinates.
    pub center_of_mass: Vec3,
    /// Inertia tensor around the centre of mass.
    pub inertia: Mat3,
    /// Inverse inertia used by the solver; zero for static compounds.
    pub inverse_inertia: Mat3,
}

/// One compound body produced by collapsing an explicit weld group.
#[derive(Clone, Debug, PartialEq)]
pub struct CompiledCompound {
    /// Canonically ordered source parts.
    pub source_parts: Vec<PartId>,
    /// Initial root position. Dynamic roots are located at their centre of mass.
    pub root_translation: Vec3,
    /// Initial root rotation.
    pub root_rotation: Quat,
    /// Whether a weld connects this group to the static ground.
    pub is_static: bool,
    /// Aggregate physical properties.
    pub mass_properties: MassProperties,
    /// Contiguous collider rows owned by this body.
    pub collider_range: Range<u32>,
}

/// Cuboid collider expressed relative to its compound root.
#[derive(Clone, Debug, PartialEq)]
pub struct LocalCollider {
    /// Source editable part.
    pub source_part: PartId,
    /// Owning compound row.
    pub compound_index: u32,
    /// Collider centroid in compound-local coordinates.
    pub local_center: Vec3,
    /// Source part's contact response.
    pub material_properties: MaterialProperties,
    /// Geometry this collider presents to the solver.
    pub shape: ColliderShape,
}

/// Geometry backing one compiled collider row.
#[derive(Clone, Debug, PartialEq)]
pub enum ColliderShape {
    /// A box. Unshaped parts and cylinder segments stay boxes so the solver's
    /// fast path is untouched by shaping existing anywhere else.
    Cuboid {
        /// Orientation relative to the compound root.
        local_rotation: Quat,
        /// Half-extents in metres.
        half_extents: Vec3,
    },
    /// A shaped cell, or a convex part of one.
    Convex(CompiledConvex),
}

impl ColliderShape {
    /// Whether this is a box.
    pub const fn is_cuboid(&self) -> bool {
        matches!(self, Self::Cuboid { .. })
    }
}

/// A convex polytope collider in compound-local coordinates.
///
/// Face normals and edge directions arrive already deduplicated, so a sheared
/// box presents the same three face axes and three edge axes an ordinary box
/// does and costs the separating-axis test exactly as much.
#[derive(Clone, Debug, PartialEq)]
pub struct CompiledConvex {
    /// Distinct vertices, relative to the compound centre of mass.
    pub vertices: Vec<Vec3>,
    /// Distinct face planes: `xyz` outward normal, `w` plane offset.
    pub face_planes: Vec<Vec4>,
    /// Distinct edge directions.
    pub edge_directions: Vec<Vec3>,
}

/// Bearing row connecting two distinct compiled compounds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CompiledBearing {
    /// Explicit motion kind and physical travel independent of drive settings.
    pub kind: crate::BearingKind,
    /// Source editable bearing.
    pub source_bearing: BearingId,
    /// Source compound row.
    pub compound_a: u32,
    /// Target compound row.
    pub compound_b: u32,
    /// Anchor relative to source root.
    pub local_anchor_a: Vec3,
    /// Anchor relative to target root.
    pub local_anchor_b: Vec3,
    /// Axis in source root coordinates.
    pub local_axis_a: Vec3,
    /// Axis in target root coordinates.
    pub local_axis_b: Vec3,
    /// Independent mechanism coordinate for a tree edge; `None` for closure edges.
    pub coordinate_index: Option<u32>,
}

/// Canonical parent metadata for one body in the reduced-coordinate forest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MechanismBodyTopology {
    /// Parent body row, or this body for a root.
    pub parent_body: u32,
    /// Tree bearing connecting this body to its parent.
    pub tree_bearing: Option<BearingId>,
    /// Zero when the bearing is traversed from A to B, one for B to A.
    pub bearing_direction: u32,
    /// Canonical connected-component row.
    pub component_index: u32,
    /// Distance from the canonical tree root.
    pub depth: u32,
    /// Stable root-before-children traversal position.
    pub preorder_index: u32,
    /// Stable children-before-root traversal position.
    pub postorder_index: u32,
    /// Whether this body owns a fixed or floating root coordinate.
    pub is_root: bool,
}

/// Canonical forest and hard loop-closure partition.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LoopTopology {
    /// Bearings that introduce independent one-dimensional coordinates.
    pub tree_bearings: Vec<BearingId>,
    /// Bearings represented by hard dependent closure equations.
    pub closure_bearings: Vec<BearingId>,
    /// Connected components, each containing sorted compound rows.
    pub mechanism_components: Vec<Vec<u32>>,
    /// Canonical roots for each connected component. Components with multiple
    /// ground anchors have one fixed root for every anchored tree.
    pub component_roots: Vec<Vec<u32>>,
    /// Canonical parent and traversal metadata indexed by compound row.
    pub body_parents: Vec<MechanismBodyTopology>,
    /// Deterministic leaf-to-root contraction rounds over non-root bodies.
    pub contraction_rounds: Vec<Vec<u32>>,
    /// Child-subtree rotational inertia about each tree bearing's own axis, in
    /// kg·m², indexed by coordinate. Infinite when the subtree is grounded.
    pub coordinate_axis_inertia: Vec<f32>,
    /// Every graph bearing, including rows collapsed into another as the same
    /// physical joint, to the coordinate it moves. Bearings that close a loop
    /// are absent.
    pub bearing_coordinates: BTreeMap<BearingId, u32>,
}

/// Complete, immutable upload image for the GPU runtime. The default is an
/// empty runtime scene, suitable for appending world-owned free bodies.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CompiledCreation {
    /// Backend-independent symbolic dynamics schedules and body-frame inertias.
    pub dynamics: crate::CompiledDynamics,
    /// Compound bodies.
    pub compounds: Vec<CompiledCompound>,
    /// Cuboid collider rows.
    pub colliders: Vec<LocalCollider>,
    /// Passive bearing rows.
    pub bearings: Vec<CompiledBearing>,
    /// Canonical mechanism topology.
    pub loop_topology: LoopTopology,
    /// Sorted compound pairs excluded from collision generation.
    pub collision_suppression: Vec<[u32; 2]>,
    /// Canonical source-part to compound-row lookup.
    pub part_to_compound: Vec<(PartId, u32)>,
    /// Resolved drive rows, one per tree bearing, in coordinate-index order.
    pub coordinate_drives: Vec<CoordinateDrive>,
    /// Analytic description of every solid full cylinder, alongside the tangent
    /// boxes that represent it in `colliders`. A solver that can take a cylinder's
    /// contact exactly uses this instead of pattern-matching the box run.
    pub cylinders: Vec<CompiledCylinder>,
}

/// One compiled solid full cylinder, in compound-local coordinates.
///
/// The sixteen tangent boxes in `colliders` circumscribe this cylinder: each box
/// face touches `outer_radius` at its midpoint and the shared corners reach
/// `outer_radius / cos(pi / 16)`. Their union is exactly [`Self::hull`], except
/// that every shared corner appears twice there, rounded once per box, which is
/// why a solver should take its geometry from here.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CompiledCylinder {
    /// Source editable part.
    pub source_part: PartId,
    /// Owning compound row.
    pub compound_index: u32,
    /// First of the [`CYLINDER_COLLIDER_COUNT`] collider rows this replaces.
    pub first_collider: u32,
    /// Axis midpoint in compound-local coordinates.
    pub local_center: Vec3,
    /// Orientation relative to the compound root; the axis is `rotation * Y`.
    pub local_rotation: Quat,
    /// Authored outer radius in metres, at each facet midpoint.
    pub outer_radius: f32,
    /// Half the axial length in metres.
    pub half_length: f32,
}

impl CompiledCylinder {
    /// Exact circumscribed prism, with each corner computed once so the two faces
    /// meeting there share the same vertex. Sixteen radial faces plus two ends.
    ///
    /// # Panics
    ///
    /// Panics only if the compiled facet count stops fitting a `u16`.
    #[must_use]
    pub fn hull(&self) -> CompiledConvex {
        let facets = CYLINDER_COLLIDER_COUNT;
        let count = u16::try_from(facets).expect("the facet count is sixteen");
        let segment = core::f32::consts::TAU / f32::from(count);
        let corner_radius = self.outer_radius / (segment * 0.5).cos();
        let corners = (0..count)
            .map(|corner| {
                let angle = segment * (f32::from(corner) - 0.5);
                Vec3::new(angle.cos(), 0.0, angle.sin()) * corner_radius
            })
            .collect::<Vec<_>>();
        let mut vertices = Vec::with_capacity(2 * facets);
        let mut edge_directions = vec![self.local_rotation * Vec3::Y];
        for (index, &corner) in corners.iter().enumerate() {
            for sign in [-1.0, 1.0] {
                vertices.push(
                    self.local_center
                        + self.local_rotation * (corner + Vec3::Y * (self.half_length * sign)),
                );
            }
            let next = corners[(index + 1) % facets];
            if let Some(direction) = (self.local_rotation * (next - corner)).try_normalize() {
                edge_directions.push(direction);
            }
        }
        let mut face_planes = Vec::with_capacity(facets + 2);
        for face in 0..count {
            let angle = segment * f32::from(face);
            let normal = self.local_rotation * Vec3::new(angle.cos(), 0.0, angle.sin());
            face_planes.push(normal.extend(normal.dot(self.local_center) + self.outer_radius));
        }
        for sign in [-1.0, 1.0] {
            let normal = self.local_rotation * (Vec3::Y * sign);
            face_planes.push(normal.extend(normal.dot(self.local_center) + self.half_length));
        }
        CompiledConvex {
            vertices,
            face_planes,
            edge_directions,
        }
    }
}

/// How the solver drives one mechanism coordinate.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DriveMode {
    /// No control block drives this coordinate; it swings freely.
    #[default]
    Passive,
    /// Hold a target speed.
    Speed,
    /// Seek and hold a target angle.
    Angle,
}

impl DriveMode {
    /// Discriminant uploaded to the GPU.
    pub const fn code(self) -> u32 {
        match self {
            Self::Passive => 0,
            Self::Speed => 1,
            Self::Angle => 2,
        }
    }
}

/// Resolved drive parameters for one mechanism coordinate.
///
/// A passive coordinate has a zero `max_acceleration` and infinite limits,
/// which is exactly free-swinging behaviour in the solver.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CoordinateDrive {
    /// What the solver does with this coordinate.
    pub mode: DriveMode,
    /// Signed target speed in radians per second, with wire reversal applied.
    pub target_speed: f32,
    /// Target angle in radians, with wire reversal applied.
    pub target_angle: f32,
    /// Fastest the joint may turn, in radians per second.
    pub max_speed: f32,
    /// Largest permitted change in joint speed per second. Infinite when the
    /// drive torque is unlimited.
    pub max_acceleration: f32,
    /// Stall acceleration supplied by the first actuator family.
    pub source_a_max_acceleration: f32,
    /// No-load speed of the first actuator family, in radians per second.
    pub source_a_no_load_speed: f32,
    /// Stall acceleration supplied by the second actuator family.
    pub source_b_max_acceleration: f32,
    /// No-load speed of the second actuator family, in radians per second.
    pub source_b_no_load_speed: f32,
    /// Lower angle limit in radians, or negative infinity.
    pub min_angle: f32,
    /// Upper angle limit in radians, or positive infinity.
    pub max_angle: f32,
}

/// Transient active gear for one Controller engine lane.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GearSelection {
    /// Controller whose directly welded engine line is being shifted.
    pub controller: PartId,
    /// Independently geared engine family.
    pub kind: EngineKind,
    /// Input-to-output ratio, or `None` while that family is disengaged.
    pub ratio: Option<f32>,
}

impl CoordinateDrive {
    /// Row describing a coordinate no control block drives.
    pub const PASSIVE: Self = Self {
        mode: DriveMode::Passive,
        target_speed: 0.0,
        target_angle: 0.0,
        max_speed: 0.0,
        max_acceleration: 0.0,
        source_a_max_acceleration: 0.0,
        source_a_no_load_speed: 0.0,
        source_b_max_acceleration: 0.0,
        source_b_no_load_speed: 0.0,
        min_angle: f32::NEG_INFINITY,
        max_angle: f32::INFINITY,
    };
}

impl Default for CoordinateDrive {
    fn default() -> Self {
        Self::PASSIVE
    }
}

/// Construction topology cannot be represented by the exact-coordinate model.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum TopologyError {
    /// Simulation requires at least one part.
    #[error("construction contains no parts")]
    EmptyConstruction,
    /// A driven bearing's endpoints were welded into the same compound, so the
    /// drive has no coordinate left to turn.
    #[error("driven bearing {bearing:?} is welded solid to compound {compound}")]
    SelfBearing {
        /// Offending bearing.
        bearing: BearingId,
        /// Collapsed compound row.
        compound: u32,
    },
    /// A driven bearing could not be given an independent coordinate.
    #[error("driven bearing {bearing:?} closes a loop and cannot carry a drive")]
    DrivenClosureBearing {
        /// Offending bearing.
        bearing: BearingId,
    },
    /// A computed mass or inertia was non-finite or singular.
    #[error("compound containing {part:?} has invalid mass properties")]
    InvalidMassProperties {
        /// One source part identifying the group.
        part: PartId,
    },
    /// A power module has more electric-driven joints than electric ports.
    #[error("control module at {controller:?} needs {required} electric ports but has {available}")]
    InsufficientElectricPorts {
        /// A controller identifying the module.
        controller: PartId,
        /// Number of assigned physical joints.
        required: u32,
        /// Number of available ports.
        available: u32,
    },
    /// A power module has more gas-driven joints than gas ports.
    #[error("control module at {controller:?} needs {required} gas ports but has {available}")]
    InsufficientGasPorts {
        /// A controller identifying the module.
        controller: PartId,
        /// Number of assigned physical joints.
        required: u32,
        /// Number of available ports.
        available: u32,
    },
    /// A creation needs more collider rows than the solver accepts.
    #[error("creation needs {required} collider rows but the budget is {available}")]
    ColliderBudgetExceeded {
        /// Rows the decomposition produced.
        required: usize,
        /// Largest supported row count.
        available: usize,
    },
    /// A power module has more servo-driven joints than Servos.
    #[error("control module at {controller:?} needs {required} Servos but has {available}")]
    InsufficientServos {
        /// A controller identifying the module.
        controller: PartId,
        /// Number of assigned physical joints.
        required: u32,
        /// Number of available Servos.
        available: u32,
    },
    /// A motor was given an angle state or a Servo was given a speed state.
    #[error("bearing {bearing:?} has a program incompatible with its assigned actuator")]
    IncompatibleActuatorProgram {
        /// Bearing carrying the incompatible program.
        bearing: BearingId,
    },
    /// Same-type engines in one Controller module have different transmission depths.
    #[error(
        "control module at {controller:?} has mismatched {kind:?} transmission depths {depths:?}"
    )]
    TransmissionDepthMismatch {
        /// Controller identifying the module.
        controller: PartId,
        /// Engine family whose physical stacks disagree.
        kind: EngineKind,
        /// Sorted depths found on the physical engines.
        depths: Vec<u8>,
    },
}

impl CompiledCreation {
    /// Re-derives the drive rows from the graph's current control blocks.
    ///
    /// The graph must be the one this creation was compiled from; only drive
    /// parameters may have changed since. This is how a running simulation is
    /// retuned without recompiling topology.
    pub fn resolve_coordinate_drives(&self, graph: &ConstructionGraph) -> Vec<CoordinateDrive> {
        self.resolve_coordinate_drives_with_gears(graph, &[])
    }

    /// Re-derives drive rows using independent active gas/electric gear ratios.
    /// A disengaged selection contributes no torque while the other family remains active.
    pub fn resolve_coordinate_drives_with_gears(
        &self,
        graph: &ConstructionGraph,
        active_gears: &[GearSelection],
    ) -> Vec<CoordinateDrive> {
        let actuation = resolve_coordinate_actuation(&self.loop_topology, graph, active_gears)
            .unwrap_or_else(|_| vec![CoordinateActuation::default(); self.coordinate_drives.len()]);
        resolve_coordinate_drives(&self.loop_topology, graph, &actuation)
    }

    /// Builds the drive row for one coordinate from a live state target.
    ///
    /// This is how a running sequencer turns the state a bearing has just
    /// entered into an upload row, without recompiling anything. Returns
    /// [`CoordinateDrive::PASSIVE`] for an unknown coordinate or a grounded
    /// subtree that no torque can accelerate, or a target with incompatible units.
    pub fn coordinate_drive_row(
        &self,
        coordinate: u32,
        target: DriveTarget,
        limits: DriveLimits,
    ) -> CoordinateDrive {
        let Some(bearing) = self
            .bearings
            .iter()
            .find(|bearing| bearing.coordinate_index == Some(coordinate))
        else {
            return CoordinateDrive::PASSIVE;
        };
        let linear = bearing.kind.is_translational();
        if target.is_linear() != linear || matches!(bearing.kind, crate::BearingKind::Suspension(_))
        {
            let [min_angle, max_angle] = bearing.kind.bounds();
            return CoordinateDrive {
                min_angle,
                max_angle,
                ..CoordinateDrive::PASSIVE
            };
        }
        let inertia = self
            .loop_topology
            .coordinate_axis_inertia
            .get(coordinate as usize)
            .copied()
            .unwrap_or(f32::INFINITY);
        if !inertia.is_finite() {
            return CoordinateDrive::PASSIVE;
        }
        let template = self
            .coordinate_drives
            .get(coordinate as usize)
            .copied()
            .unwrap_or_default();
        let mut row = coordinate_drive(
            target,
            limits,
            inertia,
            CoordinateActuation {
                source_a_torque: template.source_a_max_acceleration * inertia,
                source_a_no_load_speed: template.source_a_no_load_speed,
                source_b_torque: template.source_b_max_acceleration * inertia,
                source_b_no_load_speed: template.source_b_no_load_speed,
                max_speed: template.max_speed,
            },
        );
        if linear {
            row.min_angle = template.min_angle;
            row.max_angle = template.max_angle;
            row.target_angle = row.target_angle.clamp(row.min_angle, row.max_angle);
        }
        row
    }
}
