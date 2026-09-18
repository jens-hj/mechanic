use std::{
    collections::{BTreeMap, BTreeSet},
    ops::Range,
};

use bevy_math::{Mat3, Quat, Vec3, Vec4};
use thiserror::Error;

use crate::{
    ActuatorAssignment, BearingId, ConstructionGraph, CuboidSpec, DriveLimits, DriveTarget,
    EngineKind, FaceOwner, MACHINE_PART_DENSITY_KG_M3, MaterialProperties, PartId, PartSpec,
    RegionId, ServoSpec, ShapeRegion,
    shape::{ConvexPiece, PartPiece, decompose, decompose_part},
};

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

impl ConstructionGraph {
    /// Compiles rigid groups, mass properties, bearings, and loop equations atomically.
    ///
    /// # Errors
    ///
    /// Returns [`TopologyError`] when the graph is empty, a bearing collapses
    /// into a weld group, or derived mass properties are invalid.
    pub fn compile(&self) -> Result<CompiledCreation, TopologyError> {
        self.compile_with_suspension_sockets([], &[])
    }

    /// Compiles the graph while treating compounds containing any supplied part as static.
    ///
    /// World terrain anchors are instance state rather than authored construction data, so
    /// they must not be represented by a weld to the Garage's flat ground plane.
    ///
    /// # Errors
    ///
    /// Returns the same topology and capacity errors as [`Self::compile`].
    pub fn compile_with_static_parts(
        &self,
        static_parts: impl IntoIterator<Item = PartId>,
    ) -> Result<CompiledCreation, TopologyError> {
        self.compile_with_suspension_sockets(static_parts, &[])
    }

    /// Compiles suspension sockets as carried mass on their source compounds.
    ///
    /// An unattached socket contributes its entire assembly mass and inertia.
    /// Sockets already represented by an attached suspension bearing are ignored.
    ///
    /// # Errors
    /// Returns the same topology and capacity errors as [`Self::compile`].
    pub fn compile_with_suspension_sockets(
        &self,
        static_parts: impl IntoIterator<Item = PartId>,
        sockets: &[crate::BearingSocket],
    ) -> Result<CompiledCreation, TopologyError> {
        let static_parts = static_parts.into_iter().collect::<BTreeSet<_>>();
        compile_graph(self, &static_parts, sockets)
    }
}

#[allow(clippy::too_many_lines)]
fn compile_graph(
    graph: &ConstructionGraph,
    externally_static_parts: &BTreeSet<PartId>,
    sockets: &[crate::BearingSocket],
) -> Result<CompiledCreation, TopologyError> {
    if graph.parts.is_empty() {
        return Err(TopologyError::EmptyConstruction);
    }

    let part_rows = graph.parts.iter().collect::<Vec<_>>();
    let dense_by_part = part_rows
        .iter()
        .enumerate()
        .map(|(dense, (id, _))| (*id, dense))
        .collect::<BTreeMap<_, _>>();
    let mut weld_groups = DisjointSet::new(part_rows.len());
    let mut directly_grounded = vec![false; part_rows.len()];

    for part in externally_static_parts {
        if let Some(&dense) = dense_by_part.get(part) {
            directly_grounded[dense] = true;
        }
    }

    for (_, weld) in graph.welds.iter() {
        match (weld.first.owner, weld.second.owner) {
            (FaceOwner::Part(a), FaceOwner::Part(b)) => {
                weld_groups.union(dense_by_part[&a], dense_by_part[&b]);
            }
            (FaceOwner::Part(part), FaceOwner::Ground)
            | (FaceOwner::Ground, FaceOwner::Part(part)) => {
                directly_grounded[dense_by_part[&part]] = true;
            }
            (FaceOwner::Ground, FaceOwner::Ground) => {}
        }
    }
    for (_, link) in graph.rigid_links.iter() {
        weld_groups.union(dense_by_part[&link.first], dense_by_part[&link.second]);
    }

    let mut grouped = BTreeMap::<usize, Vec<usize>>::new();
    for dense in 0..part_rows.len() {
        grouped
            .entry(weld_groups.find(dense))
            .or_default()
            .push(dense);
    }

    let mut compounds = Vec::with_capacity(grouped.len());
    let collider_capacity = part_rows
        .iter()
        .map(|(part, spec)| {
            if graph.owner_has_shape_features(crate::SolidOwner::Part(*part)) {
                return graph
                    .evaluated_solid_shared(crate::SolidOwner::Part(*part))
                    .expect("committed feature geometry replays")
                    .cells
                    .len();
            }
            match spec {
            PartSpec::Controller(_)
            | PartSpec::Engine(_)
            | PartSpec::Transmission(_)
            | PartSpec::Servo(_)
            | PartSpec::Seat(_)
            | PartSpec::Input(_)
            | PartSpec::DimensionLink(_)
            | PartSpec::Cuboid(_) => 1,
            PartSpec::Cylinder(_) => CYLINDER_COLLIDER_COUNT,
            PartSpec::PipeBend(_) => PIPE_BEND_COLLIDER_COUNT,
            PartSpec::PipeJunction(junction) => junction.collider_count(),
            }
        })
        .sum::<usize>()
        // A region emits one row per fused convex piece, so the count is only
        // known by running its decomposition; its blocks emit nothing.
        + graph
            .regions()
            .map(|(id, region)| {
                if graph.owner_has_shape_features(crate::SolidOwner::Region(id)) {
                    graph
                        .evaluated_solid_shared(crate::SolidOwner::Region(id))
                        .expect("committed region feature geometry replays")
                        .cells
                        .len()
                } else {
                    region_pieces(region).len()
                }
            })
            .sum::<usize>();
    if collider_capacity > MAX_COMPILED_COLLIDERS {
        return Err(TopologyError::ColliderBudgetExceeded {
            required: collider_capacity,
            available: MAX_COMPILED_COLLIDERS,
        });
    }
    let mut colliders = Vec::with_capacity(collider_capacity);
    let mut cylinders = Vec::new();
    let mut compound_by_dense_part = vec![0_u32; part_rows.len()];

    // A part inside a region hands its geometry over to that region, so it must
    // not emit a box or a mass of its own as well.
    let mut covered: BTreeSet<PartId> = BTreeSet::new();
    let mut region_of_part: BTreeMap<PartId, RegionId> = BTreeMap::new();
    for (id, _) in graph.parts() {
        if let Some(region) = graph.region_of(id) {
            covered.insert(id);
            region_of_part.insert(id, region);
        }
    }

    for member_rows in grouped.values() {
        let compound_index = u32::try_from(compounds.len()).expect("compound count fits u32");
        let is_static = member_rows.iter().any(|&row| directly_grounded[row]);
        let source_parts = member_rows
            .iter()
            .map(|&row| part_rows[row].0)
            .collect::<Vec<_>>();
        // Regions whose blocks live in this compound, each counted once.
        let mut member_regions: Vec<RegionId> = Vec::new();
        for &row in member_rows {
            if let Some(&region) = region_of_part.get(&part_rows[row].0)
                && !member_regions.contains(&region)
            {
                member_regions.push(region);
            }
        }
        let region_shapes = member_regions
            .iter()
            .filter_map(|&id| graph.region(id).map(|region| (id, region)))
            .collect::<Vec<_>>();

        let mass_properties = calculate_mass_properties(
            member_rows
                .iter()
                .map(|&row| (part_rows[row].0, *part_rows[row].1)),
            is_static,
            &covered,
            &region_shapes,
            graph,
            sockets,
        )?;
        let collider_start = u32::try_from(colliders.len()).expect("collider count fits u32");
        for &row in member_rows {
            let (part, spec) = part_rows[row];
            compound_by_dense_part[row] = compound_index;
            if covered.contains(&part) {
                continue;
            }
            // Layered cuboids collide per band. An unfeatured layered cylinder
            // keeps its envelope colliders and analytic rolling contact, which
            // only its outer wall's material ever touches.
            if graph.owner_has_shape_features(crate::SolidOwner::Part(part))
                || (spec.is_layered() && spec.as_cylinder().is_none())
            {
                let solid = graph
                    .evaluated_solid_shared(crate::SolidOwner::Part(part))
                    .expect("committed feature geometry replays");
                append_evaluated_colliders(
                    &mut colliders,
                    &solid,
                    part,
                    compound_index,
                    mass_properties.center_of_mass,
                    |band| band_contact_properties(*spec, band),
                );
            } else {
                let start = colliders.len();
                append_part_colliders(&mut colliders, part, compound_index, *spec, Vec3::ZERO);
                compose_raw_colliders(
                    &mut colliders[start..],
                    graph.part_frame(part).expect("compiled part has a frame"),
                    mass_properties.center_of_mass,
                );
                cylinders.extend(solid_full_cylinder(
                    *spec,
                    part,
                    compound_index,
                    start,
                    &colliders[start..],
                ));
            }
        }
        for &(id, region) in &region_shapes {
            let source_part = member_rows
                .iter()
                .map(|&row| part_rows[row].0)
                .find(|part| region_of_part.get(part) == Some(&id))
                .expect("a region in this compound has a member part");
            if graph.owner_has_shape_features(crate::SolidOwner::Region(id)) {
                let solid = graph
                    .evaluated_solid_shared(crate::SolidOwner::Region(id))
                    .expect("committed region feature geometry replays");
                append_evaluated_colliders(
                    &mut colliders,
                    &solid,
                    source_part,
                    compound_index,
                    mass_properties.center_of_mass,
                    |_| region.material().properties(),
                );
            } else {
                let start = colliders.len();
                append_region_colliders(
                    &mut colliders,
                    id,
                    region,
                    compound_index,
                    Vec3::ZERO,
                    source_part,
                );
                compose_raw_colliders(
                    &mut colliders[start..],
                    graph.owner_frame(crate::SolidOwner::Region(id)),
                    mass_properties.center_of_mass,
                );
            }
        }
        compact_grid_aligned_cuboids(
            &mut colliders,
            usize::try_from(collider_start).expect("collider start fits usize"),
        );
        let collider_end = u32::try_from(colliders.len()).expect("collider count fits u32");
        compounds.push(CompiledCompound {
            source_parts,
            root_translation: mass_properties.center_of_mass,
            root_rotation: Quat::IDENTITY,
            is_static,
            mass_properties,
            collider_range: collider_start..collider_end,
        });
    }

    let part_to_compound = part_rows
        .iter()
        .enumerate()
        .map(|(dense, (part, _))| (*part, compound_by_dense_part[dense]))
        .collect::<Vec<_>>();
    let compound_lookup = part_to_compound.iter().copied().collect::<BTreeMap<_, _>>();

    let mut bearing_components = DisjointSet::new(compounds.len());
    let mut topology = LoopTopology::default();
    let mut bearings = Vec::with_capacity(graph.bearings.len());
    let mut suppressed = BTreeSet::new();

    let driven_bearings = graph
        .drive_links
        .iter()
        .map(|(_, link)| link.bearing)
        .collect::<BTreeSet<_>>();

    // Collapse bearings that describe the same physical joint. When one of a
    // duplicate group carries a drive, that row represents the group so its
    // control block is not silently dropped.
    let mut physical_order = Vec::new();
    let mut physical_by_key = BTreeMap::new();
    let mut bearing_keys = Vec::new();
    for (bearing_id, bearing) in graph.bearings.iter() {
        let FaceOwner::Part(part_a) = bearing.source.owner else {
            unreachable!("graph validation rejects ground bearings")
        };
        let FaceOwner::Part(part_b) = bearing.target.owner else {
            unreachable!("graph validation rejects ground bearings")
        };
        let compound_a = compound_lookup[&part_a];
        let compound_b = compound_lookup[&part_b];
        if compound_a == compound_b {
            // Welding a loop shut can leave a bearing with both sides in one
            // rigid body. The weld already fixes their relative pose, so the
            // joint constrains nothing and simply does not compile to one —
            // the player gets a locked bearing, not a build that will not run.
            // A drive is the exception: dropping the joint would kill the
            // motor with nothing to show for it.
            if driven_bearings.contains(&bearing_id) {
                return Err(TopologyError::SelfBearing {
                    bearing: bearing_id,
                    compound: compound_a,
                });
            }
            continue;
        }
        let physical_key = (
            compound_a,
            compound_b,
            bearing.shared_anchor.to_array().map(f32::to_bits),
            bearing.axis.to_array().map(f32::to_bits),
            bearing.kind.bounds().map(f32::to_bits),
            match bearing.kind {
                crate::BearingKind::Rotational => (0, 0, [0; 3], 0),
                crate::BearingKind::Suspension(_) => (2, 0, [0; 3], 0),
                crate::BearingKind::Linear(rail) => (
                    1,
                    rail.dimensions.width().to_bits(),
                    rail.mount_normal.to_array().map(f32::to_bits),
                    rail.face as u8,
                ),
            },
        );
        bearing_components.union(compound_a as usize, compound_b as usize);
        suppressed.insert(ordered_pair(compound_a, compound_b));
        bearing_keys.push((bearing_id, physical_key));
        match physical_by_key.get(&physical_key).copied() {
            None => {
                physical_by_key.insert(physical_key, physical_order.len());
                physical_order.push((bearing_id, compound_a, compound_b));
            }
            Some(existing) => {
                let (kept, ..) = physical_order[existing];
                if driven_bearings.contains(&bearing_id) && !driven_bearings.contains(&kept) {
                    physical_order[existing].0 = bearing_id;
                }
            }
        }
    }

    // Choose the spanning forest with driven bearings considered first so a
    // drive is never stranded on a loop-closure edge that a passive bearing
    // could have taken instead.
    let mut mechanism_forest = DisjointSet::new(compounds.len());
    let mut forest_has_fixed_root = compounds
        .iter()
        .map(|compound| compound.is_static)
        .collect::<Vec<_>>();
    let mut tree_edges = BTreeSet::new();
    for driven_pass in [true, false] {
        for &(bearing_id, compound_a, compound_b) in &physical_order {
            if driven_bearings.contains(&bearing_id) != driven_pass {
                continue;
            }
            let root_a = mechanism_forest.find(compound_a as usize);
            let root_b = mechanism_forest.find(compound_b as usize);
            let joins_two_fixed_trees =
                root_a != root_b && forest_has_fixed_root[root_a] && forest_has_fixed_root[root_b];
            if root_a == root_b || joins_two_fixed_trees {
                continue;
            }
            let has_fixed_root = forest_has_fixed_root[root_a] || forest_has_fixed_root[root_b];
            mechanism_forest.union(root_a, root_b);
            let joined_root = mechanism_forest.find(root_a);
            forest_has_fixed_root[joined_root] = has_fixed_root;
            tree_edges.insert(bearing_id);
        }
    }

    let mut representative_coordinates = BTreeMap::new();
    for &(bearing_id, compound_a, compound_b) in &physical_order {
        let bearing = graph
            .bearings
            .get(bearing_id)
            .copied()
            .expect("physical bearing rows come from live graph handles");
        let coordinate_index = if tree_edges.contains(&bearing_id) {
            let coordinate = u32::try_from(topology.tree_bearings.len())
                .expect("bearing coordinate count fits u32");
            topology.tree_bearings.push(bearing_id);
            representative_coordinates.insert(bearing_id, coordinate);
            Some(coordinate)
        } else {
            if driven_bearings.contains(&bearing_id) {
                return Err(TopologyError::DrivenClosureBearing {
                    bearing: bearing_id,
                });
            }
            topology.closure_bearings.push(bearing_id);
            None
        };
        let root_a = compounds[compound_a as usize].root_translation;
        let root_b = compounds[compound_b as usize].root_translation;
        bearings.push(CompiledBearing {
            kind: bearing.kind,
            source_bearing: bearing_id,
            compound_a,
            compound_b,
            local_anchor_a: bearing.shared_anchor - root_a,
            local_anchor_b: bearing.shared_anchor - root_b,
            local_axis_a: bearing.axis,
            local_axis_b: bearing.axis,
            coordinate_index,
        });
    }

    // Duplicate rows describing one physical joint share that joint's
    // coordinate, so a drive wired to any of them addresses the same row.
    topology.bearing_coordinates = bearing_keys
        .into_iter()
        .filter_map(|(bearing_id, key)| {
            let representative = physical_order[physical_by_key[&key]].0;
            let coordinate = representative_coordinates.get(&representative)?;
            Some((bearing_id, *coordinate))
        })
        .collect();

    let mut components = BTreeMap::<usize, Vec<u32>>::new();
    for compound in 0..compounds.len() {
        components
            .entry(bearing_components.find(compound))
            .or_default()
            .push(u32::try_from(compound).expect("compound count fits u32"));
    }
    topology.mechanism_components = components.into_values().collect();
    compile_tree_metadata(&compounds, &bearings, &mut topology);
    topology.coordinate_axis_inertia =
        compile_coordinate_axis_inertia(&compounds, &bearings, &topology);

    validate_transmission_depths(graph)?;
    validate_actuator_programs(graph)?;
    let actuation = resolve_coordinate_actuation(&topology, graph, &[])?;
    let coordinate_drives = resolve_coordinate_drives(&topology, graph, &actuation);

    let dynamics = crate::CompiledDynamics::compile(&compounds, &bearings, &topology);
    Ok(CompiledCreation {
        dynamics,
        compounds,
        colliders,
        bearings,
        loop_topology: topology,
        collision_suppression: suppressed.into_iter().collect(),
        part_to_compound,
        coordinate_drives,
        cylinders,
    })
}

fn validate_transmission_depths(graph: &ConstructionGraph) -> Result<(), TopologyError> {
    for (controller, spec) in graph.parts() {
        if !matches!(spec, PartSpec::Controller(_)) {
            continue;
        }
        for kind in [EngineKind::Electric, EngineKind::Gas] {
            let depths = graph
                .transmission_depths(controller, kind)
                .expect("the part was checked as a controller");
            if let Some(first) = depths.first()
                && depths.iter().any(|depth| depth != first)
            {
                return Err(TopologyError::TransmissionDepthMismatch {
                    controller,
                    kind,
                    depths,
                });
            }
        }
    }
    Ok(())
}

fn floating_component_root(compounds: &[CompiledCompound], component: &[u32]) -> Option<u32> {
    component.iter().copied().reduce(|root, candidate| {
        if compounds[candidate as usize].mass_properties.mass
            > compounds[root as usize].mass_properties.mass
        {
            candidate
        } else {
            root
        }
    })
}

fn compile_tree_metadata(
    compounds: &[CompiledCompound],
    bearings: &[CompiledBearing],
    topology: &mut LoopTopology,
) {
    let body_count = compounds.len();
    let mut adjacency = vec![Vec::<(usize, BearingId, u32)>::new(); body_count];
    for bearing in bearings {
        if bearing.coordinate_index.is_none() {
            continue;
        }
        let a = bearing.compound_a as usize;
        let b = bearing.compound_b as usize;
        adjacency[a].push((b, bearing.source_bearing, 0));
        adjacency[b].push((a, bearing.source_bearing, 1));
    }
    for neighbours in &mut adjacency {
        neighbours.sort_unstable_by_key(|&(body, bearing, direction)| {
            (body, bearing.index(), bearing.generation(), direction)
        });
    }

    let mut metadata = (0..body_count)
        .map(|body| MechanismBodyTopology {
            parent_body: u32::try_from(body).expect("body count fits u32"),
            tree_bearing: None,
            bearing_direction: 0,
            component_index: 0,
            depth: 0,
            preorder_index: 0,
            postorder_index: 0,
            is_root: true,
        })
        .collect::<Vec<_>>();
    let mut visited = vec![false; body_count];
    let mut preorder = Vec::with_capacity(body_count);
    topology.component_roots.clear();

    for (component_index, component) in topology.mechanism_components.iter().enumerate() {
        let fixed_roots = component
            .iter()
            .copied()
            .filter(|&body| compounds[body as usize].is_static)
            .collect::<Vec<_>>();
        let roots = if fixed_roots.is_empty() {
            // Rooting a floating mechanism at a light appendage makes the
            // opposite side of that joint contain almost the whole machine,
            // producing asymmetric coordinate inertia for mirrored actuators.
            floating_component_root(compounds, component)
                .into_iter()
                .collect::<Vec<_>>()
        } else {
            fixed_roots
        };
        topology.component_roots.push(roots.clone());

        for root in roots {
            let root = root as usize;
            if visited[root] {
                continue;
            }
            visited[root] = true;
            metadata[root].component_index =
                u32::try_from(component_index).expect("component count fits u32");
            let mut queue = std::collections::VecDeque::from([root]);
            while let Some(parent) = queue.pop_front() {
                metadata[parent].preorder_index =
                    u32::try_from(preorder.len()).expect("body count fits u32");
                preorder.push(parent);
                let parent_depth = metadata[parent].depth;
                for &(child, bearing, direction) in &adjacency[parent] {
                    if visited[child] {
                        continue;
                    }
                    visited[child] = true;
                    metadata[child] = MechanismBodyTopology {
                        parent_body: u32::try_from(parent).expect("body count fits u32"),
                        tree_bearing: Some(bearing),
                        bearing_direction: direction,
                        component_index: u32::try_from(component_index)
                            .expect("component count fits u32"),
                        depth: parent_depth + 1,
                        preorder_index: 0,
                        postorder_index: 0,
                        is_root: false,
                    };
                    queue.push_back(child);
                }
            }
        }
    }

    let mut postorder = preorder.clone();
    postorder.sort_unstable_by_key(|&body| (core::cmp::Reverse(metadata[body].depth), body));
    for (index, &body) in postorder.iter().enumerate() {
        metadata[body].postorder_index = u32::try_from(index).expect("body count fits u32");
    }
    let maximum_depth = metadata.iter().map(|body| body.depth).max().unwrap_or(0);
    topology.contraction_rounds = (1..=maximum_depth)
        .rev()
        .map(|depth| {
            metadata
                .iter()
                .enumerate()
                .filter(|(_, row)| row.depth == depth)
                .map(|(body, _)| u32::try_from(body).expect("body count fits u32"))
                .collect()
        })
        .collect();
    topology.body_parents = metadata;
}

/// Rotational inertia of each tree bearing's child subtree about that bearing's
/// own axis, evaluated in the compile-time bind pose.
fn compile_coordinate_axis_inertia(
    compounds: &[CompiledCompound],
    bearings: &[CompiledBearing],
    topology: &LoopTopology,
) -> Vec<f32> {
    let mut children = vec![Vec::<usize>::new(); compounds.len()];
    for (body, row) in topology.body_parents.iter().enumerate() {
        if !row.is_root {
            children[row.parent_body as usize].push(body);
        }
    }
    let child_body_by_bearing = topology
        .body_parents
        .iter()
        .enumerate()
        .filter_map(|(body, row)| row.tree_bearing.map(|bearing| (bearing, body)))
        .collect::<BTreeMap<_, _>>();

    topology
        .tree_bearings
        .iter()
        .map(|source_bearing| {
            let Some(&child_body) = child_body_by_bearing.get(source_bearing) else {
                return f32::INFINITY;
            };
            let Some(bearing) = bearings
                .iter()
                .find(|row| row.source_bearing == *source_bearing)
            else {
                return f32::INFINITY;
            };
            let axis = bearing.local_axis_a.normalize_or_zero();
            if axis == Vec3::ZERO {
                return f32::INFINITY;
            }
            let anchor =
                compounds[bearing.compound_a as usize].root_translation + bearing.local_anchor_a;

            let mut total = 0.0_f32;
            let mut stack = vec![child_body];
            while let Some(body) = stack.pop() {
                let compound = &compounds[body];
                if compound.is_static {
                    return f32::INFINITY;
                }
                let properties = compound.mass_properties;
                let offset = properties.center_of_mass - anchor;
                let radial = offset - axis * offset.dot(axis);
                total += if bearing.kind.is_translational() {
                    properties.mass
                } else {
                    axis.dot(properties.inertia * axis) + properties.mass * radial.length_squared()
                };
                stack.extend(children[body].iter().copied());
            }
            if total.is_finite() && total > 0.0 {
                total
            } else {
                f32::INFINITY
            }
        })
        .collect()
}

#[derive(Clone, Copy, Debug, Default)]
struct CoordinateActuation {
    source_a_torque: f32,
    source_a_no_load_speed: f32,
    source_b_torque: f32,
    source_b_no_load_speed: f32,
    max_speed: f32,
}

#[derive(Default)]
struct ModuleBudget {
    controller: Option<PartId>,
    electric_engines: u32,
    gas_engines: u32,
    servos: u32,
    electric_coordinates: BTreeSet<u32>,
    gas_coordinates: BTreeSet<u32>,
    servo_coordinates: BTreeSet<u32>,
}

fn validate_actuator_programs(graph: &ConstructionGraph) -> Result<(), TopologyError> {
    for (_, link) in graph.drive_links() {
        let compatible = link.program.states().iter().all(|state| {
            matches!(
                (link.actuator, state.target()),
                (ActuatorAssignment::Unpowered, _)
                    | (
                        ActuatorAssignment::Motor { .. },
                        DriveTarget::Speed(_) | DriveTarget::LinearSpeed(_)
                    )
                    | (
                        ActuatorAssignment::Servo,
                        DriveTarget::Angle(_) | DriveTarget::LinearPosition(_)
                    )
            )
        });
        if !compatible {
            return Err(TopologyError::IncompatibleActuatorProgram {
                bearing: link.bearing,
            });
        }
    }
    Ok(())
}

#[allow(clippy::too_many_lines, clippy::cast_precision_loss)]
// Graph counts are far below f32's exact-integer range in any compilable
// creation; converting them keeps the torque-sharing arithmetic readable.
fn resolve_coordinate_actuation(
    topology: &LoopTopology,
    graph: &ConstructionGraph,
    active_gears: &[GearSelection],
) -> Result<Vec<CoordinateActuation>, TopologyError> {
    let mut modules = BTreeMap::<PartId, ModuleBudget>::new();
    let mut assignment_by_coordinate = BTreeMap::<u32, (PartId, PartId, ActuatorAssignment)>::new();

    for (_, link) in graph.drive_links() {
        let Some(&coordinate) = topology.bearing_coordinates.get(&link.bearing) else {
            continue;
        };
        let members = graph.machine_module(link.controller);
        let module_key = members.iter().next().copied().unwrap_or(link.controller);
        let module = modules.entry(module_key).or_insert_with(|| {
            let mut budget = ModuleBudget {
                controller: Some(link.controller),
                ..ModuleBudget::default()
            };
            for part in &members {
                match graph.part(*part) {
                    Some(PartSpec::Engine(engine)) => match engine.kind {
                        EngineKind::Electric => budget.electric_engines += 1,
                        EngineKind::Gas => budget.gas_engines += 1,
                    },
                    Some(PartSpec::Servo(_)) => budget.servos += 1,
                    _ => {}
                }
            }
            budget
        });
        if link.actuator.uses_electric() {
            module.electric_coordinates.insert(coordinate);
        }
        if link.actuator.uses_gas() {
            module.gas_coordinates.insert(coordinate);
        }
        if link.actuator.uses_servo() {
            module.servo_coordinates.insert(coordinate);
        }
        assignment_by_coordinate.entry(coordinate).or_insert((
            module_key,
            link.controller,
            link.actuator,
        ));
    }

    for module in modules.values() {
        let controller = module
            .controller
            .expect("a module budget is created from a controller link");
        let electric_required =
            u32::try_from(module.electric_coordinates.len()).expect("coordinate count fits u32");
        let electric_available = module.electric_engines * EngineKind::Electric.bearing_capacity();
        if electric_required > electric_available {
            return Err(TopologyError::InsufficientElectricPorts {
                controller,
                required: electric_required,
                available: electric_available,
            });
        }
        let gas_required =
            u32::try_from(module.gas_coordinates.len()).expect("coordinate count fits u32");
        let gas_available = module.gas_engines * EngineKind::Gas.bearing_capacity();
        if gas_required > gas_available {
            return Err(TopologyError::InsufficientGasPorts {
                controller,
                required: gas_required,
                available: gas_available,
            });
        }
        let servo_required =
            u32::try_from(module.servo_coordinates.len()).expect("coordinate count fits u32");
        if servo_required > module.servos {
            return Err(TopologyError::InsufficientServos {
                controller,
                required: servo_required,
                available: module.servos,
            });
        }
    }

    let mut result = vec![CoordinateActuation::default(); topology.tree_bearings.len()];
    for (coordinate, (module_key, controller, assignment)) in assignment_by_coordinate {
        let module = &modules[&module_key];
        let row = &mut result[coordinate as usize];
        let active_ratio = |kind| {
            active_gears
                .iter()
                .find(|gear| gear.controller == controller && gear.kind == kind)
                .map_or(Some(1.0), |gear| gear.ratio)
        };
        match assignment {
            ActuatorAssignment::Unpowered => {}
            ActuatorAssignment::Motor {
                electric_percent,
                gas_percent,
            } => {
                if electric_percent != 0
                    && let Some(ratio) = active_ratio(EngineKind::Electric)
                {
                    let consumers = module.electric_coordinates.len() as f32;
                    row.source_a_torque = module.electric_engines as f32
                        * EngineKind::Electric.stall_torque_newton_meters()
                        / consumers
                        * (f32::from(electric_percent) / 100.0)
                        * ratio;
                    row.source_a_no_load_speed =
                        crate::rpm_to_rad_s(EngineKind::Electric.no_load_rpm()) / ratio;
                    row.max_speed = row.max_speed.max(row.source_a_no_load_speed);
                }
                if gas_percent != 0
                    && let Some(ratio) = active_ratio(EngineKind::Gas)
                {
                    let consumers = module.gas_coordinates.len() as f32;
                    row.source_b_torque = module.gas_engines as f32
                        * EngineKind::Gas.stall_torque_newton_meters()
                        / consumers
                        * (f32::from(gas_percent) / 100.0)
                        * ratio;
                    row.source_b_no_load_speed =
                        crate::rpm_to_rad_s(EngineKind::Gas.no_load_rpm()) / ratio;
                    row.max_speed = row.max_speed.max(row.source_b_no_load_speed);
                }
            }
            ActuatorAssignment::Servo => {
                row.source_a_torque = ServoSpec::STALL_TORQUE_NEWTON_METERS;
                row.source_a_no_load_speed = crate::rpm_to_rad_s(ServoSpec::NO_LOAD_RPM);
                row.max_speed = row.source_a_no_load_speed;
            }
        }
    }
    for (coordinate, bearing) in topology.tree_bearings.iter().enumerate() {
        if graph
            .bearing(*bearing)
            .is_some_and(|bearing| bearing.kind.is_translational())
        {
            let row = &mut result[coordinate];
            row.source_a_torque /= crate::LINEAR_METERS_PER_RADIAN;
            row.source_b_torque /= crate::LINEAR_METERS_PER_RADIAN;
            row.source_a_no_load_speed *= crate::LINEAR_METERS_PER_RADIAN;
            row.source_b_no_load_speed *= crate::LINEAR_METERS_PER_RADIAN;
            row.max_speed *= crate::LINEAR_METERS_PER_RADIAN;
        }
    }
    Ok(result)
}

fn resolve_coordinate_drives(
    topology: &LoopTopology,
    graph: &ConstructionGraph,
    actuation: &[CoordinateActuation],
) -> Vec<CoordinateDrive> {
    topology
        .tree_bearings
        .iter()
        .enumerate()
        .map(|(coordinate, &bearing)| {
            let kind = graph.bearing(bearing).expect("compiled bearing").kind;
            let bounds = kind.bounds();
            let passive = CoordinateDrive {
                min_angle: bounds[0],
                max_angle: bounds[1],
                ..CoordinateDrive::PASSIVE
            };
            let Some((_, link)) = graph.bearing_drive_link(bearing) else {
                return passive;
            };
            let inertia = topology
                .coordinate_axis_inertia
                .get(coordinate)
                .copied()
                .unwrap_or(f32::INFINITY);
            if !inertia.is_finite() {
                return passive;
            }
            let Some(target) = link.resolved_target(0) else {
                return passive;
            };
            let mut drive = coordinate_drive(
                target,
                link.limits,
                inertia,
                actuation.get(coordinate).copied().unwrap_or_default(),
            );
            if let Some(limits) = link.linear_limits {
                drive.min_angle = limits.minimum().max(bounds[0]);
                drive.max_angle = limits.maximum().min(bounds[1]);
                drive.max_speed = drive.max_speed.min(limits.max_speed());
                let acceleration = limits.max_force() / inertia;
                if drive.max_acceleration > acceleration {
                    let scale = acceleration / drive.max_acceleration;
                    drive.source_a_max_acceleration *= scale;
                    drive.source_b_max_acceleration *= scale;
                    drive.max_acceleration = acceleration;
                }
                drive.target_speed = drive.target_speed.clamp(-drive.max_speed, drive.max_speed);
                drive.target_angle = drive.target_angle.clamp(drive.min_angle, drive.max_angle);
            }
            drive
        })
        .collect()
}

/// Builds one GPU-bound drive row from a resolved state target.
fn coordinate_drive(
    target: DriveTarget,
    limits: DriveLimits,
    axis_inertia: f32,
    actuation: CoordinateActuation,
) -> CoordinateDrive {
    if actuation.max_speed <= 0.0 {
        return CoordinateDrive::PASSIVE;
    }
    let max_acceleration = (actuation.source_a_torque + actuation.source_b_torque) / axis_inertia;
    let (mode, target_speed, target_angle) = match target {
        DriveTarget::Speed(speed) | DriveTarget::LinearSpeed(speed) => (
            DriveMode::Speed,
            speed.clamp(-actuation.max_speed, actuation.max_speed),
            0.0,
        ),
        DriveTarget::Angle(angle) | DriveTarget::LinearPosition(angle) => {
            (DriveMode::Angle, 0.0, angle)
        }
    };
    CoordinateDrive {
        mode,
        target_speed,
        target_angle,
        max_speed: actuation.max_speed,
        max_acceleration,
        source_a_max_acceleration: actuation.source_a_torque / axis_inertia,
        source_a_no_load_speed: actuation.source_a_no_load_speed,
        source_b_max_acceleration: actuation.source_b_torque / axis_inertia,
        source_b_no_load_speed: actuation.source_b_no_load_speed,
        min_angle: limits.min_angle(),
        max_angle: limits.max_angle(),
    }
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

#[allow(clippy::too_many_lines)]
fn calculate_mass_properties<'a>(
    parts: impl Iterator<Item = (PartId, PartSpec)> + Clone + 'a,
    is_static: bool,
    covered: &BTreeSet<PartId>,
    regions: &[(RegionId, &ShapeRegion)],
    graph: &ConstructionGraph,
    sockets: &[crate::BearingSocket],
) -> Result<MassProperties, TopologyError> {
    let member_parts = parts.clone().map(|(id, _)| id).collect::<BTreeSet<_>>();
    let mut suspension_masses = Vec::new();
    let mut seen_mounts = Vec::new();
    for (_, bearing) in graph.bearings() {
        let crate::BearingKind::Suspension(spec) = bearing.kind else {
            continue;
        };
        let key = (
            bearing.source,
            bearing.shared_anchor.to_array().map(f32::to_bits),
        );
        for element in spec.mass_elements() {
            let owner = if element.opposite {
                bearing.target.owner
            } else {
                bearing.source.owner
            };
            let FaceOwner::Part(owner) = owner else {
                continue;
            };
            if !member_parts.contains(&owner) {
                continue;
            }
            // A shared mounting assembly can have several attached parts.
            let endpoint_key = (key, element.opposite);
            if seen_mounts.contains(&endpoint_key) {
                continue;
            }
            let axis = bearing.axis;
            let outer = Mat3::from_cols(axis * axis.x, axis * axis.y, axis * axis.z);
            suspension_masses.push(WorldMassProperties {
                mass: element.mass,
                center: bearing.shared_anchor + axis * element.center,
                inertia: Mat3::IDENTITY * element.transverse_inertia
                    + outer * (element.axial_inertia - element.transverse_inertia),
            });
        }
        for opposite in [false, true] {
            let owner = if opposite {
                bearing.target.owner
            } else {
                bearing.source.owner
            };
            if let FaceOwner::Part(owner) = owner
                && member_parts.contains(&owner)
            {
                seen_mounts.push((key, opposite));
            }
        }
    }
    let mut seen_sockets = Vec::new();
    for socket in sockets {
        let crate::BearingKind::Suspension(spec) = socket.kind else {
            continue;
        };
        let FaceOwner::Part(owner) = socket.source.owner else {
            continue;
        };
        if !member_parts.contains(&owner) {
            continue;
        }
        let key = (socket.source, socket.anchor.to_array().map(f32::to_bits));
        if seen_sockets.contains(&key)
            || graph.bearings().any(|(_, bearing)| {
                matches!(bearing.kind, crate::BearingKind::Suspension(_))
                    && bearing.source == socket.source
                    && bearing.shared_anchor == socket.anchor
            })
        {
            continue;
        }
        seen_sockets.push(key);
        let axis = socket.axis;
        let outer = Mat3::from_cols(axis * axis.x, axis * axis.y, axis * axis.z);
        suspension_masses.extend(spec.mass_elements().into_iter().map(|element| {
            WorldMassProperties {
                mass: element.mass,
                center: socket.anchor + axis * element.center,
                inertia: Mat3::IDENTITY * element.transverse_inertia
                    + outer * (element.axial_inertia - element.transverse_inertia),
            }
        }));
    }
    let identifying_part = parts.clone().next().expect("weld groups are non-empty").0;
    // A part inside a region has no mass of its own: the region owns its
    // geometry, so counting both would weigh the build twice.
    let contributions = parts
        .filter(|(id, _)| !covered.contains(id))
        .map(|(id, spec)| {
            if graph.owner_has_shape_features(crate::SolidOwner::Part(id)) || spec.is_layered() {
                let solid = graph
                    .evaluated_solid_shared(crate::SolidOwner::Part(id))
                    .expect("committed feature geometry replays");
                evaluated_world_mass(&solid, |band| {
                    band_contact_properties(spec, band).density_kg_m3
                })
            } else {
                compose_world_mass(
                    part_world_mass(spec),
                    graph.part_frame(id).expect("compiled part has a frame"),
                )
            }
        })
        .chain(regions.iter().map(|(id, region)| {
            if graph.owner_has_shape_features(crate::SolidOwner::Region(*id)) {
                let solid = graph
                    .evaluated_solid_shared(crate::SolidOwner::Region(*id))
                    .expect("committed region feature geometry replays");
                evaluated_world_mass(&solid, |_| region.material().properties().density_kg_m3)
            } else {
                compose_world_mass(
                    region_world_mass(region),
                    graph.owner_frame(crate::SolidOwner::Region(*id)),
                )
            }
        }))
        .collect::<Vec<_>>();

    let contributions = contributions
        .into_iter()
        .chain(suspension_masses)
        .collect::<Vec<_>>();
    let total_mass = contributions.iter().map(|body| body.mass).sum::<f32>();
    let center_of_mass = contributions
        .iter()
        .map(|body| body.center * body.mass)
        .sum::<Vec3>()
        / total_mass;
    let mut inertia = Mat3::ZERO;
    for body in &contributions {
        let offset = body.center - center_of_mass;
        let outer = Mat3::from_cols(offset * offset.x, offset * offset.y, offset * offset.z);
        inertia += body.inertia + body.mass * (Mat3::IDENTITY * offset.length_squared() - outer);
    }

    let determinant = inertia.determinant();
    if !total_mass.is_finite()
        || total_mass <= 0.0
        || !center_of_mass.is_finite()
        || !inertia.is_finite()
        || !determinant.is_finite()
        || determinant <= f32::EPSILON
    {
        return Err(TopologyError::InvalidMassProperties {
            part: identifying_part,
        });
    }

    Ok(MassProperties {
        mass: total_mass,
        inverse_mass: if is_static { 0.0 } else { total_mass.recip() },
        center_of_mass,
        inertia,
        inverse_inertia: if is_static {
            Mat3::ZERO
        } else {
            inertia.inverse()
        },
    })
}

/// Mass, centre, and inertia about that centre, all in build space.
#[derive(Clone, Copy)]
struct WorldMassProperties {
    mass: f32,
    center: Vec3,
    inertia: Mat3,
}

fn compose_world_mass(
    properties: WorldMassProperties,
    frame: crate::ConstructionFrame,
) -> WorldMassProperties {
    let basis = Mat3::from_quat(frame.rotation());
    WorldMassProperties {
        center: frame.point(properties.center),
        inertia: basis * properties.inertia * basis.transpose(),
        ..properties
    }
}

fn part_world_mass(spec: PartSpec) -> WorldMassProperties {
    let properties = part_mass_properties(spec);
    let rotation = spec.pose().rotation.quaternion();
    let basis = Mat3::from_quat(rotation);
    WorldMassProperties {
        mass: properties.mass,
        center: spec.pose().translation() + rotation * properties.local_center,
        inertia: basis * properties.local_inertia * basis.transpose(),
    }
}

#[derive(Clone, Copy)]
struct PartMassProperties {
    mass: f32,
    local_center: Vec3,
    /// Inertia about `local_center`, in the part's local frame. A shaped part
    /// has products of inertia, so this cannot be a diagonal.
    local_inertia: Mat3,
}

/// Resolves authored fixed-size parts to the cuboids physics simulates.
fn physical_spec(spec: PartSpec) -> PartSpec {
    match spec {
        PartSpec::Controller(controller) => PartSpec::Cuboid(controller.cuboid()),
        PartSpec::Engine(engine) => PartSpec::Cuboid(engine.cuboid()),
        PartSpec::Transmission(transmission) => PartSpec::Cuboid(transmission.cuboid()),
        PartSpec::Servo(servo) => PartSpec::Cuboid(servo.cuboid()),
        PartSpec::Seat(seat) => PartSpec::Cuboid(seat.cuboid()),
        PartSpec::Input(input) => PartSpec::Cuboid(input.cuboid()),
        PartSpec::DimensionLink(link) => PartSpec::Cuboid(link.cuboid()),
        other => other,
    }
}

fn part_mass_properties(spec: PartSpec) -> PartMassProperties {
    match spec {
        PartSpec::Cuboid(spec) => {
            cuboid_mass_properties(spec, spec.material.properties().density_kg_m3)
        }
        PartSpec::Cylinder(spec) => {
            // An annular sector about local Y. Layered cylinders take their mass
            // from the evaluated bands instead.
            let length = spec.dimensions.axial_length();
            let sweep = spec.dimensions.sweep_angle_radians();
            let outer = spec.dimensions.outer_diameter() * 0.5;
            let inner = spec.dimensions.inner_diameter() * 0.5;
            let radial_squared = outer * outer + inner * inner;
            let mass = spec.material.properties().density_kg_m3
                * sweep
                * (outer * outer - inner * inner)
                * length
                * 0.5;
            let center_x = 4.0 * (sweep * 0.5).sin() * (outer.powi(3) - inner.powi(3))
                / (3.0 * sweep * (outer * outer - inner * inner));
            let radial_parallel = radial_squared * (sweep + sweep.sin()) / (4.0 * sweep);
            let radial_perpendicular = radial_squared * (sweep - sweep.sin()) / (4.0 * sweep);
            let axial_variance = length * length / 12.0;
            let shift = center_x * center_x;
            PartMassProperties {
                mass,
                local_center: Vec3::new(center_x, 0.0, 0.0),
                local_inertia: Mat3::from_diagonal(
                    mass * Vec3::new(
                        axial_variance + radial_perpendicular,
                        radial_parallel + radial_perpendicular - shift,
                        radial_parallel + axial_variance - shift,
                    ),
                ),
            }
        }
        PartSpec::PipeBend(spec) => pipe_bend_mass_properties(spec),
        PartSpec::PipeJunction(spec) => pipe_junction_mass_properties(spec),
        PartSpec::Controller(controller) => {
            cuboid_mass_properties(controller.cuboid(), MACHINE_PART_DENSITY_KG_M3)
        }
        PartSpec::Engine(engine) => {
            cuboid_mass_properties(engine.cuboid(), MACHINE_PART_DENSITY_KG_M3)
        }
        PartSpec::Transmission(transmission) => {
            cuboid_mass_properties(transmission.cuboid(), MACHINE_PART_DENSITY_KG_M3)
        }
        PartSpec::Servo(servo) => {
            cuboid_mass_properties(servo.cuboid(), MACHINE_PART_DENSITY_KG_M3)
        }
        PartSpec::Seat(seat) => cuboid_mass_properties(seat.cuboid(), MACHINE_PART_DENSITY_KG_M3),
        PartSpec::Input(input) => {
            cuboid_mass_properties(input.cuboid(), MACHINE_PART_DENSITY_KG_M3)
        }
        PartSpec::DimensionLink(link) => {
            cuboid_mass_properties(link.cuboid(), MACHINE_PART_DENSITY_KG_M3)
        }
    }
}

fn pipe_bend_mass_properties(spec: crate::PipeBendSpec) -> PartMassProperties {
    let outer = spec.dimensions.outer_diameter() * 0.5;
    let inner = spec.dimensions.inner_diameter() * 0.5;
    let radius = spec.dimensions.radius();
    let sweep = core::f32::consts::FRAC_PI_2;
    let radial_square_sum = outer * outer + inner * inner;
    let volume = sweep * core::f32::consts::PI * radius * (outer * outer - inner * inner);
    let mass = spec.material.properties().density_kg_m3 * volume;

    // Integrate the torus volume element `(R + rho cos(phi)) rho d(rho)d(phi)d(theta)`.
    // `mean_q` and `mean_q_squared` are the first two centre-of-curvature
    // radial moments of the swept annulus. Symmetry then gives the complete
    // covariance over the quarter turn, including the XY product of inertia.
    let mean_q = radius + radial_square_sum / (4.0 * radius);
    let mean_q_squared = radius * radius + 0.75 * radial_square_sum;
    let mean_x = 2.0 * mean_q / core::f32::consts::PI;
    let mean_y = -mean_x;
    let planar_variance = mean_q_squared * 0.5 - mean_x * mean_x;
    let planar_covariance = -mean_q_squared / core::f32::consts::PI - mean_x * mean_y;
    let z_variance = radial_square_sum * 0.25;
    let diagonal_xy = mass * (planar_variance + z_variance);
    let product_xy = -mass * planar_covariance;

    PartMassProperties {
        mass,
        local_center: Vec3::new(-radius + mean_x, radius + mean_y, 0.0),
        local_inertia: Mat3::from_cols(
            Vec3::new(diagonal_xy, product_xy, 0.0),
            Vec3::new(product_xy, diagonal_xy, 0.0),
            Vec3::new(0.0, 0.0, mass * planar_variance * 2.0),
        ),
    }
}

/// Integrates a junction's sampled solid: a pyramid from its centre to each
/// outer triangle, minus the matching pyramid to the bore.
#[allow(clippy::cast_possible_truncation)] // Metre-scale fittings fit f32 mass properties.
fn pipe_junction_mass_properties(spec: crate::PipeJunctionSpec) -> PartMassProperties {
    use bevy_math::{DMat3, DVec3};
    let density = f64::from(spec.material.properties().density_kg_m3);
    let covariance =
        |point: DVec3| DMat3::from_cols(point * point.x, point * point.y, point * point.z);
    let mut mass = 0.0_f64;
    let mut first_moment = DVec3::ZERO;
    let mut second_moment = DMat3::ZERO;
    for triangle in crate::pipe_junction::ray_triangles(spec) {
        for (corners, sign) in [(triangle.outer, 1.0), (triangle.inner, -1.0)] {
            let [a, b, c] = corners;
            let pyramid_mass = sign * density * a.dot(b.cross(c)).abs() / 6.0;
            let sum = a + b + c;
            mass += pyramid_mass;
            first_moment += sum * (pyramid_mass / 4.0);
            second_moment += (covariance(a) + covariance(b) + covariance(c) + covariance(sum))
                * (pyramid_mass / 20.0);
        }
    }
    let trace = second_moment.x_axis.x + second_moment.y_axis.y + second_moment.z_axis.z;
    let origin_inertia = (DMat3::IDENTITY * trace - second_moment).as_mat3();
    let local_center = (first_moment / mass).as_vec3();
    let mass = mass as f32;
    PartMassProperties {
        mass,
        local_center,
        local_inertia: origin_inertia - shifted_inertia(local_center, mass),
    }
}

/// Parallel-axis term moving an inertia tensor `offset` away from its centre.
fn shifted_inertia(offset: Vec3, mass: f32) -> Mat3 {
    (Mat3::IDENTITY * offset.length_squared()
        - Mat3::from_cols(offset * offset.x, offset * offset.y, offset * offset.z))
        * mass
}

fn cuboid_mass_properties(spec: CuboidSpec, density_kg_m3: f32) -> PartMassProperties {
    let size = spec.size_meters();
    let mass = density_kg_m3 * size.x * size.y * size.z;
    PartMassProperties {
        mass,
        local_center: Vec3::ZERO,
        local_inertia: Mat3::from_diagonal(Vec3::new(
            mass * (size.y * size.y + size.z * size.z) / 12.0,
            mass * (size.x * size.x + size.z * size.z) / 12.0,
            mass * (size.x * size.x + size.y * size.y) / 12.0,
        )),
    }
}

/// Exact mass, centre of mass, and inertia of a shaped region.
///
/// Every piece is integrated over its own closed surface by the divergence
/// theorem, fanning each face into tetrahedra from one reference point. Signed
/// volumes make the sum independent of where that reference sits.
///
/// For a simplex, `∫ x⊗x dV = (V/20)(Σᵢ wᵢ⊗wᵢ + (Σᵢ wᵢ)⊗(Σᵢ wᵢ))`.
#[allow(clippy::cast_precision_loss)]
fn region_world_mass(region: &ShapeRegion) -> WorldMassProperties {
    let mut volume = 0.0_f32;
    let mut first_moment = Vec3::ZERO;
    let mut second_moment = Mat3::ZERO;
    for piece in region_pieces(region) {
        match piece {
            PartPiece::Cuboid {
                center,
                half_extents,
                rotation,
                ..
            } => {
                let size = half_extents * 2.0;
                let box_volume = size.x * size.y * size.z;
                let basis = Mat3::from_quat(rotation);
                let local = Mat3::from_diagonal(
                    Vec3::new(size.x * size.x, size.y * size.y, size.z * size.z)
                        * (box_volume / 12.0),
                );
                volume += box_volume;
                first_moment += center * box_volume;
                second_moment +=
                    basis * local * basis.transpose() + outer_product(center, center) * box_volume;
            }
            PartPiece::Convex(convex) => {
                accumulate_convex_moments(
                    &convex,
                    &mut volume,
                    &mut first_moment,
                    &mut second_moment,
                );
            }
        }
    }

    let density = region.material().properties().density_kg_m3;
    let mass = density * volume;
    let center = if volume.abs() > f32::EPSILON {
        first_moment / volume
    } else {
        Vec3::ZERO
    };
    let about_center = second_moment - outer_product(center, center) * volume;
    WorldMassProperties {
        mass,
        center,
        inertia: (Mat3::IDENTITY * trace(about_center) - about_center) * density,
    }
}

/// Mass of an evaluated solid whose cells may belong to different material
/// bands; `density` maps a cell's band to kilograms per cubic metre.
fn evaluated_world_mass(
    solid: &crate::EvaluatedSolid,
    density: impl Fn(u8) -> f32,
) -> WorldMassProperties {
    let mut mass = 0.0_f32;
    let mut first_moment = Vec3::ZERO;
    let mut second_moment = Mat3::ZERO;
    for cell in &solid.cells {
        let mut volume = 0.0_f32;
        let mut cell_first = Vec3::ZERO;
        let mut cell_second = Mat3::ZERO;
        accumulate_convex_moments(&cell.piece, &mut volume, &mut cell_first, &mut cell_second);
        let density = density(cell.band);
        mass += density * volume;
        first_moment += cell_first * density;
        second_moment += cell_second * density;
    }
    let center = first_moment / mass;
    let about_center = second_moment - outer_product(center, center) * mass;
    WorldMassProperties {
        mass,
        center,
        inertia: Mat3::IDENTITY * trace(about_center) - about_center,
    }
}

/// The convex pieces one region's cage describes.
fn region_pieces(region: &ShapeRegion) -> Vec<PartPiece> {
    let grid = region.grid();
    decompose(&grid, &|cell, corner| region.corner_steps(cell, corner))
}

fn accumulate_convex_moments(
    piece: &ConvexPiece,
    volume: &mut f32,
    first_moment: &mut Vec3,
    second_moment: &mut Mat3,
) {
    let origin = piece.vertices[0];
    for face in &piece.faces {
        for index in 1..face.indices.len() - 1 {
            let a = piece.vertices[face.indices[0] as usize];
            let b = piece.vertices[face.indices[index] as usize];
            let c = piece.vertices[face.indices[index + 1] as usize];
            let signed = (a - origin).dot((b - origin).cross(c - origin)) / 6.0;
            if signed == 0.0 {
                continue;
            }
            let corners = [origin, a, b, c];
            let sum = corners.iter().copied().sum::<Vec3>();
            let squares = corners
                .iter()
                .map(|&corner| outer_product(corner, corner))
                .fold(Mat3::ZERO, |total, term| total + term);
            *volume += signed;
            *first_moment += sum * (signed / 4.0);
            *second_moment += (squares + outer_product(sum, sum)) * (signed / 20.0);
        }
    }
}

fn outer_product(left: Vec3, right: Vec3) -> Mat3 {
    Mat3::from_cols(left * right.x, left * right.y, left * right.z)
}

fn trace(matrix: Mat3) -> f32 {
    matrix.x_axis.x + matrix.y_axis.y + matrix.z_axis.z
}

#[derive(Clone, Debug)]
struct CompactCuboid {
    source_part: PartId,
    compound_index: u32,
    center: Vec3,
    half_extents: Vec3,
    material: MaterialProperties,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct CompactCuboidKey {
    material: [u32; 6],
    other_bounds: [i64; 4],
    along_minimum: i64,
}

/// Greedily merges touching, axis-aligned boxes within one rigid body and one
/// exact material response. Authored mass properties were already integrated
/// from source geometry, so this changes only the collision decomposition.
fn compact_grid_aligned_cuboids(colliders: &mut Vec<LocalCollider>, start: usize) {
    let original = colliders.drain(start..).collect::<Vec<_>>();
    let mut rows_by_source = BTreeMap::<PartId, usize>::new();
    for collider in &original {
        *rows_by_source.entry(collider.source_part).or_default() += 1;
    }
    let mut boxes = Vec::new();
    let mut preserved = Vec::new();
    for collider in original {
        // Multi-row source geometry includes cylinders, pipe bends, and shaped
        // regions. Keep each authored decomposition contiguous and untouched.
        if rows_by_source[&collider.source_part] != 1 {
            preserved.push(collider);
            continue;
        }
        let ColliderShape::Cuboid {
            local_rotation,
            half_extents,
        } = collider.shape
        else {
            preserved.push(collider);
            continue;
        };
        let Some(axis_aligned_half_extents) =
            axis_aligned_half_extents(local_rotation, half_extents)
        else {
            preserved.push(LocalCollider {
                shape: ColliderShape::Cuboid {
                    local_rotation,
                    half_extents,
                },
                ..collider
            });
            continue;
        };
        boxes.push(CompactCuboid {
            source_part: collider.source_part,
            compound_index: collider.compound_index,
            center: collider.local_center,
            half_extents: axis_aligned_half_extents,
            material: collider.material_properties,
        });
    }

    for axis in 0..3 {
        boxes.sort_by_key(|cuboid| compact_cuboid_key(cuboid, axis));
        let mut merged: Vec<CompactCuboid> = Vec::with_capacity(boxes.len());
        for cuboid in boxes.drain(..) {
            let can_merge = merged.last().is_some_and(|previous| {
                same_compaction_lane(previous, &cuboid, axis)
                    && (cuboid_minimum(&cuboid, axis) - cuboid_maximum(previous, axis)).abs()
                        <= 1.0e-6
            });
            if can_merge {
                let previous = merged.last_mut().expect("merge candidate exists");
                let minimum = cuboid_minimum(previous, axis);
                let maximum = cuboid_maximum(&cuboid, axis);
                previous.center[axis] = (minimum + maximum) * 0.5;
                previous.half_extents[axis] = (maximum - minimum) * 0.5;
                previous.source_part = previous.source_part.min(cuboid.source_part);
            } else {
                merged.push(cuboid);
            }
        }
        boxes = merged;
    }

    colliders.extend(boxes.into_iter().map(|cuboid| LocalCollider {
        source_part: cuboid.source_part,
        compound_index: cuboid.compound_index,
        local_center: cuboid.center,
        material_properties: cuboid.material,
        shape: ColliderShape::Cuboid {
            local_rotation: Quat::IDENTITY,
            half_extents: cuboid.half_extents,
        },
    }));
    colliders.extend(preserved);
}

fn axis_aligned_half_extents(rotation: Quat, half_extents: Vec3) -> Option<Vec3> {
    let rotated_axes = [rotation * Vec3::X, rotation * Vec3::Y, rotation * Vec3::Z];
    let mut used_world_axes = 0_u8;
    for axis in rotated_axes {
        let absolute = axis.abs();
        let (world_axis, largest) = [absolute.x, absolute.y, absolute.z]
            .into_iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.total_cmp(right))?;
        if largest < 1.0 - 1.0e-5 || used_world_axes & (1 << world_axis) != 0 {
            return None;
        }
        used_world_axes |= 1 << world_axis;
    }
    Some(
        rotated_axes[0].abs() * half_extents.x
            + rotated_axes[1].abs() * half_extents.y
            + rotated_axes[2].abs() * half_extents.z,
    )
}

fn material_key(material: MaterialProperties) -> [u32; 6] {
    [
        material.density_kg_m3.to_bits(),
        material.static_friction.to_bits(),
        material.dynamic_friction.to_bits(),
        material.restitution.to_bits(),
        material.rolling_resistance.to_bits(),
        material.youngs_modulus_pa.to_bits(),
    ]
}

#[allow(clippy::cast_possible_truncation)]
fn quantized_coordinate(value: f32) -> i64 {
    f64::from(value).mul_add(1_000_000.0, 0.0).round() as i64
}

fn compact_cuboid_key(cuboid: &CompactCuboid, axis: usize) -> CompactCuboidKey {
    let other_axes = match axis {
        0 => [1, 2],
        1 => [0, 2],
        _ => [0, 1],
    };
    CompactCuboidKey {
        material: material_key(cuboid.material),
        other_bounds: [
            quantized_coordinate(cuboid_minimum(cuboid, other_axes[0])),
            quantized_coordinate(cuboid_maximum(cuboid, other_axes[0])),
            quantized_coordinate(cuboid_minimum(cuboid, other_axes[1])),
            quantized_coordinate(cuboid_maximum(cuboid, other_axes[1])),
        ],
        along_minimum: quantized_coordinate(cuboid_minimum(cuboid, axis)),
    }
}

fn same_compaction_lane(left: &CompactCuboid, right: &CompactCuboid, axis: usize) -> bool {
    left.compound_index == right.compound_index
        && material_key(left.material) == material_key(right.material)
        && (0..3).filter(|&other| other != axis).all(|other| {
            (cuboid_minimum(left, other) - cuboid_minimum(right, other)).abs() <= 1.0e-6
                && (cuboid_maximum(left, other) - cuboid_maximum(right, other)).abs() <= 1.0e-6
        })
}

fn cuboid_minimum(cuboid: &CompactCuboid, axis: usize) -> f32 {
    cuboid.center[axis] - cuboid.half_extents[axis]
}

fn cuboid_maximum(cuboid: &CompactCuboid, axis: usize) -> f32 {
    cuboid.center[axis] + cuboid.half_extents[axis]
}

const AUTHORED_CONTACT_PROPERTIES: MaterialProperties = MaterialProperties {
    density_kg_m3: MACHINE_PART_DENSITY_KG_M3,
    static_friction: 0.05,
    dynamic_friction: 0.05,
    restitution: 0.0,
    rolling_resistance: 0.0,
    youngs_modulus_pa: 200.0e9,
};

/// Contact material of one evaluated cell band. Only layered parts have more
/// than band zero; authored parts answer with their fixed properties.
fn band_contact_properties(spec: PartSpec, band: u8) -> MaterialProperties {
    spec.band(band).map_or(
        MaterialProperties {
            density_kg_m3: MACHINE_PART_DENSITY_KG_M3,
            ..AUTHORED_CONTACT_PROPERTIES
        },
        |(material, _)| material.properties(),
    )
}

fn contact_properties(spec: PartSpec) -> MaterialProperties {
    match spec {
        PartSpec::Cuboid(cuboid) => cuboid.material.properties(),
        PartSpec::Cylinder(cylinder) => cylinder.outer_contact_material().properties(),
        PartSpec::PipeBend(bend) => bend.material.properties(),
        PartSpec::PipeJunction(junction) => junction.material.properties(),
        PartSpec::Controller(_)
        | PartSpec::Engine(_)
        | PartSpec::Transmission(_)
        | PartSpec::Servo(_)
        | PartSpec::Seat(_)
        | PartSpec::Input(_)
        | PartSpec::DimensionLink(_) => AUTHORED_CONTACT_PROPERTIES,
    }
}

/// Composes raw grid geometry once, then rebases it onto the compiled root.
fn compose_raw_colliders(
    colliders: &mut [LocalCollider],
    frame: crate::ConstructionFrame,
    center_of_mass: Vec3,
) {
    let translation = frame.translation() - center_of_mass;
    for collider in colliders {
        collider.local_center = frame.vector(collider.local_center) + translation;
        match &mut collider.shape {
            ColliderShape::Cuboid { local_rotation, .. } => {
                *local_rotation = frame.rotation() * *local_rotation;
            }
            ColliderShape::Convex(convex) => {
                for vertex in &mut convex.vertices {
                    *vertex = frame.vector(*vertex) + translation;
                }
                for plane in &mut convex.face_planes {
                    let normal = frame.vector(plane.truncate());
                    *plane = normal.extend(plane.w + normal.dot(translation));
                }
                for direction in &mut convex.edge_directions {
                    *direction = frame.vector(*direction);
                }
            }
        }
    }
}

// Recovers the analytic cylinder behind a freshly emitted and rebased box run.
// The first box faces the cylinder's own zero angle, so its rotation is the
// cylinder's, its half-extents carry the radius and axial length, and its centre
// is one radius out along the radial axis. Shaped or hollow cylinders and sectors
// have no such description and are left to their boxes.
fn solid_full_cylinder(
    spec: PartSpec,
    part: PartId,
    compound_index: u32,
    first_collider: usize,
    run: &[LocalCollider],
) -> Option<CompiledCylinder> {
    let PartSpec::Cylinder(cylinder) = spec else {
        return None;
    };
    if cylinder.dimensions.inner_diameter() != 0.0
        || cylinder.dimensions.sweep_angle_degrees() != 360
        || run.len() != CYLINDER_COLLIDER_COUNT
    {
        return None;
    }
    let ColliderShape::Cuboid {
        local_rotation,
        half_extents,
    } = run[0].shape
    else {
        return None;
    };
    Some(CompiledCylinder {
        source_part: part,
        compound_index,
        first_collider: u32::try_from(first_collider).expect("collider rows fit u32"),
        local_center: run[0].local_center - (local_rotation * Vec3::X) * half_extents.x,
        local_rotation,
        outer_radius: half_extents.x * 2.0,
        half_length: half_extents.y,
    })
}

fn append_part_colliders(
    colliders: &mut Vec<LocalCollider>,
    part: PartId,
    compound_index: u32,
    spec: PartSpec,
    center_of_mass: Vec3,
) {
    let material_properties = contact_properties(spec);
    match physical_spec(spec) {
        PartSpec::Cuboid(spec) => {
            for piece in decompose_part(spec) {
                colliders.push(match piece {
                    PartPiece::Cuboid {
                        center,
                        half_extents,
                        rotation,
                        ..
                    } => LocalCollider {
                        source_part: part,
                        compound_index,
                        local_center: center - center_of_mass,
                        material_properties,
                        shape: ColliderShape::Cuboid {
                            local_rotation: rotation,
                            half_extents,
                        },
                    },
                    PartPiece::Convex(convex) => LocalCollider {
                        source_part: part,
                        compound_index,
                        local_center: convex.centroid - center_of_mass,
                        material_properties,
                        shape: ColliderShape::Convex(compile_convex(&convex, center_of_mass)),
                    },
                });
            }
        }
        PartSpec::Cylinder(spec) => {
            let outer = spec.dimensions.outer_diameter() * 0.5;
            let inner = spec.dimensions.inner_diameter() * 0.5;
            let half_radial = (outer - inner) * 0.5;
            let center_radius = (outer + inner) * 0.5;
            let sweep = spec.dimensions.sweep_angle_radians();
            let segment_angle = sweep / 16.0;
            let half_tangent = outer * (segment_angle * 0.5).tan();
            let start_angle = if spec.dimensions.sweep_angle_degrees() == 360 {
                -segment_angle * 0.5
            } else {
                -sweep * 0.5
            };
            let part_rotation = spec.pose.rotation.quaternion();
            for segment in 0_u16..16 {
                let angle = start_angle + segment_angle * (f32::from(segment) + 0.5);
                let radial = Vec3::new(angle.cos(), 0.0, angle.sin());
                colliders.push(LocalCollider {
                    source_part: part,
                    compound_index,
                    local_center: spec.pose.translation() - center_of_mass
                        + part_rotation * (radial * center_radius),
                    material_properties,
                    shape: ColliderShape::Cuboid {
                        local_rotation: part_rotation * Quat::from_rotation_y(-angle),
                        half_extents: Vec3::new(
                            half_radial,
                            spec.dimensions.axial_length() * 0.5,
                            half_tangent,
                        ),
                    },
                });
            }
        }
        PartSpec::PipeBend(spec) => append_pipe_bend_colliders(
            colliders,
            part,
            compound_index,
            spec,
            center_of_mass,
            material_properties,
        ),
        PartSpec::PipeJunction(spec) => {
            let part_rotation = spec.pose.rotation.quaternion();
            for wall in crate::pipe_junction_wall_boxes(spec) {
                colliders.push(LocalCollider {
                    source_part: part,
                    compound_index,
                    local_center: spec.pose.translation() - center_of_mass
                        + part_rotation * wall.center,
                    material_properties,
                    shape: ColliderShape::Cuboid {
                        local_rotation: part_rotation * wall.rotation,
                        half_extents: wall.half_extents,
                    },
                });
            }
        }
        PartSpec::Controller(_)
        | PartSpec::Engine(_)
        | PartSpec::Transmission(_)
        | PartSpec::Servo(_)
        | PartSpec::Seat(_)
        | PartSpec::Input(_)
        | PartSpec::DimensionLink(_) => {
            unreachable!("fixed-size authored parts resolve to cuboids")
        }
    }
}

fn append_pipe_bend_colliders(
    colliders: &mut Vec<LocalCollider>,
    part: PartId,
    compound_index: u32,
    spec: crate::PipeBendSpec,
    center_of_mass: Vec3,
    material_properties: MaterialProperties,
) {
    let outer = spec.dimensions.outer_diameter() * 0.5;
    let inner = spec.dimensions.inner_diameter() * 0.5;
    let half_radial = (outer - inner) * 0.5;
    let cross_radius = (outer + inner) * 0.5;
    let bend_radius = spec.dimensions.radius();
    let bend_step = core::f32::consts::FRAC_PI_2 / 12.0;
    let cross_step = core::f32::consts::TAU / 16.0;
    let half_bend_tangent = (bend_radius + outer) * (bend_step * 0.5).tan();
    let half_cross_tangent = outer * (cross_step * 0.5).tan();
    let part_rotation = spec.pose.rotation.quaternion();
    let corner = spec.pose.translation();
    for bend_slice in 0_u16..12 {
        let theta = -core::f32::consts::FRAC_PI_2 + bend_step * (f32::from(bend_slice) + 0.5);
        let radial = Vec3::new(theta.cos(), theta.sin(), 0.0);
        let tangent = Vec3::new(-theta.sin(), theta.cos(), 0.0);
        for sector in 0_u16..16 {
            let phi = cross_step * (f32::from(sector) + 0.5);
            let normal = radial * phi.cos() + Vec3::Z * phi.sin();
            let cross_tangent = -radial * phi.sin() + Vec3::Z * phi.cos();
            let local_center = Vec3::new(-bend_radius, bend_radius, 0.0)
                + radial * (bend_radius + cross_radius * phi.cos())
                + Vec3::Z * (cross_radius * phi.sin());
            let local_basis = Mat3::from_cols(normal, tangent, cross_tangent);
            colliders.push(LocalCollider {
                source_part: part,
                compound_index,
                local_center: corner - center_of_mass + part_rotation * local_center,
                material_properties,
                shape: ColliderShape::Cuboid {
                    local_rotation: part_rotation * Quat::from_mat3(&local_basis),
                    half_extents: Vec3::new(half_radial, half_bend_tangent, half_cross_tangent),
                },
            });
        }
    }
}

/// Emits one region's colliders, which stand in for every block it covers.
fn append_region_colliders(
    colliders: &mut Vec<LocalCollider>,
    region_id: RegionId,
    region: &ShapeRegion,
    compound_index: u32,
    center_of_mass: Vec3,
    source_part: PartId,
) {
    let _ = region_id;
    let material_properties = region.material().properties();
    for piece in region_pieces(region) {
        colliders.push(match piece {
            PartPiece::Cuboid {
                center,
                half_extents,
                rotation,
                ..
            } => LocalCollider {
                source_part,
                compound_index,
                local_center: center - center_of_mass,
                material_properties,
                shape: ColliderShape::Cuboid {
                    local_rotation: rotation,
                    half_extents,
                },
            },
            PartPiece::Convex(convex) => LocalCollider {
                source_part,
                compound_index,
                local_center: convex.centroid - center_of_mass,
                material_properties,
                shape: ColliderShape::Convex(compile_convex(&convex, center_of_mass)),
            },
        });
    }
}

fn append_evaluated_colliders(
    colliders: &mut Vec<LocalCollider>,
    solid: &crate::EvaluatedSolid,
    source_part: PartId,
    compound_index: u32,
    center_of_mass: Vec3,
    material_properties: impl Fn(u8) -> MaterialProperties,
) {
    colliders.extend(solid.cells.iter().map(|cell| LocalCollider {
        source_part,
        compound_index,
        local_center: cell.piece.centroid - center_of_mass,
        material_properties: material_properties(cell.band),
        shape: ColliderShape::Convex(compile_convex(&cell.piece, center_of_mass)),
    }));
}

/// Rebases one decomposed piece onto the compound centre of mass.
fn compile_convex(piece: &ConvexPiece, center_of_mass: Vec3) -> CompiledConvex {
    CompiledConvex {
        vertices: piece
            .vertices
            .iter()
            .map(|vertex| *vertex - center_of_mass)
            .collect(),
        face_planes: piece
            .faces
            .iter()
            .map(|face| {
                // Shifting the origin moves a plane's offset by the normal's
                // component along the shift.
                face.normal
                    .extend(face.offset - face.normal.dot(center_of_mass))
            })
            .collect(),
        edge_directions: piece.edge_directions.clone(),
    }
}

const fn ordered_pair(a: u32, b: u32) -> [u32; 2] {
    if a < b { [a, b] } else { [b, a] }
}

#[derive(Clone, Debug)]
struct DisjointSet {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl DisjointSet {
    fn new(len: usize) -> Self {
        Self {
            parent: (0..len).collect(),
            rank: vec![0; len],
        }
    }

    fn find(&mut self, mut item: usize) -> usize {
        let mut root = item;
        while self.parent[root] != root {
            root = self.parent[root];
        }
        while self.parent[item] != item {
            let parent = self.parent[item];
            self.parent[item] = root;
            item = parent;
        }
        root
    }

    fn union(&mut self, a: usize, b: usize) {
        let mut root_a = self.find(a);
        let mut root_b = self.find(b);
        if root_a == root_b {
            return;
        }
        if self.rank[root_a] < self.rank[root_b] {
            core::mem::swap(&mut root_a, &mut root_b);
        }
        self.parent[root_b] = root_a;
        if self.rank[root_a] == self.rank[root_b] {
            self.rank[root_a] += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use bevy_math::{IVec3, Vec3};

    use super::{PartPiece, outer_product};
    use crate::{
        ActuatorAssignment, BearingDimensions, BearingId, BearingSpec, BuildCommand, BuildOutcome,
        BuildPose, ColliderShape, ConstructionGraph, ConstructionMaterial, ControllerSpec,
        CoordinateDrive, CuboidSpec, CylinderDimensions, CylinderSpec, DriveLimits, DriveLinkSpec,
        DriveMode, DriveProgram, DriveState, DriveTarget, EdgeChainRef, EdgeTreatment, EngineKind,
        EngineSpec, FaceKind, FaceRef, GearSelection, GridRotation, PIPE_BEND_COLLIDER_COUNT,
        PartId, PartSpec, PipeBendDimensions, PipeBendSpec, RigidLinkSpec, ShapeFeature,
        SolidOwner, TopologyError, WeldSpec,
    };
    use bevy_math::{Mat3, Quat};

    #[test]
    fn unattached_suspension_socket_adds_carried_mass_center_and_inertia_once() {
        let mut graph = ConstructionGraph::new();
        let source = spawn(&mut graph, IVec3::new(0, 2, 0));
        let spec = crate::SuspensionSpec::new(
            Some(crate::SpringSpec::default()),
            Some(crate::ShockSpec::default()),
            None,
        )
        .unwrap();
        let socket = crate::BearingSocket {
            kind: crate::BearingKind::Suspension(spec),
            axis: Vec3::X,
            source: FaceRef::part(source, FaceKind::PositiveX),
            anchor: Vec3::new(0.5, 0.5, 0.0),
            dimensions: BearingDimensions::default(),
        };
        let bare = graph.compile().unwrap().compounds[0].mass_properties;
        let compiled = graph
            .compile_with_suspension_sockets([], &[socket])
            .unwrap();
        let actual = compiled.compounds[0].mass_properties;
        let elements = spec.mass_elements();
        let added_mass: f32 = elements.iter().map(|element| element.mass).sum();
        let total_mass = bare.mass + added_mass;
        let expected_center = (bare.center_of_mass * bare.mass
            + elements
                .iter()
                .map(|element| (socket.anchor + socket.axis * element.center) * element.mass)
                .sum::<Vec3>())
            / total_mass;
        assert!((actual.mass - total_mass).abs() < 0.001);
        assert!(actual.center_of_mass.abs_diff_eq(expected_center, 1.0e-6));
        let shift = bare.center_of_mass - expected_center;
        let mut expected_inertia = bare.inertia
            + bare.mass * (Mat3::IDENTITY * shift.length_squared() - outer_product(shift, shift));
        for element in elements {
            let offset = socket.anchor + socket.axis * element.center - expected_center;
            expected_inertia += Mat3::from_diagonal(Vec3::new(
                element.axial_inertia,
                element.transverse_inertia,
                element.transverse_inertia,
            )) + element.mass
                * (Mat3::IDENTITY * offset.length_squared() - outer_product(offset, offset));
        }
        assert!(actual.inertia.abs_diff_eq(expected_inertia, 0.001));
        assert_eq!(
            compiled,
            graph
                .compile_with_suspension_sockets([], &[socket, socket])
                .unwrap()
        );
        let anchored = graph
            .compile_with_suspension_sockets([source], &[socket])
            .unwrap();
        assert!(anchored.compounds[0].is_static);
        assert!(anchored.compounds[0].mass_properties.inverse_mass.abs() < f32::EPSILON);
        assert!((anchored.compounds[0].mass_properties.mass - total_mass).abs() < 0.001);
    }

    #[test]
    fn attached_suspension_socket_does_not_duplicate_either_endpoint_mass() {
        let mut graph = ConstructionGraph::new();
        let source = spawn(&mut graph, IVec3::new(0, 2, 0));
        let target = spawn(&mut graph, IVec3::new(6, 2, 0));
        let bare_mass: f32 = graph
            .compile()
            .unwrap()
            .compounds
            .iter()
            .map(|body| body.mass_properties.mass)
            .sum();
        let spec = crate::SuspensionSpec::new(
            Some(crate::SpringSpec::default()),
            Some(crate::ShockSpec::default()),
            None,
        )
        .unwrap();
        let socket = crate::BearingSocket {
            kind: crate::BearingKind::Suspension(spec),
            axis: Vec3::X,
            source: FaceRef::part(source, FaceKind::PositiveX),
            anchor: Vec3::new(0.5, 0.5, 0.0),
            dimensions: BearingDimensions::default(),
        };
        graph
            .apply(BuildCommand::AddBearing(
                BearingSpec::new(
                    socket.source,
                    FaceRef::part(target, FaceKind::NegativeX),
                    socket.anchor,
                    socket.axis,
                )
                .with_kind(socket.kind),
            ))
            .unwrap();
        let attached = graph.compile().unwrap();
        assert_eq!(
            attached,
            graph
                .compile_with_suspension_sockets([], &[socket])
                .unwrap()
        );
        let total_mass: f32 = attached
            .compounds
            .iter()
            .map(|body| body.mass_properties.mass)
            .sum();
        let added_mass: f32 = spec
            .mass_elements()
            .iter()
            .map(|element| element.mass)
            .sum();
        assert!((total_mass - bare_mass - added_mass).abs() < 0.001);
    }

    fn cube_at(units: IVec3) -> CuboidSpec {
        CuboidSpec::new([4, 4, 4], BuildPose::new(units, GridRotation::default())).unwrap()
    }

    fn assert_frame_compilation(mut graph: ConstructionGraph) {
        let original = graph.compile().unwrap();
        let frame = crate::ConstructionFrame::new(
            Vec3::new(2.1, -0.8, 1.3),
            Quat::from_rotation_y(0.61) * Quat::from_rotation_x(-0.37),
        )
        .unwrap();
        let parts = graph.parts().map(|(part, _)| part).collect::<Vec<_>>();
        graph.reframe_parts(parts, frame).unwrap();
        let compiled = graph.compile().unwrap();
        let basis = Mat3::from_quat(frame.rotation());
        assert_eq!(original.compounds.len(), compiled.compounds.len());
        for (old, new) in original.compounds.iter().zip(&compiled.compounds) {
            assert!(
                new.root_translation
                    .abs_diff_eq(frame.point(old.root_translation), 1.0e-4)
            );
            assert!(
                (old.mass_properties.mass - new.mass_properties.mass).abs()
                    < old.mass_properties.mass * 1.0e-4
            );
            let expected = basis * old.mass_properties.inertia * basis.transpose();
            let tolerance = expected
                .to_cols_array()
                .into_iter()
                .map(f32::abs)
                .fold(1.0, f32::max)
                * 1.0e-4;
            assert!(new.mass_properties.inertia.abs_diff_eq(expected, tolerance));
        }
        assert_eq!(original.colliders.len(), compiled.colliders.len());
        for old in &original.colliders {
            let center = frame.vector(old.local_center);
            let new = compiled
                .colliders
                .iter()
                .find(|new| new.local_center.abs_diff_eq(center, 1.0e-4))
                .expect("every original collider retains its transformed centroid");
            match (&old.shape, &new.shape) {
                (
                    ColliderShape::Cuboid {
                        local_rotation: old_rotation,
                        half_extents: old_half,
                    },
                    ColliderShape::Cuboid {
                        local_rotation: new_rotation,
                        half_extents: new_half,
                    },
                ) => {
                    assert!(new_rotation.abs_diff_eq(frame.rotation() * *old_rotation, 1.0e-5));
                    assert!(new_half.abs_diff_eq(*old_half, 1.0e-6));
                }
                (ColliderShape::Convex(old), ColliderShape::Convex(new)) => {
                    for vertex in &old.vertices {
                        assert!(
                            new.vertices
                                .iter()
                                .any(|new| new.abs_diff_eq(frame.vector(*vertex), 1.0e-4))
                        );
                    }
                    for plane in &old.face_planes {
                        let expected = frame.vector(plane.truncate()).extend(plane.w);
                        assert!(
                            new.face_planes
                                .iter()
                                .any(|new| new.abs_diff_eq(expected, 1.0e-4))
                        );
                    }
                    for direction in &old.edge_directions {
                        let expected = frame.vector(*direction);
                        assert!(
                            new.edge_directions
                                .iter()
                                .any(|new| new.abs_diff_eq(expected, 1.0e-4)
                                    || new.abs_diff_eq(-expected, 1.0e-4))
                        );
                    }
                }
                _ => panic!("rigid framing must preserve collider shape"),
            }
        }
    }

    #[test]
    fn arbitrary_frame_rotates_cuboid_inertia_and_collider() {
        let mut graph = ConstructionGraph::new();
        graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new(
                    [2, 4, 6],
                    BuildPose::new(IVec3::new(4, 8, 12), GridRotation::default()),
                )
                .unwrap(),
            ))
            .unwrap();
        assert_frame_compilation(graph);
    }

    #[test]
    fn arbitrary_frame_composes_raw_region_convex_planes_and_mass() {
        let (mut graph, region) = region_over_one_block();
        let cell = i16::try_from(crate::POSITION_TICKS_PER_GRID_UNIT).unwrap();
        graph
            .apply(BuildCommand::SetRegionVertices {
                region,
                vertices: vec![([0, 1, 1], [0, -cell, 0]), ([1, 1, 1], [0, -cell, 0])],
            })
            .unwrap();
        assert_frame_compilation(graph);
    }

    #[test]
    fn arbitrary_frame_does_not_double_transform_evaluated_geometry() {
        let mut graph = ConstructionGraph::new();
        let part = spawn(&mut graph, IVec3::ZERO);
        let owner = SolidOwner::Part(part);
        let edge = graph.evaluated_solid(owner).unwrap().logical_edges[0].key;
        graph
            .apply(BuildCommand::AddShapeFeature(ShapeFeature::new(
                [EdgeChainRef { owner, edge }],
                EdgeTreatment::Chamfer,
                20,
            )))
            .unwrap();
        assert_frame_compilation(graph);
    }

    fn spawn(graph: &mut ConstructionGraph, units: IVec3) -> crate::PartId {
        let BuildOutcome::Spawned(id) = graph.apply(BuildCommand::Spawn(cube_at(units))).unwrap()
        else {
            panic!("wrong spawn outcome")
        };
        id
    }

    fn ground(graph: &mut ConstructionGraph, part: crate::PartId) {
        graph
            .apply(BuildCommand::Weld(WeldSpec {
                first: FaceRef::part(part, FaceKind::NegativeY),
                second: FaceRef::ground(),
            }))
            .unwrap();
    }

    #[test]
    fn a_solid_cylinder_also_compiles_an_exact_hull_sharing_every_corner() {
        let mut graph = ConstructionGraph::new();
        graph
            .apply(BuildCommand::SpawnCylinder(CylinderSpec::new(
                CylinderDimensions::new(0.95, 0.0, 0.25).unwrap(),
                BuildPose::from_position_ticks(IVec3::Y * 300, GridRotation::new(1, 0, 0)),
            )))
            .unwrap();
        let compiled = graph.compile().unwrap();
        assert_eq!(compiled.colliders.len(), super::CYLINDER_COLLIDER_COUNT);
        assert_eq!(compiled.cylinders.len(), 1);
        let cylinder = compiled.cylinders[0];
        assert_eq!(cylinder.first_collider, 0);
        assert!((cylinder.outer_radius - 0.475).abs() < 1.0e-6);
        assert!((cylinder.half_length - 0.125).abs() < 1.0e-6);

        // The hull is the union of the boxes, so every box corner lies on or
        // inside it and the facet midpoints sit at the authored radius.
        let hull = cylinder.hull();
        assert_eq!(hull.vertices.len(), 2 * super::CYLINDER_COLLIDER_COUNT);
        assert_eq!(hull.face_planes.len(), super::CYLINDER_COLLIDER_COUNT + 2);
        for plane in &hull.face_planes {
            for vertex in &hull.vertices {
                assert!(plane.truncate().dot(*vertex) <= plane.w + 1.0e-5);
            }
        }
        // Each corner is one vertex, so the two faces meeting there agree exactly,
        // which sixteen independently rounded boxes cannot do.
        for vertex in &hull.vertices {
            let touching = hull
                .face_planes
                .iter()
                .filter(|plane| (plane.truncate().dot(*vertex) - plane.w).abs() < 1.0e-6)
                .count();
            assert_eq!(touching, 3, "a prism corner meets two sides and one end");
        }
    }

    #[test]
    fn a_hollow_cylinder_or_sector_has_no_analytic_description() {
        for dimensions in [
            CylinderDimensions::new(1.0, 0.5, 0.25).unwrap(),
            CylinderDimensions::new(1.0, 0.0, 0.25)
                .unwrap()
                .with_sweep_angle_degrees(255)
                .unwrap(),
        ] {
            let mut graph = ConstructionGraph::new();
            graph
                .apply(BuildCommand::SpawnCylinder(CylinderSpec::new(
                    dimensions,
                    BuildPose::default(),
                )))
                .unwrap();
            let compiled = graph.compile().unwrap();
            assert_eq!(compiled.colliders.len(), super::CYLINDER_COLLIDER_COUNT);
            assert!(compiled.cylinders.is_empty());
        }
    }

    #[test]
    fn hollow_cylinder_compiles_exact_mass_inertia_and_sixteen_colliders() {
        let mut graph = ConstructionGraph::new();
        let dimensions = CylinderDimensions::new(1.0, 0.5, 2.0).unwrap();
        let spec = CylinderSpec::new(dimensions, BuildPose::default());
        graph.apply(BuildCommand::SpawnCylinder(spec)).unwrap();

        let compiled = graph.compile().unwrap();
        let properties = compiled.compounds[0].mass_properties;
        let outer = 0.5_f32;
        let inner = 0.25_f32;
        let expected_mass = crate::ConstructionMaterial::Steel
            .properties()
            .density_kg_m3
            * core::f32::consts::PI
            * (outer * outer - inner * inner)
            * 2.0;
        let expected_axial = expected_mass * (outer * outer + inner * inner) * 0.5;
        let expected_transverse =
            expected_mass * (3.0 * (outer * outer + inner * inner) + 4.0) / 12.0;
        assert_eq!(compiled.colliders.len(), super::CYLINDER_COLLIDER_COUNT);
        assert!((properties.mass - expected_mass).abs() < 1.0e-3);
        assert!(properties.center_of_mass.abs_diff_eq(Vec3::ZERO, 1.0e-6));
        assert!((properties.inertia.x_axis.x - expected_transverse).abs() < 1.0e-3);
        assert!((properties.inertia.y_axis.y - expected_axial).abs() < 1.0e-3);
        assert!((properties.inertia.z_axis.z - expected_transverse).abs() < 1.0e-3);
        assert!(compiled.colliders.iter().all(|collider| {
            let ColliderShape::Cuboid { half_extents, .. } = collider.shape else {
                return false;
            };
            (half_extents.y - 1.0).abs() < 1.0e-6
                && collider.local_center.length() >= inner - 1.0e-6
        }));
    }

    #[test]
    fn hollow_pipe_tee_compiles_pipe_mass_and_leaves_open_passages() {
        use crate::{PipeArms, PipeJunctionDimensions, PipeJunctionSpec};
        let arms = PipeArms::single(FaceKind::NegativeX)
            .with(FaceKind::PositiveX)
            .with(FaceKind::PositiveY);
        let spec = PipeJunctionSpec::new(
            PipeJunctionDimensions::new(0.20, 0.10).unwrap(),
            arms,
            BuildPose::default(),
        );
        let mut graph = ConstructionGraph::new();
        graph.apply(BuildCommand::SpawnPipeJunction(spec)).unwrap();
        let compiled = graph.compile().unwrap();
        assert_eq!(compiled.colliders.len(), spec.collider_count());

        let (radius, bore, reach) = (0.10_f32, 0.05_f32, 0.125_f32);
        let annulus = std::f32::consts::PI * (radius * radius - bore * bore);
        let density = ConstructionMaterial::Steel.properties().density_kg_m3;
        let properties = compiled.compounds[0].mass_properties;
        // The through pipe alone, and with a whole side arm added, bracket the tee.
        let through = annulus * 2.0 * reach * density;
        assert!(properties.mass > through * 0.97, "{}", properties.mass);
        assert!(properties.mass < through * 1.5, "{}", properties.mass);
        assert!(
            properties.center_of_mass.y > 1.0e-4,
            "the side arm adds mass above"
        );
        assert!(properties.center_of_mass.x.abs() < 1.0e-4);

        // No wall box covers a point travelling along an open bore.
        let bore_points = (-10_i16..=10)
            .map(|step| Vec3::X * (f32::from(step) * reach * 0.1))
            .chain((0_i16..=10).map(|step| Vec3::Y * (f32::from(step) * reach * 0.1)))
            .collect::<Vec<_>>();
        for collider in &compiled.colliders {
            let ColliderShape::Cuboid {
                local_rotation,
                half_extents,
            } = collider.shape
            else {
                panic!("junction walls are boxes");
            };
            let centre = collider.local_center + properties.center_of_mass;
            for &point in &bore_points {
                let local = local_rotation.inverse() * (point - centre);
                assert!(
                    (local.abs() - half_extents).max_element() > -1.0e-5,
                    "a wall box covers the open bore at {point}"
                );
            }
        }
    }

    #[test]
    fn hollow_pipe_bend_compiles_exact_quarter_torus_mass_and_full_inertia() {
        let dimensions = PipeBendDimensions::new(0.50, 0.25, 4).unwrap();
        let spec = PipeBendSpec::new(dimensions, BuildPose::default());
        let mut graph = ConstructionGraph::new();
        graph.apply(BuildCommand::SpawnPipeBend(spec)).unwrap();
        let compiled = graph.compile().unwrap();
        assert_eq!(compiled.colliders.len(), PIPE_BEND_COLLIDER_COUNT);

        let outer = 0.25_f32;
        let inner = 0.125_f32;
        let radius = 0.75_f32;
        let expected_volume = std::f32::consts::FRAC_PI_2
            * std::f32::consts::PI
            * radius
            * (outer * outer - inner * inner);
        let properties = compiled.compounds[0].mass_properties;
        assert!(
            (properties.mass
                - expected_volume * ConstructionMaterial::Steel.properties().density_kg_m3)
                .abs()
                < 1.0e-3
        );
        let mean_q = radius + (outer * outer + inner * inner) / (4.0 * radius);
        let expected_center = Vec3::new(
            -radius + 2.0 * mean_q / std::f32::consts::PI,
            radius - 2.0 * mean_q / std::f32::consts::PI,
            0.0,
        );
        assert!(
            properties
                .center_of_mass
                .abs_diff_eq(expected_center, 1.0e-5)
        );
        assert!((properties.inertia.x_axis.x - properties.inertia.y_axis.y).abs() < 1.0e-4);
        assert!(properties.inertia.x_axis.y.abs() > 1.0e-3);
        assert!((properties.inertia.x_axis.y - properties.inertia.y_axis.x).abs() < 1.0e-4);
    }

    #[test]
    fn cylinder_sector_compiles_exact_offset_mass_properties_and_sixteen_colliders() {
        let mut graph = ConstructionGraph::new();
        let dimensions = CylinderDimensions::new(1.0, 0.5, 2.0)
            .unwrap()
            .with_sweep_angle_degrees(90)
            .unwrap();
        graph
            .apply(BuildCommand::SpawnCylinder(CylinderSpec::new(
                dimensions,
                BuildPose::default(),
            )))
            .unwrap();

        let compiled = graph.compile().unwrap();
        let properties = compiled.compounds[0].mass_properties;
        let outer = 0.5_f32;
        let inner = 0.25_f32;
        let length = 2.0_f32;
        let sweep = core::f32::consts::FRAC_PI_2;
        let expected_mass = crate::ConstructionMaterial::Steel
            .properties()
            .density_kg_m3
            * sweep
            * (outer * outer - inner * inner)
            * length
            * 0.5;
        let expected_center_x = 4.0 * (sweep * 0.5).sin() * (outer.powi(3) - inner.powi(3))
            / (3.0 * sweep * (outer * outer - inner * inner));
        let radial_squared = outer * outer + inner * inner;
        let radial_parallel = radial_squared * (sweep + sweep.sin()) / (4.0 * sweep);
        let radial_perpendicular = radial_squared * (sweep - sweep.sin()) / (4.0 * sweep);

        assert_eq!(compiled.colliders.len(), super::CYLINDER_COLLIDER_COUNT);
        assert!((properties.mass - expected_mass).abs() < 1.0e-3);
        assert!(
            properties
                .center_of_mass
                .abs_diff_eq(Vec3::new(expected_center_x, 0.0, 0.0), 1.0e-6)
        );
        assert!(
            (properties.inertia.x_axis.x
                - expected_mass * (length * length / 12.0 + radial_perpendicular))
                .abs()
                < 1.0e-3
        );
        assert!(
            (properties.inertia.y_axis.y
                - expected_mass
                    * (radial_parallel + radial_perpendicular
                        - expected_center_x * expected_center_x))
                .abs()
                < 1.0e-3
        );
        assert!(
            compiled
                .colliders
                .iter()
                .all(|collider| { (collider.local_center + properties.center_of_mass).x > 0.0 })
        );
    }

    fn bearing(
        graph: &mut ConstructionGraph,
        a: crate::PartId,
        face_a: FaceKind,
        b: crate::PartId,
        face_b: FaceKind,
        anchor: Vec3,
        axis: Vec3,
    ) {
        graph
            .apply(BuildCommand::AddBearing(BearingSpec::new(
                FaceRef::part(a, face_a),
                FaceRef::part(b, face_b),
                anchor,
                axis,
            )))
            .unwrap();
    }

    /// One steel block at the origin, claimed as a region.
    fn region_over_one_block() -> (ConstructionGraph, crate::RegionId) {
        let spec = CuboidSpec::new(
            [1, 1, 1],
            BuildPose::from_half_grid(IVec3::ONE, GridRotation::default()),
        )
        .unwrap();
        let mut graph = ConstructionGraph::new();
        graph.apply(BuildCommand::Spawn(spec)).unwrap();
        let region =
            crate::ShapeRegion::new(IVec3::ZERO, IVec3::ONE, ConstructionMaterial::Steel).unwrap();
        let BuildOutcome::RegionAdded(id) = graph.apply(BuildCommand::AddRegion(region)).unwrap()
        else {
            panic!("wrong outcome")
        };
        (graph, id)
    }

    #[test]
    fn a_regions_blocks_contribute_no_colliders_of_their_own() {
        // The region owns the geometry; counting the block too would render and
        // collide the same material twice.
        let (graph, _) = region_over_one_block();
        let compiled = graph.compile().unwrap();
        assert_eq!(compiled.colliders.len(), 1);
        let expected = ConstructionMaterial::Steel.properties().density_kg_m3 * 0.25_f32.powi(3);
        assert!(
            (compiled.compounds[0].mass_properties.mass - expected).abs() < 1.0e-3,
            "the region's mass must replace its block's, not add to it"
        );
    }

    #[test]
    fn collapsing_a_region_edge_halves_its_mass_and_moves_its_centroid() {
        // The bounding box means a corner can only move inward, so the exact
        // case to check is a collapse rather than a shear: the top +z edge
        // driven down onto the bottom leaves a wedge of half the material,
        // whose centroid is the triangle's.
        let (mut graph, id) = region_over_one_block();
        let cell = i16::try_from(crate::POSITION_TICKS_PER_GRID_UNIT).unwrap();
        graph
            .apply(BuildCommand::SetRegionVertices {
                region: id,
                vertices: vec![([0, 1, 1], [0, -cell, 0]), ([1, 1, 1], [0, -cell, 0])],
            })
            .unwrap();

        let properties = graph.compile().unwrap().compounds[0].mass_properties;
        let density = ConstructionMaterial::Steel.properties().density_kg_m3;
        let expected_mass = density * 0.25_f32.powi(3) * 0.5;
        assert!(
            (properties.mass - expected_mass).abs() < 1.0e-3,
            "a wedge holds half a cell: {} vs {expected_mass}",
            properties.mass
        );
        let third = 0.25_f32 / 3.0;
        let expected_center = Vec3::new(0.125, third, third);
        assert!(
            properties
                .center_of_mass
                .abs_diff_eq(expected_center, 1.0e-4),
            "wedge centroid should be the triangle's: {} vs {expected_center}",
            properties.center_of_mass
        );
    }

    #[test]
    fn a_cage_vertex_cannot_be_pushed_out_of_the_region() {
        // Shearing a face outward is exactly what the bounding box forbids.
        let (mut graph, id) = region_over_one_block();
        assert!(
            graph
                .apply(BuildCommand::SetRegionVertices {
                    region: id,
                    vertices: vec![([1, 1, 1], [5, 0, 0])],
                })
                .is_err(),
            "a corner already at the maximum has nowhere outward to go"
        );
    }

    #[test]
    #[allow(clippy::cast_precision_loss, clippy::similar_names)]
    fn region_inertia_matches_numerical_integration_of_the_same_pieces() {
        // The analytic integration is derived, so check it against a brute-force
        // sum over a dense sample of the solid it claims to describe.
        let (mut graph, id) = region_over_one_block();
        graph
            .apply(BuildCommand::SetRegionVertices {
                region: id,
                vertices: vec![
                    ([1, 1, 1], [-8, -6, -4]),
                    ([1, 1, 0], [-4, 0, 2]),
                    ([0, 0, 1], [2, 3, -5]),
                ],
            })
            .unwrap();
        let properties = graph.compile().unwrap().compounds[0].mass_properties;

        let region = graph.region(id).unwrap();
        let pieces = super::region_pieces(region);
        let inside = |point: Vec3| {
            pieces.iter().any(|piece| match piece {
                PartPiece::Cuboid {
                    center,
                    half_extents,
                    rotation,
                    ..
                } => {
                    let local = rotation.inverse() * (point - *center);
                    local.abs().cmple(*half_extents).all()
                }
                PartPiece::Convex(convex) => convex
                    .faces
                    .iter()
                    .all(|face| face.normal.dot(point) <= face.offset + 1.0e-7),
            })
        };

        let steps = 90;
        let low = Vec3::splat(-0.05);
        let cell = 0.35_f32 / steps as f32;
        let cell_volume = cell * cell * cell;
        let density = ConstructionMaterial::Steel.properties().density_kg_m3;
        let mut mass = 0.0_f32;
        let mut moment = Vec3::ZERO;
        let mut samples = Vec::new();
        for ix in 0..steps {
            for iy in 0..steps {
                for iz in 0..steps {
                    let point =
                        low + Vec3::new(ix as f32 + 0.5, iy as f32 + 0.5, iz as f32 + 0.5) * cell;
                    if inside(point) {
                        mass += density * cell_volume;
                        moment += point * (density * cell_volume);
                        samples.push(point);
                    }
                }
            }
        }
        assert!(!samples.is_empty(), "the shaped solid must contain samples");
        let center = moment / mass;
        let mut inertia = Mat3::ZERO;
        for point in &samples {
            let arm = *point - center;
            inertia += (Mat3::IDENTITY * arm.length_squared() - outer_product(arm, arm))
                * (density * cell_volume);
        }

        assert!(
            (properties.mass - mass).abs() < mass * 0.02,
            "mass {} vs sampled {mass}",
            properties.mass
        );
        assert!(
            properties.center_of_mass.abs_diff_eq(center, 2.0e-3),
            "centre {} vs sampled {center}",
            properties.center_of_mass
        );
        for axis in 0..3 {
            let analytic = properties.inertia.col(axis)[axis];
            let sampled = inertia.col(axis)[axis];
            assert!(
                (analytic - sampled).abs() < sampled.abs() * 0.05,
                "inertia axis {axis}: {analytic} vs sampled {sampled}"
            );
        }
    }

    #[test]
    fn an_unshaped_creation_still_compiles_to_one_collider_per_block() {
        // The regression guard: regions must cost nothing where none exist.
        let mut graph = ConstructionGraph::new();
        for offset in 0..4 {
            graph
                .apply(BuildCommand::Spawn(cube_at(IVec3::new(offset * 4, 0, 0))))
                .unwrap();
        }
        let compiled = graph.compile().unwrap();
        assert_eq!(compiled.colliders.len(), 4);
        assert!(
            compiled
                .colliders
                .iter()
                .all(|collider| collider.shape.is_cuboid()),
            "an unshaped creation must produce only boxes"
        );
    }

    #[test]
    fn featured_cuboid_compiles_evaluated_mass_and_convex_collision() {
        let mut graph = ConstructionGraph::new();
        let part = spawn(&mut graph, IVec3::ZERO);
        let owner = SolidOwner::Part(part);
        let edge = graph.evaluated_solid(owner).unwrap().logical_edges[0].key;
        graph
            .apply(BuildCommand::AddShapeFeature(ShapeFeature::new(
                [EdgeChainRef { owner, edge }],
                EdgeTreatment::Chamfer,
                20,
            )))
            .unwrap();

        let compiled = graph.compile().unwrap();
        let uncut_mass = ConstructionMaterial::Steel.properties().density_kg_m3;
        assert!(compiled.compounds[0].mass_properties.mass < uncut_mass);
        assert!(
            compiled
                .colliders
                .iter()
                .all(|collider| matches!(collider.shape, ColliderShape::Convex(_)))
        );
    }

    #[test]
    fn welded_same_material_cuboids_compact_without_changing_mass_or_inertia() {
        let mut graph = ConstructionGraph::new();
        let a = spawn(&mut graph, IVec3::ZERO);
        let b = spawn(&mut graph, IVec3::new(4, 0, 0));
        graph
            .apply(BuildCommand::Weld(WeldSpec {
                first: FaceRef::part(a, FaceKind::PositiveX),
                second: FaceRef::part(b, FaceKind::NegativeX),
            }))
            .unwrap();

        let compiled = graph.compile().unwrap();
        let properties = compiled.compounds[0].mass_properties;
        assert_eq!(compiled.compounds.len(), 1);
        assert_eq!(compiled.colliders.len(), 1);
        let ColliderShape::Cuboid {
            local_rotation,
            half_extents,
        } = compiled.colliders[0].shape
        else {
            panic!("compacted collider must remain a cuboid")
        };
        assert!(local_rotation.abs_diff_eq(Quat::IDENTITY, 1.0e-6));
        assert!(half_extents.abs_diff_eq(Vec3::new(1.0, 0.5, 0.5), 1.0e-6));
        let cube_mass = crate::ConstructionMaterial::Steel
            .properties()
            .density_kg_m3;
        assert!((properties.mass - cube_mass * 2.0).abs() < 1.0e-3);
        assert!(
            properties
                .center_of_mass
                .abs_diff_eq(Vec3::new(0.5, 0.0, 0.0), 1.0e-6)
        );
        assert!((properties.inertia.x_axis.x - cube_mass / 3.0).abs() < 1.0e-3);
        assert!((properties.inertia.y_axis.y - cube_mass * 5.0 / 6.0).abs() < 1.0e-3);
        assert!((properties.inertia.z_axis.z - cube_mass * 5.0 / 6.0).abs() < 1.0e-3);
    }

    #[test]
    fn every_material_scales_cuboid_and_cylinder_mass() {
        for material in ConstructionMaterial::ALL {
            let density = material.properties().density_kg_m3;
            let mut cuboid_graph = ConstructionGraph::new();
            cuboid_graph
                .apply(BuildCommand::Spawn(
                    CuboidSpec::new([4; 3], BuildPose::default())
                        .unwrap()
                        .with_material(material),
                ))
                .unwrap();
            let cuboid = cuboid_graph.compile().unwrap();
            assert!((cuboid.compounds[0].mass_properties.mass - density).abs() < 1.0e-3);
            assert!(
                (cuboid.compounds[0].mass_properties.inertia.x_axis.x - density / 6.0).abs()
                    < 1.0e-3
            );

            let mut cylinder_graph = ConstructionGraph::new();
            cylinder_graph
                .apply(BuildCommand::SpawnCylinder(
                    CylinderSpec::new(
                        CylinderDimensions::new(1.0, 0.0, 1.0).unwrap(),
                        BuildPose::default(),
                    )
                    .with_material(material),
                ))
                .unwrap();
            let cylinder = cylinder_graph.compile().unwrap();
            let expected = density * core::f32::consts::PI * 0.25;
            assert!((cylinder.compounds[0].mass_properties.mass - expected).abs() < 1.0e-3);
        }
    }

    #[test]
    fn mixed_material_welds_sum_mass_and_keep_collider_contact_properties() {
        let mut graph = ConstructionGraph::new();
        let BuildOutcome::Spawned(aluminium) = graph
            .apply(BuildCommand::Spawn(
                cube_at(IVec3::ZERO).with_material(ConstructionMaterial::Aluminium),
            ))
            .unwrap()
        else {
            unreachable!()
        };
        let BuildOutcome::Spawned(wood) = graph
            .apply(BuildCommand::Spawn(
                cube_at(IVec3::new(4, 0, 0)).with_material(ConstructionMaterial::Wood),
            ))
            .unwrap()
        else {
            unreachable!()
        };
        graph
            .apply(BuildCommand::Weld(WeldSpec {
                first: FaceRef::part(aluminium, FaceKind::PositiveX),
                second: FaceRef::part(wood, FaceKind::NegativeX),
            }))
            .unwrap();

        let compiled = graph.compile().unwrap();
        let mass = compiled.compounds[0].mass_properties;
        assert!((mass.mass - 3_400.0).abs() < 1.0e-3);
        assert!(
            mass.center_of_mass
                .abs_diff_eq(Vec3::new(700.0 / 3_400.0, 0.0, 0.0), 1.0e-6)
        );
        let contacts = compiled
            .colliders
            .iter()
            .map(|collider| collider.material_properties)
            .collect::<Vec<_>>();
        assert!(contacts.contains(&ConstructionMaterial::Aluminium.properties()));
        assert!(contacts.contains(&ConstructionMaterial::Wood.properties()));
    }

    #[test]
    fn shared_bearing_attachments_compile_as_one_rotor_and_one_joint() {
        let mut graph = ConstructionGraph::new();
        let support = CuboidSpec::new(
            [4, 4, 4],
            BuildPose::new(IVec3::new(0, 2, 0), GridRotation::default()),
        )
        .unwrap();
        let BuildOutcome::Spawned(support) = graph.apply(BuildCommand::Spawn(support)).unwrap()
        else {
            unreachable!()
        };
        let targets = [IVec3::new(0, 9, 0), IVec3::new(2, 9, 0)].map(|center| {
            let spec = CuboidSpec::new(
                [1, 1, 1],
                BuildPose::from_half_grid(center, GridRotation::default()),
            )
            .unwrap();
            let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(spec)).unwrap()
            else {
                unreachable!()
            };
            part
        });
        let dimensions = BearingDimensions::new(0.80, 0.10).unwrap();
        for target in targets {
            graph
                .apply(BuildCommand::AddBearing(
                    BearingSpec::new(
                        FaceRef::part(support, FaceKind::PositiveY),
                        FaceRef::part(target, FaceKind::NegativeY),
                        Vec3::Y,
                        Vec3::Y,
                    )
                    .with_dimensions(dimensions),
                ))
                .unwrap();
        }
        graph
            .apply(BuildCommand::RigidLink(RigidLinkSpec {
                first: targets[0],
                second: targets[1],
            }))
            .unwrap();

        let compiled = graph.compile().unwrap();
        let compound_for = |part| {
            compiled
                .part_to_compound
                .iter()
                .find_map(|&(candidate, compound)| (candidate == part).then_some(compound))
                .unwrap()
        };
        assert_eq!(compiled.compounds.len(), 2);
        assert_eq!(compiled.bearings.len(), 1);
        assert_eq!(compound_for(targets[0]), compound_for(targets[1]));
        assert_ne!(compound_for(support), compound_for(targets[0]));
    }

    #[test]
    fn welding_to_ground_makes_only_that_group_static() {
        let mut graph = ConstructionGraph::new();
        let grounded = spawn(&mut graph, IVec3::new(0, 2, 0));
        spawn(&mut graph, IVec3::new(8, 2, 0));
        graph
            .apply(BuildCommand::Weld(WeldSpec {
                first: FaceRef::part(grounded, FaceKind::NegativeY),
                second: FaceRef::ground(),
            }))
            .unwrap();

        let compiled = graph.compile().unwrap();
        assert_eq!(compiled.compounds.len(), 2);
        assert!(compiled.compounds[0].is_static);
        assert!(compiled.compounds[0].mass_properties.inverse_mass.abs() < f32::EPSILON);
        assert!(!compiled.compounds[1].is_static);
    }

    #[test]
    fn external_world_anchor_makes_only_its_rigid_group_static() {
        let mut graph = ConstructionGraph::new();
        let anchored = spawn(&mut graph, IVec3::new(0, 20, 0));
        let floating = spawn(&mut graph, IVec3::new(8, 20, 0));

        let ordinary = graph.compile().unwrap();
        assert!(
            ordinary
                .compounds
                .iter()
                .all(|compound| !compound.is_static)
        );

        let compiled = graph.compile_with_static_parts([anchored]).unwrap();
        let compound_for = |part| {
            compiled
                .part_to_compound
                .iter()
                .find_map(|&(candidate, compound)| (candidate == part).then_some(compound))
                .unwrap()
        };
        assert!(compiled.compounds[compound_for(anchored) as usize].is_static);
        assert!(!compiled.compounds[compound_for(floating) as usize].is_static);
    }

    /// Two blocks on a bearing, then welded to each other as well.
    fn bearing_welded_shut() -> (ConstructionGraph, BearingId) {
        let mut graph = ConstructionGraph::new();
        let a = spawn(&mut graph, IVec3::ZERO);
        let b = spawn(&mut graph, IVec3::new(4, 0, 0));
        let BuildOutcome::BearingAdded(bearing) = graph
            .apply(BuildCommand::AddBearing(BearingSpec::new(
                FaceRef::part(a, FaceKind::PositiveX),
                FaceRef::part(b, FaceKind::NegativeX),
                Vec3::new(0.5, 0.0, 0.0),
                Vec3::X,
            )))
            .unwrap()
        else {
            panic!("wrong bearing outcome")
        };
        graph
            .apply(BuildCommand::Weld(WeldSpec {
                first: FaceRef::part(a, FaceKind::PositiveX),
                second: FaceRef::part(b, FaceKind::NegativeX),
            }))
            .unwrap();
        (graph, bearing)
    }

    #[test]
    fn bearing_collapsed_by_a_later_weld_stops_being_a_joint() {
        let (graph, _) = bearing_welded_shut();

        let compiled = graph.compile().unwrap();

        // The weld already holds the two sides together, so the bearing adds
        // no coordinate and no constraint — it is locked, not unbuildable.
        assert_eq!(compiled.compounds.len(), 1);
        assert!(compiled.bearings.is_empty());
        assert!(compiled.loop_topology.tree_bearings.is_empty());
        assert!(compiled.loop_topology.closure_bearings.is_empty());
    }

    #[test]
    fn a_driven_bearing_collapsed_by_a_later_weld_is_rejected() {
        let (mut graph, bearing) = bearing_welded_shut();
        wire_with(
            &mut graph,
            bearing,
            DriveLimits::default(),
            DriveProgram::default(),
            false,
        );

        assert_eq!(
            graph.compile(),
            Err(TopologyError::SelfBearing {
                bearing,
                compound: 0
            })
        );
    }

    #[test]
    fn bearing_dimensions_do_not_change_compiled_physics() {
        let compile_with = |dimensions: BearingDimensions| {
            let mut graph = ConstructionGraph::new();
            let a = spawn(&mut graph, IVec3::ZERO);
            let b = spawn(&mut graph, IVec3::new(4, 0, 0));
            graph
                .apply(BuildCommand::AddBearing(
                    BearingSpec::new(
                        FaceRef::part(a, FaceKind::PositiveX),
                        FaceRef::part(b, FaceKind::NegativeX),
                        Vec3::new(0.5, 0.0, 0.0),
                        Vec3::X,
                    )
                    .with_dimensions(dimensions),
                ))
                .unwrap();
            graph.compile().unwrap()
        };

        assert_eq!(
            compile_with(BearingDimensions::default()),
            compile_with(BearingDimensions::new(0.50, 0.20).unwrap())
        );
    }

    #[test]
    fn closed_square_has_one_hard_closure_edge() {
        let mut graph = ConstructionGraph::new();
        let a = spawn(&mut graph, IVec3::ZERO);
        let b = spawn(&mut graph, IVec3::new(4, 0, 0));
        let c = spawn(&mut graph, IVec3::new(4, 4, 0));
        let d = spawn(&mut graph, IVec3::new(0, 4, 0));
        let edges = [
            (
                a,
                FaceKind::PositiveX,
                b,
                FaceKind::NegativeX,
                Vec3::new(0.5, 0.0, 0.0),
                Vec3::X,
            ),
            (
                b,
                FaceKind::PositiveY,
                c,
                FaceKind::NegativeY,
                Vec3::new(1.0, 0.5, 0.0),
                Vec3::Y,
            ),
            (
                c,
                FaceKind::NegativeX,
                d,
                FaceKind::PositiveX,
                Vec3::new(0.5, 1.0, 0.0),
                Vec3::NEG_X,
            ),
            (
                d,
                FaceKind::NegativeY,
                a,
                FaceKind::PositiveY,
                Vec3::new(0.0, 0.5, 0.0),
                Vec3::NEG_Y,
            ),
        ];
        for (source, source_face, target, target_face, anchor, axis) in edges {
            graph
                .apply(BuildCommand::AddBearing(BearingSpec::new(
                    FaceRef::part(source, source_face),
                    FaceRef::part(target, target_face),
                    anchor,
                    axis,
                )))
                .unwrap();
        }

        let compiled = graph.compile().unwrap();
        assert_eq!(compiled.loop_topology.tree_bearings.len(), 3);
        assert_eq!(compiled.loop_topology.closure_bearings.len(), 1);
        assert_eq!(
            compiled.loop_topology.mechanism_components,
            vec![vec![0, 1, 2, 3]]
        );
        assert_eq!(compiled.collision_suppression.len(), 4);
    }

    #[test]
    fn floating_branch_has_one_canonical_root_and_stable_traversals() {
        let mut graph = ConstructionGraph::new();
        let root = spawn(&mut graph, IVec3::ZERO);
        let x_child = spawn(&mut graph, IVec3::new(4, 0, 0));
        let y_child = spawn(&mut graph, IVec3::new(0, 4, 0));
        bearing(
            &mut graph,
            root,
            FaceKind::PositiveX,
            x_child,
            FaceKind::NegativeX,
            Vec3::new(0.5, 0.0, 0.0),
            Vec3::X,
        );
        bearing(
            &mut graph,
            root,
            FaceKind::PositiveY,
            y_child,
            FaceKind::NegativeY,
            Vec3::new(0.0, 0.5, 0.0),
            Vec3::Y,
        );

        let topology = graph.compile().unwrap().loop_topology;
        assert_eq!(topology.component_roots, vec![vec![0]]);
        assert!(topology.body_parents[0].is_root);
        assert_eq!(topology.body_parents[1].parent_body, 0);
        assert_eq!(topology.body_parents[2].parent_body, 0);
        assert_eq!(topology.body_parents[1].preorder_index, 1);
        assert_eq!(topology.body_parents[2].preorder_index, 2);
        assert_eq!(topology.contraction_rounds, vec![vec![1, 2]]);
    }

    #[test]
    fn mirrored_floating_branches_use_heavy_chassis_root_and_equal_inertia() {
        let mut graph = ConstructionGraph::new();
        let left = spawn(&mut graph, IVec3::new(-2, 0, 0));
        let right = spawn(&mut graph, IVec3::new(2, 0, 0));
        let BuildOutcome::Spawned(chassis) = graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new(
                    [8, 4, 8],
                    BuildPose::new(IVec3::new(0, 4, 0), GridRotation::default()),
                )
                .unwrap(),
            ))
            .unwrap()
        else {
            unreachable!()
        };
        bearing(
            &mut graph,
            chassis,
            FaceKind::NegativeY,
            left,
            FaceKind::PositiveY,
            Vec3::new(-0.5, 0.5, 0.0),
            Vec3::NEG_Y,
        );
        bearing(
            &mut graph,
            chassis,
            FaceKind::NegativeY,
            right,
            FaceKind::PositiveY,
            Vec3::new(0.5, 0.5, 0.0),
            Vec3::NEG_Y,
        );

        let topology = graph.compile().unwrap().loop_topology;
        assert_eq!(topology.component_roots, vec![vec![2]]);
        assert!(topology.body_parents[2].is_root);
        assert_eq!(topology.body_parents[0].parent_body, 2);
        assert_eq!(topology.body_parents[1].parent_body, 2);
        assert_eq!(topology.coordinate_axis_inertia.len(), 2);
        assert!(
            (topology.coordinate_axis_inertia[0] - topology.coordinate_axis_inertia[1]).abs()
                < 1.0e-4,
            "mirrored branches compiled unequal inertia: {:?}",
            topology.coordinate_axis_inertia
        );
    }

    #[test]
    fn grounded_body_is_root_even_when_not_the_first_compound() {
        let mut graph = ConstructionGraph::new();
        let child = spawn(&mut graph, IVec3::new(0, 6, 0));
        let root = spawn(&mut graph, IVec3::new(0, 2, 0));
        ground(&mut graph, root);
        bearing(
            &mut graph,
            child,
            FaceKind::NegativeY,
            root,
            FaceKind::PositiveY,
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::NEG_Y,
        );

        let topology = graph.compile().unwrap().loop_topology;
        assert_eq!(topology.component_roots, vec![vec![1]]);
        assert_eq!(topology.body_parents[0].parent_body, 1);
        assert_eq!(topology.body_parents[0].bearing_direction, 1);
        assert!(topology.body_parents[1].is_root);
    }

    #[test]
    fn multiple_ground_anchors_remain_fixed_roots_with_a_closure_edge() {
        let mut graph = ConstructionGraph::new();
        let left = spawn(&mut graph, IVec3::new(0, 2, 0));
        let middle = spawn(&mut graph, IVec3::new(4, 2, 0));
        let right = spawn(&mut graph, IVec3::new(8, 2, 0));
        ground(&mut graph, left);
        ground(&mut graph, right);
        bearing(
            &mut graph,
            left,
            FaceKind::PositiveX,
            middle,
            FaceKind::NegativeX,
            Vec3::new(0.5, 0.5, 0.0),
            Vec3::X,
        );
        bearing(
            &mut graph,
            middle,
            FaceKind::PositiveX,
            right,
            FaceKind::NegativeX,
            Vec3::new(1.5, 0.5, 0.0),
            Vec3::X,
        );

        let topology = graph.compile().unwrap().loop_topology;
        assert_eq!(topology.component_roots, vec![vec![0, 2]]);
        assert_eq!(topology.tree_bearings.len(), 1);
        assert_eq!(topology.closure_bearings.len(), 1);
        assert!(topology.body_parents[0].is_root);
        assert_eq!(topology.body_parents[1].parent_body, 0);
        assert!(topology.body_parents[2].is_root);
    }

    fn add_bearing(
        graph: &mut ConstructionGraph,
        source: PartId,
        source_face: FaceKind,
        target: PartId,
        target_face: FaceKind,
        anchor: Vec3,
        axis: Vec3,
    ) -> BearingId {
        let BuildOutcome::BearingAdded(bearing) = graph
            .apply(BuildCommand::AddBearing(BearingSpec::new(
                FaceRef::part(source, source_face),
                FaceRef::part(target, target_face),
                anchor,
                axis,
            )))
            .unwrap()
        else {
            unreachable!()
        };
        bearing
    }

    fn wire_with(
        graph: &mut ConstructionGraph,
        bearing: BearingId,
        limits: DriveLimits,
        program: DriveProgram,
        reversed: bool,
    ) -> PartId {
        let BuildOutcome::Spawned(controller) = graph
            .apply(BuildCommand::SpawnController(ControllerSpec::new(
                BuildPose::new(IVec3::new(0, 40, 0), GridRotation::default()),
            )))
            .unwrap()
        else {
            unreachable!()
        };
        let mut spec = DriveLinkSpec::new(controller, bearing);
        spec.limits = limits;
        spec.program = program;
        spec.reversed = reversed;
        spec.actuator = ActuatorAssignment::motor(100, 0).unwrap();
        graph.apply(BuildCommand::AddDriveLink(spec)).unwrap();
        let BuildOutcome::Spawned(engine) = graph
            .apply(BuildCommand::SpawnEngine(EngineSpec::new(
                EngineKind::Electric,
                BuildPose::new(IVec3::new(0, 42, 0), GridRotation::default()),
            )))
            .unwrap()
        else {
            unreachable!()
        };
        graph
            .apply(BuildCommand::Weld(WeldSpec {
                first: FaceRef::part(controller, FaceKind::PositiveY),
                second: FaceRef::part(engine, FaceKind::NegativeY),
            }))
            .unwrap();
        controller
    }

    fn wire(graph: &mut ConstructionGraph, bearing: BearingId, reversed: bool) -> PartId {
        wire_with(
            graph,
            bearing,
            DriveLimits::default(),
            DriveProgram::default(),
            reversed,
        )
    }

    fn square_loop(graph: &mut ConstructionGraph) -> [BearingId; 4] {
        let a = spawn(graph, IVec3::ZERO);
        let b = spawn(graph, IVec3::new(4, 0, 0));
        let c = spawn(graph, IVec3::new(4, 4, 0));
        let d = spawn(graph, IVec3::new(0, 4, 0));
        [
            add_bearing(
                graph,
                a,
                FaceKind::PositiveX,
                b,
                FaceKind::NegativeX,
                Vec3::new(0.5, 0.0, 0.0),
                Vec3::X,
            ),
            add_bearing(
                graph,
                b,
                FaceKind::PositiveY,
                c,
                FaceKind::NegativeY,
                Vec3::new(1.0, 0.5, 0.0),
                Vec3::Y,
            ),
            add_bearing(
                graph,
                c,
                FaceKind::NegativeX,
                d,
                FaceKind::PositiveX,
                Vec3::new(0.5, 1.0, 0.0),
                Vec3::NEG_X,
            ),
            add_bearing(
                graph,
                d,
                FaceKind::NegativeY,
                a,
                FaceKind::PositiveY,
                Vec3::new(0.0, 0.5, 0.0),
                Vec3::NEG_Y,
            ),
        ]
    }

    #[test]
    fn control_block_compiles_as_one_half_by_half_by_quarter_metre_collider() {
        let mut graph = ConstructionGraph::new();
        graph
            .apply(BuildCommand::SpawnController(ControllerSpec::new(
                BuildPose::default(),
            )))
            .unwrap();

        let compiled = graph.compile().unwrap();
        assert_eq!(compiled.colliders.len(), 1);
        assert_eq!(
            compiled.colliders[0].shape,
            ColliderShape::Cuboid {
                local_rotation: Quat::IDENTITY,
                half_extents: Vec3::new(0.25, 0.25, 0.125),
            }
        );
        let expected_mass = crate::MACHINE_PART_DENSITY_KG_M3 * 0.5 * 0.5 * 0.25;
        assert!((compiled.compounds[0].mass_properties.mass - expected_mass).abs() < 1.0e-4);
    }

    #[test]
    fn driven_bearing_is_preferred_as_a_tree_edge_over_a_passive_one() {
        // The last edge of the square would normally become the closure. Driving
        // it must push the closure onto a passive edge instead.
        let mut graph = ConstructionGraph::new();
        let bearings = square_loop(&mut graph);
        let driven = bearings[3];
        wire(&mut graph, driven, false);

        let compiled = graph.compile().unwrap();
        assert!(compiled.loop_topology.tree_bearings.contains(&driven));
        assert_eq!(compiled.loop_topology.closure_bearings.len(), 1);
        assert!(!compiled.loop_topology.closure_bearings.contains(&driven));
    }

    #[test]
    fn driven_bearing_forced_onto_a_closure_edge_is_rejected() {
        let mut graph = ConstructionGraph::new();
        let bearings = square_loop(&mut graph);
        for bearing in bearings {
            wire(&mut graph, bearing, false);
        }

        let Err(TopologyError::DrivenClosureBearing { bearing }) = graph.compile() else {
            panic!("a fully driven loop cannot give every edge a coordinate")
        };
        assert!(bearings.contains(&bearing));
    }

    #[test]
    fn coordinate_axis_inertia_matches_a_hand_computed_arm() {
        let mut graph = ConstructionGraph::new();
        let base = spawn(&mut graph, IVec3::new(0, 2, 0));
        ground(&mut graph, base);
        let arm = spawn(&mut graph, IVec3::new(4, 2, 0));
        add_bearing(
            &mut graph,
            base,
            FaceKind::PositiveX,
            arm,
            FaceKind::NegativeX,
            Vec3::new(0.5, 0.5, 0.0),
            Vec3::X,
        );

        let compiled = graph.compile().unwrap();
        let inertia = compiled.loop_topology.coordinate_axis_inertia[0];
        // The arm is a 1 m cube centred 0.5 m along the +x hinge axis, so the
        // radial offset is zero and only its own x inertia contributes.
        let mass = crate::ConstructionMaterial::Steel
            .properties()
            .density_kg_m3;
        let expected = mass * (1.0 + 1.0) / 12.0;
        assert!(
            (inertia - expected).abs() < 1.0e-2,
            "axis inertia {inertia} should be about {expected}"
        );
    }

    #[test]
    fn coordinate_axis_inertia_includes_the_whole_child_subtree() {
        let mut graph = ConstructionGraph::new();
        let base = spawn(&mut graph, IVec3::new(0, 2, 0));
        ground(&mut graph, base);
        let first = spawn(&mut graph, IVec3::new(0, 6, 0));
        let second = spawn(&mut graph, IVec3::new(0, 10, 0));
        add_bearing(
            &mut graph,
            base,
            FaceKind::PositiveY,
            first,
            FaceKind::NegativeY,
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::Y,
        );
        add_bearing(
            &mut graph,
            first,
            FaceKind::PositiveY,
            second,
            FaceKind::NegativeY,
            Vec3::new(0.0, 2.0, 0.0),
            Vec3::Y,
        );

        let inertia = graph
            .compile()
            .unwrap()
            .loop_topology
            .coordinate_axis_inertia;
        assert_eq!(inertia.len(), 2);
        assert!(
            inertia[0] > inertia[1],
            "the lower joint carries both links: {inertia:?}"
        );
    }

    #[test]
    fn duplicate_bearing_rows_share_one_coordinate() {
        let mut graph = ConstructionGraph::new();
        let base = spawn(&mut graph, IVec3::new(0, 2, 0));
        ground(&mut graph, base);
        let arm = spawn(&mut graph, IVec3::new(4, 2, 0));
        let first = add_bearing(
            &mut graph,
            base,
            FaceKind::PositiveX,
            arm,
            FaceKind::NegativeX,
            Vec3::new(0.5, 0.5, 0.0),
            Vec3::X,
        );
        // A second row describing the same physical joint: same compounds, same
        // anchor, same axis. Compilation collapses it, but a drive may still be
        // wired to whichever row the app happens to hold.
        let duplicate = add_bearing(
            &mut graph,
            base,
            FaceKind::PositiveX,
            arm,
            FaceKind::NegativeX,
            Vec3::new(0.5, 0.5, 0.0),
            Vec3::X,
        );

        let compiled = graph.compile().unwrap();
        let coordinates = &compiled.loop_topology.bearing_coordinates;
        assert_eq!(compiled.loop_topology.tree_bearings.len(), 1);
        assert_eq!(coordinates.get(&first), coordinates.get(&duplicate));
        assert!(coordinates.contains_key(&duplicate));
    }

    #[test]
    fn closure_bearings_have_no_coordinate_to_drive() {
        let mut graph = ConstructionGraph::new();
        let bearings = square_loop(&mut graph);
        let compiled = graph.compile().unwrap();

        let coordinates = &compiled.loop_topology.bearing_coordinates;
        for closure in &compiled.loop_topology.closure_bearings {
            assert!(!coordinates.contains_key(closure));
        }
        for tree in &compiled.loop_topology.tree_bearings {
            assert!(coordinates.contains_key(tree));
        }
        assert_eq!(
            coordinates.len(),
            bearings.len() - compiled.loop_topology.closure_bearings.len()
        );
    }

    #[test]
    fn coordinate_drives_apply_per_wire_reverse_and_leave_undriven_rows_passive() {
        let mut graph = ConstructionGraph::new();
        let base = spawn(&mut graph, IVec3::new(0, 2, 0));
        ground(&mut graph, base);
        let first = spawn(&mut graph, IVec3::new(0, 6, 0));
        let second = spawn(&mut graph, IVec3::new(0, 10, 0));
        let lower = add_bearing(
            &mut graph,
            base,
            FaceKind::PositiveY,
            first,
            FaceKind::NegativeY,
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::Y,
        );
        add_bearing(
            &mut graph,
            first,
            FaceKind::PositiveY,
            second,
            FaceKind::NegativeY,
            Vec3::new(0.0, 2.0, 0.0),
            Vec3::Y,
        );
        let limits = DriveLimits::new(4.0, 20.0, Some((-1.0, 1.0))).unwrap();
        let program =
            DriveProgram::new(&[DriveState::new(DriveTarget::Speed(3.0)).unwrap()], false).unwrap();
        wire_with(&mut graph, lower, limits, program, true);

        let compiled = graph.compile().unwrap();
        let drives = compiled.resolve_coordinate_drives(&graph);
        assert_eq!(drives.len(), 2);
        let driven_index = compiled
            .loop_topology
            .tree_bearings
            .iter()
            .position(|&bearing| bearing == lower)
            .unwrap();
        let motor_row = drives[driven_index];
        assert_eq!(motor_row.mode, DriveMode::Speed);
        assert!((motor_row.target_speed + 3.0).abs() < f32::EPSILON);
        assert!((motor_row.min_angle + 1.0).abs() < f32::EPSILON);
        assert!((motor_row.max_angle - 1.0).abs() < f32::EPSILON);
        let expected = 500.0 / compiled.loop_topology.coordinate_axis_inertia[driven_index];
        assert!((motor_row.max_acceleration - expected).abs() < 1.0e-3);

        let passive_index = 1 - driven_index;
        assert_eq!(drives[passive_index], CoordinateDrive::PASSIVE);
    }

    #[test]
    fn hardware_torque_replaces_the_legacy_arbitrary_torque_limit() {
        let mut graph = ConstructionGraph::new();
        let base = spawn(&mut graph, IVec3::new(0, 2, 0));
        ground(&mut graph, base);
        let arm = spawn(&mut graph, IVec3::new(4, 2, 0));
        let bearing = add_bearing(
            &mut graph,
            base,
            FaceKind::PositiveX,
            arm,
            FaceKind::NegativeX,
            Vec3::new(0.5, 0.5, 0.0),
            Vec3::X,
        );
        wire(&mut graph, bearing, false);

        let compiled = graph.compile().unwrap();
        let drives = compiled.resolve_coordinate_drives(&graph);
        assert!(drives[0].max_acceleration.is_finite());
        assert!(drives[0].max_acceleration > 0.0);
        assert!(drives[0].min_angle.is_infinite() && drives[0].min_angle < 0.0);
    }

    #[test]
    fn one_engine_splits_its_torque_across_its_assigned_coordinates() {
        let mut graph = ConstructionGraph::new();
        let base = spawn(&mut graph, IVec3::new(0, 2, 0));
        ground(&mut graph, base);
        let right = spawn(&mut graph, IVec3::new(4, 2, 0));
        let left = spawn(&mut graph, IVec3::new(-4, 2, 0));
        let bearings = [
            add_bearing(
                &mut graph,
                base,
                FaceKind::PositiveX,
                right,
                FaceKind::NegativeX,
                Vec3::new(0.5, 0.5, 0.0),
                Vec3::X,
            ),
            add_bearing(
                &mut graph,
                base,
                FaceKind::NegativeX,
                left,
                FaceKind::PositiveX,
                Vec3::new(-0.5, 0.5, 0.0),
                -Vec3::X,
            ),
        ];
        let BuildOutcome::Spawned(controller) = graph
            .apply(BuildCommand::SpawnController(ControllerSpec::new(
                BuildPose::new(IVec3::new(0, 40, 0), GridRotation::default()),
            )))
            .unwrap()
        else {
            unreachable!()
        };
        let BuildOutcome::Spawned(engine) = graph
            .apply(BuildCommand::SpawnEngine(EngineSpec::new(
                EngineKind::Electric,
                BuildPose::new(IVec3::new(0, 42, 0), GridRotation::default()),
            )))
            .unwrap()
        else {
            unreachable!()
        };
        graph
            .apply(BuildCommand::Weld(WeldSpec {
                first: FaceRef::part(controller, FaceKind::PositiveY),
                second: FaceRef::part(engine, FaceKind::NegativeY),
            }))
            .unwrap();
        for bearing in bearings {
            let mut link = DriveLinkSpec::new(controller, bearing);
            link.actuator = ActuatorAssignment::motor(100, 0).unwrap();
            link.program =
                DriveProgram::new(&[DriveState::new(DriveTarget::Speed(20.0)).unwrap()], false)
                    .unwrap();
            graph.apply(BuildCommand::AddDriveLink(link)).unwrap();
        }

        let compiled = graph.compile().unwrap();
        let drives = compiled.resolve_coordinate_drives(&graph);
        assert_eq!(drives.len(), 2);
        for (coordinate, drive) in drives.iter().enumerate() {
            let torque = drive.source_a_max_acceleration
                * compiled.loop_topology.coordinate_axis_inertia[coordinate];
            assert!((torque - 250.0).abs() < 1.0e-3);
            assert!((drive.max_speed - 4.0 * core::f32::consts::PI).abs() < 1.0e-5);
            assert!((drive.target_speed - 4.0 * core::f32::consts::PI).abs() < 1.0e-5);
        }
    }

    #[test]
    fn ideal_gearing_multiplies_torque_divides_speed_and_can_disengage_one_family() {
        let mut graph = ConstructionGraph::new();
        let base = spawn(&mut graph, IVec3::new(0, 2, 0));
        ground(&mut graph, base);
        let arm = spawn(&mut graph, IVec3::new(4, 2, 0));
        let bearing = add_bearing(
            &mut graph,
            base,
            FaceKind::PositiveX,
            arm,
            FaceKind::NegativeX,
            Vec3::new(0.5, 0.5, 0.0),
            Vec3::X,
        );
        let controller = wire(&mut graph, bearing, false);
        let compiled = graph.compile().unwrap();
        let direct = compiled.resolve_coordinate_drives(&graph)[0];
        let geared = compiled.resolve_coordinate_drives_with_gears(
            &graph,
            &[GearSelection {
                controller,
                kind: EngineKind::Electric,
                ratio: Some(4.0),
            }],
        )[0];
        assert!(
            (geared.source_a_max_acceleration / direct.source_a_max_acceleration - 4.0).abs()
                < 1.0e-5
        );
        assert!(
            (direct.source_a_no_load_speed / geared.source_a_no_load_speed - 4.0).abs() < 1.0e-5
        );

        let disengaged = compiled.resolve_coordinate_drives_with_gears(
            &graph,
            &[GearSelection {
                controller,
                kind: EngineKind::Electric,
                ratio: None,
            }],
        )[0];
        assert_eq!(disengaged, CoordinateDrive::PASSIVE);
    }

    #[test]
    fn assigned_motor_without_a_touching_engine_is_rejected() {
        let mut graph = ConstructionGraph::new();
        let base = spawn(&mut graph, IVec3::new(0, 2, 0));
        ground(&mut graph, base);
        let arm = spawn(&mut graph, IVec3::new(4, 2, 0));
        let bearing = add_bearing(
            &mut graph,
            base,
            FaceKind::PositiveX,
            arm,
            FaceKind::NegativeX,
            Vec3::new(0.5, 0.5, 0.0),
            Vec3::X,
        );
        let BuildOutcome::Spawned(controller) = graph
            .apply(BuildCommand::SpawnController(ControllerSpec::new(
                BuildPose::new(IVec3::new(0, 40, 0), GridRotation::default()),
            )))
            .unwrap()
        else {
            unreachable!()
        };
        let mut link = DriveLinkSpec::new(controller, bearing);
        link.actuator = ActuatorAssignment::motor(100, 0).unwrap();
        link.program =
            DriveProgram::new(&[DriveState::new(DriveTarget::Speed(1.0)).unwrap()], false).unwrap();
        graph.apply(BuildCommand::AddDriveLink(link)).unwrap();

        assert_eq!(
            graph.compile(),
            Err(TopologyError::InsufficientElectricPorts {
                controller,
                required: 1,
                available: 0,
            })
        );
    }

    fn layered_steel_wheel() -> CylinderSpec {
        CylinderSpec::new(
            CylinderDimensions::new(1.0, 0.0, 2.0).unwrap(),
            BuildPose::default(),
        )
        .with_layer(
            crate::LayerFace::OuterWall,
            0.25,
            crate::ConstructionMaterial::Rubber,
            crate::MaterialAppearance::BAKED,
        )
        .unwrap()
    }

    #[test]
    fn layered_cylinder_mass_sums_band_densities() {
        let mut graph = ConstructionGraph::new();
        graph
            .apply(BuildCommand::SpawnCylinder(layered_steel_wheel()))
            .unwrap();
        let properties = graph.compile().unwrap().compounds[0].mass_properties;
        let density = |material: crate::ConstructionMaterial| material.properties().density_kg_m3;
        let pi = core::f32::consts::PI;
        let steel = density(crate::ConstructionMaterial::Steel) * pi * 0.25 * 2.0;
        let rubber = density(crate::ConstructionMaterial::Rubber) * pi * (0.5625 - 0.25) * 2.0;
        let axial = steel * 0.25 * 0.5 + rubber * (0.5625 + 0.25) * 0.5;
        // Bands are 24-sided prisms, a percent or two under the true circle.
        assert!((properties.mass - (steel + rubber)).abs() < 0.02 * properties.mass);
        assert!(properties.center_of_mass.abs_diff_eq(Vec3::ZERO, 1.0e-4));
        assert!((properties.inertia.y_axis.y - axial).abs() < 0.04 * axial);
    }

    #[test]
    fn layered_cuboid_mass_sums_band_densities() {
        let core = CuboidSpec::new([4, 4, 4], BuildPose::default()).unwrap();
        let layered = PartSpec::Cuboid(core)
            .with_layer(
                crate::LayerFace::Face(crate::FaceKind::PositiveY),
                0.25,
                crate::ConstructionMaterial::Rubber,
                crate::MaterialAppearance::BAKED,
            )
            .unwrap()
            .as_cuboid()
            .unwrap();
        let mut graph = ConstructionGraph::new();
        graph.apply(BuildCommand::Spawn(layered)).unwrap();
        let compiled = graph.compile().unwrap();
        let properties = compiled.compounds[0].mass_properties;
        let steel = crate::ConstructionMaterial::Steel
            .properties()
            .density_kg_m3;
        let rubber = crate::ConstructionMaterial::Rubber
            .properties()
            .density_kg_m3
            * 0.25;
        assert!((properties.mass - (steel + rubber)).abs() < 1.0e-3 * properties.mass);
        // The core's centre sits at y = 0, the layer's at 0.625 m.
        let center_y = rubber * 0.625 / (steel + rubber);
        assert!((properties.center_of_mass.y - center_y).abs() < 1.0e-3);
        for material in [
            crate::ConstructionMaterial::Steel,
            crate::ConstructionMaterial::Rubber,
        ] {
            assert!(
                compiled
                    .colliders
                    .iter()
                    .any(|collider| collider.material_properties == material.properties()),
                "{material:?} band has colliders"
            );
        }
    }

    #[test]
    fn layered_wheel_keeps_analytic_contact_with_its_tyre_material() {
        let mut graph = ConstructionGraph::new();
        graph
            .apply(BuildCommand::SpawnCylinder(layered_steel_wheel()))
            .unwrap();
        let compiled = graph.compile().unwrap();
        assert_eq!(compiled.cylinders.len(), 1);
        assert!((compiled.cylinders[0].outer_radius - 0.75).abs() < 1.0e-4);
        assert!(compiled.colliders.iter().all(|collider| {
            collider.material_properties == crate::ConstructionMaterial::Rubber.properties()
        }));
    }

    #[test]
    fn featured_layer_colliders_carry_their_band_material() {
        let mut graph = ConstructionGraph::new();
        let BuildOutcome::Spawned(part) = graph
            .apply(BuildCommand::SpawnCylinder(layered_steel_wheel()))
            .unwrap()
        else {
            panic!("spawning a cylinder reports its part");
        };
        let owner = crate::SolidOwner::Part(part);
        let rim = graph
            .evaluated_solid(owner)
            .unwrap()
            .logical_edges
            .iter()
            .find(|edge| edge.closed && edge.convex)
            .unwrap()
            .key;
        graph
            .apply(BuildCommand::AddShapeFeature(crate::ShapeFeature::new(
                [crate::EdgeChainRef { owner, edge: rim }],
                crate::EdgeTreatment::Fillet,
                120,
            )))
            .unwrap();
        let compiled = graph.compile().unwrap();
        for material in [
            crate::ConstructionMaterial::Steel,
            crate::ConstructionMaterial::Rubber,
        ] {
            assert!(
                compiled
                    .colliders
                    .iter()
                    .any(|collider| collider.material_properties == material.properties()),
                "{material:?} band has colliders"
            );
        }
    }
}
