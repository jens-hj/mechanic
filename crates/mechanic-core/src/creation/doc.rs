//! The serialized rows of a creation file.

use crate::{
    ActuatorAssignment, BuildPose, CageIndex, ConstructionMaterial, DimensionLinkId, DriveRelease,
    DriveTarget, EdgeTreatment, EngineKind, FaceKind, GearKeyChord, MaterialAppearance, ShiftMode,
};
use serde::{Deserialize, Serialize};

/// Grid-aligned pose in its serialized form.
///
/// Translation uses exact 2.5 mm ticks. Older coordinate encodings are rejected.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PoseDoc {
    /// Centre in exact integer 2.5 mm ticks.
    pub translation_ticks: [i32; 3],
    /// Quarter turns around local x, y, and z.
    pub rotation: [u8; 3],
}

impl From<BuildPose> for PoseDoc {
    fn from(pose: BuildPose) -> Self {
        let translation = pose.translation_position_ticks();
        Self {
            translation_ticks: [translation.x, translation.y, translation.z],
            rotation: pose.rotation.quarter_turns_xyz(),
        }
    }
}

/// One material layer in its serialized form.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct MaterialLayerDoc {
    /// Surface the layer was laid on.
    pub face: crate::LayerFace,
    /// Thickness in metres.
    pub thickness: f32,
    /// Physical material.
    pub material: ConstructionMaterial,
    /// Color and finish treatment.
    pub appearance: MaterialAppearance,
}

/// One construction part in its serialized form.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum PartDoc {
    /// Rectangular cuboid, sized in quarter-metre grid units.
    Cuboid {
        /// Integer x/y/z core side lengths.
        dimensions: [u8; 3],
        /// Core centre and orientation.
        pose: PoseDoc,
        /// Core physical material.
        material: ConstructionMaterial,
        /// Core color and finish treatment.
        appearance: MaterialAppearance,
        /// Material layers replayed over the core, oldest first.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        layers: Vec<MaterialLayerDoc>,
    },
    /// Solid or hollow cylinder whose axis is local Y.
    Cylinder {
        /// Core outer diameter in metres.
        outer_diameter: f32,
        /// Core inner diameter in metres. Zero is solid.
        inner_diameter: f32,
        /// Core axial length in quarter-metre grid units.
        length_units: u8,
        /// Retained angular sector in degrees.
        sweep_degrees: u16,
        /// Core centre and orientation.
        pose: PoseDoc,
        /// Core physical material.
        material: ConstructionMaterial,
        /// Core color and finish treatment.
        appearance: MaterialAppearance,
        /// Material layers replayed over the core, oldest first.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        layers: Vec<MaterialLayerDoc>,
    },
    /// Cardinal 90-degree quarter-torus pipe bend.
    PipeBend {
        /// Outer diameter in metres.
        outer_diameter: f32,
        /// Inner diameter in metres. Zero is solid.
        inner_diameter: f32,
        /// Side of the bend's square footprint, in blocks.
        span_blocks: u8,
        /// Sharp-corner position and cardinal orientation.
        pose: PoseDoc,
        /// Physical material.
        material: ConstructionMaterial,
        /// Color and finish treatment.
        appearance: MaterialAppearance,
    },
    /// Cube fitting joining pipe ends on any of its faces.
    PipeJunction {
        /// Outer diameter of each pipe end in metres.
        outer_diameter: f32,
        /// Bore diameter in metres. Zero is solid.
        inner_diameter: f32,
        /// Open faces as bits ordered +X, −X, +Y, −Y, +Z, −Z.
        arms: u8,
        /// Cube centre and cardinal orientation.
        pose: PoseDoc,
        /// Physical material.
        material: ConstructionMaterial,
        /// Color and finish treatment.
        appearance: MaterialAppearance,
    },
    /// Fixed-size control block.
    Controller {
        /// Centre and orientation.
        pose: PoseDoc,
    },
    /// Fixed-size gas or electric engine.
    Engine {
        /// Authored engine family.
        kind: EngineKind,
        /// Centre and orientation.
        pose: PoseDoc,
    },
    /// Fixed-size transmission with a graph-owned upstream relation.
    Transmission {
        /// Index of the engine or transmission whose positive-Z output it extends.
        parent: u32,
        /// Centre and inherited root-engine orientation.
        pose: PoseDoc,
    },
    /// Fixed-size servo.
    Servo {
        /// Centre and orientation.
        pose: PoseDoc,
    },
    /// Fixed-size seat cushion.
    Seat {
        /// Centre and orientation.
        pose: PoseDoc,
    },
    /// Fixed-size keyboard Input block.
    Input {
        /// Centre and orientation.
        pose: PoseDoc,
    },
    /// Fixed-size Dimension Link portal anchor.
    DimensionLink {
        /// Stable identity within its owning world and Garage.
        id: DimensionLinkId,
        /// Centre and orientation.
        pose: PoseDoc,
    },
}

/// One shape region in its serialized form.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RegionDoc {
    /// Minimum corner in shape steps (1.25 cm).
    pub origin_steps: [i32; 3],
    /// Extent in construction cells.
    pub size_cells: [i32; 3],
    /// Material every block in the region shares.
    pub material: ConstructionMaterial,
    /// Color and finish shared by every region member.
    pub appearance: MaterialAppearance,
    /// Cage planes beyond the two the extent implies, in cells, per axis.
    #[serde(default)]
    pub divisions: [Vec<i32>; 3],
    /// Displaced cage vertices.
    #[serde(default)]
    pub vertices: Vec<(CageIndex, [i16; 3])>,
}

/// Owner of a serialized face: a part index, or the ground plane.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FaceOwnerDoc {
    /// Index into the document's part list.
    Part(u32),
    /// The static ground plane.
    Ground,
}

/// Reference to one oriented face in its serialized form.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FaceRefDoc {
    /// Part index or ground.
    pub owner: FaceOwnerDoc,
    /// Oriented face on that owner.
    pub face: FaceKind,
    /// Stable evaluated surface patch, when this connection uses trimmed or
    /// feature-generated geometry.
    #[serde(default)]
    pub patch: Option<TopologyKeyDoc>,
}

/// Serialized owner of a parametric feature target.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SolidOwnerDoc {
    /// Index into the document's part list.
    Part(u32),
    /// Index into the document's Shape-region list.
    Region(u32),
}

/// Serialized provenance of a stable topology key.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TopologySourceDoc {
    /// Base-generator topology.
    Base,
    /// Index of an earlier record in the ordered feature list.
    Feature(u32),
}

/// Serialized stable logical-curve key.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TopologyKeyDoc {
    /// Base or generating-feature provenance.
    pub source: TopologySourceDoc,
    /// Deterministic identity within that provenance.
    pub local: u32,
}

/// One serialized feature target.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EdgeChainRefDoc {
    /// Part or region carrying the chain.
    pub owner: SolidOwnerDoc,
    /// Stable logical edge key.
    pub edge: TopologyKeyDoc,
}

/// One record in explicit global feature order.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShapeFeatureDoc {
    /// Logical chains treated together.
    pub targets: Vec<EdgeChainRefDoc>,
    /// Chamfer or fillet.
    pub treatment: EdgeTreatment,
    /// Equal setback or radius in exact position ticks.
    pub amount_ticks: u32,
}

/// Weld between two touching faces.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WeldDoc {
    /// First welded face.
    pub first: FaceRefDoc,
    /// Second welded face.
    pub second: FaceRefDoc,
}

/// Non-geometric rigid membership between two part indices.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RigidLinkDoc {
    /// First part index.
    pub first: u32,
    /// Second part index.
    pub second: u32,
}

/// One-degree-of-freedom bearing in its serialized form.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct BearingDoc {
    /// Motion kind and complete rail configuration.
    pub kind: crate::BearingKind,
    /// Face whose outward normal establishes the axis.
    pub source: FaceRefDoc,
    /// Compatible face on the attached side.
    pub target: FaceRefDoc,
    /// Shared world-space anchor.
    pub anchor: [f32; 3],
    /// Unit world-space axis.
    pub axis: [f32; 3],
    /// Visual outer diameter in metres.
    pub outer_diameter: f32,
    /// Visual inner diameter in metres.
    pub inner_diameter: f32,
}

/// Unattached bearing ring in its serialized form.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct BearingSocketDoc {
    /// Physical motion variant and rail frame.
    pub kind: crate::BearingKind,
    /// World-space travel axis for a linear socket.
    pub axis: [f32; 3],
    /// Face the ring sits on.
    pub source: FaceRefDoc,
    /// World-space point the ring is centred on.
    pub anchor: [f32; 3],
    /// Visual outer diameter in metres.
    pub outer_diameter: f32,
    /// Visual inner diameter in metres.
    pub inner_diameter: f32,
}

/// Speed, torque, and travel envelope in its serialized form.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct DriveLimitsDoc {
    /// Fastest the bearing may turn, in radians per second.
    pub max_speed_rad_s: f32,
    /// Maximum applied torque in newton metres. `None` is unlimited, which is
    /// how the panel's `inf` reads on disk without encoding a float infinity.
    pub max_torque_newton_meters: Option<f32>,
    /// Travel limits in radians, when the bearing stops and holds at its ends.
    pub angle_limits: Option<(f32, f32)>,
}

/// Automatic handoff in its serialized form.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct DriveDwellDoc {
    /// How long the state stays active, in seconds.
    pub seconds: f32,
    /// Explicit handoff target, or `None` for the following state.
    pub next: Option<u8>,
}

/// Key binding in its serialized form.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DriveTriggerDoc {
    /// Bound character, always an uppercase letter or a digit.
    pub key: char,
    /// What happens when the key is released.
    pub release: DriveRelease,
}

/// One drive state in its serialized form.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct DriveStateDoc {
    /// What this state asks of the bearing.
    pub target: DriveTarget,
    /// Automatic handoff, when this state advances on its own.
    pub dwell: Option<DriveDwellDoc>,
    /// Key binding, when this state can be triggered by hand.
    pub trigger: Option<DriveTriggerDoc>,
}

/// Ordered states of one driven bearing in their serialized form.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DriveProgramDoc {
    /// Whether the last state hands back to the first.
    pub loops: bool,
    /// The states, in order.
    pub states: Vec<DriveStateDoc>,
}

/// Wire from a control block to one bearing, in its serialized form.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DriveLinkDoc {
    /// Typed linear limits in SI units.
    pub linear_limits: Option<crate::LinearDriveLimits>,
    /// Index of the control-block part this wire belongs to.
    pub controller: u32,
    /// Index of the bearing driven through this wire.
    pub bearing: u32,
    /// Whether this bearing runs opposite the programmed direction.
    pub reversed: bool,
    /// Hardware family assigned to this joint. Older files load unpowered.
    #[serde(default)]
    pub actuator: ActuatorAssignment,
    /// Speed, torque, and travel envelope.
    pub limits: DriveLimitsDoc,
    /// Ordered states this bearing moves through.
    pub program: DriveProgramDoc,
    /// What the panel calls this joint. Absent in files written before joints
    /// could be named, which read back as unnamed.
    #[serde(default)]
    pub name: String,
}

/// Logical link from an Input block to a Seat.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct InputSeatLinkDoc {
    /// Index of the Input part.
    pub input: u32,
    /// Index of the Seat part.
    pub seat: u32,
}

/// Logical link from a Seat to a Controller.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SeatControllerLinkDoc {
    /// Index of the Seat part.
    pub seat: u32,
    /// Index of the Controller part.
    pub controller: u32,
}

/// Persistent gearbox settings for one Controller and engine family.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GearboxConfigDoc {
    /// Index of the Controller part owning this lane.
    pub controller: u32,
    /// Engine family being configured.
    pub kind: EngineKind,
    /// Automatic or manual shifting.
    pub mode: ShiftMode,
    /// Strictly descending input-to-output ratios.
    pub ratios: Vec<f32>,
    /// Number of leading gas ratios assigned to reverse.
    pub reverse_gears: u8,
    /// Manual upshift chord.
    pub gear_up: GearKeyChord,
    /// Manual downshift chord.
    pub gear_down: GearKeyChord,
}

/// Rigid transform of an authored local construction grid.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConstructionFrameDoc {
    /// Translation from the local grid into build space, in metres.
    pub translation: [f32; 3],
    /// Unit quaternion in x/y/z/w order.
    pub rotation: [f32; 4],
}
