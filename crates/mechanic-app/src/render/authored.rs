//! Authored machine parts: orientations, cube geometry, and their texture atlases.

use crate::hotbar::Tool;
use bevy::prelude::{Color, Component};
use mechanic_core::{ConstructionGraph, EngineKind, GridRotation, PartId, PartSpec};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AuthoredPart {
    Controller,
    GasEngine,
    ElectricEngine,
    GasTransmission,
    ElectricTransmission,
    Servo,
    Seat,
    Input,
    DimensionLinkDisabled,
    DimensionLinkEnabled,
}

pub(crate) const CONTROLLER_SURFACE_COLOR: Color = Color::srgb(0.10, 0.78, 0.68);

impl AuthoredPart {
    pub(crate) const ALL: [Self; 10] = [
        Self::Controller,
        Self::GasEngine,
        Self::ElectricEngine,
        Self::GasTransmission,
        Self::ElectricTransmission,
        Self::Servo,
        Self::Seat,
        Self::Input,
        Self::DimensionLinkDisabled,
        Self::DimensionLinkEnabled,
    ];

    pub(crate) const fn index(self) -> usize {
        match self {
            Self::Controller => 0,
            Self::GasEngine => 1,
            Self::ElectricEngine => 2,
            Self::GasTransmission => 3,
            Self::ElectricTransmission => 4,
            Self::Servo => 5,
            Self::Seat => 6,
            Self::Input => 7,
            Self::DimensionLinkDisabled => 8,
            Self::DimensionLinkEnabled => 9,
        }
    }

    pub(crate) const fn from_tool(tool: Tool) -> Option<Self> {
        match tool {
            Tool::Controller => Some(Self::Controller),
            Tool::GasEngine => Some(Self::GasEngine),
            Tool::ElectricEngine => Some(Self::ElectricEngine),
            Tool::Servo => Some(Self::Servo),
            Tool::Seat => Some(Self::Seat),
            Tool::Input => Some(Self::Input),
            Tool::DimensionLink => Some(Self::DimensionLinkDisabled),
            _ => None,
        }
    }

    pub(crate) fn matches(
        self,
        graph: &ConstructionGraph,
        part: PartId,
        spec: PartSpec,
        active: Option<mechanic_core::DimensionLinkId>,
    ) -> bool {
        if let PartSpec::DimensionLink(link) = spec {
            return match self {
                Self::DimensionLinkDisabled => Some(link.id) != active,
                Self::DimensionLinkEnabled => Some(link.id) == active,
                _ => false,
            };
        }
        matches!(
            (self, spec),
            (Self::Controller, PartSpec::Controller(_))
                | (
                    Self::GasEngine,
                    PartSpec::Engine(mechanic_core::EngineSpec {
                        kind: EngineKind::Gas,
                        ..
                    }),
                )
                | (Self::Servo, PartSpec::Servo(_))
                | (Self::Seat, PartSpec::Seat(_))
                | (Self::Input, PartSpec::Input(_))
                | (
                    Self::ElectricEngine,
                    PartSpec::Engine(mechanic_core::EngineSpec {
                        kind: EngineKind::Electric,
                        ..
                    }),
                )
        ) || matches!(spec, PartSpec::Transmission(_))
            && match self {
                Self::GasTransmission => graph.transmission_kind(part) == Some(EngineKind::Gas),
                Self::ElectricTransmission => {
                    graph.transmission_kind(part) == Some(EngineKind::Electric)
                }
                _ => false,
            }
    }
}

pub(crate) const AUTHORED_ORIENTATION_COUNT: u8 = 24;

pub(crate) const AUTHORED_ORIENTATIONS: [GridRotation; 24] = [
    GridRotation::new(0, 0, 0),
    GridRotation::new(0, 1, 0),
    GridRotation::new(0, 2, 0),
    GridRotation::new(0, 3, 0),
    GridRotation::new(0, 0, 1),
    GridRotation::new(0, 0, 2),
    GridRotation::new(0, 0, 3),
    GridRotation::new(0, 1, 1),
    GridRotation::new(0, 1, 2),
    GridRotation::new(0, 1, 3),
    GridRotation::new(0, 2, 1),
    GridRotation::new(0, 2, 2),
    GridRotation::new(0, 2, 3),
    GridRotation::new(0, 3, 1),
    GridRotation::new(0, 3, 2),
    GridRotation::new(0, 3, 3),
    GridRotation::new(1, 0, 0),
    GridRotation::new(1, 0, 1),
    GridRotation::new(1, 0, 2),
    GridRotation::new(1, 0, 3),
    GridRotation::new(1, 2, 0),
    GridRotation::new(1, 2, 1),
    GridRotation::new(1, 2, 2),
    GridRotation::new(1, 2, 3),
];

pub(crate) fn authored_orientation(index: u8) -> GridRotation {
    AUTHORED_ORIENTATIONS[usize::from(index) % AUTHORED_ORIENTATIONS.len()]
}

#[derive(Component)]
pub(crate) struct AuthoredPartVisual(pub(crate) AuthoredPart);

pub(crate) const CUBE_NORMALS: [[f32; 3]; 24] = [
    [0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, -1.0, 0.0],
    [0.0, -1.0, 0.0],
    [0.0, -1.0, 0.0],
    [0.0, -1.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, 0.0, 1.0],
    [0.0, 0.0, 1.0],
    [0.0, 0.0, 1.0],
    [0.0, 0.0, -1.0],
    [0.0, 0.0, -1.0],
    [0.0, 0.0, -1.0],
    [0.0, 0.0, -1.0],
];

pub(crate) const CUBE_INDICES: [u32; 36] = [
    0, 3, 1, 1, 3, 2, 4, 5, 7, 5, 6, 7, 8, 11, 9, 9, 11, 10, 12, 13, 15, 13, 14, 15, 16, 19, 17,
    17, 19, 18, 20, 21, 23, 21, 22, 23,
];

// Authored machine GLBs use +X, -X, +Y, -Y, +Z, -Z face order. Keeping that
// template here lets their UV atlases stay exact while placed parts remain in
// the app's batched meshes rather than becoming one entity per part.
pub(crate) const AUTHORED_CUBE_POSITIONS: [[f32; 3]; 24] = [
    [0.5, 0.5, 0.5],
    [0.5, -0.5, 0.5],
    [0.5, -0.5, -0.5],
    [0.5, 0.5, -0.5],
    [-0.5, 0.5, -0.5],
    [-0.5, -0.5, -0.5],
    [-0.5, -0.5, 0.5],
    [-0.5, 0.5, 0.5],
    [-0.5, 0.5, -0.5],
    [-0.5, 0.5, 0.5],
    [0.5, 0.5, 0.5],
    [0.5, 0.5, -0.5],
    [-0.5, -0.5, 0.5],
    [-0.5, -0.5, -0.5],
    [0.5, -0.5, -0.5],
    [0.5, -0.5, 0.5],
    [-0.5, 0.5, 0.5],
    [-0.5, -0.5, 0.5],
    [0.5, -0.5, 0.5],
    [0.5, 0.5, 0.5],
    [0.5, 0.5, -0.5],
    [0.5, -0.5, -0.5],
    [-0.5, -0.5, -0.5],
    [-0.5, 0.5, -0.5],
];

pub(crate) const AUTHORED_CUBE_NORMALS: [[f32; 3]; 24] = [
    [1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, -1.0, 0.0],
    [0.0, -1.0, 0.0],
    [0.0, -1.0, 0.0],
    [0.0, -1.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, 0.0, 1.0],
    [0.0, 0.0, 1.0],
    [0.0, 0.0, 1.0],
    [0.0, 0.0, -1.0],
    [0.0, 0.0, -1.0],
    [0.0, 0.0, -1.0],
    [0.0, 0.0, -1.0],
];

pub(crate) const AUTHORED_CUBE_TANGENTS: [[f32; 4]; 24] = [
    [0.0, 0.0, -1.0, -1.0],
    [0.0, 0.0, -1.0, -1.0],
    [0.0, 0.0, -1.0, -1.0],
    [0.0, 0.0, -1.0, -1.0],
    [0.0, 0.0, 1.0, -1.0],
    [0.0, 0.0, 1.0, -1.0],
    [0.0, 0.0, 1.0, -1.0],
    [0.0, 0.0, 1.0, -1.0],
    [1.0, 0.0, 0.0, -1.0],
    [1.0, 0.0, 0.0, -1.0],
    [1.0, 0.0, 0.0, -1.0],
    [1.0, 0.0, 0.0, -1.0],
    [1.0, 0.0, 0.0, -1.0],
    [1.0, 0.0, 0.0, -1.0],
    [1.0, 0.0, 0.0, -1.0],
    [1.0, 0.0, 0.0, -1.0],
    [1.0, 0.0, 0.0, -1.0],
    [1.0, 0.0, 0.0, -1.0],
    [1.0, 0.0, 0.0, -1.0],
    [1.0, 0.0, 0.0, -1.0],
    [-1.0, 0.0, 0.0, -1.0],
    [-1.0, 0.0, 0.0, -1.0],
    [-1.0, 0.0, 0.0, -1.0],
    [-1.0, 0.0, 0.0, -1.0],
];

pub(crate) const AUTHORED_CUBE_INDICES: [u32; 36] = [
    0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7, 8, 9, 10, 8, 10, 11, 12, 13, 14, 12, 14, 15, 16, 17, 18,
    16, 18, 19, 20, 21, 22, 20, 22, 23,
];

pub(crate) const CONTROLLER_UVS: [[f32; 2]; 24] = [
    [0.0, 0.5],
    [0.0, 0.0],
    [0.25, 0.0],
    [0.25, 0.5],
    [0.25, 0.5],
    [0.25, 0.0],
    [0.5, 0.0],
    [0.5, 0.5],
    [0.5, 0.5],
    [0.5, 0.25],
    [1.0, 0.25],
    [1.0, 0.5],
    [0.5, 0.25],
    [0.5, 0.0],
    [1.0, 0.0],
    [1.0, 0.25],
    [0.0, 1.0],
    [0.0, 0.5],
    [0.5, 0.5],
    [0.5, 1.0],
    [0.5, 1.0],
    [0.5, 0.5],
    [1.0, 0.5],
    [1.0, 1.0],
];

// Both transmission GLBs use this same six-tile atlas layout. The imported
// vertex order differs within each face, but remapping it onto
// `AUTHORED_CUBE_POSITIONS` produces the controller-style ordering below.
pub(crate) const TRANSMISSION_UVS: [[f32; 2]; 24] = CONTROLLER_UVS;

pub(crate) const GAS_ENGINE_UVS: [[f32; 2]; 24] = [
    [0.0, 1.0],
    [0.0, 0.666_667],
    [0.5, 0.666_667],
    [0.5, 1.0],
    [0.5, 1.0],
    [0.5, 0.666_667],
    [1.0, 0.666_667],
    [1.0, 1.0],
    [0.0, 0.666_667],
    [0.0, 0.166_667],
    [0.333_333, 0.166_667],
    [0.333_333, 0.666_667],
    [0.333_333, 0.666_667],
    [0.333_333, 0.166_667],
    [0.666_667, 0.166_667],
    [0.666_667, 0.666_667],
    [0.666_667, 0.666_667],
    [0.666_667, 0.333_333],
    [1.0, 0.333_333],
    [1.0, 0.666_667],
    [0.666_667, 0.333_333],
    [0.666_667, 0.0],
    [1.0, 0.0],
    [1.0, 0.333_333],
];

pub(crate) const ELECTRIC_ENGINE_UVS: [[f32; 2]; 24] = [
    [0.666_667, 1.0],
    [0.666_667, 0.5],
    [1.0, 0.5],
    [1.0, 1.0],
    [0.0, 0.5],
    [0.0, 0.0],
    [0.333_333, 0.0],
    [0.333_333, 0.5],
    [0.333_333, 0.5],
    [0.333_333, 0.0],
    [0.666_667, 0.0],
    [0.666_667, 0.5],
    [0.666_667, 0.5],
    [0.666_667, 0.0],
    [1.0, 0.0],
    [1.0, 0.5],
    [0.0, 1.0],
    [0.0, 0.5],
    [0.333_333, 0.5],
    [0.333_333, 1.0],
    [0.333_333, 1.0],
    [0.333_333, 0.5],
    [0.666_667, 0.5],
    [0.666_667, 1.0],
];

pub(crate) const SERVO_UVS: [[f32; 2]; 24] = [
    [0.666_667, 0.5],
    [0.666_667, 1.0],
    [1.0, 1.0],
    [1.0, 0.5],
    [0.0, 0.0],
    [0.0, 0.5],
    [0.333_333, 0.5],
    [0.333_333, 0.0],
    [0.333_333, 0.0],
    [0.333_333, 0.5],
    [0.666_667, 0.5],
    [0.666_667, 0.0],
    [0.666_667, 0.0],
    [0.666_667, 0.5],
    [1.0, 0.5],
    [1.0, 0.0],
    [0.0, 0.5],
    [0.0, 1.0],
    [0.333_333, 1.0],
    [0.333_333, 0.5],
    [0.333_333, 0.5],
    [0.333_333, 1.0],
    [0.666_667, 1.0],
    [0.666_667, 0.5],
];

pub(crate) const SEAT_UVS: [[f32; 2]; 24] = [
    [0.5, 0.25],
    [0.5, 0.5],
    [1.0, 0.5],
    [1.0, 0.25],
    [0.5, 0.0],
    [0.5, 0.25],
    [1.0, 0.25],
    [1.0, 0.0],
    [0.0, 0.5],
    [0.0, 1.0],
    [0.5, 1.0],
    [0.5, 0.5],
    [0.5, 0.5],
    [0.5, 1.0],
    [1.0, 1.0],
    [1.0, 0.5],
    [0.0, 0.25],
    [0.0, 0.5],
    [0.5, 0.5],
    [0.5, 0.25],
    [0.0, 0.0],
    [0.0, 0.25],
    [0.5, 0.25],
    [0.5, 0.0],
];

pub(crate) const INPUT_UVS: [[f32; 2]; 24] = [
    [0.0, 0.25],
    [0.0, 0.5],
    [0.25, 0.5],
    [0.25, 0.25],
    [0.25, 0.25],
    [0.25, 0.5],
    [0.5, 0.5],
    [0.5, 0.25],
    [0.0, 0.5],
    [0.0, 0.75],
    [0.5, 0.75],
    [0.5, 0.5],
    [0.5, 0.5],
    [0.5, 0.75],
    [1.0, 0.75],
    [1.0, 0.5],
    [0.0, 0.75],
    [0.0, 1.0],
    [0.5, 1.0],
    [0.5, 0.75],
    [0.5, 0.75],
    [0.5, 1.0],
    [1.0, 1.0],
    [1.0, 0.75],
];

pub(crate) const DIMENSION_LINK_UVS: [[f32; 2]; 24] = [
    [0.0, 0.5],
    [0.0, 0.25],
    [0.25, 0.25],
    [0.25, 0.5],
    [0.25, 0.5],
    [0.25, 0.25],
    [0.5, 0.25],
    [0.5, 0.5],
    [0.0, 0.75],
    [0.0, 0.5],
    [0.5, 0.5],
    [0.5, 0.75],
    [0.5, 0.75],
    [0.5, 0.5],
    [1.0, 0.5],
    [1.0, 0.75],
    [0.0, 1.0],
    [0.0, 0.75],
    [0.5, 0.75],
    [0.5, 1.0],
    [0.5, 1.0],
    [0.5, 0.75],
    [1.0, 0.75],
    [1.0, 1.0],
];

pub(crate) fn authored_uvs(appearance: AuthoredPart) -> [[f32; 2]; 24] {
    let assimp_uvs = match appearance {
        AuthoredPart::Controller => CONTROLLER_UVS,
        AuthoredPart::GasEngine => GAS_ENGINE_UVS,
        AuthoredPart::ElectricEngine => ELECTRIC_ENGINE_UVS,
        AuthoredPart::GasTransmission | AuthoredPart::ElectricTransmission => TRANSMISSION_UVS,
        AuthoredPart::Servo => SERVO_UVS,
        AuthoredPart::Seat => SEAT_UVS,
        AuthoredPart::Input => INPUT_UVS,
        AuthoredPart::DimensionLinkDisabled | AuthoredPart::DimensionLinkEnabled => {
            DIMENSION_LINK_UVS
        }
    };
    // Assimp's dump uses an OpenGL-style bottom-left texture origin. Bevy
    // samples the PNG atlases from the top left, matching the original glTF
    // accessor, so restore that V coordinate before building the runtime mesh.
    assimp_uvs.map(|[u, v]| [u, 1.0 - v])
}
