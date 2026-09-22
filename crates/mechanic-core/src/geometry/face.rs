//! Part faces: their kinds, owners, references, and outlines.

use super::CuboidSpec;
use super::cylinder::{CylinderSpec, MAX_CYLINDER_SWEEP_DEGREES};
use super::grid::{Axis, snap_cardinal};
use super::pipe::{PipeBendSpec, PipeJunctionSpec};
use crate::PartId;
use bevy_math::{Vec2, Vec3};
use serde::{Deserialize, Serialize};

/// One of the six oriented local faces. Cylinders expose only their Y ends.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FaceKind {
    /// Positive local x face.
    PositiveX,
    /// Negative local x face.
    NegativeX,
    /// Positive local y face.
    PositiveY,
    /// Negative local y face.
    NegativeY,
    /// Positive local z face.
    PositiveZ,
    /// Negative local z face.
    NegativeZ,
}

impl FaceKind {
    /// Face with the opposite local normal.
    #[must_use]
    pub const fn opposite(self) -> Self {
        match self {
            Self::PositiveX => Self::NegativeX,
            Self::NegativeX => Self::PositiveX,
            Self::PositiveY => Self::NegativeY,
            Self::NegativeY => Self::PositiveY,
            Self::PositiveZ => Self::NegativeZ,
            Self::NegativeZ => Self::PositiveZ,
        }
    }

    /// The local axis this face is perpendicular to.
    pub const fn axis(self) -> Axis {
        match self {
            Self::PositiveX | Self::NegativeX => Axis::X,
            Self::PositiveY | Self::NegativeY => Axis::Y,
            Self::PositiveZ | Self::NegativeZ => Axis::Z,
        }
    }

    /// Which way along its axis the face looks: `1.0` or `-1.0`.
    pub const fn sign(self) -> f32 {
        match self {
            Self::PositiveX | Self::PositiveY | Self::PositiveZ => 1.0,
            Self::NegativeX | Self::NegativeY | Self::NegativeZ => -1.0,
        }
    }

    pub(crate) const fn tangent_axes(self) -> (Axis, Axis) {
        match self.axis() {
            Axis::X => (Axis::Y, Axis::Z),
            Axis::Y => (Axis::X, Axis::Z),
            Axis::Z => (Axis::X, Axis::Y),
        }
    }
}

/// Object owning a selectable face.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FaceOwner {
    /// A user-created construction part.
    Part(PartId),
    /// The central static ground plane. Only its positive-y face is valid.
    Ground,
}

/// Stable reference to a part or ground face.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FaceRef {
    /// Face owner.
    pub owner: FaceOwner,
    /// Oriented face on that owner.
    pub face: FaceKind,
    /// Stable evaluated planar patch. `None` names the primitive face and lets
    /// feature evaluation resolve its trimmed remainder.
    pub patch: Option<crate::SurfacePatchKey>,
}

impl FaceRef {
    /// Creates a part-face reference.
    pub const fn part(part: PartId, face: FaceKind) -> Self {
        Self {
            owner: FaceOwner::Part(part),
            face,
            patch: None,
        }
    }

    /// Creates a reference to one evaluated planar surface patch.
    pub const fn patch(part: PartId, face: FaceKind, patch: crate::SurfacePatchKey) -> Self {
        Self {
            owner: FaceOwner::Part(part),
            face,
            patch: Some(patch),
        }
    }

    /// The ground's upward-facing plane.
    pub const fn ground() -> Self {
        Self {
            owner: FaceOwner::Ground,
            face: FaceKind::PositiveY,
            patch: None,
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct FaceGeometry {
    pub(crate) center: Vec3,
    pub(crate) normal: Vec3,
    pub(crate) tangent_u: Vec3,
    pub(crate) tangent_v: Vec3,
    pub(crate) profile: FaceProfile,
}

#[derive(Clone, Debug)]
pub(crate) enum FaceProfile {
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

pub(crate) fn cuboid_face(spec: CuboidSpec, face: FaceKind) -> FaceGeometry {
    envelope_face(spec.pose, spec.size_meters(), face)
}

pub(crate) fn envelope_face(pose: super::BuildPose, size: Vec3, face: FaceKind) -> FaceGeometry {
    let rotation = pose.rotation.quaternion();
    let axis = face.axis();
    let (u_axis, v_axis) = face.tangent_axes();
    let normal = snap_cardinal(rotation * axis.unit()) * face.sign();
    let tangent_u = snap_cardinal(rotation * u_axis.unit());
    let tangent_v = snap_cardinal(rotation * v_axis.unit());
    FaceGeometry {
        center: pose.translation() + normal * size[axis.index()] * 0.5,
        normal,
        tangent_u,
        tangent_v,
        profile: FaceProfile::Rectangle {
            half_u: size[u_axis.index()] * 0.5,
            half_v: size[v_axis.index()] * 0.5,
        },
    }
}

pub(crate) fn cylinder_face(spec: CylinderSpec, face: FaceKind) -> Option<FaceGeometry> {
    if !matches!(face, FaceKind::PositiveY | FaceKind::NegativeY) {
        return None;
    }
    let rotation = spec.pose.rotation.quaternion();
    let normal = snap_cardinal(rotation * Vec3::Y) * face.sign();
    let profile = if spec.dimensions.sweep_angle_degrees() == MAX_CYLINDER_SWEEP_DEGREES {
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

pub(crate) fn pipe_bend_face(spec: PipeBendSpec, face: FaceKind) -> Option<FaceGeometry> {
    let (local_center, local_normal, local_u, local_v) = match face {
        FaceKind::NegativeX => (
            Vec3::new(-spec.dimensions.radius(), 0.0, 0.0),
            Vec3::NEG_X,
            Vec3::Y,
            Vec3::Z,
        ),
        FaceKind::PositiveY => (
            Vec3::new(0.0, spec.dimensions.radius(), 0.0),
            Vec3::Y,
            Vec3::X,
            Vec3::Z,
        ),
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

/// Open arm end of a junction; closed faces are not connection faces.
pub(crate) fn pipe_junction_face(spec: PipeJunctionSpec, face: FaceKind) -> Option<FaceGeometry> {
    if !spec.arms.contains(face) {
        return None;
    }
    let rotation = spec.pose.rotation.quaternion();
    let (u_axis, v_axis) = face.tangent_axes();
    let normal = snap_cardinal(rotation * face.axis().unit()) * face.sign();
    Some(FaceGeometry {
        center: spec.pose.translation() + normal * spec.dimensions.half_side(),
        normal,
        tangent_u: snap_cardinal(rotation * u_axis.unit()),
        tangent_v: snap_cardinal(rotation * v_axis.unit()),
        profile: FaceProfile::Annulus {
            inner_radius: spec.dimensions.inner_diameter() * 0.5,
            outer_radius: spec.dimensions.outer_diameter() * 0.5,
        },
    })
}

pub(crate) const fn ground_face() -> FaceGeometry {
    FaceGeometry {
        center: Vec3::ZERO,
        normal: Vec3::Y,
        tangent_u: Vec3::X,
        tangent_v: Vec3::Z,
        profile: FaceProfile::Ground,
    }
}
