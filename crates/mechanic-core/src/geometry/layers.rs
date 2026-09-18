//! Material layers laid over a part's faces.

use super::cylinder::{CylinderDimensionError, MAX_CYLINDER_OUTER_DIAMETER};
use super::face::FaceKind;
use super::grid::{BuildPose, POSITION_TICK_METERS};
use super::material::ConstructionMaterial;
use crate::MaterialAppearance;
use bevy_math::{IVec3, Vec2, Vec3};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Most material layers one part carries over its core.
pub const MAX_PART_LAYERS: usize = 4;

/// Thinnest material layer, in metres.
pub const MIN_LAYER_THICKNESS_METERS: f32 = 0.01;

/// Surface a material layer grows from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LayerFace {
    /// A flat cuboid face or cylinder end cap, growing outward along its normal.
    Face(FaceKind),
    /// A cylinder's curved outer wall, growing its outer diameter.
    OuterWall,
    /// A hollow cylinder's bore, shrinking its inner diameter.
    Bore,
}

/// One material layer laid over a part's envelope at the time it was added.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MaterialLayer {
    /// Surface the layer was laid on.
    pub face: LayerFace,
    /// Thickness in metres, along the face normal or the radius.
    pub thickness: f32,
    /// Material used for appearance, mass, and contact response.
    pub material: ConstructionMaterial,
    /// Independent color and finish treatment.
    pub appearance: MaterialAppearance,
}

impl MaterialLayer {
    pub(super) const UNUSED: Self = Self {
        face: LayerFace::OuterWall,
        thickness: 0.0,
        material: ConstructionMaterial::Steel,
        appearance: MaterialAppearance::BAKED,
    };
}

/// A part's material layers, oldest first.
///
/// A layered part is still one solid. Its envelope already includes every
/// layer; the layers only divide that envelope's material. The part's own
/// material is band zero, and layer `i` is band `i + 1`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MaterialLayers {
    pub(super) layers: [MaterialLayer; MAX_PART_LAYERS],
    pub(super) count: u8,
}

impl MaterialLayers {
    /// No layers: the whole part is its own material.
    pub const NONE: Self = Self {
        layers: [MaterialLayer::UNUSED; MAX_PART_LAYERS],
        count: 0,
    };

    /// Number of layers.
    pub const fn len(self) -> usize {
        self.count as usize
    }

    /// Whether there are no layers.
    pub const fn is_empty(self) -> bool {
        self.count == 0
    }

    /// Layer `index`, oldest first.
    pub const fn get(self, index: usize) -> Option<MaterialLayer> {
        if index < self.len() {
            Some(self.layers[index])
        } else {
            None
        }
    }

    /// Every layer, oldest first.
    pub fn iter(self) -> impl DoubleEndedIterator<Item = MaterialLayer> {
        self.layers.into_iter().take(self.len())
    }

    pub(super) fn pushed(mut self, layer: MaterialLayer) -> Result<Self, LayerError> {
        let len = self.len();
        let slot = self.layers.get_mut(len).ok_or(LayerError::TooManyLayers)?;
        *slot = layer;
        self.count += 1;
        Ok(self)
    }

    pub(super) fn with_appearance(
        mut self,
        index: usize,
        appearance: MaterialAppearance,
    ) -> Option<Self> {
        let len = self.len();
        self.layers[..len].get_mut(index)?.appearance = appearance;
        Some(self)
    }
}

/// Invalid material layer.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum LayerError {
    /// The part already carries the maximum number of layers.
    #[error("a part holds at most {MAX_PART_LAYERS} material layers")]
    TooManyLayers,
    /// The layer is thinner than the minimum or off the 5 mm step of flat layers.
    #[error("a material layer must be at least 1 cm thick, and a flat layer a whole 5 mm step")]
    ThicknessOutOfRange,
    /// A bore layer needs a bore to line.
    #[error("only a hollow cylinder can take a layer inside its bore")]
    BoreRequired,
    /// The part has no such layerable surface.
    #[error("this surface cannot take a material layer")]
    UnsupportedFace,
    /// The layered envelope would outgrow the largest part.
    #[error("a layered part must stay within 8 m")]
    TooLarge,
    /// The layered cylinder envelope is invalid.
    #[error(transparent)]
    Cylinder(#[from] CylinderDimensionError),
}

/// Part-local space owned by one layer: beyond the envelope it was laid on.
///
/// Later layers win where regions overlap, which is exactly the corner a later
/// layer's full face covers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LayerRegion {
    /// Points with `sign * local[axis] > distance`.
    Beyond {
        /// Local axis index.
        axis: usize,
        /// Outward direction along that axis, `1.0` or `-1.0`.
        sign: f32,
        /// Distance of the earlier face from the part centre.
        distance: f32,
    },
    /// Points farther than this radius from the local Y axis.
    OutsideRadius(f32),
    /// Points nearer than this radius to the local Y axis.
    InsideRadius(f32),
}

impl LayerRegion {
    /// Whether a part-local point lies in this layer's region.
    pub fn contains(self, local: Vec3) -> bool {
        match self {
            Self::Beyond {
                axis,
                sign,
                distance,
            } => sign * local[axis] > distance,
            Self::OutsideRadius(radius) => Vec2::new(local.x, local.z).length() > radius,
            Self::InsideRadius(radius) => Vec2::new(local.x, local.z).length() < radius,
        }
    }
}

/// Part-local envelope unwound one layer at a time.
#[derive(Clone, Copy)]
pub(super) struct LayerEnvelope {
    pub(super) minimum: Vec3,
    pub(super) maximum: Vec3,
    pub(super) outer_radius: f32,
    pub(super) inner_radius: f32,
}

pub(super) fn unwind_layer_regions(
    mut envelope: LayerEnvelope,
    layers: MaterialLayers,
) -> Vec<LayerRegion> {
    let mut regions = layers
        .iter()
        .rev()
        .map(|layer| match layer.face {
            LayerFace::Face(face) => {
                let axis = face.axis().index();
                if face.sign() > 0.0 {
                    envelope.maximum[axis] -= layer.thickness;
                    LayerRegion::Beyond {
                        axis,
                        sign: 1.0,
                        distance: envelope.maximum[axis],
                    }
                } else {
                    envelope.minimum[axis] += layer.thickness;
                    LayerRegion::Beyond {
                        axis,
                        sign: -1.0,
                        distance: -envelope.minimum[axis],
                    }
                }
            }
            LayerFace::OuterWall => {
                envelope.outer_radius -= layer.thickness;
                LayerRegion::OutsideRadius(envelope.outer_radius)
            }
            LayerFace::Bore => {
                envelope.inner_radius += layer.thickness;
                LayerRegion::InsideRadius(envelope.inner_radius)
            }
        })
        .collect::<Vec<_>>();
    regions.reverse();
    regions
}

/// Validates a layer thickness; flat layers also return their even tick count,
/// so the half-thickness centre shift stays on the position grid.
#[expect(
    clippy::cast_possible_truncation,
    reason = "checked against the 8 m envelope first"
)]
pub(super) fn layer_thickness_ticks(thickness: f32, flat: bool) -> Result<i32, LayerError> {
    if !thickness.is_finite() || thickness < MIN_LAYER_THICKNESS_METERS - 1.0e-5 {
        return Err(LayerError::ThicknessOutOfRange);
    }
    if thickness > MAX_CYLINDER_OUTER_DIAMETER {
        return Err(LayerError::TooLarge);
    }
    let ticks = thickness / POSITION_TICK_METERS;
    let rounded = ticks.round();
    if flat && ((ticks - rounded).abs() > 1.0e-3 || rounded as i32 % 2 != 0) {
        return Err(LayerError::ThicknessOutOfRange);
    }
    Ok(rounded as i32)
}

/// Moves a pose along one of its rotated local axes by whole ticks.
#[expect(
    clippy::cast_possible_truncation,
    reason = "cardinal rotations keep whole ticks"
)]
pub(super) fn shifted_pose(pose: BuildPose, local_ticks: Vec3) -> BuildPose {
    let shift = (pose.rotation.quaternion() * local_ticks).round();
    BuildPose::from_position_ticks(
        pose.translation_position_ticks()
            + IVec3::new(shift.x as i32, shift.y as i32, shift.z as i32),
        pose.rotation,
    )
}
