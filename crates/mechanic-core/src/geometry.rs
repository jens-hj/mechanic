mod cylinder;
mod face;
mod grid;
mod layers;
mod machine;
mod material;
mod pipe;

pub use cylinder::{
    CYLINDER_SWEEP_STEP_DEGREES, CylinderDimensionError, CylinderDimensions, CylinderSpec,
    MAX_CYLINDER_OUTER_DIAMETER, MAX_CYLINDER_SWEEP_DEGREES, MIN_CYLINDER_DIAMETER_GAP,
    MIN_CYLINDER_OUTER_DIAMETER, MIN_CYLINDER_SWEEP_DEGREES,
};
pub(crate) use face::{
    FaceGeometry, FaceProfile, cuboid_face, cylinder_face, ground_face, pipe_bend_face,
    pipe_junction_face,
};
pub use face::{FaceKind, FaceOwner, FaceRef};
pub use grid::{
    Axis, BuildPose, DimensionError, GRID_UNIT_METERS, GridDimension, GridRotation, MAX_GRID_UNITS,
    POSITION_TICK_METERS, POSITION_TICKS_PER_GRID_UNIT, POSITION_TICKS_PER_HALF_GRID_UNIT,
    snap_world_to_grid,
};
use layers::{LayerEnvelope, layer_thickness_ticks, shifted_pose, unwind_layer_regions};
pub use layers::{
    LayerError, LayerFace, LayerRegion, MAX_PART_LAYERS, MIN_LAYER_THICKNESS_METERS, MaterialLayer,
    MaterialLayers,
};
pub use machine::{
    ControllerSpec, DimensionLinkId, DimensionLinkSpec, EngineKind, EngineSpec, InputSpec,
    SeatSpec, ServoSpec, TransmissionSpec,
};
pub use material::{ConstructionMaterial, MaterialProperties, SurfaceResponse};
pub use pipe::{
    PIPE_BEND_ARC_SLICES, PIPE_BEND_RADIAL_SIDES, PipeArms, PipeBendDimensionError,
    PipeBendDimensions, PipeBendSpec, PipeJunctionDimensions, PipeJunctionError, PipeJunctionSpec,
};

use bevy_math::Vec3;

use crate::MaterialAppearance;

/// Editable cuboid dimensions and build pose.
///
/// A layered cuboid is one solid. `dimensions` are its grid-aligned core, while
/// `pose` and [`CuboidSpec::size_meters`] describe the whole envelope including
/// every [`MaterialLayer`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CuboidSpec {
    /// Validated x/y/z core dimensions.
    pub dimensions: [GridDimension; 3],
    /// Envelope centre and orientation.
    pub pose: BuildPose,
    /// Core material used for appearance, mass, and contact response.
    pub material: ConstructionMaterial,
    /// Core color and finish treatment.
    pub appearance: MaterialAppearance,
    layers: MaterialLayers,
}

/// A construction part with shape-specific dimensions and a shared build pose.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PartSpec {
    /// Rectangular cuboid.
    Cuboid(CuboidSpec),
    /// Solid or hollow cylinder whose axis is local Y.
    Cylinder(CylinderSpec),
    /// Cardinal 90-degree quarter-torus pipe bend.
    PipeBend(PipeBendSpec),
    /// Fitting joining pipe arms on any of its six faces.
    PipeJunction(PipeJunctionSpec),
    /// Fixed-size control block driving the bearings wired to it.
    Controller(ControllerSpec),
    /// Fixed-size inert engine with an authored appearance.
    Engine(EngineSpec),
    /// Fixed-size transmission whose appearance comes from its root engine.
    Transmission(TransmissionSpec),
    /// Fixed-size servo angle actuator.
    Servo(ServoSpec),
    /// Fixed-size seat cushion.
    Seat(SeatSpec),
    /// Fixed-size keyboard input router.
    Input(InputSpec),
    /// Fixed-size Dimension Link portal anchor.
    DimensionLink(DimensionLinkSpec),
}

impl PartSpec {
    /// Part build pose.
    pub const fn pose(self) -> BuildPose {
        match self {
            Self::Cuboid(spec) => spec.pose,
            Self::Cylinder(spec) => spec.pose,
            Self::PipeBend(spec) => spec.pose,
            Self::PipeJunction(spec) => spec.pose,
            Self::Controller(spec) => spec.pose,
            Self::Engine(spec) => spec.pose,
            Self::Transmission(spec) => spec.pose,
            Self::Servo(spec) => spec.pose,
            Self::Seat(spec) => spec.pose,
            Self::Input(spec) => spec.pose,
            Self::DimensionLink(spec) => spec.pose,
        }
    }

    /// Construction appearance, or `None` for authored machine parts.
    pub const fn appearance(self) -> Option<MaterialAppearance> {
        match self {
            Self::Cuboid(spec) => Some(spec.appearance),
            Self::Cylinder(spec) => Some(spec.appearance),
            Self::PipeBend(spec) => Some(spec.appearance),
            Self::PipeJunction(spec) => Some(spec.appearance),
            Self::Controller(_)
            | Self::Engine(_)
            | Self::Transmission(_)
            | Self::Servo(_)
            | Self::Seat(_)
            | Self::Input(_)
            | Self::DimensionLink(_) => None,
        }
    }

    /// Material layers over an ordinary part's core; none for other parts.
    pub const fn material_layers(self) -> MaterialLayers {
        match self {
            Self::Cuboid(spec) => spec.layers,
            Self::Cylinder(spec) => spec.layers,
            Self::PipeBend(_)
            | Self::PipeJunction(_)
            | Self::Controller(_)
            | Self::Engine(_)
            | Self::Transmission(_)
            | Self::Servo(_)
            | Self::Seat(_)
            | Self::Input(_)
            | Self::DimensionLink(_) => MaterialLayers::NONE,
        }
    }

    /// Whether the part carries any material layer.
    pub const fn is_layered(self) -> bool {
        !self.material_layers().is_empty()
    }

    /// Material and appearance of one band: zero is the core, `i + 1` layer `i`.
    pub fn band(self, band: u8) -> Option<(ConstructionMaterial, MaterialAppearance)> {
        let core = match self {
            Self::Cuboid(spec) => (spec.material, spec.appearance),
            Self::Cylinder(spec) => (spec.material, spec.appearance),
            Self::PipeBend(spec) => (spec.material, spec.appearance),
            Self::PipeJunction(spec) => (spec.material, spec.appearance),
            Self::Controller(_)
            | Self::Engine(_)
            | Self::Transmission(_)
            | Self::Servo(_)
            | Self::Seat(_)
            | Self::Input(_)
            | Self::DimensionLink(_) => return None,
        };
        match band.checked_sub(1) {
            None => Some(core),
            Some(index) => self
                .material_layers()
                .get(usize::from(index))
                .map(|layer| (layer.material, layer.appearance)),
        }
    }

    /// Returns the part with one band's appearance replaced, or `None` when the
    /// band does not exist.
    pub fn with_band_appearance(self, band: u8, appearance: MaterialAppearance) -> Option<Self> {
        let Some(index) = band.checked_sub(1) else {
            return self.with_appearance(appearance);
        };
        let index = usize::from(index);
        match self {
            Self::Cuboid(mut spec) => {
                spec.layers = spec.layers.with_appearance(index, appearance)?;
                Some(Self::Cuboid(spec))
            }
            Self::Cylinder(mut spec) => {
                spec.layers = spec.layers.with_appearance(index, appearance)?;
                Some(Self::Cylinder(spec))
            }
            Self::PipeBend(_)
            | Self::PipeJunction(_)
            | Self::Controller(_)
            | Self::Engine(_)
            | Self::Transmission(_)
            | Self::Servo(_)
            | Self::Seat(_)
            | Self::Input(_)
            | Self::DimensionLink(_) => None,
        }
    }

    /// Adds a material layer to a cuboid face or a cylinder wall, bore, or cap.
    ///
    /// # Errors
    ///
    /// Returns [`LayerError::UnsupportedFace`] for parts that take no layers, or
    /// the cuboid or cylinder layer error.
    pub fn with_layer(
        self,
        face: LayerFace,
        thickness: f32,
        material: ConstructionMaterial,
        appearance: MaterialAppearance,
    ) -> Result<Self, LayerError> {
        match self {
            Self::Cuboid(spec) => spec
                .with_layer(face, thickness, material, appearance)
                .map(Self::Cuboid),
            Self::Cylinder(spec) => spec
                .with_layer(face, thickness, material, appearance)
                .map(Self::Cylinder),
            Self::PipeBend(_)
            | Self::PipeJunction(_)
            | Self::Controller(_)
            | Self::Engine(_)
            | Self::Transmission(_)
            | Self::Servo(_)
            | Self::Seat(_)
            | Self::Input(_)
            | Self::DimensionLink(_) => Err(LayerError::UnsupportedFace),
        }
    }

    /// The core an ordinary part's layers were laid on; other parts unchanged.
    #[must_use]
    pub fn without_layers(self) -> Self {
        match self {
            Self::Cuboid(spec) => Self::Cuboid(spec.without_layers()),
            Self::Cylinder(spec) => Self::Cylinder(spec.without_layers()),
            other => other,
        }
    }

    /// Whether two parts are the same core, differing at most in their layers.
    pub fn shares_core_with(self, other: Self) -> bool {
        match (self.without_layers(), other.without_layers()) {
            (Self::Cylinder(first), Self::Cylinder(second)) => {
                let (a, b) = (first.dimensions, second.dimensions);
                first.pose == second.pose
                    && first.material == second.material
                    && a.axial_length_ticks == b.axial_length_ticks
                    && a.sweep_angle_degrees == b.sweep_angle_degrees
                    && (a.outer_diameter - b.outer_diameter).abs() < 1.0e-4
                    && (a.inner_diameter - b.inner_diameter).abs() < 1.0e-4
            }
            (Self::Cuboid(first), Self::Cuboid(second)) => {
                Self::Cuboid(first.with_appearance(second.appearance)) == Self::Cuboid(second)
            }
            _ => false,
        }
    }

    /// Each layer's part-local region, oldest first.
    pub fn layer_regions(self) -> Vec<LayerRegion> {
        match self {
            Self::Cuboid(spec) => spec.layer_regions(),
            Self::Cylinder(spec) => spec.layer_regions(),
            _ => Vec::new(),
        }
    }

    /// Band owning a part-local point: the newest layer whose region contains
    /// it, or the core.
    #[expect(clippy::cast_possible_truncation, reason = "at most MAX_PART_LAYERS")]
    pub fn band_at_local_point(self, local: Vec3) -> u8 {
        self.layer_regions()
            .iter()
            .rposition(|region| region.contains(local))
            .map_or(0, |index| index as u8 + 1)
    }

    /// Returns an ordinary construction part with a replacement appearance.
    pub(crate) const fn with_appearance(self, appearance: MaterialAppearance) -> Option<Self> {
        match self {
            Self::Cuboid(spec) => Some(Self::Cuboid(spec.with_appearance(appearance))),
            Self::Cylinder(spec) => Some(Self::Cylinder(spec.with_appearance(appearance))),
            Self::PipeBend(spec) => Some(Self::PipeBend(spec.with_appearance(appearance))),
            Self::PipeJunction(spec) => Some(Self::PipeJunction(spec.with_appearance(appearance))),
            Self::Controller(_)
            | Self::Engine(_)
            | Self::Transmission(_)
            | Self::Servo(_)
            | Self::Seat(_)
            | Self::Input(_)
            | Self::DimensionLink(_) => None,
        }
    }

    /// Returns this part with a replacement authored pose.
    #[must_use]
    pub const fn with_pose(self, pose: BuildPose) -> Self {
        match self {
            Self::Cuboid(mut spec) => {
                spec.pose = pose;
                Self::Cuboid(spec)
            }
            Self::Cylinder(mut spec) => {
                spec.pose = pose;
                Self::Cylinder(spec)
            }
            Self::PipeBend(mut spec) => {
                spec.pose = pose;
                Self::PipeBend(spec)
            }
            Self::PipeJunction(mut spec) => {
                spec.pose = pose;
                Self::PipeJunction(spec)
            }
            Self::Controller(mut spec) => {
                spec.pose = pose;
                Self::Controller(spec)
            }
            Self::Engine(mut spec) => {
                spec.pose = pose;
                Self::Engine(spec)
            }
            Self::Transmission(mut spec) => {
                spec.pose = pose;
                Self::Transmission(spec)
            }
            Self::Servo(mut spec) => {
                spec.pose = pose;
                Self::Servo(spec)
            }
            Self::Seat(mut spec) => {
                spec.pose = pose;
                Self::Seat(spec)
            }
            Self::Input(mut spec) => {
                spec.pose = pose;
                Self::Input(spec)
            }
            Self::DimensionLink(mut spec) => {
                spec.pose = pose;
                Self::DimensionLink(spec)
            }
        }
    }

    /// Returns the cuboid shape backing this part, when it has one. Control
    /// blocks report their fixed cube.
    pub fn as_cuboid(self) -> Option<CuboidSpec> {
        match self {
            Self::Cuboid(spec) => Some(spec),
            Self::Controller(spec) => Some(spec.cuboid()),
            Self::Engine(spec) => Some(spec.cuboid()),
            Self::Transmission(spec) => Some(spec.cuboid()),
            Self::Servo(spec) => Some(spec.cuboid()),
            Self::Seat(spec) => Some(spec.cuboid()),
            Self::Input(spec) => Some(spec.cuboid()),
            Self::DimensionLink(spec) => Some(spec.cuboid()),
            Self::Cylinder(_) | Self::PipeBend(_) | Self::PipeJunction(_) => None,
        }
    }

    /// Returns the control-block shape, when this part is a control block.
    pub const fn as_controller(self) -> Option<ControllerSpec> {
        match self {
            Self::Controller(spec) => Some(spec),
            Self::Cuboid(_)
            | Self::Cylinder(_)
            | Self::PipeBend(_)
            | Self::PipeJunction(_)
            | Self::Engine(_)
            | Self::Transmission(_)
            | Self::Servo(_)
            | Self::Seat(_)
            | Self::Input(_)
            | Self::DimensionLink(_) => None,
        }
    }

    /// Returns the cylinder shape, when this part is a cylinder.
    pub const fn as_cylinder(self) -> Option<CylinderSpec> {
        match self {
            Self::Cylinder(spec) => Some(spec),
            Self::Cuboid(_)
            | Self::PipeBend(_)
            | Self::PipeJunction(_)
            | Self::Controller(_)
            | Self::Engine(_)
            | Self::Transmission(_)
            | Self::Servo(_)
            | Self::Seat(_)
            | Self::Input(_)
            | Self::DimensionLink(_) => None,
        }
    }

    /// Returns the pipe-bend shape, when this part is a bend.
    pub const fn as_pipe_bend(self) -> Option<PipeBendSpec> {
        match self {
            Self::PipeBend(spec) => Some(spec),
            Self::Cuboid(_)
            | Self::Cylinder(_)
            | Self::PipeJunction(_)
            | Self::Controller(_)
            | Self::Engine(_)
            | Self::Transmission(_)
            | Self::Servo(_)
            | Self::Seat(_)
            | Self::Input(_)
            | Self::DimensionLink(_) => None,
        }
    }

    /// Returns the pipe-junction shape, when this part is a junction.
    pub const fn as_pipe_junction(self) -> Option<PipeJunctionSpec> {
        match self {
            Self::PipeJunction(spec) => Some(spec),
            Self::Cuboid(_)
            | Self::Cylinder(_)
            | Self::PipeBend(_)
            | Self::Controller(_)
            | Self::Engine(_)
            | Self::Transmission(_)
            | Self::Servo(_)
            | Self::Seat(_)
            | Self::Input(_)
            | Self::DimensionLink(_) => None,
        }
    }

    /// Axis-aligned local dimensions. Cylinders return diameter/length/diameter.
    pub fn size_meters(self) -> Vec3 {
        match self {
            Self::Cuboid(spec) => spec.size_meters(),
            Self::Controller(spec) => spec.cuboid().size_meters(),
            Self::Engine(spec) => spec.cuboid().size_meters(),
            Self::Transmission(spec) => spec.cuboid().size_meters(),
            Self::Servo(spec) => spec.cuboid().size_meters(),
            Self::Seat(spec) => spec.cuboid().size_meters(),
            Self::Input(spec) => spec.cuboid().size_meters(),
            Self::DimensionLink(spec) => spec.cuboid().size_meters(),
            Self::Cylinder(spec) => Vec3::new(
                spec.dimensions.outer_diameter(),
                spec.dimensions.axial_length(),
                spec.dimensions.outer_diameter(),
            ),
            Self::PipeBend(spec) => {
                let outer = spec.dimensions.outer_diameter();
                let radius = spec.dimensions.radius();
                Vec3::new(radius + outer * 0.5, radius + outer * 0.5, outer)
            }
            Self::PipeJunction(spec) => Vec3::splat(spec.dimensions.half_side() * 2.0),
        }
    }
}

impl PartialEq<CuboidSpec> for PartSpec {
    fn eq(&self, other: &CuboidSpec) -> bool {
        matches!(self, Self::Cuboid(spec) if spec == other)
    }
}

impl From<CuboidSpec> for PartSpec {
    fn from(value: CuboidSpec) -> Self {
        Self::Cuboid(value)
    }
}

impl From<CylinderSpec> for PartSpec {
    fn from(value: CylinderSpec) -> Self {
        Self::Cylinder(value)
    }
}

impl From<PipeBendSpec> for PartSpec {
    fn from(value: PipeBendSpec) -> Self {
        Self::PipeBend(value)
    }
}

impl From<PipeJunctionSpec> for PartSpec {
    fn from(value: PipeJunctionSpec) -> Self {
        Self::PipeJunction(value)
    }
}

impl From<ControllerSpec> for PartSpec {
    fn from(value: ControllerSpec) -> Self {
        Self::Controller(value)
    }
}

impl From<EngineSpec> for PartSpec {
    fn from(value: EngineSpec) -> Self {
        Self::Engine(value)
    }
}

impl From<TransmissionSpec> for PartSpec {
    fn from(value: TransmissionSpec) -> Self {
        Self::Transmission(value)
    }
}

impl From<ServoSpec> for PartSpec {
    fn from(value: ServoSpec) -> Self {
        Self::Servo(value)
    }
}

impl From<SeatSpec> for PartSpec {
    fn from(value: SeatSpec) -> Self {
        Self::Seat(value)
    }
}

impl From<InputSpec> for PartSpec {
    fn from(value: InputSpec) -> Self {
        Self::Input(value)
    }
}

impl From<DimensionLinkSpec> for PartSpec {
    fn from(value: DimensionLinkSpec) -> Self {
        Self::DimensionLink(value)
    }
}

impl CuboidSpec {
    /// Creates a cuboid from integer quarter-metre dimensions.
    ///
    /// # Errors
    ///
    /// Returns [`DimensionError`] when any dimension is outside 1..=32.
    pub fn new(dimensions: [u8; 3], pose: BuildPose) -> Result<Self, DimensionError> {
        let [x, y, z] = dimensions;
        Ok(Self {
            dimensions: [
                GridDimension::new(x)?,
                GridDimension::new(y)?,
                GridDimension::new(z)?,
            ],
            pose,
            material: ConstructionMaterial::Steel,
            appearance: MaterialAppearance::BAKED,
            layers: MaterialLayers::NONE,
        })
    }

    /// Uses an explicit construction material.
    #[must_use]
    pub const fn with_material(mut self, material: ConstructionMaterial) -> Self {
        self.material = material;
        self
    }

    /// Uses an explicit construction appearance.
    #[must_use]
    pub const fn with_appearance(mut self, appearance: MaterialAppearance) -> Self {
        self.appearance = appearance;
        self
    }

    /// Envelope side lengths in metres, including every layer.
    pub fn size_meters(self) -> Vec3 {
        let mut size = Vec3::new(
            self.dimensions[0].meters(),
            self.dimensions[1].meters(),
            self.dimensions[2].meters(),
        );
        for layer in self.layers.iter() {
            if let LayerFace::Face(face) = layer.face {
                size[face.axis().index()] += layer.thickness;
            }
        }
        size
    }

    /// Material layers over the core, oldest first.
    pub const fn layers(self) -> MaterialLayers {
        self.layers
    }

    /// Adds a flat material layer to one face, moving the centre outward by
    /// half the thickness so the opposite face stays put.
    ///
    /// # Errors
    ///
    /// Returns [`LayerError`] when the face is not flat, the thickness is not a
    /// whole 5 mm step of at least 1 cm, the part has the most layers, or the
    /// envelope would exceed 8 m.
    #[expect(clippy::cast_precision_loss, reason = "tick counts within 8 m")]
    pub fn with_layer(
        self,
        face: LayerFace,
        thickness: f32,
        material: ConstructionMaterial,
        appearance: MaterialAppearance,
    ) -> Result<Self, LayerError> {
        let LayerFace::Face(kind) = face else {
            return Err(LayerError::UnsupportedFace);
        };
        let ticks = layer_thickness_ticks(thickness, true)?;
        let thickness = ticks as f32 * POSITION_TICK_METERS;
        let layers = self.layers.pushed(MaterialLayer {
            face,
            thickness,
            material,
            appearance,
        })?;
        if self.size_meters()[kind.axis().index()] + thickness
            > f32::from(MAX_GRID_UNITS) * GRID_UNIT_METERS + 1.0e-4
        {
            return Err(LayerError::TooLarge);
        }
        Ok(Self {
            pose: shifted_pose(
                self.pose,
                kind.axis().unit() * kind.sign() * (ticks / 2) as f32,
            ),
            layers,
            ..self
        })
    }

    /// The grid-aligned core this cuboid's layers were laid on.
    #[must_use]
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        reason = "whole ticks within 8 m"
    )]
    pub fn without_layers(self) -> Self {
        let shift = self
            .layers
            .iter()
            .filter_map(|layer| match layer.face {
                LayerFace::Face(face) => Some(
                    face.axis().unit()
                        * face.sign()
                        * ((layer.thickness / POSITION_TICK_METERS).round() as i32 / 2) as f32,
                ),
                LayerFace::OuterWall | LayerFace::Bore => None,
            })
            .sum::<Vec3>();
        Self {
            pose: shifted_pose(self.pose, -shift),
            layers: MaterialLayers::NONE,
            ..self
        }
    }

    /// Each layer's part-local region, oldest first.
    pub fn layer_regions(self) -> Vec<LayerRegion> {
        let half = self.size_meters() * 0.5;
        unwind_layer_regions(
            LayerEnvelope {
                minimum: -half,
                maximum: half,
                outer_radius: 0.0,
                inner_radius: 0.0,
            },
            self.layers,
        )
    }
}

#[cfg(test)]
mod tests;
