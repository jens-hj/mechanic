//! Meshes between toothed parts: which pairs may mesh and where they touch.
//!
//! A mesh is a magnetic coupling, not a contact. Two meshing parts never touch;
//! the solver holds the no-slip condition at their pitch point, and the ratio
//! comes from tooth counts rather than from the built centre distance. That is
//! why a mesh only has to be roughly tangent: a gap of up to one module or an
//! overlap of half a module is accepted, and the teeth do not have to be phased.

use crate::{ConstructionFrame, GearKind, GearSpec, PartSpec, RackSpec, SpiralSpec};
use bevy_math::Vec3;
use thiserror::Error;

/// Largest gap between two pitch surfaces that still meshes, in modules.
pub const GEAR_MESH_GAP_MODULES: f32 = 1.0;

/// Largest overlap of two pitch surfaces that still meshes, in modules.
pub const GEAR_MESH_OVERLAP_MODULES: f32 = 0.5;

/// Cosine above which two axes count as parallel, about 1.1 degrees.
pub const GEAR_PARALLEL_COSINE: f32 = 0.9998;

/// Cosine below which two axes count as perpendicular, about 1.1 degrees.
pub const GEAR_PERPENDICULAR_COSINE: f32 = 0.02;

/// Why two parts do not mesh.
#[derive(Clone, Copy, Debug, Error, PartialEq)]
pub enum GearLinkError {
    /// Neither part has teeth, a thread, or is a nut on a thread.
    #[error("a mesh needs teeth on at least one side")]
    NeedsTeeth,
    /// Two racks cannot mesh.
    #[error("two racks do not mesh")]
    TwoRacks,
    /// Two threads cannot mesh.
    #[error("two spirals do not mesh")]
    TwoSpirals,
    /// A rack and a thread cannot mesh.
    #[error("a rack does not mesh with a spiral")]
    RackOnSpiral,
    /// Two ring gears cannot mesh.
    #[error("two ring gears do not mesh")]
    TwoRings,
    /// A bevel gear only meshes with another bevel gear.
    #[error("a bevel gear meshes only with another bevel gear")]
    BevelWithSpur,
    /// Racks and worms take straight external teeth.
    #[error("a rack or worm meshes only with a spur gear")]
    NeedsSpur,
    /// The teeth are different sizes.
    #[error("the teeth must be the same module")]
    ModuleMismatch,
    /// A worm's pitch does not match the wheel's teeth.
    #[error("the spiral's pitch must match the wheel's tooth spacing")]
    PitchMismatch,
    /// A spiral cut only into its bore, or a taper alone, has no thread to mesh.
    #[error("the spiral needs a thread on its outer wall")]
    NeedsThread,
    /// Spur and ring gears mesh on parallel axes.
    #[error("the axes must be parallel")]
    AxesNotParallel,
    /// Bevel gears mesh on intersecting axes.
    #[error("the axes must meet")]
    AxesDoNotMeet,
    /// A worm meshes across perpendicular axes.
    #[error("the axes must be perpendicular")]
    AxesNotPerpendicular,
    /// A pinion's axis runs across a rack, parallel to its face.
    #[error("the gear's axis must lie along the rack's face and across its teeth")]
    AxisNotAcrossRack,
    /// The pitch surfaces are too far apart or too deep into each other.
    #[error("the pitch circles must touch; they are {gap_millimeters:.0} mm apart")]
    NotTangent {
        /// Signed pitch-surface gap, negative when overlapping.
        gap_millimeters: f32,
    },
    /// The teeth do not pass each other along the axis.
    #[error("the teeth must overlap along the axis")]
    NoFaceOverlap,
    /// The pinion is not over the rack's face.
    #[error("the gear must sit over the rack's face")]
    OffRack,
    /// A pinion larger than the ring it should sit in.
    #[error("the gear must fit inside the ring")]
    LargerThanRing,
    /// A nut must surround its thread.
    #[error("a nut must surround the spiral")]
    NutOffAxis,
}

/// How a mesh couples its two sides, with the special side second.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GearLinkKind {
    /// Two toothed wheels: spur, ring, or bevel pairs.
    Gears,
    /// A spur gear on a rack.
    Rack,
    /// A spur gear driven across a threaded cylinder.
    Worm,
    /// Any part riding a thread as a nut.
    Screw,
}

/// One part as a mesh sees it, in world space at the rest pose.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GearEnd {
    /// A toothed cylinder.
    Gear {
        /// The teeth.
        spec: GearSpec,
        /// Point on the axis in the pitch plane; the large end of a bevel gear.
        center: Vec3,
        /// Unit axis.
        axis: Vec3,
        /// Half the face width.
        half_length: f32,
    },
    /// A toothed cuboid face.
    Rack {
        /// The teeth.
        spec: RackSpec,
        /// Centre of the pitch plane, one addendum below the face.
        pitch_center: Vec3,
        /// Outward face normal.
        normal: Vec3,
        /// Unit tooth direction across the face.
        across: Vec3,
        /// Half the face's extent along the teeth.
        half_along: f32,
        /// Half the face's extent across the teeth.
        half_across: f32,
    },
    /// A threaded cylinder.
    Screw {
        /// Axis midpoint.
        center: Vec3,
        /// Unit axis.
        axis: Vec3,
        /// Half the thread's length.
        half_length: f32,
        /// Radius halfway down the thread.
        pitch_radius: f32,
        /// Distance between neighbouring ridges.
        pitch: f32,
        /// Signed axial advance per radian about the axis.
        advance: f32,
    },
    /// Anything else, which can only ride a thread.
    Plain {
        /// Envelope centre.
        center: Vec3,
        /// Largest half extent of the envelope.
        half_extent: f32,
    },
}

/// One side of a resolved mesh, in world space at the rest pose.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GearMeshSide {
    /// Point the solver measures surface speed from: the axis point in the
    /// pitch plane for a gear or thread, the pitch-plane centre for a rack,
    /// the point on the thread's axis for a nut.
    pub center: Vec3,
    /// Unit axis for a gear or thread; the outward pitch-plane normal for a rack.
    pub axis: Vec3,
    /// Pitch radius for a gear, negative for internal teeth; zero otherwise.
    pub pitch_radius: f32,
}

/// A mesh two parts admit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GearMesh {
    /// How the sides couple.
    pub kind: GearLinkKind,
    /// The sides in the order the kind names them.
    pub sides: [GearMeshSide; 2],
    /// Thread advance per radian, for worms and screws; zero otherwise.
    pub advance: f32,
    /// Whether the resolved sides are the caller's ends in reverse order.
    pub swapped: bool,
}

impl GearMesh {
    /// Turns of the first side per turn of the second. `None` for racks and nuts.
    pub fn ratio(self) -> Option<f32> {
        match self.kind {
            GearLinkKind::Gears => {
                let [a, b] = self.sides;
                Some((b.pitch_radius / a.pitch_radius).abs())
            }
            GearLinkKind::Worm => {
                // One worm turn moves the wheel's pitch circle one lead.
                let [wheel, _] = self.sides;
                Some(self.advance.abs() / wheel.pitch_radius)
            }
            GearLinkKind::Rack | GearLinkKind::Screw => None,
        }
    }
}

impl GearEnd {
    /// How a part takes part in a mesh, at its rest pose in `frame`.
    pub fn of_part(spec: PartSpec, frame: ConstructionFrame) -> Self {
        let pose = spec.pose();
        let rotation = pose.rotation.quaternion();
        let center = frame.point(pose.translation());
        let world = |local: Vec3| frame.vector(rotation * local);
        let plain = || Self::Plain {
            center,
            half_extent: spec.size_meters().max_element() * 0.5,
        };
        match spec {
            PartSpec::Cylinder(cylinder) => match (cylinder.gear(), cylinder.spiral()) {
                (Some(gear), _) => {
                    let axis = world(Vec3::Y);
                    let half_length = cylinder.dimensions.axial_length() * 0.5;
                    let center = match gear.kind() {
                        GearKind::Bevel { large_end, .. } => {
                            center + axis * large_end.sign() * half_length
                        }
                        GearKind::Spur | GearKind::Internal => center,
                    };
                    Self::Gear {
                        spec: gear,
                        center,
                        axis,
                        half_length,
                    }
                }
                (None, Some(spiral)) => Self::Screw {
                    center,
                    axis: world(Vec3::Y),
                    half_length: cylinder.dimensions.axial_length() * 0.5,
                    pitch_radius: screw_pitch_radius(cylinder.dimensions.outer_diameter(), spiral),
                    pitch: spiral.pitch_meters(),
                    advance: spiral.hand().sign() * spiral.lead_meters() / core::f32::consts::TAU,
                },
                (None, None) => plain(),
            },
            PartSpec::Cuboid(cuboid) => {
                let Some(rack) = cuboid.rack() else {
                    return plain();
                };
                let size = cuboid.size_meters();
                let face = rack.face();
                let normal = world(face.axis().unit() * face.sign());
                let across_axis = match (face.axis(), rack.along()) {
                    (crate::Axis::X, crate::Axis::Y) | (crate::Axis::Y, crate::Axis::X) => {
                        crate::Axis::Z
                    }
                    (crate::Axis::X, crate::Axis::Z) | (crate::Axis::Z, crate::Axis::X) => {
                        crate::Axis::Y
                    }
                    _ => crate::Axis::X,
                };
                let half = size * 0.5;
                Self::Rack {
                    spec: rack,
                    pitch_center: center
                        + normal * (half[face.axis().index()] - rack.pitch_depth()),
                    normal,
                    across: world(across_axis.unit()),
                    half_along: half[rack.along().index()],
                    half_across: half[across_axis.index()],
                }
            }
            _ => plain(),
        }
    }

    /// Teeth on this end, if it is a gear.
    pub const fn gear(self) -> Option<GearSpec> {
        match self {
            Self::Gear { spec, .. } => Some(spec),
            _ => None,
        }
    }
}

/// Radius halfway down a thread cut into a cylinder's outer wall.
fn screw_pitch_radius(outer_diameter: f32, spiral: SpiralSpec) -> f32 {
    let depth = f32::from(spiral.outer().max_depth_ticks()) * crate::POSITION_TICK_METERS;
    outer_diameter * 0.5 - depth * 0.5
}

fn perpendicular_part(vector: Vec3, axis: Vec3) -> Vec3 {
    vector - axis * vector.dot(axis)
}

/// Checks that two pitch surfaces `gap` apart are close enough to mesh.
fn tangent(gap: f32, module: f32) -> Result<(), GearLinkError> {
    if gap > GEAR_MESH_GAP_MODULES * module || gap < -GEAR_MESH_OVERLAP_MODULES * module {
        Err(GearLinkError::NotTangent {
            gap_millimeters: gap * 1000.0,
        })
    } else {
        Ok(())
    }
}

/// Resolves the mesh two parts admit, in either order.
///
/// # Errors
///
/// Returns [`GearLinkError`] naming the first reason the parts do not mesh.
pub fn mesh(first: GearEnd, second: GearEnd) -> Result<GearMesh, GearLinkError> {
    match (first, second) {
        (GearEnd::Gear { .. }, GearEnd::Gear { .. }) => gears(first, second, false),
        (GearEnd::Gear { .. }, GearEnd::Rack { .. }) => rack(first, second, false),
        (GearEnd::Rack { .. }, GearEnd::Gear { .. }) => rack(second, first, true),
        (GearEnd::Gear { .. }, GearEnd::Screw { .. }) => worm(first, second, false),
        (GearEnd::Screw { .. }, GearEnd::Gear { .. }) => worm(second, first, true),
        (GearEnd::Plain { .. }, GearEnd::Screw { .. }) => screw(first, second, false),
        (GearEnd::Screw { .. }, GearEnd::Plain { .. }) => screw(second, first, true),
        (GearEnd::Rack { .. }, GearEnd::Rack { .. }) => Err(GearLinkError::TwoRacks),
        (GearEnd::Screw { .. }, GearEnd::Screw { .. }) => Err(GearLinkError::TwoSpirals),
        (GearEnd::Rack { .. }, GearEnd::Screw { .. })
        | (GearEnd::Screw { .. }, GearEnd::Rack { .. }) => Err(GearLinkError::RackOnSpiral),
        (GearEnd::Plain { .. }, _) | (_, GearEnd::Plain { .. }) => Err(GearLinkError::NeedsTeeth),
    }
}

fn gears(first: GearEnd, second: GearEnd, swapped: bool) -> Result<GearMesh, GearLinkError> {
    let (
        GearEnd::Gear {
            spec: spec_a,
            center: center_a,
            axis: axis_a,
            half_length: half_a,
        },
        GearEnd::Gear {
            spec: spec_b,
            center: center_b,
            axis: axis_b,
            half_length: half_b,
        },
    ) = (first, second)
    else {
        unreachable!("both ends are gears")
    };
    if spec_a.module_ticks() != spec_b.module_ticks() {
        return Err(GearLinkError::ModuleMismatch);
    }
    let module = spec_a.module_meters();
    let bevel = |spec: GearSpec| matches!(spec.kind(), GearKind::Bevel { .. });
    match (bevel(spec_a), bevel(spec_b)) {
        (true, true) => {
            let cross = axis_a.cross(axis_b);
            if cross.length() < GEAR_PERPENDICULAR_COSINE {
                return Err(GearLinkError::AxesDoNotMeet);
            }
            let offset = center_b - center_a;
            if (offset.dot(cross) / cross.length()).abs() > GEAR_MESH_GAP_MODULES * module {
                return Err(GearLinkError::AxesDoNotMeet);
            }
            let toward_b = perpendicular_part(offset, axis_a).normalize_or_zero();
            let toward_a = perpendicular_part(-offset, axis_b).normalize_or_zero();
            let pitch_a = center_a + toward_b * spec_a.pitch_radius();
            let pitch_b = center_b + toward_a * spec_b.pitch_radius();
            tangent(pitch_a.distance(pitch_b), module)?;
        }
        (false, false) => {
            if axis_a.dot(axis_b).abs() < GEAR_PARALLEL_COSINE {
                return Err(GearLinkError::AxesNotParallel);
            }
            if spec_a.is_internal() && spec_b.is_internal() {
                return Err(GearLinkError::TwoRings);
            }
            let offset = center_b - center_a;
            if offset.dot(axis_a).abs() >= half_a + half_b {
                return Err(GearLinkError::NoFaceOverlap);
            }
            let distance = perpendicular_part(offset, axis_a).length();
            let (radius_a, radius_b) = (spec_a.pitch_radius(), spec_b.pitch_radius());
            let target = if spec_a.is_internal() || spec_b.is_internal() {
                let (ring, pinion) = if spec_a.is_internal() {
                    (radius_a, radius_b)
                } else {
                    (radius_b, radius_a)
                };
                if pinion >= ring {
                    return Err(GearLinkError::LargerThanRing);
                }
                ring - pinion
            } else {
                radius_a + radius_b
            };
            tangent(distance - target, module)?;
        }
        _ => return Err(GearLinkError::BevelWithSpur),
    }
    let signed = |spec: GearSpec| {
        if spec.is_internal() {
            -spec.pitch_radius()
        } else {
            spec.pitch_radius()
        }
    };
    Ok(GearMesh {
        kind: GearLinkKind::Gears,
        sides: [
            GearMeshSide {
                center: center_a,
                axis: axis_a,
                pitch_radius: signed(spec_a),
            },
            GearMeshSide {
                center: center_b,
                axis: axis_b,
                pitch_radius: signed(spec_b),
            },
        ],
        advance: 0.0,
        swapped,
    })
}

fn rack(gear: GearEnd, rack: GearEnd, swapped: bool) -> Result<GearMesh, GearLinkError> {
    let (
        GearEnd::Gear {
            spec,
            center,
            axis,
            half_length,
        },
        GearEnd::Rack {
            spec: rack_spec,
            pitch_center,
            normal,
            across,
            half_along,
            half_across,
        },
    ) = (gear, rack)
    else {
        unreachable!("a gear and a rack")
    };
    if spec.kind() != GearKind::Spur {
        return Err(GearLinkError::NeedsSpur);
    }
    if spec.module_ticks() != rack_spec.module_ticks() {
        return Err(GearLinkError::ModuleMismatch);
    }
    if axis.dot(across).abs() < GEAR_PARALLEL_COSINE {
        return Err(GearLinkError::AxisNotAcrossRack);
    }
    let offset = center - pitch_center;
    let along = normal.cross(across);
    if offset.dot(along).abs() > half_along {
        return Err(GearLinkError::OffRack);
    }
    if offset.dot(across).abs() >= half_across + half_length {
        return Err(GearLinkError::NoFaceOverlap);
    }
    tangent(
        offset.dot(normal) - spec.pitch_radius(),
        spec.module_meters(),
    )?;
    Ok(GearMesh {
        kind: GearLinkKind::Rack,
        sides: [
            GearMeshSide {
                center,
                axis,
                pitch_radius: spec.pitch_radius(),
            },
            GearMeshSide {
                center: pitch_center,
                axis: normal,
                pitch_radius: 0.0,
            },
        ],
        advance: 0.0,
        swapped,
    })
}

fn worm(wheel: GearEnd, worm: GearEnd, swapped: bool) -> Result<GearMesh, GearLinkError> {
    let (
        GearEnd::Gear {
            spec,
            center,
            axis,
            half_length,
        },
        GearEnd::Screw {
            center: worm_center,
            axis: worm_axis,
            half_length: worm_half,
            pitch_radius,
            pitch,
            advance,
        },
    ) = (wheel, worm)
    else {
        unreachable!("a gear and a screw")
    };
    if spec.kind() != GearKind::Spur {
        return Err(GearLinkError::NeedsSpur);
    }
    if pitch_radius <= 0.0 || advance == 0.0 {
        return Err(GearLinkError::NeedsThread);
    }
    if (pitch - spec.circular_pitch()).abs() > GEAR_MESH_OVERLAP_MODULES * spec.module_meters() {
        return Err(GearLinkError::PitchMismatch);
    }
    if axis.dot(worm_axis).abs() > GEAR_PERPENDICULAR_COSINE {
        return Err(GearLinkError::AxesNotPerpendicular);
    }
    // Closest points of the two axis lines.
    let offset = worm_center - center;
    let cross = axis.cross(worm_axis).normalize();
    let distance = offset.dot(cross).abs();
    let along_wheel = offset.dot(axis);
    let along_worm = -offset.dot(worm_axis);
    if along_wheel.abs() > half_length || along_worm.abs() > worm_half {
        return Err(GearLinkError::NoFaceOverlap);
    }
    tangent(
        distance - spec.pitch_radius() - pitch_radius,
        spec.module_meters(),
    )?;
    Ok(GearMesh {
        kind: GearLinkKind::Worm,
        sides: [
            GearMeshSide {
                center: center + axis * along_wheel,
                axis,
                pitch_radius: spec.pitch_radius(),
            },
            GearMeshSide {
                center: worm_center + worm_axis * along_worm,
                axis: worm_axis,
                pitch_radius,
            },
        ],
        advance,
        swapped,
    })
}

fn screw(nut: GearEnd, screw: GearEnd, swapped: bool) -> Result<GearMesh, GearLinkError> {
    let (
        GearEnd::Plain {
            center,
            half_extent,
        },
        GearEnd::Screw {
            center: screw_center,
            axis,
            pitch_radius,
            advance,
            ..
        },
    ) = (nut, screw)
    else {
        unreachable!("a plain part and a screw")
    };
    if pitch_radius <= 0.0 || advance == 0.0 {
        return Err(GearLinkError::NeedsThread);
    }
    let offset = center - screw_center;
    let on_axis = screw_center + axis * offset.dot(axis);
    if center.distance(on_axis) > half_extent {
        return Err(GearLinkError::NutOffAxis);
    }
    Ok(GearMesh {
        kind: GearLinkKind::Screw,
        sides: [
            GearMeshSide {
                center: on_axis,
                axis,
                pitch_radius: 0.0,
            },
            GearMeshSide {
                center: screw_center,
                axis,
                pitch_radius,
            },
        ],
        advance,
        swapped,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gear(teeth: u16, center: Vec3, axis: Vec3) -> GearEnd {
        GearEnd::Gear {
            spec: GearSpec::new(4, teeth, GearKind::Spur).expect("valid"),
            center,
            axis,
            half_length: 0.125,
        }
    }

    #[test]
    fn spur_gears_mesh_when_their_pitch_circles_touch() {
        // 12 and 36 teeth at module 1 cm: pitch radii 6 cm and 18 cm.
        let a = gear(12, Vec3::ZERO, Vec3::Y);
        let b = gear(36, Vec3::new(0.24, 0.0, 0.0), Vec3::Y);
        let mesh = mesh(a, b).expect("tangent");
        assert_eq!(mesh.kind, GearLinkKind::Gears);
        assert!((mesh.ratio().expect("gears") - 3.0).abs() < 1.0e-6);
        let far = gear(36, Vec3::new(0.26, 0.0, 0.0), Vec3::Y);
        assert!(matches!(
            super::mesh(a, far),
            Err(GearLinkError::NotTangent { .. })
        ));
        let tilted = gear(36, Vec3::new(0.24, 0.0, 0.0), Vec3::X);
        assert_eq!(super::mesh(a, tilted), Err(GearLinkError::AxesNotParallel));
        let raised = gear(36, Vec3::new(0.24, 0.3, 0.0), Vec3::Y);
        assert_eq!(super::mesh(a, raised), Err(GearLinkError::NoFaceOverlap));
    }

    #[test]
    fn a_pinion_meshes_inside_a_ring_on_the_difference_of_their_radii() {
        let pinion = gear(12, Vec3::new(0.30, 0.0, 0.0), Vec3::Y);
        let ring = GearEnd::Gear {
            spec: GearSpec::new(4, 72, GearKind::Internal).expect("valid"),
            center: Vec3::ZERO,
            axis: Vec3::Y,
            half_length: 0.125,
        };
        assert!(mesh(ring, pinion).is_ok());
        assert!(!mesh(pinion, ring).expect("either order").swapped);
        let big = gear(80, Vec3::ZERO, Vec3::Y);
        assert_eq!(mesh(ring, big), Err(GearLinkError::LargerThanRing));
    }

    #[test]
    fn a_pinion_meshes_a_rack_with_its_axis_across_the_teeth() {
        let rack = GearEnd::Rack {
            spec: RackSpec::new(4, crate::FaceKind::PositiveY, crate::Axis::X).expect("valid"),
            pitch_center: Vec3::new(0.0, -0.01, 0.0),
            normal: Vec3::Y,
            across: Vec3::Z,
            half_along: 1.0,
            half_across: 0.125,
        };
        let pinion = gear(12, Vec3::new(0.3, 0.05, 0.0), Vec3::Z);
        let mesh = mesh(pinion, rack).expect("tangent");
        assert_eq!(mesh.kind, GearLinkKind::Rack);
        let wrong_way = gear(12, Vec3::new(0.3, 0.05, 0.0), Vec3::X);
        assert_eq!(
            super::mesh(wrong_way, rack),
            Err(GearLinkError::AxisNotAcrossRack)
        );
        let beyond = gear(12, Vec3::new(1.3, 0.05, 0.0), Vec3::Z);
        assert_eq!(super::mesh(beyond, rack), Err(GearLinkError::OffRack));
    }

    #[test]
    fn a_wheel_meshes_a_worm_across_perpendicular_axes() {
        let wheel = gear(30, Vec3::ZERO, Vec3::Y);
        let pitch = GearSpec::new(4, 30, GearKind::Spur)
            .expect("valid")
            .circular_pitch();
        let worm = GearEnd::Screw {
            center: Vec3::new(0.15 + 0.04, 0.0, 0.0),
            axis: Vec3::Z,
            half_length: 0.25,
            pitch_radius: 0.04,
            pitch,
            advance: pitch / core::f32::consts::TAU,
        };
        let mesh = mesh(wheel, worm).expect("tangent");
        assert_eq!(mesh.kind, GearLinkKind::Worm);
        let nut = GearEnd::Plain {
            center: Vec3::new(0.19, 0.0, 0.1),
            half_extent: 0.125,
        };
        assert_eq!(
            super::mesh(nut, worm).expect("nut").kind,
            GearLinkKind::Screw
        );
        let away = GearEnd::Plain {
            center: Vec3::new(0.5, 0.0, 0.1),
            half_extent: 0.125,
        };
        assert_eq!(super::mesh(away, worm), Err(GearLinkError::NutOffAxis));
        assert_eq!(super::mesh(nut, wheel), Err(GearLinkError::NeedsTeeth));
    }
}
