//! Collider rows emitted for parts, pipe bends, regions, and evaluated solids.

use super::mass::{physical_spec, region_pieces};
use super::model::{
    CYLINDER_COLLIDER_COUNT, ColliderShape, CompiledConvex, CompiledCylinder, LocalCollider,
};
use crate::shape::{ConvexPiece, PartPiece, decompose_part};
use crate::{
    MACHINE_PART_DENSITY_KG_M3, MaterialProperties, PartId, PartSpec, RegionId, ShapeRegion,
};
use bevy_math::{Mat3, Quat, Vec3};

pub(super) const AUTHORED_CONTACT_PROPERTIES: MaterialProperties = MaterialProperties {
    density_kg_m3: MACHINE_PART_DENSITY_KG_M3,
    static_friction: 0.05,
    dynamic_friction: 0.05,
    restitution: 0.0,
    rolling_resistance: 0.0,
    youngs_modulus_pa: 200.0e9,
};

/// Contact material of one evaluated cell band. Only layered parts have more
/// than band zero; authored parts answer with their fixed properties.
pub(super) fn band_contact_properties(spec: PartSpec, band: u8) -> MaterialProperties {
    spec.band(band).map_or(
        MaterialProperties {
            density_kg_m3: MACHINE_PART_DENSITY_KG_M3,
            ..AUTHORED_CONTACT_PROPERTIES
        },
        |(material, _)| material.properties(),
    )
}

pub(super) fn contact_properties(spec: PartSpec) -> MaterialProperties {
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
pub(super) fn compose_raw_colliders(
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
pub(super) fn solid_full_cylinder(
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

/// Most collider rows a cylinder compiles to: sixteen boxes, and for a spiral
/// its ridge runs and the narrowing core under a taper.
pub fn cylinder_collider_count(spec: crate::CylinderSpec) -> usize {
    CYLINDER_COLLIDER_COUNT
        + spec.spiral().map_or(0, |spiral| {
            let length = spec.dimensions.axial_length();
            let steps = collider_steps_per_turn(spec, spiral);
            spiral.ridge_segments(length, steps) + usize::from(steps)
        })
}

pub(super) fn append_part_colliders(
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
        PartSpec::Cylinder(spec) => append_cylinder_colliders(
            colliders,
            part,
            compound_index,
            spec,
            center_of_mass,
            material_properties,
        ),
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

// A validated spiral always fits the ridge budget at some step count.
fn collider_steps_per_turn(spec: crate::CylinderSpec, spiral: crate::SpiralSpec) -> u16 {
    spiral
        .collider_steps_per_turn(spec.dimensions.axial_length())
        .unwrap_or(crate::MIN_SPIRAL_COLLIDER_STEPS_PER_TURN)
}

fn append_cylinder_colliders(
    colliders: &mut Vec<LocalCollider>,
    part: PartId,
    compound_index: u32,
    spec: crate::CylinderSpec,
    center_of_mass: Vec3,
    material_properties: MaterialProperties,
) {
    // A spiral cylinder collides as its straight core, boxed like any
    // cylinder, plus one convex piece per ridge run.
    let core = crate::spiral_core(spec);
    let (outer, inner, half_length, center_y) = match (spec.spiral(), core) {
        (Some(_), Some(core)) => (
            core.outer_radius,
            core.inner_radius,
            core.length * 0.5,
            core.center_y,
        ),
        (Some(_), None) => (0.0, 0.0, 0.0, 0.0),
        (None, _) => (
            spec.dimensions.outer_diameter() * 0.5,
            spec.dimensions.inner_diameter() * 0.5,
            spec.dimensions.axial_length() * 0.5,
            0.0,
        ),
    };
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
    for segment in 0_u16..if half_length > 0.0 { 16 } else { 0 } {
        let angle = start_angle + segment_angle * (f32::from(segment) + 0.5);
        let radial = Vec3::new(angle.cos(), 0.0, angle.sin());
        colliders.push(LocalCollider {
            source_part: part,
            compound_index,
            local_center: spec.pose.translation() - center_of_mass
                + part_rotation * (radial * center_radius + Vec3::Y * center_y),
            material_properties,
            shape: ColliderShape::Cuboid {
                local_rotation: part_rotation * Quat::from_rotation_y(-angle),
                half_extents: Vec3::new(half_radial, half_length, half_tangent),
            },
        });
    }
    if let Some(spiral) = spec.spiral() {
        let steps = collider_steps_per_turn(spec, spiral);
        colliders.extend(
            crate::spiral_pieces(spec, steps)
                .iter()
                .map(|piece| LocalCollider {
                    source_part: part,
                    compound_index,
                    local_center: piece.centroid - center_of_mass,
                    material_properties,
                    shape: ColliderShape::Convex(compile_convex(piece, center_of_mass)),
                }),
        );
    }
}

pub(super) fn append_pipe_bend_colliders(
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
pub(super) fn append_region_colliders(
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

pub(super) fn append_evaluated_colliders(
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
pub(super) fn compile_convex(piece: &ConvexPiece, center_of_mass: Vec3) -> CompiledConvex {
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
