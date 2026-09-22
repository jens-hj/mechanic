//! Mass, centre of mass, and inertia of parts, regions, and compounds.

use super::colliders::band_contact_properties;
use super::model::{MassProperties, TopologyError};
use crate::shape::{ConvexPiece, PartPiece, decompose};
use crate::{
    ConstructionGraph, CuboidSpec, FaceOwner, MACHINE_PART_DENSITY_KG_M3, PartId, PartSpec,
    RegionId, ShapeRegion,
};
use bevy_math::{Mat3, Vec3};
use std::collections::BTreeSet;

#[expect(clippy::too_many_lines)]
pub(super) fn calculate_mass_properties<'a>(
    parts: impl Iterator<Item = (PartId, PartSpec)> + Clone + 'a,
    is_static: bool,
    covered: &BTreeSet<PartId>,
    regions: &[(RegionId, &ShapeRegion)],
    graph: &ConstructionGraph,
    sockets: &[crate::BearingSocket],
    heads: &[super::HeadKey],
) -> Result<MassProperties, TopologyError> {
    let member_parts = parts.clone().map(|(id, _)| id).collect::<BTreeSet<_>>();
    let mut hardware_masses = Vec::new();
    let mut seen_mounts = Vec::new();
    for (_, bearing) in graph.bearings() {
        let elements = bearing.kind.mass_elements();
        if elements.is_empty() {
            continue;
        }
        let origin = bearing
            .kind
            .mass_origin(bearing.shared_anchor, bearing.axis);
        let key = (
            bearing.source,
            bearing.shared_anchor.to_array().map(f32::to_bits),
        );
        // Which side of this joint the compound holds: a head it owns, or
        // the part on that side.
        let holds = |opposite: bool| {
            if opposite && bearing.kind.owns_head() {
                return heads.contains(&super::head_key(bearing));
            }
            let face = if opposite {
                bearing.target
            } else {
                Some(bearing.source)
            };
            matches!(face.map(|face| face.owner), Some(FaceOwner::Part(owner)) if member_parts.contains(&owner))
        };
        for element in elements {
            if !holds(element.opposite) {
                continue;
            }
            // A shared mounting assembly can have several attached parts.
            let endpoint_key = (key, element.opposite);
            if seen_mounts.contains(&endpoint_key) {
                continue;
            }
            let axis = bearing.axis;
            let outer = Mat3::from_cols(axis * axis.x, axis * axis.y, axis * axis.z);
            hardware_masses.push(WorldMassProperties {
                mass: element.mass,
                center: origin + axis * element.center,
                inertia: Mat3::IDENTITY * element.transverse_inertia
                    + outer * (element.axial_inertia - element.transverse_inertia),
            });
        }
        for opposite in [false, true] {
            if holds(opposite) {
                seen_mounts.push((key, opposite));
            }
        }
    }
    let mut seen_sockets = Vec::new();
    for socket in sockets {
        let elements = socket.kind.mass_elements();
        if elements.is_empty() {
            continue;
        }
        let FaceOwner::Part(owner) = socket.source.owner else {
            continue;
        };
        if !member_parts.contains(&owner) {
            continue;
        }
        let key = (socket.source, socket.anchor.to_array().map(f32::to_bits));
        if seen_sockets.contains(&key)
            || graph.bearings().any(|(_, bearing)| {
                !bearing.kind.mass_elements().is_empty()
                    && bearing.source == socket.source
                    && bearing.shared_anchor == socket.anchor
            })
        {
            continue;
        }
        seen_sockets.push(key);
        let axis = socket.axis;
        let origin = socket.kind.mass_origin(socket.anchor, axis);
        let outer = Mat3::from_cols(axis * axis.x, axis * axis.y, axis * axis.z);
        hardware_masses.extend(elements.into_iter().map(|element| WorldMassProperties {
            mass: element.mass,
            center: origin + axis * element.center,
            inertia: Mat3::IDENTITY * element.transverse_inertia
                + outer * (element.axial_inertia - element.transverse_inertia),
        }));
    }
    // A bare head has no part of its own; its support identifies it.
    let identifying_part = parts
        .clone()
        .next()
        .map(|(id, _)| id)
        .or_else(|| heads.first().map(|&(support, _)| support))
        .expect("a compound has a part or a mounted head");
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
        .chain(hardware_masses)
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
    // The determinant scales with the cube of inertia; an absolute epsilon
    // rejects small, well-conditioned physical controls. Validate the inverse.
    let inverse_inertia = inertia.inverse();
    if !total_mass.is_finite()
        || total_mass <= 0.0
        || !center_of_mass.is_finite()
        || !inertia.is_finite()
        || !determinant.is_finite()
        || determinant <= 0.0
        || !inverse_inertia.is_finite()
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
            inverse_inertia
        },
    })
}

/// Mass, centre, and inertia about that centre, all in build space.
#[derive(Clone, Copy)]
pub(super) struct WorldMassProperties {
    pub(super) mass: f32,
    pub(super) center: Vec3,
    pub(super) inertia: Mat3,
}

pub(super) fn compose_world_mass(
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

pub(super) fn part_world_mass(spec: PartSpec) -> WorldMassProperties {
    if let PartSpec::Cylinder(cylinder) = spec
        && cylinder.spiral().is_some()
    {
        return spiral_world_mass(cylinder);
    }
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
pub(super) struct PartMassProperties {
    pub(super) mass: f32,
    pub(super) local_center: Vec3,
    /// Inertia about `local_center`, in the part's local frame. A shaped part
    /// has products of inertia, so this cannot be a diagonal.
    pub(super) local_inertia: Mat3,
}

/// Resolves authored fixed-size parts to the cuboids physics simulates.
pub(super) fn physical_spec(spec: PartSpec) -> PartSpec {
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

pub(super) fn part_mass_properties(spec: PartSpec) -> PartMassProperties {
    match spec {
        PartSpec::Cuboid(spec) => {
            cuboid_mass_properties(spec, spec.material.properties().density_kg_m3)
        }
        PartSpec::Cylinder(spec) => {
            // An annular sector about local Y. Layered cylinders take their mass
            // from the evaluated bands instead, and spiral ones from their core
            // and ridges.
            annular_sector_mass_properties(
                spec.material.properties().density_kg_m3,
                spec.dimensions.outer_diameter() * 0.5,
                spec.dimensions.inner_diameter() * 0.5,
                spec.dimensions.axial_length(),
                spec.dimensions.sweep_angle_radians(),
            )
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
        PartSpec::Dial(spec) => {
            envelope_mass_properties(spec.size_meters(), MACHINE_PART_DENSITY_KG_M3)
        }
        PartSpec::Button(spec) => {
            envelope_mass_properties(spec.size_meters(), MACHINE_PART_DENSITY_KG_M3)
        }
        PartSpec::Input(input) => {
            cuboid_mass_properties(input.cuboid(), MACHINE_PART_DENSITY_KG_M3)
        }
        PartSpec::DimensionLink(link) => {
            cuboid_mass_properties(link.cuboid(), MACHINE_PART_DENSITY_KG_M3)
        }
    }
}

fn annular_sector_mass_properties(
    density_kg_m3: f32,
    outer: f32,
    inner: f32,
    length: f32,
    sweep: f32,
) -> PartMassProperties {
    let radial_squared = outer * outer + inner * inner;
    let mass = density_kg_m3 * sweep * (outer * outer - inner * inner) * length * 0.5;
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

/// A spiral cylinder: its straight core as a plain annulus, and everything
/// else from the same convex pieces a fine collider run would use.
fn spiral_world_mass(spec: crate::CylinderSpec) -> WorldMassProperties {
    /// Angular steps per turn the ridges are weighed at.
    const MASS_STEPS_PER_TURN: u16 = 24;
    let density = spec.material.properties().density_kg_m3;
    let rotation = spec.pose.rotation.quaternion();
    let basis = Mat3::from_quat(rotation);
    let mut bodies = Vec::new();
    if let Some(core) = crate::spiral_core(spec) {
        let properties = annular_sector_mass_properties(
            density,
            core.outer_radius,
            core.inner_radius,
            core.length,
            core::f32::consts::TAU,
        );
        bodies.push(WorldMassProperties {
            mass: properties.mass,
            center: spec.pose.translation() + rotation * (Vec3::Y * core.center_y),
            inertia: basis * properties.local_inertia * basis.transpose(),
        });
    }
    let mut volume = 0.0_f32;
    let mut first_moment = Vec3::ZERO;
    let mut second_moment = Mat3::ZERO;
    for piece in crate::spiral_pieces(spec, MASS_STEPS_PER_TURN) {
        accumulate_convex_moments(&piece, &mut volume, &mut first_moment, &mut second_moment);
    }
    if volume > f32::EPSILON {
        let center = first_moment / volume;
        let about_center = second_moment - outer_product(center, center) * volume;
        bodies.push(WorldMassProperties {
            mass: density * volume,
            center,
            inertia: (Mat3::IDENTITY * trace(about_center) - about_center) * density,
        });
    }
    let mass = bodies.iter().map(|body| body.mass).sum::<f32>();
    let center = bodies
        .iter()
        .map(|body| body.center * body.mass)
        .sum::<Vec3>()
        / mass;
    let inertia = bodies.iter().fold(Mat3::ZERO, |total, body| {
        let offset = body.center - center;
        total
            + body.inertia
            + body.mass * (Mat3::IDENTITY * offset.length_squared() - outer_product(offset, offset))
    });
    WorldMassProperties {
        mass,
        center,
        inertia,
    }
}

pub(super) fn pipe_bend_mass_properties(spec: crate::PipeBendSpec) -> PartMassProperties {
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
#[expect(
    clippy::cast_possible_truncation,
    reason = "metre-scale fittings fit f32 mass properties"
)]
pub(super) fn pipe_junction_mass_properties(spec: crate::PipeJunctionSpec) -> PartMassProperties {
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
pub(super) fn shifted_inertia(offset: Vec3, mass: f32) -> Mat3 {
    (Mat3::IDENTITY * offset.length_squared()
        - Mat3::from_cols(offset * offset.x, offset * offset.y, offset * offset.z))
        * mass
}

pub(super) fn cuboid_mass_properties(spec: CuboidSpec, density_kg_m3: f32) -> PartMassProperties {
    envelope_mass_properties(spec.size_meters(), density_kg_m3)
}

fn envelope_mass_properties(size: Vec3, density_kg_m3: f32) -> PartMassProperties {
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
pub(super) fn region_world_mass(region: &ShapeRegion) -> WorldMassProperties {
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
pub(super) fn evaluated_world_mass(
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
pub(super) fn region_pieces(region: &ShapeRegion) -> Vec<PartPiece> {
    let grid = region.grid();
    decompose(&grid, &|cell, corner| region.corner_steps(cell, corner))
}

pub(super) fn accumulate_convex_moments(
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

pub(super) fn outer_product(left: Vec3, right: Vec3) -> Mat3 {
    Mat3::from_cols(left * right.x, left * right.y, left * right.z)
}

pub(super) fn trace(matrix: Mat3) -> f32 {
    matrix.x_axis.x + matrix.y_axis.y + matrix.z_axis.z
}
