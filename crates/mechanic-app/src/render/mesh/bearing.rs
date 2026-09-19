//! Bearing ring geometry and its terraced texture profile.

use crate::builder::{BEARING_DEPTH, face_geometry_from_ref};
use crate::editor::build_actions::{PlacedBearing, bearing_uses_socket};
use crate::render::materials::BEARING_RENDER_RADIAL_SKIN;
use bevy::asset::RenderAssetUsages;
use bevy::mesh::Indices;
use bevy::prelude::{Mesh, Vec3};
use bevy::render::render_resource::PrimitiveTopology;
use mechanic_core::{BearingDimensions, ConstructionGraph};

pub(crate) fn combined_bearing_mesh(
    graph: &ConstructionGraph,
    placed_bearings: &[PlacedBearing],
) -> Mesh {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut tangents = Vec::new();
    let mut indices = Vec::new();
    for (_, bearing) in graph.bearings().filter(|(_, bearing)| {
        matches!(bearing.kind, mechanic_core::JointKind::Rotational)
            && !placed_bearings
                .iter()
                .any(|&socket| bearing_uses_socket(bearing, socket))
    }) {
        append_bearing_cylinder(
            bearing.shared_anchor,
            bearing.axis,
            bearing.dimensions,
            &mut positions,
            &mut normals,
            &mut uvs,
            &mut tangents,
            &mut indices,
        );
    }
    for bearing in placed_bearings {
        if bearing.kind.is_translational() {
            continue;
        }
        let axis = face_geometry_from_ref(bearing.source, Some(graph)).normal;
        append_bearing_cylinder(
            bearing.anchor,
            axis,
            bearing.dimensions,
            &mut positions,
            &mut normals,
            &mut uvs,
            &mut tangents,
            &mut indices,
        );
    }

    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
    .with_inserted_attribute(Mesh::ATTRIBUTE_TANGENT, tangents)
    .with_inserted_indices(Indices::U32(indices))
}

pub(crate) fn single_bearing_mesh(dimensions: BearingDimensions) -> Mesh {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut tangents = Vec::new();
    let mut indices = Vec::new();
    append_bearing_cylinder(
        Vec3::ZERO,
        Vec3::Y,
        dimensions,
        &mut positions,
        &mut normals,
        &mut uvs,
        &mut tangents,
        &mut indices,
    );
    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
    .with_inserted_attribute(Mesh::ATTRIBUTE_TANGENT, tangents)
    .with_inserted_indices(Indices::U32(indices))
}

pub(crate) const BEARING_SEGMENTS: u16 = 24;

pub(crate) const BEARING_ATLAS_PIXELS: f32 = 1_024.0;

pub(crate) const BEARING_ARC_METERS_PER_TILE: f32 = 0.05;

pub(crate) const BEARING_LAND_METERS: f32 = 0.008;

pub(crate) const BEARING_LIP_METERS: f32 = 0.006;

pub(crate) const BEARING_RELIEF_MIN_METERS: f32 = 0.005;

pub(crate) const BEARING_RELIEF_MAX_METERS: f32 = 0.040;

pub(crate) const BEARING_RELIEF_WALL_FRACTION: f32 = 0.10;

pub(crate) const BEARING_TERRACE_NOMINAL_METERS: f32 = 0.014;

pub(crate) const BEARING_TURN_METERS: f32 = 0.009;

pub(crate) const BEARING_STEP_SPLIT: f32 = 0.66;

#[derive(Clone, Copy, Debug)]
pub(crate) struct BearingProfilePlan {
    pub(crate) steps: u8,
    pub(crate) terrace_meters: f32,
    pub(crate) turns: u16,
    pub(crate) relief_meters: f32,
}

#[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub(crate) fn bearing_profile_plan(outer_radius: f32, inner_radius: f32) -> BearingProfilePlan {
    let wall = outer_radius - inner_radius;
    let middle = (wall - BEARING_LAND_METERS - BEARING_LIP_METERS).max(0.0005);
    let steps = (middle / BEARING_TERRACE_NOMINAL_METERS)
        .round()
        .clamp(1.0, 4.0) as u8;
    let unit = middle / f32::from(steps);
    let relief_meters = (wall * BEARING_RELIEF_WALL_FRACTION)
        .clamp(BEARING_RELIEF_MIN_METERS, BEARING_RELIEF_MAX_METERS);
    let terrace_meters = (unit - relief_meters).max(0.0015);
    let turns = (terrace_meters / BEARING_TURN_METERS).round().max(1.0) as u16;
    BearingProfilePlan {
        steps,
        terrace_meters,
        turns,
        relief_meters,
    }
}

pub(crate) fn bearing_band_v(start_row: f32, end_row: f32, t: f32) -> f32 {
    1.0 - (start_row + (end_row - start_row) * t) / BEARING_ATLAS_PIXELS
}

pub(crate) fn bearing_u_repeat(radius: f32) -> f32 {
    (std::f32::consts::TAU * radius.max(0.02) / BEARING_ARC_METERS_PER_TILE)
        .round()
        .max(1.0)
}

#[expect(clippy::too_many_arguments, clippy::too_many_lines)]
pub(crate) fn append_bearing_cylinder(
    anchor: Vec3,
    axis: Vec3,
    dimensions: BearingDimensions,
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    uvs: &mut Vec<[f32; 2]>,
    tangents: &mut Vec<[f32; 4]>,
    indices: &mut Vec<u32>,
) {
    const LAND_TS: [f32; 5] = [0.0, 0.17, 0.34, 0.62, 1.0];
    const RELIEF_TS: [f32; 7] = [0.0, 0.12, 0.235, 0.40, 0.55, 0.75, 1.0];
    const LIP_TS: [f32; 7] = [0.0, 0.24, 0.46, 0.60, 0.74, 0.87, 1.0];

    let axis = axis.normalize();
    let radial_u = if axis.y.abs() < 0.9 {
        axis.cross(Vec3::Y).normalize()
    } else {
        axis.cross(Vec3::X).normalize()
    };
    let radial_v = axis.cross(radial_u);
    let outer_radius = dimensions.outer_diameter() * 0.5 - BEARING_RENDER_RADIAL_SKIN;
    let inner_radius = if dimensions.inner_diameter() > 0.0 {
        dimensions.inner_diameter() * 0.5 + BEARING_RENDER_RADIAL_SKIN
    } else {
        0.0
    };
    let plan = bearing_profile_plan(outer_radius, inner_radius);
    let repeat = bearing_u_repeat(outer_radius);

    for (center, normal, front) in [
        (anchor + axis * BEARING_DEPTH * 0.5, axis, true),
        (anchor - axis * BEARING_DEPTH * 0.5, -axis, false),
    ] {
        let mut radius = outer_radius;
        let land = LAND_TS.map(|t| {
            (
                radius - BEARING_LAND_METERS * t,
                bearing_band_v(192.0, 320.0, t),
            )
        });
        append_bearing_face_strip(
            center, normal, front, radial_u, radial_v, repeat, &land, positions, normals, uvs,
            tangents, indices,
        );
        radius -= BEARING_LAND_METERS;

        let terrace_tile = plan.terrace_meters / f32::from(plan.turns);
        for _ in 0..plan.steps {
            for _ in 0..plan.turns {
                let terrace = [
                    (radius, bearing_band_v(320.0, 704.0, 0.0)),
                    (
                        radius - terrace_tile,
                        bearing_band_v(320.0, 704.0, BEARING_STEP_SPLIT),
                    ),
                ];
                append_bearing_face_strip(
                    center, normal, front, radial_u, radial_v, repeat, &terrace, positions,
                    normals, uvs, tangents, indices,
                );
                radius -= terrace_tile;
            }
            let relief = RELIEF_TS.map(|t| {
                (
                    radius - plan.relief_meters * t,
                    bearing_band_v(
                        320.0,
                        704.0,
                        BEARING_STEP_SPLIT + t * (1.0 - BEARING_STEP_SPLIT),
                    ),
                )
            });
            append_bearing_face_strip(
                center, normal, front, radial_u, radial_v, repeat, &relief, positions, normals,
                uvs, tangents, indices,
            );
            radius -= plan.relief_meters;
        }

        let lip_span = radius - inner_radius;
        let lip = LIP_TS.map(|t| (radius - lip_span * t, bearing_band_v(704.0, 832.0, t)));
        append_bearing_face_strip(
            center, normal, front, radial_u, radial_v, repeat, &lip, positions, normals, uvs,
            tangents, indices,
        );
    }

    let upper = anchor + axis * BEARING_DEPTH * 0.5;
    let lower = anchor - axis * BEARING_DEPTH * 0.5;
    let outer_upper = append_bearing_profile_ring(
        upper,
        outer_radius,
        radial_u,
        radial_v,
        repeat,
        bearing_band_v(0.0, 192.0, 0.0),
        BearingRingNormal::Radial(1.0),
        1.0,
        positions,
        normals,
        uvs,
        tangents,
    );
    let outer_lower = append_bearing_profile_ring(
        lower,
        outer_radius,
        radial_u,
        radial_v,
        repeat,
        bearing_band_v(0.0, 192.0, 1.0),
        BearingRingNormal::Radial(1.0),
        1.0,
        positions,
        normals,
        uvs,
        tangents,
    );
    stitch_bearing_side(outer_upper, outer_lower, true, indices);

    if inner_radius > 0.0 {
        let bore_repeat = bearing_u_repeat(inner_radius);
        let inner_upper = append_bearing_profile_ring(
            upper,
            inner_radius,
            radial_u,
            radial_v,
            bore_repeat,
            bearing_band_v(832.0, 1_024.0, 0.0),
            BearingRingNormal::Radial(-1.0),
            -1.0,
            positions,
            normals,
            uvs,
            tangents,
        );
        let inner_lower = append_bearing_profile_ring(
            lower,
            inner_radius,
            radial_u,
            radial_v,
            bore_repeat,
            bearing_band_v(832.0, 1_024.0, 1.0),
            BearingRingNormal::Radial(-1.0),
            -1.0,
            positions,
            normals,
            uvs,
            tangents,
        );
        stitch_bearing_side(inner_upper, inner_lower, false, indices);
    }
}

#[derive(Clone, Copy)]
pub(crate) enum BearingRingNormal {
    Face(Vec3),
    Radial(f32),
}

#[expect(clippy::too_many_arguments)]
pub(crate) fn append_bearing_face_strip(
    center: Vec3,
    normal: Vec3,
    front: bool,
    radial_u: Vec3,
    radial_v: Vec3,
    repeat: f32,
    rings: &[(f32, f32)],
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    uvs: &mut Vec<[f32; 2]>,
    tangents: &mut Vec<[f32; 4]>,
    indices: &mut Vec<u32>,
) {
    let handedness = if front { -1.0 } else { 1.0 };
    let mut previous = None;
    for &(radius, v) in rings {
        let current = append_bearing_profile_ring(
            center,
            radius.max(0.0),
            radial_u,
            radial_v,
            repeat,
            v,
            BearingRingNormal::Face(normal),
            handedness,
            positions,
            normals,
            uvs,
            tangents,
        );
        if let Some((previous_start, previous_radius)) = previous {
            stitch_bearing_face(
                previous_start,
                current,
                front,
                radius <= f32::EPSILON && previous_radius > f32::EPSILON,
                indices,
            );
        }
        previous = Some((current, radius));
    }
}

#[expect(clippy::too_many_arguments)]
pub(crate) fn append_bearing_profile_ring(
    center: Vec3,
    radius: f32,
    radial_u: Vec3,
    radial_v: Vec3,
    repeat: f32,
    v: f32,
    ring_normal: BearingRingNormal,
    handedness: f32,
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    uvs: &mut Vec<[f32; 2]>,
    tangents: &mut Vec<[f32; 4]>,
) -> u32 {
    let start = u32::try_from(positions.len()).expect("prototype mesh fits 32-bit indices");
    for segment in 0..=BEARING_SEGMENTS {
        let phase = f32::from(segment) / f32::from(BEARING_SEGMENTS);
        let angle = std::f32::consts::TAU * phase;
        let radial = radial_u * angle.cos() + radial_v * angle.sin();
        let tangent = -radial_u * angle.sin() + radial_v * angle.cos();
        let normal = match ring_normal {
            BearingRingNormal::Face(normal) => normal,
            BearingRingNormal::Radial(sign) => radial * sign,
        };
        positions.push((center + radial * radius).to_array());
        normals.push(normal.to_array());
        uvs.push([phase * repeat, v]);
        tangents.push([tangent.x, tangent.y, tangent.z, handedness]);
    }
    start
}

pub(crate) fn stitch_bearing_face(
    outer: u32,
    inner: u32,
    front: bool,
    inner_is_center: bool,
    indices: &mut Vec<u32>,
) {
    for segment in 0..BEARING_SEGMENTS {
        let current = u32::from(segment);
        let next = current + 1;
        if front {
            indices.extend([outer + current, outer + next, inner + current]);
            if !inner_is_center {
                indices.extend([outer + next, inner + next, inner + current]);
            }
        } else {
            indices.extend([outer + current, inner + current, outer + next]);
            if !inner_is_center {
                indices.extend([outer + next, inner + current, inner + next]);
            }
        }
    }
}

pub(crate) fn stitch_bearing_side(upper: u32, lower: u32, outward: bool, indices: &mut Vec<u32>) {
    for segment in 0..BEARING_SEGMENTS {
        let current = u32::from(segment);
        let next = current + 1;
        if outward {
            indices.extend([
                upper + current,
                lower + current,
                upper + next,
                upper + next,
                lower + current,
                lower + next,
            ]);
        } else {
            indices.extend([
                upper + current,
                upper + next,
                lower + current,
                upper + next,
                lower + next,
                lower + current,
            ]);
        }
    }
}

pub(crate) fn append_bearing_face_ring(
    center: Vec3,
    normal: Vec3,
    radius: f32,
    tangent_u: Vec3,
    tangent_v: Vec3,
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
) -> u32 {
    const SEGMENTS: u16 = 24;
    let base = u32::try_from(positions.len()).expect("prototype mesh fits 32-bit indices");
    for segment in 0..SEGMENTS {
        let angle = std::f32::consts::TAU * f32::from(segment) / f32::from(SEGMENTS);
        let radial = tangent_u * angle.cos() + tangent_v * angle.sin();
        positions.push((center + radial * radius).to_array());
        normals.push(normal.to_array());
    }
    base
}
