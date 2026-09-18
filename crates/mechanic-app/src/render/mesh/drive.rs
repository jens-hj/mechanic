//! Drive indicator, wire, and drive x-ray overlay meshes.

use crate::render::mesh::primitives::{
    append_mesh_quad, append_mesh_triangle, degenerate_overlay_mesh,
};
use crate::sequencer::DriveSequencer;
use crate::{PlacedBearing, bearing_uses_socket, simulation_bearing_pose, simulation_part_pose};
use bevy::asset::RenderAssetUsages;
use bevy::mesh::Indices;
use bevy::prelude::{Mesh, Vec3};
use bevy::render::render_resource::PrimitiveTopology;
use mechanic_core::{
    BearingDimensions, CompiledCreation, ConstructionGraph, DriveState, DriveTarget, PartId,
};
use mechanic_gpu::GpuTransform;

/// Radius multiplier placing the spin arc just outside a driven bearing's ring.
pub(crate) const DRIVE_ARC_RADIUS_SCALE: f32 = 1.4;

/// Half-thickness of every drive overlay ribbon, in metres.
pub(crate) const DRIVE_OVERLAY_HALF_WIDTH: f32 = 0.012;

/// Arc sweep of the spin-direction indicator, in radians.
pub(crate) const DRIVE_ARC_SWEEP: f32 = core::f32::consts::PI * 1.25;

/// Orthonormal pair spanning the plane perpendicular to `axis`. Matches the
/// basis every bearing ring is already built from.
pub(crate) fn axis_tangents(axis: Vec3) -> (Vec3, Vec3) {
    let axis = axis.normalize();
    let tangent_u = if axis.y.abs() < 0.9 {
        axis.cross(Vec3::Y).normalize()
    } else {
        axis.cross(Vec3::X).normalize()
    };
    (tangent_u, axis.cross(tangent_u))
}

/// Flat ribbon between two points, kept broadside to the viewer-independent
/// `face` normal so it stays visible in the unlit x-ray pass.
pub(crate) fn append_overlay_segment(
    start: Vec3,
    end: Vec3,
    face: Vec3,
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    indices: &mut Vec<u32>,
) {
    let along = end - start;
    if along.length_squared() <= f32::EPSILON {
        return;
    }
    let side = along.normalize().cross(face.normalize());
    if side.length_squared() <= f32::EPSILON {
        return;
    }
    let offset = side.normalize() * DRIVE_OVERLAY_HALF_WIDTH;
    append_mesh_quad(
        [start - offset, end - offset, end + offset, start + offset],
        face.normalize(),
        positions,
        normals,
        indices,
    );
}

/// Spin arc, arrow head, and optional angle-limit ticks for one driven bearing.
#[expect(
    clippy::too_many_arguments,
    reason = "overlay builders thread three mesh buffers"
)]
pub(crate) fn append_drive_indicator(
    anchor: Vec3,
    axis: Vec3,
    dimensions: BearingDimensions,
    state: DriveState,
    travel: Option<(f32, f32)>,
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    indices: &mut Vec<u32>,
) {
    const ARC_SEGMENTS: usize = 20;
    let axis = axis.normalize();
    let (tangent_u, tangent_v) = axis_tangents(axis);
    let radius = dimensions.outer_diameter() * 0.5 * DRIVE_ARC_RADIUS_SCALE;
    let radial = |angle: f32| tangent_u * angle.cos() + tangent_v * angle.sin();
    // The arc sweeps the way the joint is being asked to turn.
    let signed = match state.target() {
        DriveTarget::Angle(angle) | DriveTarget::LinearPosition(angle) => angle,
        DriveTarget::Speed(speed) | DriveTarget::LinearSpeed(speed) => speed,
    };
    let winding = if signed.is_sign_negative() { -1.0 } else { 1.0 };

    let mut previous = anchor + radial(0.0) * radius;
    for segment in 1..=ARC_SEGMENTS {
        let fraction = f32::from(u8::try_from(segment).unwrap_or(u8::MAX))
            / f32::from(u8::try_from(ARC_SEGMENTS).unwrap_or(u8::MAX));
        let angle = winding * DRIVE_ARC_SWEEP * fraction;
        let point = anchor + radial(angle) * radius;
        append_overlay_segment(previous, point, axis, positions, normals, indices);
        previous = point;
    }

    let tip_angle = winding * DRIVE_ARC_SWEEP;
    let tangent = (radial(tip_angle + winding * 0.01) - radial(tip_angle)).normalize_or_zero();
    if tangent != Vec3::ZERO {
        let outward = radial(tip_angle);
        let head = radius * 0.32;
        append_mesh_triangle(
            [
                previous + tangent * head,
                previous - outward * head * 0.5,
                previous + outward * head * 0.5,
            ],
            axis,
            positions,
            normals,
            indices,
        );
    }

    if let Some((minimum, maximum)) = travel {
        for angle in [minimum, maximum] {
            let direction = radial(angle);
            append_overlay_segment(
                anchor + direction * radius * 0.9,
                anchor + direction * radius * 1.25,
                axis,
                positions,
                normals,
                indices,
            );
        }
    }
}

/// Straight wire from a driven bearing to the control block steering it.
pub(crate) fn append_drive_wire(
    anchor: Vec3,
    controller_center: Vec3,
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    indices: &mut Vec<u32>,
) {
    let along = controller_center - anchor;
    if along.length_squared() <= f32::EPSILON {
        return;
    }
    // Two crossed ribbons so the wire reads from any camera angle.
    let (tangent_u, tangent_v) = axis_tangents(along);
    for face in [tangent_u, tangent_v] {
        append_overlay_segment(anchor, controller_center, face, positions, normals, indices);
    }
}

/// Mesh for the wire being dragged out by the pointer.
///
/// A wire with no length still yields one degenerate triangle so the visual
/// always has vertex data to allocate, which keeps it renderable-but-invisible
/// while no drag is in progress.
pub(crate) fn wire_drag_preview_mesh(from: Vec3, to: Vec3) -> Mesh {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut indices = Vec::new();
    append_drive_wire(from, to, &mut positions, &mut normals, &mut indices);
    if positions.is_empty() {
        return degenerate_overlay_mesh();
    }
    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_indices(Indices::U32(indices))
}

pub(crate) fn combined_drive_xray_mesh(
    graph: &ConstructionGraph,
    placed_bearings: &[PlacedBearing],
    sequencer: &DriveSequencer,
) -> Mesh {
    drive_xray_mesh(
        graph,
        placed_bearings,
        sequencer,
        |bearing, controller| {
            Some((
                bearing.shared_anchor,
                bearing.axis,
                graph.part_position(controller)?,
            ))
        },
        |part| graph.part_position(part),
    )
}

/// The same overlay in simulation space, following the bodies as they move.
///
/// Bearings and control blocks are read from the published snapshot rather than
/// the build pose, so the arcs and wires stay attached to a running mechanism.
pub(crate) fn combined_simulation_drive_xray_mesh(
    graph: &ConstructionGraph,
    creation: &CompiledCreation,
    transforms: &[GpuTransform],
    placed_bearings: &[PlacedBearing],
    sequencer: &DriveSequencer,
) -> Mesh {
    let part_position =
        |part: PartId| simulation_part_pose(graph, creation, transforms, part).map(|pose| pose.0);
    drive_xray_mesh(
        graph,
        placed_bearings,
        sequencer,
        |bearing, controller| {
            let (anchor, axis) = simulation_bearing_pose(graph, creation, transforms, bearing)?;
            Some((anchor, axis, part_position(controller)?))
        },
        part_position,
    )
}

/// Shared overlay builder. `pose` resolves one bearing and its control block to
/// world space, which is the only thing that differs between build and
/// simulation.
pub(crate) fn drive_xray_mesh(
    graph: &ConstructionGraph,
    placed_bearings: &[PlacedBearing],
    sequencer: &DriveSequencer,
    pose: impl Fn(&mechanic_core::BearingSpec, PartId) -> Option<(Vec3, Vec3, Vec3)>,
    part_position: impl Fn(PartId) -> Option<Vec3>,
) -> Mesh {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut indices = Vec::new();
    let mut drawn = Vec::new();

    for (id, bearing) in graph.bearings() {
        let Some((link, spec)) = graph.bearing_drive_link(id) else {
            continue;
        };
        // While simulating, the arc shows the state the bearing is actually in.
        let active = sequencer.active_state(link).unwrap_or(0);
        let Some(state) = spec.program.state(active) else {
            continue;
        };
        let state = if spec.reversed {
            state
                .with_target(state.target().reversed())
                .unwrap_or(state)
        } else {
            state
        };
        let travel = spec.limits.angle_limits();
        // A socket carrying several rotor rows would otherwise draw its arc more
        // than once at the same place.
        if drawn
            .iter()
            .any(|previous: &Vec3| previous.abs_diff_eq(bearing.shared_anchor, 1.0e-5))
        {
            continue;
        }
        drawn.push(bearing.shared_anchor);

        let Some((anchor, axis, controller_center)) = pose(bearing, spec.controller) else {
            continue;
        };
        let dimensions = placed_bearings
            .iter()
            .find(|socket| bearing_uses_socket(bearing, **socket))
            .map_or(bearing.dimensions, |socket| socket.dimensions);
        append_drive_indicator(
            anchor,
            axis,
            dimensions,
            state,
            travel,
            &mut positions,
            &mut normals,
            &mut indices,
        );
        append_drive_wire(
            anchor,
            controller_center,
            &mut positions,
            &mut normals,
            &mut indices,
        );
    }

    for (_, link) in graph.input_seat_links() {
        if let (Some(input), Some(seat)) = (part_position(link.input), part_position(link.seat)) {
            append_drive_wire(input, seat, &mut positions, &mut normals, &mut indices);
        }
    }
    for (_, link) in graph.seat_controller_links() {
        if let (Some(seat), Some(controller)) =
            (part_position(link.seat), part_position(link.controller))
        {
            append_drive_wire(seat, controller, &mut positions, &mut normals, &mut indices);
        }
    }

    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_indices(Indices::U32(indices))
}
