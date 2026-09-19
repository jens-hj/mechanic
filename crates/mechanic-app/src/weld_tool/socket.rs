//! Weld placement onto the moving attachment surface of a bearing socket.
use super::{Pick, motion};
use crate::builder;
use crate::editor::build_actions::PlacedBearing;
use crate::simulation::state::AppSimulation;
use bevy::prelude::*;
use mechanic_core::{
    BearingSpec, BuildCommand, CarriageFace, ConstructionFrame, ConstructionGraph, FaceOwner,
    JointKind, PartId, RigidLinkSpec, WeldFeature, WeldSelection,
};

pub(super) fn append_outline(
    socket: PlacedBearing,
    frame: ConstructionFrame,
    geometry: &mut crate::editor::overlay::OverlayGeometry,
) {
    let mut line = |a, b| {
        crate::editor::overlay::append_overlay_bar(frame.point(a), frame.point(b), 0.010, geometry);
    };
    match socket.kind {
        JointKind::Rotational => {
            let (u, v) = crate::render::mesh::drive::axis_tangents(socket.axis);
            for diameter in [
                socket.dimensions.inner_diameter(),
                socket.dimensions.outer_diameter(),
            ] {
                for i in 0..32_u16 {
                    let point = |i| {
                        let angle = f32::from(i) * std::f32::consts::TAU / 32.0;
                        socket.anchor + (u * angle.cos() + v * angle.sin()) * diameter * 0.5
                    };
                    line(point(i), point(i + 1));
                }
            }
        }
        JointKind::Suspension(_) | JointKind::Piston(_) => {
            let Some((center, radius)) = socket.moving_plate() else {
                return;
            };
            let (u, v) = crate::render::mesh::drive::axis_tangents(socket.axis);
            for i in 0..32_u16 {
                let point = |i| {
                    let angle = f32::from(i) * std::f32::consts::TAU / 32.0;
                    center + (u * angle.cos() + v * angle.sin()) * radius
                };
                line(point(i), point(i + 1));
            }
        }
        JointKind::Linear(rail) => {
            let Ok(rotation) = rail.rotation(socket.axis) else {
                return;
            };
            let center = socket.anchor + rotation * rail.face.origin(rail.dimensions);
            let size = rail.face.size(rail.dimensions);
            let u = socket.axis * size.x * 0.5;
            let v = socket.axis.cross(rotation * rail.face.normal()) * size.y * 0.5;
            let corners = [
                center - u - v,
                center + u - v,
                center + u + v,
                center - u + v,
            ];
            for i in 0..4 {
                line(corners[i], corners[(i + 1) % 4]);
            }
        }
    }
}

pub(crate) fn pick(
    graph: &ConstructionGraph,
    simulation: &AppSimulation,
    sockets: &[PlacedBearing],
    ray: Ray3d,
) -> Option<Pick> {
    let mut selected = super::pick(graph, simulation, ray);
    // Include unselectable curved parts in occlusion.
    let mut nearest = graph
        .parts()
        .filter_map(|(part, _)| {
            let inverse = motion(simulation, part, false).ok()?.inverse();
            builder::raycast_part_in_construction(
                graph,
                part,
                inverse.point(ray.origin),
                inverse.vector(ray.direction.as_vec3()),
            )
            .map(|hit| hit.distance)
        })
        .min_by(f32::total_cmp)
        .unwrap_or(f32::INFINITY);
    for &socket in sockets {
        let FaceOwner::Part(support) = socket.source.owner else {
            continue;
        };
        let targets = crate::editor::build_actions::bearing_socket_targets(graph, socket);
        let part = targets.first().copied().unwrap_or(support);
        let Ok(frame) = motion(simulation, part, false) else {
            continue;
        };
        // Housing occlusion uses the same body motion as the mating surface,
        // including paused world publications.
        if let Some((rail, carriage)) = crate::linear_editor::build_poses(socket)
            && let Ok(support_frame) = motion(simulation, support, false)
        {
            let place = |pose: Transform, frame: ConstructionFrame| {
                Transform::from_translation(frame.point(pose.translation))
                    .with_rotation(frame.rotation() * pose.rotation)
            };
            if let Some(distance) = crate::linear_editor::raycast(
                socket,
                place(rail, support_frame),
                place(carriage, frame),
                ray.origin,
                ray.direction.as_vec3(),
            ) && distance < nearest
            {
                nearest = distance;
                selected = None;
            }
        }
        let inverse = frame.inverse();
        let origin = inverse.point(ray.origin);
        let direction = inverse.vector(ray.direction.as_vec3());
        let occupied = graph.bearings().find_map(|(_, joint)| {
            if crate::editor::build_actions::bearing_uses_socket(joint, socket) {
                Some(joint.kind)
            } else {
                None
            }
        });
        let Some((distance, socket, point, normal, tangent)) =
            surface(socket, occupied, origin, direction)
        else {
            continue;
        };
        if distance > nearest + 1.0e-5 {
            continue;
        }
        nearest = distance;
        selected = Some(Pick {
            part,
            face: socket.source,
            socket: Some(socket),
            selection: WeldSelection {
                feature: WeldFeature::Face,
                point,
                normal,
                tangent,
            },
        });
    }
    selected
}

fn surface(
    socket: PlacedBearing,
    occupied: Option<JointKind>,
    origin: Vec3,
    direction: Vec3,
) -> Option<(f32, PlacedBearing, Vec3, Vec3, Vec3)> {
    match socket.kind {
        JointKind::Rotational => {
            if direction.dot(socket.axis) >= -1.0e-6 {
                return None;
            }
            let distance = crate::editor::raycast::raycast_bearing_annulus(
                origin,
                direction,
                socket.anchor,
                socket.axis,
                socket.dimensions,
            )?;
            let point = origin + direction * distance;
            let point = point - socket.axis * (point - socket.anchor).dot(socket.axis);
            Some((
                distance,
                socket,
                point,
                socket.axis,
                crate::render::mesh::drive::axis_tangents(socket.axis).0,
            ))
        }
        JointKind::Suspension(_) | JointKind::Piston(_) => {
            let normal = socket.axis;
            let denominator = direction.dot(normal);
            if denominator >= -1.0e-6 {
                return None;
            }
            let (center, radius) = socket.moving_plate()?;
            let distance = (center - origin).dot(normal) / denominator;
            let point = origin + direction * distance;
            if distance < 0.0 || point.distance(center) > radius {
                return None;
            }
            Some((
                distance,
                socket,
                point,
                normal,
                crate::render::mesh::drive::axis_tangents(normal).0,
            ))
        }
        JointKind::Linear(rail) => {
            let rotation = rail.rotation(socket.axis).ok()?;
            [
                CarriageFace::Top,
                CarriageFace::PositiveSide,
                CarriageFace::NegativeSide,
            ]
            .into_iter()
            .filter_map(|face| {
                if matches!(occupied, Some(JointKind::Linear(other)) if other.face != face) {
                    return None;
                }
                let normal = rotation * face.normal();
                let denominator = direction.dot(normal);
                if denominator >= -1.0e-6 {
                    return None;
                }
                let center = socket.anchor + rotation * face.origin(rail.dimensions);
                let distance = (center - origin).dot(normal) / denominator;
                if distance < 0.0 {
                    return None;
                }
                let point = origin + direction * distance;
                let offset = point - center;
                let size = face.size(rail.dimensions);
                if offset.dot(socket.axis).abs() > size.x / 2.0 + 1.0e-5
                    || offset.dot(socket.axis.cross(normal)).abs() > size.y / 2.0 + 1.0e-5
                {
                    return None;
                }
                let mut socket = socket;
                socket.kind = JointKind::Linear(mechanic_core::LinearBearing { face, ..rail });
                Some((distance, socket, point, normal, socket.axis))
            })
            .min_by(|a, b| a.0.total_cmp(&b.0))
        }
    }
}

pub(crate) fn stage(
    graph: &ConstructionGraph,
    source: &Pick,
    socket: PlacedBearing,
    transform: ConstructionFrame,
    anchored: impl IntoIterator<Item = PartId>,
) -> Result<ConstructionGraph, String> {
    let component = graph
        .structural_component(source.part, anchored)
        .map_err(|e| e.to_string())?;
    if component.touches_authored_ground() {
        return Err("Terrain-anchored sources cannot relocate".to_owned());
    }
    if matches!(socket.source.owner, FaceOwner::Part(part) if component.contains(part)) {
        return Err("Select a separate assembly to attach to this bearing".to_owned());
    }
    let mut staged = graph.clone();
    staged
        .reframe_parts(component.parts(), transform)
        .map_err(|e| e.to_string())?;
    staged
        .apply(BuildCommand::AddBearing(
            BearingSpec::new(socket.source, source.face, socket.anchor, socket.axis)
                .with_dimensions(socket.dimensions)
                .with_kind(socket.kind),
        ))
        .map_err(|e| e.to_string())?;
    for target in crate::editor::build_actions::bearing_socket_targets(graph, socket) {
        staged
            .apply(BuildCommand::RigidLink(RigidLinkSpec {
                first: target,
                second: source.part,
            }))
            .map_err(|e| e.to_string())?;
    }
    staged
        .apply(BuildCommand::CancelPending)
        .map_err(|e| e.to_string())?;
    staged.compile().map_err(|e| e.to_string())?;
    Ok(staged)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mechanic_core::{BuildOutcome, BuildPose, CuboidSpec, FaceKind, FaceRef, GridRotation};

    fn spawn(graph: &mut ConstructionGraph, x: i32) -> PartId {
        let BuildOutcome::Spawned(part) = graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new(
                    [2; 3],
                    BuildPose::new(IVec3::new(x, 28, 0), GridRotation::default()),
                )
                .unwrap(),
            ))
            .unwrap()
        else {
            panic!("spawn expected")
        };
        part
    }

    #[test]
    fn occupied_bearing_attaches_to_the_moving_body_and_rejects_invalid_contact() {
        let mut graph = ConstructionGraph::new();
        let support = spawn(&mut graph, 8);
        let first = spawn(&mut graph, 0);
        let second = spawn(&mut graph, -8);
        let face = FaceRef::part(support, FaceKind::PositiveY);
        let socket = PlacedBearing {
            source: face,
            anchor: builder::face_geometry_from_ref(face, Some(&graph)).center,
            axis: Vec3::Y,
            dimensions: mechanic_core::BearingDimensions::new(1.0, 0.05).unwrap(),
            kind: JointKind::Rotational,
        };
        for (part, x, offset) in [(first, 0.0, -0.26), (second, -2.0, 0.26)] {
            let source = super::super::pick(
                &graph,
                &AppSimulation::default(),
                Ray3d::new(Vec3::new(x, 9.0, 0.0), Dir3::NEG_Y),
            )
            .unwrap();
            assert_eq!(source.part, part);
            let target = WeldSelection {
                feature: WeldFeature::Face,
                point: socket.anchor + Vec3::X * offset,
                normal: Vec3::Y,
                tangent: source.selection.tangent,
            };
            let transform = mechanic_core::WeldAlignment::new(source.selection, target)
                .unwrap()
                .initial();
            assert!(stage(&graph, &source, socket, transform, [part]).is_err());
            let off_socket = ConstructionFrame::new(Vec3::X * 10.0, Quat::IDENTITY)
                .unwrap()
                .compose(transform);
            assert!(stage(&graph, &source, socket, off_socket, []).is_err());
            assert!((graph.part_position(part).unwrap().x - x).abs() < 1.0e-5);
            graph = stage(&graph, &source, socket, transform, []).unwrap();
        }
        let creation = graph.compile().unwrap();
        assert_eq!(creation.compounds.len(), 2);
        let body = |part| {
            creation
                .part_to_compound
                .iter()
                .find(|(id, _)| *id == part)
                .unwrap()
                .1
        };
        assert_eq!(body(first), body(second));
        assert_ne!(body(first), body(support));
    }

    #[test]
    fn linear_side_picking_respects_the_occupied_carriage_face() {
        let mut graph = ConstructionGraph::new();
        let support = spawn(&mut graph, 0);
        let rail = mechanic_core::LinearBearing {
            dimensions: mechanic_core::LinearBearingDimensions::default(),
            mount_normal: Vec3::Y,
            face: CarriageFace::Top,
        };
        let socket = PlacedBearing {
            source: FaceRef::part(support, FaceKind::PositiveY),
            anchor: Vec3::Y * 7.25,
            axis: Vec3::X,
            dimensions: mechanic_core::BearingDimensions::default(),
            kind: JointKind::Linear(rail),
        };
        assert!(
            pick(
                &graph,
                &AppSimulation::default(),
                &[socket],
                Ray3d::new(socket.anchor + Vec3::new(0.15, 1.0, 0.0), Dir3::NEG_Y),
            )
            .is_none(),
            "rail housing must occlude its supporting block"
        );
        let rotation = rail.rotation(socket.axis).unwrap();
        for face in [
            CarriageFace::Top,
            CarriageFace::PositiveSide,
            CarriageFace::NegativeSide,
        ] {
            let normal = rotation * face.normal();
            let center = socket.anchor + rotation * face.origin(rail.dimensions);
            let hit = surface(socket, None, center + normal, -normal).unwrap();
            assert!(matches!(hit.1.kind, JointKind::Linear(selected) if selected.face == face));
            assert!(hit.2.abs_diff_eq(center, 1.0e-5));
            let occupied = surface(socket, Some(socket.kind), center + normal, -normal);
            assert_eq!(occupied.is_some(), face == CarriageFace::Top);
        }
    }
}
