//! Entering, leaving, and locating seats.

use crate::builder::raycast::raycast_construction;
use crate::camera::{
    MainCamera, MaterialWheelState, PlayerCamera, PlayerState, SEATED_EYE_HEIGHT,
    seated_view_rotation,
};
use crate::controls::GameAction;
use crate::editor::raycast::raycast_simulation;
use crate::editor::state::{EditorGraph, EditorState};
use crate::simulation::state::AppSimulation;
use crate::{camera, ui, world};
use bevy::prelude::{
    ButtonInput, Camera, GlobalTransform, Quat, Res, ResMut, Single, State, ToOwned, Transform,
    Vec2, Vec3, Window, With,
};
use bevy::window::PrimaryWindow;
use mechanic_core::{ConstructionGraph, FaceOwner, PartId, PartSpec};

#[expect(clippy::too_many_arguments)]
pub(crate) fn handle_seat_interaction(
    actions: Res<ButtonInput<GameAction>>,
    overlay: Res<ui::UiInput>,
    wheel: Res<MaterialWheelState>,
    graph: Res<EditorGraph>,
    simulation: Res<AppSimulation>,
    space: Res<State<world::AppSpace>>,
    window: Single<&Window, With<PrimaryWindow>>,
    mut camera: Single<
        (
            &Camera,
            &mut PlayerCamera,
            &mut Transform,
            &mut GlobalTransform,
        ),
        With<MainCamera>,
    >,
    mut player: ResMut<PlayerState>,
    mut state: ResMut<EditorState>,
) {
    if actions.just_pressed(GameAction::Interact)
        && player.world_input_active()
        && !overlay.blocks_keyboard()
        && !wheel.open
    {
        if let Some(seat) = player.seat {
            let exit_position =
                seat_exit_position(&graph.0, &simulation, seat).unwrap_or(camera.2.translation);
            player.leave_seat_at(exit_position, *space.get());
            state.feedback = Some("Left Seat".to_owned());
            return;
        }
        let (camera_component, _, _, camera_global) = &mut *camera;
        let cursor_position = camera::viewport_center(Vec2::new(window.width(), window.height()));
        let hit = camera_component
            .viewport_to_world(camera_global, cursor_position)
            .ok()
            .and_then(|ray| {
                raycast_seat_interaction(&graph.0, &simulation, ray.origin, ray.direction.as_vec3())
            });
        let seat = hit.and_then(|(part, _)| {
            seat_surface_distance(&graph.0, &simulation, part, player.position)
                .map(|distance| (part, distance))
        });
        if let Some((part, distance)) = seat
            && camera::seat_entry_allowed(distance, true)
        {
            player.seat = Some(part);
            camera.1.yaw = 0.0;
            camera.1.pitch = 0.0;
            state.feedback = Some("Seated — mouse looks around, E leaves the Seat".to_owned());
        } else {
            state.feedback = Some(if hit.is_some() {
                "Seat must be under the reticle and within 3 m".to_owned()
            } else {
                "Aim the reticle at a Seat within 3 m and press E".to_owned()
            });
        }
    }

    let Some(seat) = player.seat else {
        return;
    };
    let Some((seat_center, seat_rotation)) = seat_world_pose(&graph.0, &simulation, seat) else {
        player.seat = None;
        return;
    };
    let (_, view, transform, global) = &mut *camera;
    let rotation = seated_view_rotation(seat_rotation, view.yaw, view.pitch);
    **transform = view.apply_pullback(
        seat_center + seat_rotation * (Vec3::Y * SEATED_EYE_HEIGHT),
        rotation,
    );
    **global = GlobalTransform::from(**transform);
}

pub(crate) fn seat_interaction_graph<'a>(
    graph: &'a ConstructionGraph,
    simulation: &'a AppSimulation,
) -> &'a ConstructionGraph {
    if simulation.creation.is_some() {
        &simulation.published_graph
    } else {
        graph
    }
}

pub(crate) fn raycast_seat_interaction(
    graph: &ConstructionGraph,
    simulation: &AppSimulation,
    origin: Vec3,
    direction: Vec3,
) -> Option<(PartId, f32)> {
    let interaction_graph = seat_interaction_graph(graph, simulation);
    if let Some(creation) = simulation.creation.as_ref() {
        return raycast_simulation(
            interaction_graph,
            creation,
            &simulation.transforms,
            origin,
            direction,
        )
        .map(|hit| (hit.part, hit.distance));
    }
    let hit = raycast_construction(interaction_graph, origin, direction)?;
    let FaceOwner::Part(part) = hit.face.owner else {
        return None;
    };
    Some((part, hit.distance))
}

pub(crate) fn seat_world_pose(
    graph: &ConstructionGraph,
    simulation: &AppSimulation,
    seat: PartId,
) -> Option<(Vec3, Quat)> {
    seat_interaction_graph(graph, simulation)
        .is_seat(seat)
        .then(|| simulation.live_part_pose(graph, seat))?
}

pub(crate) fn seat_exit_position(
    graph: &ConstructionGraph,
    simulation: &AppSimulation,
    seat: PartId,
) -> Option<Vec3> {
    let PartSpec::Seat(spec) = seat_interaction_graph(graph, simulation)
        .part(seat)
        .copied()?
    else {
        return None;
    };
    let (centre, rotation) = seat_world_pose(graph, simulation, seat)?;
    Some(
        centre
            + rotation
                * (Vec3::Y * (spec.cuboid().size_meters().y * 0.5 + SEAT_EXIT_CLEARANCE_METERS)),
    )
}

pub(crate) fn seat_surface_distance(
    graph: &ConstructionGraph,
    simulation: &AppSimulation,
    seat: PartId,
    point: Vec3,
) -> Option<f32> {
    let PartSpec::Seat(spec) = seat_interaction_graph(graph, simulation)
        .part(seat)
        .copied()?
    else {
        return None;
    };
    let (centre, rotation) = seat_world_pose(graph, simulation, seat)?;
    let local = rotation.inverse() * (point - centre);
    let outside = (local.abs() - spec.cuboid().size_meters() * 0.5).max(Vec3::ZERO);
    Some(outside.length())
}

pub(crate) const SEAT_EXIT_CLEARANCE_METERS: f32 = 0.01;
