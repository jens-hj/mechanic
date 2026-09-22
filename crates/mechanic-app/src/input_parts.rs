//! Physical input placement and its plain, transient preview.

use crate::builder::inputs::{input_surface_transform, stage_input_part, validate_input_part};
use crate::camera::{MaterialWheelState, PlayerState};
use crate::controls::GameAction;
use crate::editor::history::{EditorHistory, EditorSnapshot};
use crate::editor::state::{EditorGraph, EditorState};
use crate::hotbar::{SelectedTool, Tool};
use crate::{live_edit, ui};
use bevy::prelude::*;
use mechanic_core::{BuildPose, ButtonSpec, DialSpec, FaceRef, PartSpec};

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct InputPreview {
    pub spec: PartSpec,
    pub transform: Transform,
    pub support: Option<FaceRef>,
    pub valid: bool,
}

#[derive(Resource, Default)]
pub(crate) struct InputParts {
    pub preview: Option<InputPreview>,
}

#[expect(
    clippy::too_many_arguments,
    reason = "placement consumes existing editor input and history resources"
)]
pub(crate) fn update_input_placement(
    mut parts: ResMut<InputParts>,
    selected: Res<SelectedTool>,
    mut graph: ResMut<EditorGraph>,
    mut state: ResMut<EditorState>,
    mut history: ResMut<EditorHistory>,
    actions: Res<ButtonInput<GameAction>>,
    overlay: Res<ui::UiInput>,
    player: Res<PlayerState>,
    wheel: Res<MaterialWheelState>,
) {
    parts.preview = None;
    if overlay.blocks_pointer() || !player.world_input_active() || wheel.open {
        return;
    }
    let spec = match selected.active_editor_tool() {
        Some(Tool::Dial(size)) => PartSpec::Dial(DialSpec::new(size, BuildPose::default())),
        Some(Tool::Button(size)) => PartSpec::Button(ButtonSpec::new(size, BuildPose::default())),
        _ => return,
    };
    let mut view = live_edit::EditorView::new(&mut graph, &mut state);
    let (graph, state) = view.parts();
    let yaw = Quat::from_rotation_y(
        f32::from(state.authored_orientation % 4) * std::f32::consts::FRAC_PI_2,
    );
    let candidate = state
        .hovered
        .and_then(|hit| {
            input_surface_transform(&graph.0, spec, hit, yaw, state.placement_grid)
                .map(|transform| (transform, Some(hit.face)))
        })
        .or_else(|| {
            state
                .free_placement_point
                .map(|point| (Transform::from_translation(point).with_rotation(yaw), None))
        });
    let Some((transform, support)) = candidate else {
        return;
    };
    let validated = validate_input_part(&graph.0, spec, transform, state.placement_bounds);
    parts.preview = Some(InputPreview {
        spec,
        transform: state.edit_context.map_or(transform, |context| {
            Transform::from_translation(context.frame_to_world.point(transform.translation))
                .with_rotation(context.frame_to_world.rotation() * transform.rotation)
        }),
        support,
        valid: validated.is_ok(),
    });
    if actions.just_pressed(GameAction::Primary) {
        match validated.and_then(|()| {
            stage_input_part(&graph.0, spec, transform, support, state.placement_bounds)
        }) {
            Ok(staged) => {
                let previous = EditorSnapshot::capture(&graph.0, state);
                graph.0 = staged;
                history.commit(previous);
                state.construction_mesh_dirty = true;
                state.feedback = Some("Placed physical input".to_owned());
            }
            Err(error) => state.feedback = Some(error.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::ecs::system::RunSystemOnce;
    use mechanic_core::{
        BuildCommand, BuildOutcome, ConstructionFrame, ConstructionGraph, CuboidSpec, FaceKind,
        InputSize,
    };

    #[test]
    fn physical_input_click_uses_local_surface_and_world_preview() {
        for tool in [Tool::Dial(InputSize::Panel), Tool::Button(InputSize::Panel)] {
            let mut graph = ConstructionGraph::new();
            let BuildOutcome::Spawned(anchor) = graph
                .apply(BuildCommand::Spawn(
                    CuboidSpec::new([4; 3], BuildPose::default()).unwrap(),
                ))
                .unwrap()
            else {
                panic!("support")
            };
            let authored = ConstructionFrame::new(
                Vec3::Y * (crate::garage::BUILD_MIN_Y + 3.0),
                Quat::from_rotation_z(0.7),
            )
            .unwrap();
            let frame = graph.add_construction_frame(authored).unwrap();
            graph.assign_part_frame(anchor, frame).unwrap();
            let local = graph.in_edit_frame(frame).unwrap();
            let face = FaceRef::part(anchor, FaceKind::PositiveX);
            let geometry =
                crate::builder::faces::try_face_geometry_from_ref(face, Some(&local)).unwrap();
            let moving = ConstructionFrame::new(
                Vec3::new(2.0, crate::garage::BUILD_MIN_Y + 4.0, 1.0),
                Quat::from_rotation_x(0.9),
            )
            .unwrap();
            let state = EditorState {
                hovered: Some(crate::builder::SurfaceHit {
                    distance: 1.0,
                    point: geometry.center,
                    face,
                }),
                edit_context: Some(live_edit::EditContext {
                    anchor,
                    frame,
                    frame_to_world: moving,
                }),
                placement_bounds: crate::builder::PlacementBounds::GarageBuild
                    .in_edit_frame(moving),
                ..Default::default()
            };
            let mut selected = SelectedTool::default();
            selected.select_editor_tool(tool);
            let mut actions = ButtonInput::default();
            actions.press(GameAction::Primary);
            let mut world = World::new();
            world.insert_resource(EditorGraph(graph));
            world.insert_resource(state);
            world.insert_resource(selected);
            world.insert_resource(actions);
            world.insert_resource(PlayerState {
                input_captured: true,
                ..Default::default()
            });
            world.init_resource::<InputParts>();
            world.init_resource::<EditorHistory>();
            world.init_resource::<ui::UiInput>();
            world.init_resource::<MaterialWheelState>();
            world.run_system_once(update_input_placement).unwrap();
            let preview = world.resource::<InputParts>().preview.unwrap();
            assert!(preview.valid);
            assert!(
                (preview.transform.rotation * Vec3::Y)
                    .abs_diff_eq(moving.vector(geometry.normal), 1e-5)
            );
            let graph = &world.resource::<EditorGraph>().0;
            assert_eq!(graph.parts().count(), 2);
            assert_eq!(graph.welds().count(), 1);
            let (input, _) = graph.parts().find(|(id, _)| *id != anchor).unwrap();
            assert_eq!(graph.edit_source(input), Some(anchor));
            assert!(
                (graph.part_frame(input).unwrap().rotation() * Vec3::Y)
                    .abs_diff_eq(authored.vector(geometry.normal), 1e-5)
            );
            graph.compile().unwrap();
            assert_eq!(world.resource::<EditorHistory>().undo.len(), 1);
        }
    }
}
