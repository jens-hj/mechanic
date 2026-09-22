//! Plain button configuration callouts rendered by Mosaic.
#![allow(clippy::wildcard_imports, reason = "Mosaic authoring vocabulary")]
use super::{Handles, styles::*};
use crate::button_config::Control;
use bevy_mosaic::ui::*;
use mosaic_macros::{component, view};

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Model {
    pub at: Option<(f32, f32)>,
    pub key: String,
    pub mode: String,
    pub status: String,
}
pub(crate) type Layout = std::rc::Rc<std::cell::RefCell<std::collections::BTreeMap<Control, Rect>>>;

#[component]
pub(crate) fn ButtonOverlay(handles: Handles) -> Element {
    let model = handles.button_config;
    let layout = handles.button_layout.clone();
    view! {
        stack width:fill height:fill align:start justify:start nohit {
            if model.with(|m| m.at.is_some()) {
                stack width:0px height:0px nohit translate:(x:{ Length::px(model.with(|m| m.at.unwrap_or_default().0)) } y:{ Length::px(model.with(|m| m.at.unwrap_or_default().1)) }) {
                    col width:220px height:min-content shrink:0 gap:4px nohit {
                        (control(model, layout.clone(), Control::Key))
                        (control(model, layout.clone(), Control::Clear))
                        (control(model, layout.clone(), Control::Mode))
                        text #mechanic.value width:fill height:min-content { model.with(|m| m.status.clone()) }
                    }
                }
            }
        }
    }
}
fn control(model: State<Model>, layout: Layout, control: Control) -> Element {
    view! {
        row #mechanic.badge width:220px height:32px shrink:0 pad:6px nohit @layout:{ move |rect: Rect| { layout.borrow_mut().insert(control, rect); } } {
            text #mechanic.value { model.with(|m| match control { Control::Key => m.key.clone(), Control::Clear => "Clear".into(), Control::Mode => m.mode.clone() }) }
        }
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "snapshot and reticle intent resolution at the ECS/UI boundary"
)]
pub(crate) fn push(
    ui: Option<bevy::prelude::NonSend<super::AppUi>>,
    mut config: bevy::prelude::ResMut<crate::button_config::ButtonConfiguration>,
    mut graph: bevy::prelude::ResMut<crate::editor::state::EditorGraph>,
    mut editor: bevy::prelude::ResMut<crate::editor::state::EditorState>,
    mut history: bevy::prelude::ResMut<crate::editor::history::EditorHistory>,
    mut actions: bevy::prelude::ResMut<bevy::prelude::ButtonInput<crate::controls::GameAction>>,
    selection: bevy::prelude::Res<crate::hotbar::SelectedTool>,
    mut simulation: bevy::prelude::ResMut<crate::simulation::state::AppSimulation>,
    camera: bevy::prelude::Single<
        (&bevy::prelude::Camera, &bevy::prelude::GlobalTransform),
        bevy::prelude::With<crate::camera::MainCamera>,
    >,
) {
    let Some(ui) = ui else { return };
    let centre = camera
        .0
        .logical_viewport_rect()
        .map_or(bevy::math::Vec2::ZERO, |r| r.center());
    config.aim = if config.selected.is_some() {
        ui.handles
            .button_layout
            .borrow()
            .iter()
            .find(|(_, rect)| rect.contains(Vector2::new(centre.x, centre.y)))
            .map(|(control, _)| *control)
    } else {
        None
    };
    if selection.active_editor_tool() != Some(crate::hotbar::Tool::Connector) {
        config.selected = None;
        config.capturing = false;
        config.aim = None;
    } else if config.aim.is_none() && !config.capturing {
        let hit = editor.world_hovered_part.or_else(|| {
            editor.hovered.and_then(|hit| {
                if let mechanic_core::FaceOwner::Part(part) = hit.face.owner {
                    Some(part)
                } else {
                    None
                }
            })
        });
        if let Some(part) = hit.filter(|part| {
            matches!(
                graph.0.part(*part),
                Some(mechanic_core::PartSpec::Button(_))
            )
        }) {
            config.selected = Some(part);
        }
    }
    let before = (config.aim.is_some()
        && actions.just_pressed(crate::controls::GameAction::Primary))
    .then(|| graph.0.clone());
    let revision = history.current_revision;
    crate::button_config::act(
        &mut config,
        &mut graph,
        &mut editor,
        &mut history,
        &mut actions,
    );
    if let Some(before) = before
        && revision != history.current_revision
        && simulation.is_running()
    {
        simulation.reconcile_controller_values(&before, &graph.0);
        simulation.accept_controller_edit(revision, history.current_revision, &graph.0);
    }
    let model = config
        .selected
        .and_then(|part| {
            let setting = graph.0.input_configuration(part)?;
            let (position, _) = simulation.live_part_pose(&graph.0, part)?;
            let at = camera.0.world_to_viewport(camera.1, position).ok()?
                + bevy::math::Vec2::new(60.0, -60.0);
            Some(Model {
                at: Some((at.x, at.y)),
                key: if config.capturing {
                    "Press A–Z or 0–9 · Esc cancels".into()
                } else {
                    format!(
                        "Key: {}",
                        setting
                            .key
                            .map_or_else(|| "Unassigned".into(), |key| key.to_string())
                    )
                },
                mode: format!("Mode: {:?}", setting.button_mode),
                status: if setting.controller.is_some() {
                    "Connected to Controller".into()
                } else {
                    "Not connected".into()
                },
            })
        })
        .unwrap_or_default();
    if ui.handles.button_config.get_untracked() != model {
        ui.handles.button_config.set(model);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projected_button_controls_keep_separate_click_targets() {
        let overlay = crate::ui::testing::Overlay::mount();
        overlay.handles.button_config.set(Model {
            at: Some((200.0, 100.0)),
            key: "Key: W".into(),
            mode: "Mode: Momentary".into(),
            status: "Connected".into(),
        });
        overlay.settle();
        let layout = overlay.handles.button_layout.borrow();
        let rows = [Control::Key, Control::Clear, Control::Mode].map(|key| layout[&key]);
        for row in rows {
            assert!(row.size.width >= 220.0 && row.size.height >= 32.0);
        }
        for pair in rows.windows(2) {
            assert!(pair[0].origin.y + pair[0].size.height <= pair[1].origin.y);
        }
    }
}
