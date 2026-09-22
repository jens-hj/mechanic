//! Dial assignment snapshots and editor; configuration transactions live at the root.
#![allow(clippy::wildcard_imports, reason = "Mosaic authoring vocabulary")]
use super::theme::{accent, chip, control, port};
use super::{control_block::Handles, styles::*};
use crate::dial_assignment::Draft;
use bevy_mosaic::ui::*;
use mechanic_core::{DriveParameter, GearParameter, NumericParameter, PartId};
use mosaic_core::theme::color;
use mosaic_macros::view;

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Field {
    pub draft: Draft,
    pub badge: String,
}
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Dial {
    pub id: PartId,
    pub name: String,
    pub position: String,
    pub count: usize,
}
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Model {
    pub fields: Vec<Field>,
    pub dials: Vec<Dial>,
    pub draft: Option<Draft>,
}
#[derive(Clone, Debug)]
pub(crate) enum Intent {
    Open(NumericParameter),
    Select(PartId),
    Reverse,
    Apply(String, String),
    Cancel,
    Unlink,
    Rename(PartId, String),
    Locate(PartId),
}
pub(crate) type Queue = std::rc::Rc<std::cell::RefCell<Vec<Intent>>>;

pub(crate) fn badge(
    handles: &Handles,
    target: impl Fn() -> Option<NumericParameter> + Clone + 'static,
) -> Element {
    let model = handles.dials;
    let queue = handles.dial_intents.clone();
    let clicked = target.clone();
    view! {
        row #mechanic.action width:max-content max-width:96px height:20px shrink:0 align:center justify:center pad:(horizontal:5px vertical:0px) clip role:button label:"Assign dial" @pointer-down.stop:{} @click.stop:{ if let Some(target) = clicked() { queue.borrow_mut().push(Intent::Open(target)); } } {
            text #mechanic.caption width:max-content height:14px shrink:0 font-size:10px text-wrap:none { model.with(|m| target().and_then(|target| m.fields.iter().find(|f| f.draft.targets.contains(&target))).map_or_else(|| "◉".into(), |field| field.badge.clone())) }
            tooltip "Assign dial"
        }
    }
}

pub(crate) fn panel(handles: &Handles) -> Element {
    let model = handles.dials;
    let opened = State::new(false);
    let overview_handles = handles.clone();
    let editor_handles = handles.clone();
    view! {
        col width:fill height:min-content shrink:0 align:start gap:6px pad:10px {
            row #mechanic.action width:100px height:28px shrink:0 @click:{ opened.set(!opened.get_untracked()); } { text "Dials" }
            if opened.get() { (overview(&overview_handles)) }
            if model.with(|m| m.draft.is_some()) { (assignment_editor(&editor_handles)) }
        }
    }
}
fn overview(handles: &Handles) -> Element {
    let model = handles.dials;
    let summaries = handles.clone();
    view! {
        col width:fill height:min-content shrink:0 align:start gap:4px {
            for (id, ()) in { model.with(|m| m.dials.iter().map(|dial| (dial.id, ())).collect::<Vec<_>>()) } {
                (dial_summary(&summaries, *id))
            }
        }
    }
}
fn field_row(handles: &Handles, target: NumericParameter) -> Element {
    let model = handles.dials;
    let binding = badge(handles, move || Some(target));
    view! { row width:fill height:28px shrink:0 align:center gap:6px {
        text { model.with(|m| m.fields.iter().find(|field| field.draft.targets.contains(&target)).map_or_else(String::new, |field| format!("{}: {} {}", field.draft.name, field.draft.current, field.draft.unit))) }
        (binding)
    } }
}
fn dial_summary(handles: &Handles, id: PartId) -> Element {
    let model = handles.dials;
    let queue = handles.dial_intents.clone();
    let locate = queue.clone();
    let assignments = handles.clone();
    let expanded = State::new(false);
    let name = State::new(String::new());
    let previous = State::new(None::<String>);
    // Live feedback updates the model every frame. Only an authored name change
    // may replace the text being edited (including undo and redo).
    Effect::new(move || {
        let wanted = model.with(|m| {
            m.dials
                .iter()
                .find(|dial| dial.id == id)
                .map(|dial| dial.name.clone())
        });
        if previous.get_untracked() != wanted {
            name.set(wanted.clone().unwrap_or_default());
            previous.set(wanted);
        }
    });
    view! {
        col width:fill height:min-content shrink:0 align:start gap:4px {
            row width:fill height:min-content shrink:0 align:center gap:6px {
                row #mechanic.action width:24px height:28px shrink:0 @click.stop:{ expanded.set(!expanded.get_untracked()); } {
                    text { if expanded.get() { "▾" } else { "▸" } }
                }
                input #mechanic.field width:180px height:28px name
                row #mechanic.action width:90px height:28px shrink:0 @click.stop:{ queue.borrow_mut().push(Intent::Rename(id, name.get_untracked())); } { text "Save name" }
                row #mechanic.action width:60px height:28px shrink:0 @click.stop:{ locate.borrow_mut().push(Intent::Locate(id)); } { text "Locate" }
                text #mechanic.caption { model.with(|m| m.dials.iter().find(|dial| dial.id == id).map_or_else(String::new, |dial| format!("{} assignments · {}", dial.count, dial.position))) }
            }
            if expanded.get() {
                (dial_fields(&assignments, id))
            }
        }
    }
}
fn dial_fields(handles: &Handles, id: PartId) -> Element {
    let model = handles.dials;
    let handles = handles.clone();
    view! {
        col width:fill height:min-content shrink:0 align:start gap:4px pad:6px {
            for (target, ()) in { model.with(|m| m.fields.iter().filter(|field| field.draft.dial == Some(id)).filter_map(|field| field.draft.targets.first().map(|target| (*target, ()))).collect::<Vec<_>>()) } {
                (field_row(&handles, *target))
            }
            if model.with(|m| m.dials.iter().find(|dial| dial.id == id).is_some_and(|dial| dial.count == 0)) { text #mechanic.caption "No assignments" }
        }
    }
}
fn dial_choice(model: State<Model>, queue: Queue, id: PartId) -> Element {
    let selected = move || model.with(|m| m.draft.as_ref().is_some_and(|d| d.dial == Some(id)));
    view! { row #mechanic.action width:fill height:28px shrink:0 justify:start pad:(horizontal:8px vertical:0px)
        fill:{ if selected() { color(port.fill) } else { color(control.rest) } }
        stroke:(width:1px color:{ if selected() { color(accent.key) } else { color(chip.edge) } })
        role:button @click:{ queue.borrow_mut().push(Intent::Select(id)); } {
        text { model.with(|m| m.dials.iter().find(|dial| dial.id == id).map_or_else(String::new, |dial| format!("{} {}{}", if selected() { "●" } else { "○" }, dial.name, if selected() { " · Selected" } else { "" }))) }
    } }
}

fn selected_dial(model: &Model) -> Option<&Dial> {
    let id = model.draft.as_ref()?.dial?;
    model.dials.iter().find(|dial| dial.id == id)
}

fn assignment_editor(handles: &Handles) -> Element {
    let model = handles.dials;
    let queue = handles.dial_intents.clone();
    let reverse = queue.clone();
    let apply = queue.clone();
    let cancel = queue.clone();
    let unlink = queue.clone();
    let minimum = handles.dial_minimum;
    let maximum = handles.dial_maximum;
    let ready = move || model.with(|m| selected_dial(m).is_some());
    view! {
        col #mechanic.panel width:fill max-width:640px height:min-content shrink:0 align:start justify:start pad:10px gap:6px {
            text #mechanic.value width:fill height:min-content { model.with(|m| m.draft.as_ref().map_or_else(String::new, |d| format!("{} · {} {}", d.name, d.current, d.unit))) }
            text #mechanic.caption width:fill height:min-content "Dial to assign"
            if model.with(|m| m.dials.is_empty()) { text "Connect a dial to this controller with the Connector." }
            if model.with(|m| !m.dials.is_empty() && selected_dial(m).is_none()) { text width:fill height:min-content "Choose a dial below to enable Apply." }
            for (id, ()) in { model.with(|m| m.dials.iter().map(|dial| (dial.id, ())).collect::<Vec<_>>()) } {
                (dial_choice(model, queue.clone(), *id))
            }
            row width:fill height:min-content shrink:0 align:center gap:6px {
                col width:1fr height:min-content gap:4px { text height:min-content "Minimum" input #mechanic.field width:fill height:28px minimum }
                col width:1fr height:min-content gap:4px { text height:min-content "Maximum" input #mechanic.field width:fill height:28px maximum }
            }
            row #mechanic.action width:max-content height:28px shrink:0 pad:(horizontal:8px vertical:0px) @click:{ reverse.borrow_mut().push(Intent::Reverse); } { text width:max-content height:20px text-wrap:none { model.with(|m| if m.draft.as_ref().is_some_and(|d| d.reverse) { "☑ Reverse direction" } else { "☐ Reverse direction" }) } }
            text width:fill height:min-content { model.with(|m| m.draft.as_ref().map_or_else(String::new, |d| { let low = minimum.get().parse::<f32>().ok().filter(|value| value.is_finite()).unwrap_or(d.minimum); let high = maximum.get().parse::<f32>().ok().filter(|value| value.is_finite()).unwrap_or(d.maximum); let ends = if d.reverse { (high, low) } else { (low, high) }; format!("Left: {} · Right: {} {}", ends.0, ends.1, d.unit) })) }
            text width:fill height:min-content { model.with(|m| m.draft.as_ref().and_then(|d| d.error.clone()).unwrap_or_default()) }
            row width:fill height:28px shrink:0 align:center gap:8px {
                row #mechanic.action width:76px height:28px shrink:0 opacity:{ if ready() { 1.0 } else { 0.38 } } role:button label:"Apply dial assignment" @click:{ if ready() { apply.borrow_mut().push(Intent::Apply(minimum.get_untracked(), maximum.get_untracked())); } } { text "Apply" }
                row #mechanic.action width:76px height:28px shrink:0 @click:{ cancel.borrow_mut().push(Intent::Cancel); } { text "Cancel" }
                if model.with(|m| m.draft.as_ref().is_some_and(|draft| m.fields.iter().any(|field| field.draft.targets == draft.targets && field.draft.dial.is_some()))) { (unlink_button(unlink.clone())) }
            }
        }
    }
}

fn unlink_button(queue: Queue) -> Element {
    view! { row #mechanic.action width:76px height:28px shrink:0 @click:{ queue.borrow_mut().push(Intent::Unlink); } { text "Unlink" } }
}

#[expect(
    clippy::too_many_lines,
    reason = "exhaustive snapshot of enabled numeric controller fields and display units"
)]
fn capture(
    graph: &mechanic_core::ConstructionGraph,
    panel: &crate::control_panel::ControlPanelState,
    draft: Option<Draft>,
) -> Model {
    let Some(controller) = panel.controller() else {
        return Model::default();
    };
    let mut fields = Vec::new();
    let mut add = |targets: Vec<NumericParameter>, name: String, factor: f32, unit: &str| {
        let Some(&target) = targets.first() else {
            return;
        };
        let Some(current) = graph.numeric_value(target) else {
            return;
        };
        let Some(metadata) = graph.numeric_metadata(target) else {
            return;
        };
        let minimum = targets
            .iter()
            .filter_map(|target| graph.numeric_metadata(*target))
            .fold(metadata.minimum, |bound, m| bound.max(m.minimum));
        let maximum = targets
            .iter()
            .filter_map(|target| graph.numeric_metadata(*target))
            .fold(metadata.maximum, |bound, m| bound.min(m.maximum));
        let bound = graph.physical_inputs().find_map(|(id, config)| {
            config
                .analog
                .iter()
                .find(|mapping| mapping.target == target)
                .map(|mapping| (id, config.name.clone(), mapping.range))
        });
        fields.push(Field {
            badge: bound
                .as_ref()
                .map_or_else(|| "◉".into(), |(_, name, _)| name.clone()),
            draft: Draft {
                targets,
                name,
                current: current * factor,
                dial: bound.as_ref().map(|(id, _, _)| *id),
                minimum: bound
                    .as_ref()
                    .map_or(minimum, |(_, _, range)| range.endpoints()[0])
                    * factor,
                maximum: bound
                    .as_ref()
                    .map_or(maximum, |(_, _, range)| range.endpoints()[1])
                    * factor,
                reverse: bound.as_ref().is_some_and(|(_, _, range)| range.inverted()),
                factor,
                unit: unit.into(),
                integer: metadata.integer_step.is_some(),
                error: None,
            },
        });
    };
    for row in crate::control_panel::panel_rows(graph, controller) {
        let Some(spec) = graph.drive_link(row.primary) else {
            continue;
        };
        let mut drive = |parameter, name: String, factor, unit| {
            add(
                row.links
                    .iter()
                    .map(|&link| NumericParameter::Drive { link, parameter })
                    .collect(),
                name,
                factor,
                unit,
            );
        };
        let linear = spec.linear_limits.is_some();
        let angular_factor = match panel.speed_unit() {
            crate::control_panel::SpeedUnit::Rpm => mechanic_core::rad_s_to_rpm(1.0),
            crate::control_panel::SpeedUnit::DegreesPerSecond => 1.0f32.to_degrees(),
        };
        for (index, state) in spec.program.states().iter().enumerate() {
            let Ok(index) = u8::try_from(index) else {
                continue;
            };
            let (parameter, factor, unit) = match state.target() {
                mechanic_core::DriveTarget::Angle(_) => (
                    DriveParameter::AngularPosition(index),
                    1.0f32.to_degrees(),
                    "°",
                ),
                mechanic_core::DriveTarget::Speed(_) => (
                    DriveParameter::AngularSpeed(index),
                    angular_factor,
                    if panel.speed_unit() == crate::control_panel::SpeedUnit::Rpm {
                        "RPM"
                    } else {
                        "°/s"
                    },
                ),
                mechanic_core::DriveTarget::LinearPosition(_) => {
                    (DriveParameter::LinearPosition(index), 1.0, "m")
                }
                mechanic_core::DriveTarget::LinearSpeed(_) => {
                    (DriveParameter::LinearSpeed(index), 1.0, "m/s")
                }
            };
            drive(
                parameter,
                format!("State {} target", index + 1),
                factor,
                unit,
            );
            drive(
                DriveParameter::Dwell(index),
                format!("State {} dwell", index + 1),
                1.0,
                "s",
            );
        }
        for (parameter, name) in [
            (DriveParameter::TravelMinimum, "Minimum travel"),
            (DriveParameter::TravelMaximum, "Maximum travel"),
        ] {
            drive(
                parameter,
                name.into(),
                if linear { 1.0 } else { 1.0f32.to_degrees() },
                if linear { "m" } else { "°" },
            );
        }
        drive(
            DriveParameter::ElectricContribution,
            "Electric contribution".into(),
            1.0,
            "%",
        );
        drive(
            DriveParameter::GasContribution,
            "Gas contribution".into(),
            1.0,
            "%",
        );
    }
    for kind in [
        mechanic_core::EngineKind::Electric,
        mechanic_core::EngineKind::Gas,
    ] {
        if let Ok(config) = graph.gearbox_config(controller, kind) {
            for index in 0..config.ratios().len() {
                let Ok(index) = u8::try_from(index) else {
                    continue;
                };
                add(
                    vec![NumericParameter::Gear {
                        controller,
                        kind,
                        parameter: GearParameter::Ratio(index),
                    }],
                    format!("Gear {} ratio", index + 1),
                    1.0,
                    ":1",
                );
            }
            add(
                vec![NumericParameter::Gear {
                    controller,
                    kind,
                    parameter: GearParameter::ReverseCount,
                }],
                "Reverse gears".into(),
                1.0,
                "",
            );
        }
    }
    let dials = graph
        .physical_inputs()
        .filter(|(id, config)| {
            config.controller == Some(controller)
                && matches!(graph.part(*id), Some(mechanic_core::PartSpec::Dial(_)))
        })
        .map(|(id, config)| {
            let position = match mechanic_core::DialFeedback::from_values(
                config.analog.iter().filter_map(|mapping| {
                    graph
                        .numeric_value(mapping.target)
                        .map(|value| (mapping.range, value))
                }),
            ) {
                mechanic_core::DialFeedback::Uniform(value) => format!("{:.0}%", value * 100.0),
                mechanic_core::DialFeedback::Mixed => "Mixed".into(),
            };
            Dial {
                id,
                name: config.name.clone(),
                position,
                count: config.analog.len(),
            }
        })
        .collect();
    Model {
        fields,
        dials,
        draft,
    }
}

fn draft_for(model: &Model, target: NumericParameter) -> Option<Draft> {
    let mut draft = model
        .fields
        .iter()
        .find(|field| field.draft.targets.contains(&target))?
        .draft
        .clone();
    if draft.dial.is_none()
        && let [dial] = model.dials.as_slice()
    {
        draft.dial = Some(dial.id);
    }
    Some(draft)
}

fn sync_opened_draft(handles: &Handles, draft: Option<&Draft>) {
    if let Some(draft) = draft
        && handles
            .dials
            .get_untracked()
            .draft
            .as_ref()
            .is_none_or(|previous| previous.targets != draft.targets)
    {
        handles.dial_minimum.set(draft.minimum.to_string());
        handles.dial_maximum.set(draft.maximum.to_string());
    }
}

pub(crate) fn push(
    ui: Option<bevy::prelude::NonSend<super::AppUi>>,
    panel: bevy::prelude::Res<crate::control_panel::ControlPanelState>,
    mut assignments: bevy::prelude::ResMut<crate::dial_assignment::DialAssignments>,
    mut simulation: bevy::prelude::ResMut<crate::simulation::state::AppSimulation>,
    mut graph: bevy::prelude::ResMut<crate::editor::state::EditorGraph>,
    mut editor: bevy::prelude::ResMut<crate::editor::state::EditorState>,
    mut history: bevy::prelude::ResMut<crate::editor::history::EditorHistory>,
) {
    let Some(ui) = ui else { return };
    sync_opened_draft(&ui.handles.block, assignments.draft.as_ref());
    let revision = history.current_revision;
    let intents = ui
        .handles
        .block
        .dial_intents
        .borrow_mut()
        .drain(..)
        .collect::<Vec<_>>();
    let before = (!intents.is_empty()).then(|| graph.0.clone());
    let effective = if simulation.is_running() {
        simulation.effective_graph()
    } else {
        &graph.0
    };
    let mut model = capture(effective, &panel, assignments.draft.clone());
    for intent in intents {
        match intent {
            Intent::Open(target) => {
                assignments.draft = draft_for(&model, target);
                if let Some(draft) = &assignments.draft {
                    ui.handles.block.dial_minimum.set(draft.minimum.to_string());
                    ui.handles.block.dial_maximum.set(draft.maximum.to_string());
                }
            }
            Intent::Cancel => assignments.draft = None,
            Intent::Locate(dial) => assignments.located = Some(dial),
            Intent::Rename(dial, name) => {
                crate::dial_assignment::rename(dial, &name, &mut graph, &mut editor, &mut history);
            }
            Intent::Apply(minimum, maximum) => {
                if let Some(draft) = &mut assignments.draft {
                    draft.error = None;
                    edit_endpoint(draft, &minimum, true);
                    edit_endpoint(draft, &maximum, false);
                    if draft.error.is_none() {
                        crate::dial_assignment::apply(
                            &mut assignments,
                            &mut graph,
                            &mut editor,
                            &mut history,
                            false,
                        );
                    }
                }
            }
            Intent::Unlink => crate::dial_assignment::apply(
                &mut assignments,
                &mut graph,
                &mut editor,
                &mut history,
                true,
            ),
            _ => {
                if let Some(draft) = &mut assignments.draft {
                    match intent {
                        Intent::Select(dial) => {
                            draft.dial = Some(dial);
                            draft.error = None;
                        }
                        Intent::Reverse => draft.reverse = !draft.reverse,
                        _ => {}
                    }
                }
            }
        }
    }
    if let Some(before) = before
        && graph.0.physical_inputs().ne(before.physical_inputs())
        && simulation.is_running()
    {
        simulation.reconcile_controller_values(&before, &graph.0);
        simulation.accept_controller_edit(revision, history.current_revision, &graph.0);
    }
    if !panel.is_open() {
        assignments.draft = None;
    }
    if !panel.is_open()
        || assignments.located.is_some_and(|dial| {
            !matches!(graph.0.part(dial), Some(mechanic_core::PartSpec::Dial(_)))
                || graph
                    .0
                    .input_configuration(dial)
                    .is_none_or(|configuration| {
                        configuration.controller.is_none()
                            || configuration.controller != panel.controller()
                    })
        })
    {
        assignments.located = None;
    }
    model.draft.clone_from(&assignments.draft);
    if ui.handles.block.dials.get_untracked() != model {
        ui.handles.block.dials.set(model);
    }
}

fn edit_endpoint(draft: &mut Draft, text: &str, minimum: bool) {
    match text.parse::<f32>() {
        Ok(value) if value.is_finite() && (!draft.integer || value.fract() == 0.0) => {
            if minimum {
                draft.minimum = value;
            } else {
                draft.maximum = value;
            }
        }
        _ => draft.error = Some("Enter a valid endpoint".into()),
    }
}

#[cfg(test)]
mod tests;
