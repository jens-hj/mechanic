//! The help and status panel: what mode this is, what the keys do, and what
//! just happened.
//!
//! Eight lines, each one a sentence about the editor's current state. The work
//! is in deciding what they say — the view is a column of text — so this module
//! is mostly [`capture`], reading the whole editor once a frame and writing
//! down what it found.

#![allow(
    clippy::wildcard_imports,
    reason = "Mosaic's authoring vocabulary is meant to be globbed"
)]

use bevy::ecs::system::{Res, SystemParam};
use bevy_mosaic::ui::*;
use mosaic_core::theme::color;
use mosaic_core::theme::{LengthToken, typed};
use mosaic_macros::{component, view};

use super::Handles;
use super::components::{PanelSurface, PanelSurfaceProps};
#[expect(
    unused_imports,
    reason = "style constants are consumed by `view!` expansion"
)]
use super::styles::*;
use super::theme::*;
use crate::camera::PlayerState;
use crate::controls::GameAction;
use crate::editor::build_actions::visible_bearing_count;
use crate::editor::dimensions::{BearingToolSettings, CylinderToolSettings};
use crate::editor::hammer::{HAMMER_CHARGE_SECONDS, HammerInteraction};
use crate::editor::hover::BlockAttachment;
use crate::editor::state::{EditorGraph, EditorState};
use crate::editor::wiring::WireEnd;
use crate::hotbar::{MainTool, MatterMode, SelectedMaterial, SelectedTool, Tool};
use crate::simulation::state::AppSimulation;
use crate::world::AppSpace;

/// How loudly one line speaks, and about what.
///
/// Named for meaning rather than for a colour: the palette decides what amber
/// looks like, and this decides what deserves it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Tone {
    /// The panel's own title.
    Title,
    /// Ordinary prose.
    #[default]
    Body,
    /// Secondary prose, there when wanted and quiet when not.
    Muted,
    /// Everything is fine.
    Good,
    /// Something is in progress, or about to need attention.
    Warn,
    /// Something went wrong.
    Bad,
    /// A tool that works in angles and positions.
    Angle,
    /// A tool that works in speeds.
    Speed,
    /// A tool that binds things together.
    Key,
}

impl Tone {
    /// The palette entry this tone reads in.
    fn paint(self) -> Color {
        match self {
            Tone::Title => color(help.title),
            Tone::Body => color(help.body),
            Tone::Muted => color(help.muted),
            Tone::Good => color(help.good),
            Tone::Warn => color(help.warn),
            Tone::Bad => color(help.bad),
            Tone::Angle => color(accent.angle),
            Tone::Speed => color(accent.speed),
            Tone::Key => color(accent.key),
        }
    }
}

/// One line of the panel.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Line {
    /// What it says.
    text: String,
    /// How it says it.
    tone: Tone,
}

impl Line {
    /// A line in a given tone.
    fn new(text: impl Into<String>, tone: Tone) -> Self {
        Line {
            text: text.into(),
            tone,
        }
    }
}

/// Everything the panel says, decided once a frame.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Model {
    /// `MECHANIC • BUILDING`, and what mode that is.
    title: Line,
    /// The keys that always work.
    primary: Line,
    /// The keys that edit, or why they are unavailable.
    edit: Line,
    /// What the mouse does.
    pointer: Line,
    /// The tool in hand and how it is set up.
    tool: Line,
    /// How much has been built.
    counts: Line,
    /// What to do next.
    hint: Line,
    /// What just happened.
    status: Line,
}

/// Everything [`capture`] reads.
#[derive(SystemParam)]
pub(crate) struct Sources<'w> {
    graph: Res<'w, EditorGraph>,
    state: Res<'w, EditorState>,
    simulation: Res<'w, AppSimulation>,
    hammer: Res<'w, HammerInteraction>,
    selection: Res<'w, SelectedTool>,
    material: Res<'w, SelectedMaterial>,
    bearing: Res<'w, BearingToolSettings>,
    cylinder: Res<'w, CylinderToolSettings>,
    player: Res<'w, PlayerState>,
    settings: Res<'w, crate::settings::AppSettings>,
    app_space: Res<'w, bevy::prelude::State<AppSpace>>,
}

fn linear_controls(controls: &crate::controls::Controls, rotate: &str) -> String {
    format!(
        "{rotate}  Four travel directions · {} / {}  Length 250 mm · {} / {}  Width 25 mm · {} / {}  Fine length 2.5 mm · {} / {}  Fine width 2.5 mm",
        controls.label(GameAction::LinearLengthDecrease),
        controls.label(GameAction::LinearLengthIncrease),
        controls.label(GameAction::LinearWidthDecrease),
        controls.label(GameAction::LinearWidthIncrease),
        controls.label(GameAction::LinearLengthFineDecrease),
        controls.label(GameAction::LinearLengthFineIncrease),
        controls.label(GameAction::LinearWidthFineDecrease),
        controls.label(GameAction::LinearWidthFineIncrease),
    )
}

/// Reads the editor into what the panel says.
#[expect(
    clippy::too_many_lines,
    reason = "one line of prose per editor state, in the order they are shown"
)]
pub(crate) fn capture(sources: &Sources) -> Model {
    let Sources {
        graph,
        state,
        simulation,
        hammer,
        selection,
        material,
        bearing,
        cylinder,
        player,
        settings,
        app_space,
    } = sources;
    let selected_tool = selection.active_editor_tool().unwrap_or(Tool::Block);
    let linear_attachment = state
        .attachment_bearing
        .and_then(|index| state.placed_bearings.get(index))
        .is_some_and(|socket| matches!(socket.kind, mechanic_core::JointKind::Linear(_)));
    let terrain_mode = selection.tool == Some(MainTool::MatterManipulator)
        && selection.matter_mode == MatterMode::Terrain;
    let item_mode = selection.tool == Some(MainTool::MatterManipulator)
        && selection.matter_mode == MatterMode::Item;
    let controls = settings.controls();
    let rotate = controls.label(GameAction::Rotate);
    let selector_controls = match (selection.tool, selection.matter_mode) {
        (
            Some(MainTool::MatterManipulator),
            MatterMode::Block | MatterMode::Cylinder | MatterMode::Layer,
        ) => {
            format!(
                "     {}  Choose material",
                controls.label(GameAction::MaterialWheel)
            )
        }
        (Some(MainTool::MatterManipulator), MatterMode::Item) => {
            format!(
                "     {}  Choose item",
                controls.label(GameAction::MaterialWheel)
            )
        }
        (Some(MainTool::MatterManipulator), MatterMode::Terrain) => {
            format!(
                "     {}  Choose terrain",
                controls.label(GameAction::MaterialWheel)
            )
        }
        (Some(MainTool::MatterManipulator), MatterMode::Chroma) => {
            format!(
                "     {}  Toggle appearance workbench",
                controls.label(GameAction::MaterialWheel)
            )
        }
        _ => String::new(),
    };

    let active_error = state
        .delete_drag
        .as_ref()
        .and_then(|drag| drag.error.as_ref())
        .or(state.preview_error.as_ref());
    // A warning is not a refusal: it only speaks when nothing is refused.
    let active_warning = active_error
        .is_none()
        .then_some(state.preview_warning.as_ref())
        .flatten();
    let status = if let Some(charge) = hammer.charging {
        format!(
            "Hammer charge: {:.0}% — release to strike",
            charge.elapsed_seconds / HAMMER_CHARGE_SECONDS * 100.0
        )
    } else if let Some(warning) = active_warning {
        warning.clone()
    } else {
        active_error.map_or_else(
            || state.feedback.clone().unwrap_or_else(|| "Ready".to_owned()),
            ToString::to_string,
        )
    };
    let status_tone = if hammer.charging.is_some() || active_warning.is_some() {
        Tone::Warn
    } else if active_error.is_some()
        || ["Cannot", "Could not", "Simulation stopped"]
            .iter()
            .any(|prefix| status.starts_with(prefix))
    {
        Tone::Bad
    } else {
        Tone::Good
    };
    let in_world = *app_space.get() == AppSpace::World;
    let live_hammer = in_world && selected_tool == Tool::Hammer;
    let tool_hint = if terrain_mode {
        if in_world {
            "Hold left to add terrain; hold right to remove; wheel changes brush radius"
                .to_owned()
        } else {
            "Terrain editing is available in the World; press F6 to switch spaces".to_owned()
        }
    } else if let Some(drag) = state.delete_drag.as_ref() {
        format!(
            "Release to delete {} cuboid(s) — Q rotates the {} plane and keeps the extent",
            drag.parts.len(),
            drag.plane.label()
        )
    } else if let Some(drag) = state.pipe_drag.as_ref() {
        if drag.bearing_offset.is_some() {
            "Bearing and pipe centred — drag to offset, release to place; R edits length"
                .to_owned()
        } else {
            "Release to place pipe run — R cycles dimensions, F adds bends".to_owned()
        }
    } else if selected_tool == Tool::Weld {
        let selector = controls.label(GameAction::MaterialWheel);
        match (selection.weld_mode, state.weld.busy()) {
            (crate::hotbar::WeldMode::Join, false) => format!("Join: click a body to weld where it touches another · Tap {selector} for Place"),
            (crate::hotbar::WeldMode::Join, true) => "Click a touching body; faces must share a 5 × 5 cm square".to_owned(),
            (crate::hotbar::WeldMode::Place, true) => "Press a destination feature, drag, and release to weld; R rotates 15°, Shift snaps 5 cm".to_owned(),
            (crate::hotbar::WeldMode::Place, false) => format!("Place: select a source face, straight edge, or corner · Tap {selector} for Join"),
        }
    } else {
        match (
            live_hammer,
            selected_tool,
            graph.0.pending(),
            state.block_drag.as_ref(),
            state.attachment_bearing,
        ) {
            (false, Tool::Block, _, Some(drag), _) => {
                if matches!(drag.attachment, BlockAttachment::Linear { .. }) {
                    format!("Release to attach {} block(s) to the carriage on the 25 mm lattice", drag.volume.count())
                } else if matches!(drag.attachment, BlockAttachment::Bearing { .. }) {
                    format!(
                        "Green bearing attachment active — release to connect {} block(s)",
                        drag.volume.count()
                    )
                } else {
                    format!(
                        "Release to place {} blocks on the {} plane",
                        drag.volume.count(),
                        drag.plane.label()
                    )
                }
            }
            (false, Tool::Block, _, None, Some(_)) => {
                if linear_attachment {
                    "Carriage face selected — click or drag to attach blocks on the 25 mm lattice".to_owned()
                } else { "Green bearing attachment active — click or drag to connect blocks".to_owned() }
            }
            (false, Tool::Layer, _, _, _) => {
                "Press a face, wall, or bore and drag out to thicken or back to thin — Shift 5 cm, Shift+Ctrl 1 cm".to_owned()
            }
            (false, Tool::Cylinder, _, _, Some(_)) => {
                if linear_attachment {
                    "Carriage face selected — attach the cylinder on the 25 mm lattice; further attachments use this face".to_owned()
                } else {
                    "Green bearing attachment active — click centres the pipe; hold and drag to offset it".to_owned()
                }
            }
            (true, Tool::Hammer, _, _, _) => {
                "Hold left mouse on a moving cuboid; release to strike. Freeze controls require an active dimension link".to_owned()
            }
            (true, _, _, _, _) => unreachable!("only the Hammer is a live-only tool"),
            (false, Tool::Block, _, _, _) => {
                "Click for one block or drag to place a welded sheet".to_owned()
            }
            (false, Tool::Cylinder, _, _, _) => {
                "Hold on a pipe end to extend it or on a pipe side to branch (R turns the branch); F bends, wheel sizes bends"
                    .to_owned()
            }
            (false, Tool::Weld, None, _, _) => "Left click selects the first object".to_owned(),
            (false, Tool::Weld, Some(_), _, _) => {
                "Left click a second object the highlighted one touches".to_owned()
            }
            (false, Tool::Bearing, _, _, _) => {
                "Left click places a bearing; use Blocker Placer to attach it".to_owned()
            }
            (false, Tool::Spring | Tool::Shock, _, _, _) => "Hold left-click and move to size; R cycles dimensions; release to place. Equip Connector to adjust placed suspension.".to_owned(),
            (false, Tool::LinearBearing, _, _, _) => {
                "Place the rail underside on a flat face; attach blocks or cylinders to one carriage face: top or either side. End faces are unavailable; both rail ends remain physical stops.".to_owned()
            }
            (false, Tool::Piston, _, _, _) => {
                "R picks end mount or one of four side mounts. Hold left-click and move to set closed length; R switches to stages; release to place. Attach blocks or cylinders to the head.".to_owned()
            }
            (false, Tool::Hammer, _, _, _) => {
                "The Hammer is available in the live World".to_owned()
            }
            (false, Tool::Controller, _, _, _) => {
                "Q cycles all 24 orientations; left click places a control block; click one to retune it"
                    .to_owned()
            }
            (false, Tool::GasEngine, _, _, _) => {
                "Q rotates; place a 6000 N·m, 360 RPM gas engine (4 bearing ports)".to_owned()
            }
            (false, Tool::ElectricEngine, _, _, _) => {
                "Q rotates; place a 500 N·m, 120 RPM electric engine (4 bearing ports)"
                    .to_owned()
            }
            (false, Tool::Transmission, _, _, _) => {
                "Attach to an engine or transmission +Z output; orientation is inherited"
                    .to_owned()
            }
            (false, Tool::Servo, _, _, _) => {
                "Q rotates; place a 12000 N·m, 30 RPM Servo (one angle-controlled bearing)"
                    .to_owned()
            }
            (false, Tool::Seat, _, _, _) => {
                "Q cycles all 24 orientations; place a cushion, then wire it with Connector"
                    .to_owned()
            }
            (false, Tool::Input, _, _, _) => {
                "Q cycles all 24 orientations; place Input, then wire it to a Seat".to_owned()
            }
            (false, Tool::DimensionLink, _, _, _) => {
                "Q rotates; place a Dimension Link, aim at it, then press E to toggle it on or off"
                    .to_owned()
            }
            (false, Tool::Shape, _, _, _) => {
                "Drag an area (Q changes plane); Shift+left paints corners; left drag moves on one axis (Q changes axis); arrows nudge"
                    .to_owned()
            }
            (false, Tool::Chroma, _, _, _) => {
                format!(
                    "Left-drag paints; right-drag restores baked appearance; Q samples; press {} to configure",
                    controls.label(GameAction::MaterialWheel),
                )
            }
            (false, Tool::Connector, _, _, _) => match state.wire_drag.map(|drag| drag.from) {
                None => "Wire Controller↔Bearing, Input↔Seat, or Seat↔Controller".to_owned(),
                Some(WireEnd::Controller(_)) => {
                    "Release on a bearing to wire it — drop it on the same block to reverse"
                        .to_owned()
                }
                Some(WireEnd::Bearing(_)) => "Release on a control block to wire it".to_owned(),
                Some(WireEnd::Input(_)) => "Release on a Seat to link keyboard input".to_owned(),
                Some(WireEnd::Seat(_)) => {
                    "Release on an Input or Controller to complete the chain".to_owned()
                }
            },
        }
    }
    .replace('Q', &rotate);
    let tool_hint = if item_mode {
        format!(
            "{tool_hint} — hold {}, point at an item, and release to select another",
            controls.label(GameAction::MaterialWheel)
        )
    } else {
        tool_hint
    };
    let (phase, title_tone, primary_controls) = if in_world {
        (
            "WORLD  •  LIVE",
            Tone::Good,
            format!(
                "{}  Garage     SPACE  Jump     SHIFT  Sprint     {}  Crouch     {}  Hide help",
                controls.label(GameAction::ToggleSpace),
                controls.label(GameAction::Crouch),
                controls.label(GameAction::ToggleHelp)
            ),
        )
    } else {
        (
            "GARAGE  •  EDITOR",
            Tone::Title,
            format!(
                "{}  World     {}  Creations     {}  Save     {}  Hide help",
                controls.label(GameAction::ToggleSpace),
                controls.label(GameAction::Creations),
                controls.label(GameAction::Save),
                controls.label(GameAction::ToggleHelp)
            ),
        )
    };
    let action_controls = if state.delete_drag.is_some() {
        "RELEASE RIGHT  Delete     ESC  Cancel"
    } else {
        match (live_hammer, selected_tool, state.block_drag.is_some()) {
            (false, Tool::Block, true) => "RELEASE  Place     RIGHT / ESC  Cancel",
            (true, Tool::Hammer, _) => "HOLD LEFT  Charge hammer",
            (true, _, _) => unreachable!("only the Hammer is a live-only tool"),
            (false, _, _) => "LEFT  Action     RIGHT DRAG  Delete",
        }
    };
    let plane_controls = if let Some(drag) = state.block_drag.as_ref() {
        format!("{rotate}  Cycle plane ({})", drag.plane.label())
    } else if let Some(drag) = state.delete_drag.as_ref() {
        format!("{rotate}  Cycle delete plane ({})", drag.plane.label())
    } else if selected_tool == Tool::LinearBearing {
        format!("{rotate}  Rotate travel direction (4 directions)")
    } else if selected_tool == Tool::Piston {
        format!(
            "{rotate}  Cycle mount (end, 4 side directions); closed length / stages while holding"
        )
    } else {
        format!("{rotate}  Cycle plane while dragging or deleting")
    };
    let edit_controls = if live_hammer {
        format!(
            "{}  Freeze / release active link     {} / {}  Raise / lower 25 cm (hold to repeat)",
            controls.label(GameAction::FreezeCreation),
            controls.label(GameAction::RaiseFrozenCreation),
            controls.label(GameAction::LowerFrozenCreation),
        )
    } else if selected_tool == Tool::LinearBearing {
        let controls = linear_controls(controls, &rotate);
        if in_world {
            format!("{controls}     EDIT GROUNDED/STATIC ONLY")
        } else {
            controls
        }
    } else if in_world {
        format!("{plane_controls}     EDIT GROUNDED/STATIC ONLY     CTRL/CMD+Z  Undo")
    } else {
        format!(
            "{plane_controls}     SHIFT+WHEEL  Free range     TAP ALT  Object snap     HOLD ALT  Show range     ALT+WHEEL  Adjust snap range     CTRL/CMD+Z  Undo     SHIFT+CTRL/CMD+Z  Redo"
        )
    };
    let selected_wires = state
        .selected_controller
        .filter(|&part| graph.0.is_controller(part))
        .map(|part| graph.0.controller_links(part).count());

    Model {
        title: Line::new(format!("MECHANIC  •  {phase}"), title_tone),
        primary: Line::new(primary_controls, Tone::Body),
        edit: Line::new(edit_controls, Tone::Muted),
        pointer: Line::new(
            format!(
                "{action_controls}     WASD  Walk     MOUSE  Look     WHEEL  FP↔TP{selector_controls}{}",
                if in_world && simulation.is_running() {
                    if player.seat.is_some() {
                        "     E  Leave Seat"
                    } else {
                        "     E  Enter Seat"
                    }
                } else {
                    ""
                }
            ),
            Tone::Muted,
        ),
        tool: if terrain_mode {
            Line::new("Matter Manipulator · Terrain", Tone::Speed)
        } else {
            let tool_status = if selected_tool == Tool::Cylinder
                && material.0 == mechanic_core::ConstructionMaterial::Rubber
                && state.suspension.insertion.is_some()
            {
                let stop = state.suspension.stop;
                format!(
                    "Bump Stop · {:.1} mm long × {:.1} mm OD · hold to size · R cycles length/OD · release to place · bore follows shaft",
                    stop.length() * 1000.0,
                    stop.od() * 1000.0
                )
            } else if matches!(selected_tool, Tool::Spring | Tool::Shock) {
                state.suspension.preview.and_then(|s| if let mechanic_core::JointKind::Suspension(spec) = s.kind { Some(spec) } else { None }).map_or_else(|| "Suspension · choose a flat construction face or compatible shared mounts".into(), |spec| format!("Suspension · {:.1} mm extended · {:.1} mm travel · limited by {:?} · Connector to adjust", spec.extended_length()*1000.0, spec.compression_limit().0*1000.0, spec.compression_limit().1))
            } else if selected_tool == Tool::Piston {
                let dimensions = state.piston.dimensions;
                format!(
                    "Piston · {} blocks closed · {} stages · {:.2} m stroke · extension measured from collapsed",
                    dimensions.blocks(),
                    dimensions.stages(),
                    dimensions.stroke()
                )
            } else if selected_tool == Tool::LinearBearing {
                let dimensions = state.linear.dimensions;
                format!(
                    "Linear Bearing · {:.3} m × {:.3} m · {:.3} m usable travel · position measured from centre",
                    dimensions.length(),
                    dimensions.width(),
                    dimensions.travel()
                )
            } else {
                crate::editor::preview::tool_status_line(
                    selection.active_editor_tool(),
                    bearing.dimensions,
                    cylinder.dimensions,
                    selected_wires,
                    material.0,
                )
            };
            let placement_status = matches!(
                selection.active_editor_tool(),
                Some(
                    Tool::Block
                        | Tool::Cylinder
                        | Tool::Bearing
                        | Tool::LinearBearing
                        | Tool::Piston
                        | Tool::Spring
                        | Tool::Shock
                        | Tool::Controller
                        | Tool::GasEngine
                        | Tool::ElectricEngine
                        | Tool::Transmission
                        | Tool::Servo
                        | Tool::Seat
                        | Tool::Input
                        | Tool::DimensionLink
                )
            )
            .then(|| {
                let free_range = if !in_world
                    && selection
                        .active_editor_tool()
                        .is_some_and(crate::editor::placement::tool_supports_free_placement)
                {
                    format!("    Free range: {:.2} m", state.free_placement.range)
                } else {
                    String::new()
                };
                format!(
                    "    Grid: {}    Object snap: {} ({:.2} m){free_range}",
                    state.placement_grid.label(),
                    if state.smart_snap.enabled {
                        "On"
                    } else {
                        "Off"
                    },
                    state.smart_snap.range,
                )
            })
            .unwrap_or_default();
            Line::new(
                format!("{tool_status}{placement_status}"),
                tool_tone(selection.active_editor_tool()),
            )
        },
        counts: Line::new(
            format!(
                "{} parts  •  {} welds  •  {} bearings",
                graph.0.part_count(),
                graph.0.weld_count(),
                visible_bearing_count(&graph.0, &state.placed_bearings),
            ),
            Tone::Muted,
        ),
        hint: Line::new(tool_hint, Tone::Body),
        status: Line::new(format!("STATUS  •  {status}"), status_tone),
    }
}

/// What each tool reads in, following what the tool works on rather than what
/// it is called: positions are amber, speeds cyan, connections teal.
const fn tool_tone(tool: Option<Tool>) -> Tone {
    match tool {
        Some(
            Tool::Bearing
            | Tool::LinearBearing
            | Tool::Piston
            | Tool::Spring
            | Tool::Shock
            | Tool::Hammer
            | Tool::GasEngine
            | Tool::Servo,
        ) => Tone::Angle,
        Some(
            Tool::Weld | Tool::Controller | Tool::Connector | Tool::Input | Tool::DimensionLink,
        ) => Tone::Key,
        Some(
            Tool::Block
            | Tool::Cylinder
            | Tool::Layer
            | Tool::ElectricEngine
            | Tool::Transmission
            | Tool::Seat
            | Tool::Shape
            | Tool::Chroma,
        ) => Tone::Speed,
        None => Tone::Muted,
    }
}

/// The panel.
///
/// Hugging rather than filling: it sits in the top-left corner and the world
/// keeps the pointer everywhere it is not.
#[component]
pub(crate) fn HelpPanel(handles: Handles) -> Element {
    let model = handles.help;
    view! {
        PanelSurface elevated:false width:(panel_size.help_width) height:min-content gap:7px
            translate:(Translate::new(panel_size.inset, panel_size.inset))
            pad:(Edges::all(pad.panel)) {
            (line(model, text_size.title, text_tracking.help_title, true, |found| &found.title))
            (line(model, text_size.value, text_tracking.tight, false, |found| &found.primary))
            (line(model, text_size.body, text_tracking.tight, false, |found| &found.edit))
            (line(model, text_size.body, text_tracking.tight, false, |found| &found.pointer))
            (line(model, text_size.section, text_tracking.tight, false, |found| &found.tool))
            (line(model, text_size.label, text_tracking.tight, false, |found| &found.counts))
            (line(model, text_size.value, text_tracking.tight, false, |found| &found.hint))
            (line(model, text_size.label, text_tracking.tight, false, |found| &found.status))
        }
    }
}

/// One line of the panel, read out of the model rather than handed to it, so an
/// edit re-evaluates a binding instead of rebuilding the column.
fn line(
    model: State<Model>,
    size_token: LengthToken,
    spacing_token: LengthToken,
    display: bool,
    of: impl Fn(&Model) -> &Line + Copy + 'static,
) -> Element {
    let text = move || model.with(|found| of(found).text.clone());
    let tone = move || model.with(|found| of(found).tone).paint();
    let family = move || {
        if display {
            typed(typeface.display, FontFamily::default)
        } else {
            typed(typeface.body, FontFamily::default)
        }
    };
    view! {
        text width:fill font-family:{ family() }
            font-size:{ mosaic_core::theme::length(size_token) }
            letter-spacing:{ mosaic_core::theme::length(spacing_token) } font-color:{ tone() }
            { text() }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn linear_help_reports_steps_and_current_bindings() {
        let mut controls = crate::controls::Controls::default();
        controls.set(
            crate::controls::GameAction::LinearWidthFineIncrease,
            0,
            Some(crate::controls::InputChord::key(
                bevy::prelude::KeyCode::KeyP,
            )),
        );
        let help = super::linear_controls(&controls, "R");
        for expected in [
            "Four travel directions",
            "Length 250 mm",
            "Width 25 mm",
            "Fine length 2.5 mm",
            "Fine width 2.5 mm",
            " / P ",
        ] {
            assert!(help.contains(expected), "missing {expected}: {help}");
        }
    }
    use super::{Line, Model, Tone};
    use crate::ui::testing::Overlay;

    /// A panel with something to say in every line.
    fn filled() -> Model {
        Model {
            title: Line::new("MECHANIC  •  BUILDING", Tone::Title),
            primary: Line::new("SPACE  Start simulation", Tone::Body),
            edit: Line::new("R  Cycle plane", Tone::Muted),
            pointer: Line::new("LEFT  Action", Tone::Muted),
            tool: Line::new("Tool: Blocker Placer    Material: Steel", Tone::Speed),
            counts: Line::new("3 parts", Tone::Muted),
            hint: Line::new("Click for one block", Tone::Body),
            status: Line::new("STATUS  •  Ready", Tone::Good),
        }
    }

    #[test]
    fn the_panel_is_put_away_until_it_is_asked_for() {
        let overlay = Overlay::mount();
        overlay.handles.help.set(filled());
        overlay.settle();
        let hidden = overlay.element_count();

        overlay.handles.help_open.set(true);
        overlay.settle();
        assert!(
            overlay.element_count() > hidden,
            "asking for help puts the panel up",
        );

        overlay.handles.help_open.set(false);
        overlay.settle();
        assert_eq!(
            overlay.element_count(),
            hidden,
            "and asking again takes it away entirely",
        );
    }

    /// It sits in the corner rather than filling the window, or the world
    /// behind it would stop taking the pointer everywhere.
    #[test]
    fn the_panel_hugs_its_corner() {
        let overlay = Overlay::mount();
        overlay.handles.help.set(filled());
        overlay.handles.help_open.set(true);
        overlay.settle();

        let panel = overlay
            .reachable_boxes()
            .into_iter()
            .find(|rect| (rect.size.width - 720.0).abs() < 0.5)
            .expect("the panel is on screen");
        assert!((panel.origin.x - 16.0).abs() < 0.5);
        assert!((panel.origin.y - 16.0).abs() < 0.5);
        assert!(
            panel.size.height < 400.0,
            "the panel hugs its eight lines; it was {} tall",
            panel.size.height,
        );
    }

    /// Each line reads its own text out of the model, so a status change
    /// re-evaluates one binding rather than rebuilding the column.
    #[test]
    fn every_line_follows_the_model_it_reads() {
        let overlay = Overlay::mount();
        overlay.handles.help.set(filled());
        overlay.handles.help_open.set(true);
        overlay.settle();
        let built = overlay.element_count();

        overlay.handles.help.update(|model| {
            model.status = Line::new("STATUS  •  Cannot place that", Tone::Bad);
        });
        overlay.settle();
        assert_eq!(
            overlay.element_count(),
            built,
            "a new status is a new value, not a new panel",
        );
    }
}
