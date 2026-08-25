//! The help and status panel: what mode this is, what the keys do, and what
//! just happened.
//!
//! Eight lines, each one a sentence about the editor's current state. The work
//! is in deciding what they say — the view is a column of text — so this module
//! is mostly [`capture`], reading the whole editor once a frame and writing
//! down what it found.

#![allow(clippy::wildcard_imports)] // Mosaic's authoring vocabulary is meant to be globbed.

use bevy::ecs::system::{Res, SystemParam};
use bevy_mosaic::ui::*;
use mosaic_core::theme::color;
use mosaic_macros::view;

use super::Handles;
#[allow(clippy::wildcard_imports)] // The design tokens are read as bare names.
use super::theme::*;
use crate::hotbar::{SelectedMaterial, SelectedTool, Tool};
use crate::{
    AppSimulation, BearingToolSettings, BlockAttachment, CylinderToolSettings, EditorGraph,
    EditorState, HAMMER_CHARGE_SECONDS, HammerInteraction, WireEnd, visible_bearing_count,
};

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
}

/// Reads the editor into what the panel says.
#[allow(clippy::too_many_lines)] // One line of prose per editor state, in the order they are shown.
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
    } = sources;

    let active_error = state
        .delete_drag
        .as_ref()
        .and_then(|drag| drag.error.as_ref())
        .or(state.preview_error.as_ref());
    let status = if let Some(charge) = hammer.charging {
        format!(
            "Hammer charge: {:.0}% — release to strike",
            charge.elapsed_seconds / HAMMER_CHARGE_SECONDS * 100.0
        )
    } else {
        active_error.map_or_else(
            || state.feedback.clone().unwrap_or_else(|| "Ready".to_owned()),
            ToString::to_string,
        )
    };
    let status_tone = if hammer.charging.is_some() {
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
    let tool_hint = if simulation.is_paused() {
        "Simulation paused at the current pose — press Space to resume".to_owned()
    } else if let Some(drag) = state.delete_drag.as_ref() {
        format!(
            "Release to delete {} cuboid(s) on the {} plane",
            drag.parts.len(),
            drag.plane.label()
        )
    } else {
        match (
            simulation.is_running(),
            selection.0,
            graph.0.pending(),
            state.block_drag.as_ref(),
            state.attachment_bearing,
        ) {
            (false, Tool::Block, _, Some(drag), _) => {
                if matches!(drag.attachment, BlockAttachment::Bearing { .. }) {
                    format!(
                        "Green bearing attachment active — release to connect {} block(s)",
                        drag.specs.len()
                    )
                } else {
                    format!(
                        "Release to place {} blocks on the {} plane",
                        drag.specs.len(),
                        drag.plane.label()
                    )
                }
            }
            (false, Tool::Block, _, None, Some(_)) => {
                "Green bearing attachment active — click or drag to connect blocks".to_owned()
            }
            (false, Tool::Cylinder, _, _, Some(_)) => {
                "Green bearing attachment active — click to connect cylinder".to_owned()
            }
            (true, Tool::Hammer, _, _, _) => {
                "Hold left mouse on a moving cuboid; release to strike".to_owned()
            }
            (true, _, _, _, _) => {
                "This tool is build-only — press Escape to return to build mode".to_owned()
            }
            (false, Tool::Block, _, _, _) => {
                "Click for one block or drag to place a welded sheet".to_owned()
            }
            (false, Tool::Cylinder, _, _, _) => {
                "Click a flat face to place; Shift+Down/Up changes the slice angle".to_owned()
            }
            (false, Tool::Weld, None, _, _) => "Left click selects the first object".to_owned(),
            (false, Tool::Weld, Some(_), _, _) => "Left click a touching second object".to_owned(),
            (false, Tool::Bearing, _, _, _) => {
                "Left click places a bearing; use Blocker Placer to attach it".to_owned()
            }
            (false, Tool::Hammer, _, _, _) => {
                "Hammer is available while simulating — press Space to start".to_owned()
            }
            (false, Tool::Controller, _, _, _) => {
                "Q cycles all 24 orientations; left click places a control block; click one to retune it"
                    .to_owned()
            }
            (false, Tool::GasEngine, _, _, _) => {
                "Q rotates; place a 200 N·m, 220 RPM gas engine (4 bearing ports)".to_owned()
            }
            (false, Tool::ElectricEngine, _, _, _) => {
                "Q rotates; place a 500 N·m, 120 RPM electric engine (4 bearing ports)"
                    .to_owned()
            }
            (false, Tool::Servo, _, _, _) => {
                "Q rotates; place a 150 N·m, 30 RPM Servo (one angle-controlled bearing)"
                    .to_owned()
            }
            (false, Tool::Seat, _, _, _) => {
                "Q cycles all 24 orientations; place a cushion, then wire it with Connector"
                    .to_owned()
            }
            (false, Tool::Input, _, _, _) => {
                "Q cycles all 24 orientations; place Input, then wire it to a Seat".to_owned()
            }
            (false, Tool::Shape, _, _, _) => {
                "Drag an area of blocks (Q changes plane), then drag its corners; arrows/WASD nudge; G sets the step"
                    .to_owned()
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
    };
    let (phase, title_tone, primary_controls) = if simulation.is_paused() {
        (
            "PAUSED",
            Tone::Warn,
            "SPACE  Resume     SHIFT+SPACE  Restart     ESC  Build mode     ?  Hide help",
        )
    } else if simulation.is_running() {
        (
            "SIMULATING",
            Tone::Good,
            "SPACE  Pause     SHIFT+SPACE  Restart     ESC  Build mode     ?  Hide help",
        )
    } else {
        (
            "BUILDING",
            Tone::Title,
            "SPACE  Start simulation     P  Creations     CTRL/CMD+S  Save     ?  Hide help",
        )
    };
    let action_controls = if state.delete_drag.is_some() {
        "RELEASE RIGHT  Delete     ESC  Cancel"
    } else {
        match (
            simulation.is_running(),
            selection.0,
            state.block_drag.is_some(),
        ) {
            (false, Tool::Block, true) => "RELEASE  Place     RIGHT / ESC  Cancel",
            (true, Tool::Hammer, _) if !simulation.is_paused() => "HOLD LEFT  Charge hammer",
            (true, _, _) => "Construction actions unavailable",
            (false, _, _) => "LEFT  Action     RIGHT DRAG  Delete",
        }
    };
    let plane_controls = if let Some(drag) = state.block_drag.as_ref() {
        format!("Q  Cycle plane ({})", drag.plane.label())
    } else if let Some(drag) = state.delete_drag.as_ref() {
        format!("Q  Cycle delete plane ({})", drag.plane.label())
    } else {
        "Q  Cycle plane while dragging or deleting".to_owned()
    };
    let edit_controls = if simulation.is_running() {
        if simulation.is_paused() {
            "Current pose is frozen; construction editing remains locked".to_owned()
        } else {
            "Physics is live; construction editing remains locked".to_owned()
        }
    } else {
        format!("{plane_controls}     CTRL/CMD+Z  Undo     SHIFT+CTRL/CMD+Z  Redo")
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
            format!("{action_controls}     ALT+LEFT  Orbit     SHIFT+LEFT  Pan     WHEEL  Zoom"),
            Tone::Muted,
        ),
        tool: Line::new(
            crate::tool_status_line(
                selection.0,
                bearing.dimensions,
                cylinder.dimensions,
                selected_wires,
                material.0,
            ),
            tool_tone(selection.0),
        ),
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
const fn tool_tone(tool: Tool) -> Tone {
    match tool {
        Tool::Bearing | Tool::Hammer | Tool::GasEngine | Tool::Servo => Tone::Angle,
        Tool::Weld | Tool::Controller | Tool::Connector | Tool::Input => Tone::Key,
        Tool::Block | Tool::Cylinder | Tool::ElectricEngine | Tool::Seat | Tool::Shape => {
            Tone::Speed
        }
    }
}

/// The panel.
///
/// Hugging rather than filling: it sits in the top-left corner and the world
/// keeps the pointer everywhere it is not.
pub(crate) fn view(handles: &Handles) -> Element {
    let model = handles.help;
    view! {
        col width:720px height:min-content gap:7px
            translate:(x:16px y:16px) pad:16px radius:10px
            fill:shell stroke:(width:1px color:shell-edge)
            font-color:help.body {
            (line(model, 19.0, 1.4, |found| &found.title))
            (line(model, 15.0, 0.0, |found| &found.primary))
            (line(model, 14.0, 0.0, |found| &found.edit))
            (line(model, 14.0, 0.0, |found| &found.pointer))
            (line(model, 16.0, 0.0, |found| &found.tool))
            (line(model, 13.0, 0.0, |found| &found.counts))
            (line(model, 15.0, 0.0, |found| &found.hint))
            (line(model, 13.0, 0.0, |found| &found.status))
        }
    }
}

/// One line of the panel, read out of the model rather than handed to it, so an
/// edit re-evaluates a binding instead of rebuilding the column.
fn line(
    model: State<Model>,
    size: f32,
    spacing: f32,
    of: impl Fn(&Model) -> &Line + Copy + 'static,
) -> Element {
    let text = move || model.with(|found| of(found).text.clone());
    let tone = move || model.with(|found| of(found).tone).paint();
    view! {
        text width:fill font-size:{ Length::px(size) } letter-spacing:{ Length::px(spacing) }
            font-color:{ tone() } { text() }
    }
}

#[cfg(test)]
mod tests {
    use super::{Line, Model, Tone};
    use crate::ui::testing::Overlay;

    /// A panel with something to say in every line.
    fn filled() -> Model {
        Model {
            title: Line::new("MECHANIC  •  BUILDING", Tone::Title),
            primary: Line::new("SPACE  Start simulation", Tone::Body),
            edit: Line::new("Q  Cycle plane", Tone::Muted),
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
