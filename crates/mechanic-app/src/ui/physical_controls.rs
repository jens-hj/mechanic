//! Plain feedback for physical dial operation; input stays owned by the world.

#![allow(
    clippy::wildcard_imports,
    reason = "Mosaic's authoring vocabulary is meant to be globbed"
)]

use bevy_mosaic::ui::*;
use mosaic_macros::{component, view};

use super::components::{OverlayBadge, OverlayBadgeProps};
use super::styles::*;

/// Formats native physical-control state using the controller's display units.
pub(crate) fn capture(
    controls: &crate::physical_controls::PhysicalControls,
    speed_unit: crate::control_panel::SpeedUnit,
) -> Vec<String> {
    let mut lines = controls.feedback.clone();
    if let Some((choices, selected)) = controls.choices() {
        lines.extend(choices.iter().enumerate().map(|(index, choice)| {
            format!(
                "{} {} · {}",
                if index == selected { "›" } else { " " },
                index + 1,
                choice_label(choice.target, choice.value, choice.linear, speed_unit)
            )
        }));
    }
    lines
}

fn choice_label(
    target: mechanic_core::NumericParameter,
    value: f32,
    linear: bool,
    speed_unit: crate::control_panel::SpeedUnit,
) -> String {
    use crate::control_panel::SpeedUnit;
    use mechanic_core::{DriveParameter, EngineKind, GearParameter, NumericParameter};
    let (name, value, unit, integer) = match target {
        NumericParameter::Drive { parameter, .. } => match parameter {
            DriveParameter::AngularPosition(index) => (
                format!("State {} target", index + 1),
                value.to_degrees(),
                "°",
                false,
            ),
            DriveParameter::AngularSpeed(index) => {
                let (value, unit) = match speed_unit {
                    SpeedUnit::Rpm => (mechanic_core::rad_s_to_rpm(value), "rpm"),
                    SpeedUnit::DegreesPerSecond => (value.to_degrees(), "°/s"),
                };
                (format!("State {} speed", index + 1), value, unit, false)
            }
            DriveParameter::LinearPosition(index) => {
                (format!("State {} target", index + 1), value, "m", false)
            }
            DriveParameter::LinearSpeed(index) => {
                (format!("State {} speed", index + 1), value, "m/s", false)
            }
            DriveParameter::Dwell(index) => {
                (format!("State {} dwell", index + 1), value, "s", false)
            }
            DriveParameter::TravelMinimum | DriveParameter::TravelMaximum => (
                if parameter == DriveParameter::TravelMinimum {
                    "Minimum travel"
                } else {
                    "Maximum travel"
                }
                .into(),
                if linear { value } else { value.to_degrees() },
                if linear { "m" } else { "°" },
                false,
            ),
            DriveParameter::ElectricContribution => {
                ("Electric contribution".into(), value, "%", true)
            }
            DriveParameter::GasContribution => ("Gas contribution".into(), value, "%", true),
        },
        NumericParameter::Gear {
            kind, parameter, ..
        } => {
            let engine = match kind {
                EngineKind::Electric => "Electric",
                EngineKind::Gas => "Gas",
            };
            match parameter {
                GearParameter::Ratio(index) => (
                    format!("{engine} gear {} ratio", index + 1),
                    value,
                    ":1",
                    false,
                ),
                GearParameter::ReverseCount => (format!("{engine} reverse gears"), value, "", true),
            }
        }
    };
    let value = if integer {
        format!("{value:.0}")
    } else {
        format!("{value:.3}")
    };
    format!("{name}: {value} {unit}").trim_end().to_owned()
}

const PANEL_WIDTH: f32 = 360.0;
// Leave space for both rows of the Matter Manipulator hotbar.
const BOTTOM_INSET: f32 = 220.0;

#[component]
pub(crate) fn PhysicalControlOverlay(model: State<Vec<String>>, viewport: State<Size>) -> Element {
    let size = State::new(Size::ZERO);
    let at = move || {
        let window = viewport.get();
        (
            Length::px(((window.width - PANEL_WIDTH) * 0.5).max(8.0)),
            Length::px((window.height - size.get().height - BOTTOM_INSET).max(8.0)),
        )
    };
    view! {
        col width:min-content height:min-content nohit
            translate:(x:{ at().0 } y:{ at().1 })
            @layout:{ move |bounds: Rect| {
                if size.get_untracked() != bounds.size {
                    size.set(bounds.size);
                }
            } } {
            OverlayBadge width:(Length::px(PANEL_WIDTH)) height:min-content
                gap:6px pad:12px {
                for (index, ()) in { (0..model.with(Vec::len)).map(|index| (index, ())) } {
                    (feedback_line(model, *index))
                }
            }
        }
    }
}

fn feedback_line(model: State<Vec<String>>, index: usize) -> Element {
    view! {
        text #mechanic.caption width:fill {
            model.with(|lines| lines.get(index).cloned().unwrap_or_default())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::testing::{Overlay, VIEWPORT};

    #[test]
    fn feedback_updates_without_capturing_world_input_and_empty_hides_it() {
        let overlay = Overlay::mount();
        // The default Matter Manipulator hotbar already occupies part of this
        // area. A passive overlay must preserve underlying hits, not erase them.
        let points = [-160.0, -80.0, 0.0, 80.0, 160.0]
            .into_iter()
            .flat_map(|x| {
                [-140.0, -100.0, -60.0, -20.0].into_iter().map(move |y| {
                    Vector2::new(VIEWPORT.width * 0.5 + x, VIEWPORT.height - BOTTOM_INSET + y)
                })
            })
            .collect::<Vec<_>>();
        let hits_before = points
            .iter()
            .map(|&point| overlay.wants_pointer_at(point))
            .collect::<Vec<_>>();
        overlay.handles.physical_controls.set(vec![
            "Mixed — select a target".into(),
            "1. State 1 target: 30°".into(),
        ]);
        overlay.settle();
        assert!(overlay.labels().contains(&"Mixed — select a target".into()));
        assert_eq!(
            points
                .iter()
                .map(|&point| overlay.wants_pointer_at(point))
                .collect::<Vec<_>>(),
            hits_before,
            "feedback must leave existing hotbar and world hits unchanged",
        );
        overlay
            .handles
            .physical_controls
            .set(vec!["Drag horizontally; Shift adjusts finely".into()]);
        overlay.settle();
        assert!(!overlay.labels().contains(&"Mixed — select a target".into()));
        assert!(
            overlay
                .labels()
                .contains(&"Drag horizontally; Shift adjusts finely".into())
        );
        overlay.handles.physical_controls.set(Vec::new());
        overlay.settle();
        assert!(
            !overlay
                .labels()
                .contains(&"Drag horizontally; Shift adjusts finely".into())
        );
    }
}
