//! Screen-upright controls anchored to suspension geometry; the reticle owns input.

#![allow(
    clippy::wildcard_imports,
    reason = "Mosaic's authoring vocabulary is meant to be globbed"
)]

use super::Handles;
use super::styles::*;
use super::theme::{accent, dial};
use crate::suspension_controls::{Aim, Control, Parameter, Target};
use bevy::prelude::{Camera, GlobalTransform, Vec2, Vec3};
use bevy_mosaic::ui::*;
use mosaic_core::theme::color;
use mosaic_macros::{component, view};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum Information {
    Summary,
    Initial,
    Current,
    Limit,
    BumpContact,
    SourceMount,
    OppositeMount,
    Feedback,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum CalloutKey {
    Control(Control, bool),
    Information(Information),
}
impl CalloutKey {
    fn control(self) -> Option<Control> {
        if let Self::Control(control, _) = self {
            Some(control)
        } else {
            None
        }
    }
    fn grip(self) -> bool {
        matches!(self, Self::Control(_, true))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Callout {
    target: Target,
    key: CalloutKey,
    at: Vec2,
    text: String,
    direction: Vec2,
    pixels_per_step: f32,
    invalid: bool,
    width: f32,
}
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Model {
    callouts: Vec<Callout>,
    lines: Vec<(Vec2, Vec2)>,
}
#[derive(Clone, Debug)]
pub(crate) struct Bounds {
    target: Target,
    rect: Rect,
}
pub(crate) type Layout =
    std::rc::Rc<std::cell::RefCell<std::collections::BTreeMap<CalloutKey, Bounds>>>;

#[expect(clippy::too_many_lines)]
pub(crate) fn capture(
    state: &crate::editor::state::EditorState,
    graph: &crate::ConstructionGraph,
    simulation: &crate::simulation::state::AppSimulation,
    camera: &(&Camera, &GlobalTransform),
) -> Model {
    let controls = &state.suspension.controls;
    let Some(target) = controls.selected else {
        return Model::default();
    };
    let Some(index) = target.resolve(state) else {
        return Model::default();
    };
    let socket = state.placed_bearings[index];
    let mechanic_core::JointKind::Suspension(committed) = socket.kind else {
        return Model::default();
    };
    let spec = controls.gesture.as_ref().map_or(committed, |g| g.draft);
    let (pose, compression) =
        crate::suspension_render::socket_pose(graph, Some(simulation), socket);
    let (camera, transform) = *camera;
    let project = |p| {
        camera
            .world_to_viewport(transform, pose.transform_point(p))
            .ok()
    };
    let length = spec.extended_length();
    let current_separation = committed.extended_length() - compression;
    let separation = controls.gesture.as_ref().map_or(current_separation, |g| {
        g.draft.extended_length()
            - crate::suspension_render::draft_compression(
                g.original,
                g.draft,
                compression,
                !crate::editor::build_actions::bearing_socket_targets(graph, socket).is_empty(),
            )
    });
    let Some(origin) = project(Vec3::ZERO) else {
        return Model::default();
    };
    let Some(end) = project(Vec3::Y * length) else {
        return Model::default();
    };
    let axis = (end - origin).normalize_or_zero();
    let side = if axis.length_squared() > 0.5 {
        Vec2::new(-axis.y, axis.x)
    } else {
        Vec2::X
    };
    let mut callouts = Vec::new();
    let mut lines = Vec::new();
    let mut grips = Vec::new();
    let mut add = |key, at, text, direction, pixels_per_step, invalid| {
        callouts.push(Callout {
            target,
            key,
            at,
            text,
            direction,
            pixels_per_step,
            invalid,
            width: 260.0,
        });
    };
    for (component, present, name) in [
        (0, spec.spring().is_some(), "Spring"),
        (1, spec.shock().is_some(), "Shock"),
        (2, spec.bump_stop().is_some(), "Bump Stop"),
    ] {
        if present {
            #[expect(clippy::cast_precision_loss)]
            let offset = component as f32 * 34.0;
            add(
                CalloutKey::Control(Control::Component(component), false),
                end + side * 240.0 + Vec2::Y * offset,
                format!(
                    "{} {name}",
                    if controls.component == component {
                        "●"
                    } else {
                        "○"
                    }
                ),
                Vec2::X,
                6.0,
                false,
            );
        }
    }
    let mut row = 0.0;
    for parameter in Parameter::ALL
        .into_iter()
        .filter(|p| p.component() == controls.component && *p != Parameter::StartingCompression)
    {
        let (value, error) = controls.attempted(parameter, spec);
        let radius = match parameter {
            Parameter::SpringId => spec.spring().map_or(0.0, |s| s.id() / 2.0),
            _ => match parameter.component() {
                0 => spec.spring().map_or(0.0, |s| s.od() / 2.0),
                1 => spec.shock().map_or(0.0, |s| s.od() / 2.0),
                _ => spec.bump_stop().map_or(0.0, |s| s.od() / 2.0),
            },
        };
        let y = match parameter {
            Parameter::Preload => {
                spec.plates().thickness
                    + spec
                        .spring()
                        .map_or(0.0, mechanic_core::SpringSpec::preload)
            }
            Parameter::SpringLength => spec
                .spring()
                .map_or(length, mechanic_core::SpringSpec::length),
            Parameter::ShockLength => spec
                .shock()
                .map_or(length, mechanic_core::ShockSpec::length),
            Parameter::StartingCompression => length - spec.starting_compression(),
            Parameter::StopLength | Parameter::StopOd => {
                let fraction = if parameter == Parameter::StopLength {
                    1.0
                } else {
                    0.5
                };
                let reach = spec.plates().thickness
                    + spec
                        .bump_stop()
                        .map_or(0.0, mechanic_core::BumpStopSpec::length)
                        * fraction;
                if spec
                    .shock()
                    .is_some_and(|s| s.body_end() == mechanic_core::ShockBodyEnd::Opposite)
                {
                    reach
                } else {
                    separation - reach
                }
            }
            Parameter::Compression | Parameter::Rebound | Parameter::ShockOd => {
                spec.shock().map_or(separation * 0.5, |s| {
                    let distance = spec.plates().thickness
                        + s.geometry(spec.plates())
                            .expect("validated shock")
                            .body_length
                            * 0.5;
                    if s.body_end() == mechanic_core::ShockBodyEnd::Opposite {
                        separation - distance
                    } else {
                        distance
                    }
                })
            }
            _ => separation * 0.5,
        };
        let anchor = Vec3::new(radius, y, 0.0);
        let Some(at) = project(anchor) else {
            continue;
        };
        let direction3 = if parameter.axial() {
            if parameter == Parameter::StartingCompression
                || (parameter == Parameter::StopLength
                    && spec
                        .shock()
                        .is_some_and(|s| s.body_end() == mechanic_core::ShockBodyEnd::Source))
            {
                -Vec3::Y
            } else {
                Vec3::Y
            }
        } else {
            Vec3::X
        };
        let projected = project(anchor + direction3 * 0.1).map_or(Vec2::ZERO, |b| b - at);
        let fallback = !parameter.dimension() || projected.length() < 8.0;
        let direction = if fallback {
            Vec2::X
        } else {
            projected.normalize()
        };
        let scale = if fallback {
            6.0
        } else {
            projected.length() / 0.1 * parameter.step()
        };
        let text = format!(
            "{} {}",
            if fallback { "↔" } else { "◇" },
            parameter.readout(value)
        );
        // Callout leaders retain their geometry anchor while the readable grips fan out.
        let position = (origin + end) * 0.5 - side * 250.0 + Vec2::Y * (row - 80.0);
        lines.push((at, position + side * 130.0));
        row += 36.0;
        if parameter.dimension() {
            grips.push(Callout {
                target,
                key: CalloutKey::Control(Control::Parameter(parameter), true),
                at,
                text: String::new(),
                direction,
                pixels_per_step: scale,
                invalid: error.is_some(),
                width: 26.0,
            });
            let start = if parameter == Parameter::StopLength {
                Vec3::new(
                    radius,
                    if spec
                        .shock()
                        .is_some_and(|s| s.body_end() == mechanic_core::ShockBodyEnd::Opposite)
                    {
                        spec.plates().thickness
                    } else {
                        separation - spec.plates().thickness
                    },
                    0.0,
                )
            } else if parameter.axial() {
                Vec3::new(
                    radius,
                    if parameter == Parameter::Preload {
                        spec.plates().thickness
                    } else {
                        0.0
                    },
                    0.0,
                )
            } else {
                Vec3::new(-radius, y, 0.0)
            };
            if let Some(start) = project(start) {
                lines.push((start, at));
            }
        }
        add(
            CalloutKey::Control(Control::Parameter(parameter), false),
            position,
            text,
            direction,
            scale,
            error.is_some(),
        );
    }
    let (limit, limiting) = spec.compression_limit();
    let stroke = spec.shock().map_or(limit, |s| {
        s.geometry(spec.plates()).expect("validated shock").stroke
    });
    let summary = format!(
        "Rate {:.1} N/mm · stroke {:.1} mm\nSeparation {:.1} mm · remaining {:.1} mm · {limiting:?}",
        spec.spring().map_or(0.0, |s| s.rate() / 1000.0),
        stroke * 1000.0,
        current_separation * 1000.0,
        (limit - compression).max(0.0) * 1000.0
    );
    let right = end + side * 240.0;
    add(
        CalloutKey::Information(Information::Summary),
        right + Vec2::Y * 278.0,
        summary,
        Vec2::X,
        1.0,
        false,
    );
    let marks = [
        ("Initial", Some(spec.starting_compression())),
        ("Current", Some(compression)),
        ("Limit", Some(limit)),
        ("Bump contact", spec.bump_contact()),
    ];
    for (index, (name, value)) in marks.into_iter().enumerate() {
        let Some(value) = value else {
            continue;
        };
        let Some(at) = project(
            Vec3::Y
                * (if index == 1 {
                    committed.extended_length()
                } else {
                    length
                } - value),
        ) else {
            continue;
        };
        #[expect(clippy::cast_precision_loss)]
        let position = right + Vec2::Y * (112.0 + index as f32 * 36.0);
        lines.push((at + side * 64.0, at + side * 76.0));
        lines.push((at + side * 76.0, position - side * 130.0));
        let parameter = (index == 0 && controls.component == 1 && spec.shock().is_some())
            .then_some(Parameter::StartingCompression);
        let (attempted, error) = parameter.map_or((value, None), |p| controls.attempted(p, spec));
        let fallback = origin.distance(end) < 40.0;
        if let Some(parameter) = parameter {
            grips.push(Callout {
                target,
                key: CalloutKey::Control(Control::Parameter(parameter), true),
                at: at + side * 70.0,
                text: String::new(),
                direction: if fallback { Vec2::X } else { -axis },
                pixels_per_step: if fallback {
                    6.0
                } else {
                    origin.distance(end) / length * 0.0025
                },
                invalid: error.is_some(),
                width: 26.0,
            });
        }
        add(
            parameter.map_or(
                CalloutKey::Information(
                    [
                        Information::Initial,
                        Information::Current,
                        Information::Limit,
                        Information::BumpContact,
                    ][index],
                ),
                |p| CalloutKey::Control(Control::Parameter(p), false),
            ),
            position,
            format!(
                "{name} {} {:.1} mm",
                if parameter.is_some() { "◇" } else { "┃" },
                attempted * 1000.0
            ),
            if fallback { Vec2::X } else { -axis },
            if fallback {
                6.0
            } else {
                origin.distance(end) / length * 0.0025
            },
            error.is_some(),
        );
    }
    let feedback = controls
        .gesture
        .as_ref()
        .and_then(|g| g.error.as_ref().map(|e| (g.parameter, e.as_str())))
        .or_else(|| controls.feedback.as_ref().map(|(p, _, e)| (*p, e.as_str())));
    if let Some((parameter, message)) = feedback {
        add(
            CalloutKey::Information(Information::Feedback),
            (origin + end) * 0.5 - side * 250.0 + Vec2::Y * (row - 68.0),
            format!("{}: {message}", parameter.label()),
            Vec2::X,
            1.0,
            true,
        );
    }
    for (id, text, at) in [
        (
            Information::SourceMount,
            "Source mount",
            Some(origin + Vec2::Y * 24.0),
        ),
        (
            Information::OppositeMount,
            "Opposite mount",
            project(Vec3::Y * separation).map(|p| p - Vec2::Y * 24.0),
        ),
    ] {
        if let Some(at) = at {
            callouts.push(Callout {
                target,
                key: CalloutKey::Information(id),
                at,
                text: text.into(),
                direction: Vec2::X,
                pixels_per_step: 1.0,
                invalid: false,
                width: 140.0,
            });
        }
    }
    lines.push((origin + side * 70.0, end + side * 70.0));
    callouts.extend(grips);
    Model { callouts, lines }
}

pub(crate) fn aim(model: &Model, layout: &Layout, point: Vec2) -> Option<Aim> {
    layout.borrow().iter().find_map(|(key, b)| {
        let c = model
            .callouts
            .iter()
            .find(|c| c.target == b.target && c.key == *key)?;
        b.rect
            .contains(Vector2::new(point.x, point.y))
            .then_some(Aim {
                control: key.control()?,
                direction: c.direction,
                pixels_per_step: c.pixels_per_step,
            })
    })
}

#[component]
pub(crate) fn SuspensionOverlay(handles: Handles) -> Element {
    let model = handles.suspension;
    let layout = handles.suspension_layout.clone();
    view! {
        stack width:fill height:fill align:start justify:start nohit {
            for (index, ()) in { (0..model.with(|m| m.lines.len())).map(|i| (i, ())) } {
                (leader(model, *index))
            }
            for (key, ()) in { model.with(|m| m.callouts.iter().map(|c| (c.key, ())).collect::<Vec<_>>()) } {
                (callout(model, layout.clone(), *key))
            }
        }
    }
}
fn leader(model: State<Model>, index: usize) -> Element {
    let ends = move || model.with(|m| m.lines.get(index).copied().unwrap_or_default());
    view! {
        canvas width:fill height:fill nohit {
            line from:(x:{ Length::px(ends().0.x) } y:{ Length::px(ends().0.y) })
                to:(x:{ Length::px(ends().1.x) } y:{ Length::px(ends().1.y) }) stroke:(width:1px color:dial.tick)
            circle at:(x:{ Length::px(ends().0.x) } y:{ Length::px(ends().0.y) })
                radius:4px fill:dial.knob stroke:(width:2px color:dial.grip)
        }
    }
}
fn callout(model: State<Model>, layout: Layout, key: CalloutKey) -> Element {
    let found = move || model.with(|m| m.callouts.iter().find(|c| c.key == key).cloned());
    let at = move || found().map_or(Vec2::ZERO, |c| c.at);
    let half_width = move || found().map_or(130.0, |c| c.width / 2.0);
    view! {
        stack width:0px height:0px nohit translate:(x:{ Length::px(at().x - half_width()) } y:{ Length::px(at().y - 15.0) }) {
            row #mechanic.badge stroke:(width:1px color:{ if found().is_some_and(|c| c.invalid) { color(accent.danger) } else { color(dial.grip) } }) width:{ Length::px(half_width()*2.0) } height:min-content min-height:30px pad:6px nohit
                @layout:{ move |rect: Rect| {
                    let mut bounds = layout.borrow_mut();
                    if let Some(c) = found().filter(|c| c.key.control().is_some()) { bounds.insert(key, Bounds { target: c.target, rect }); }
                    else { bounds.remove(&key); }
                } } {
                text #mechanic.value width:fill { found().map_or(String::new(), |c| if c.key.grip() { "◇".into() } else { format!("{}{}", if c.invalid && c.key.control().is_none() { "Invalid · " } else { "" }, c.text) }) }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn leader_geometry_follows_projected_positions_without_remounting() {
        let overlay = crate::ui::testing::Overlay::mount();
        for origin in [Vec2::new(600.0, 250.0), Vec2::new(850.0, 450.0)] {
            overlay.handles.suspension.set(Model {
                callouts: Vec::new(),
                lines: vec![(origin, origin + Vec2::new(120.0, 80.0))],
            });
            overlay.settle();
            assert!(
                overlay.shapes().iter().any(|s| {
                    (s.rect.center().x - origin.x).abs() < 0.5
                        && (s.rect.center().y - origin.y).abs() < 0.5
                        && (s.rect.size.width - 8.0).abs() < 0.5
                }),
                "leader marker missing at {origin:?}"
            );
        }
    }
    #[test]
    fn reticle_hits_actual_mosaic_bounds_without_blocking_world_input() {
        let (_, state) = crate::suspension_controls::tests::fixture();
        let target = state.suspension.controls.selected.unwrap();
        let overlay = crate::ui::testing::Overlay::mount();
        let model = Model {
            callouts: vec![Callout {
                target,
                key: CalloutKey::Control(Control::Parameter(Parameter::Preload), false),
                at: Vec2::new(800.0, 450.0),
                text: "Additional natural length 0.0 mm".into(),
                direction: Vec2::Y,
                pixels_per_step: 2.0,
                invalid: false,
                width: 260.0,
            }],
            lines: Vec::new(),
        };
        overlay.handles.suspension.set(model.clone());
        overlay.settle();
        let layout = &overlay.handles.suspension_layout;
        let aimed = aim(&model, layout, Vec2::new(800.0, 450.0))
            .expect("reticle intersects laid-out control");
        assert_eq!(aimed.control, Control::Parameter(Parameter::Preload));
        assert!(!overlay.wants_pointer_at(Vector2::new(800.0, 450.0)));
        assert!(aim(&model, layout, Vec2::ZERO).is_none());
        assert!(aim(&Model::default(), layout, Vec2::new(800.0, 450.0)).is_none());
        let count = overlay.element_count();
        overlay.handles.suspension.set(Model::default());
        overlay.settle();
        assert!(overlay.element_count() < count);
    }
    #[test]
    fn changing_component_replaces_grips_with_full_callouts_at_the_same_slot() {
        let (_, state) = crate::suspension_controls::tests::fixture();
        let target = state.suspension.controls.selected.unwrap();
        let overlay = crate::ui::testing::Overlay::mount();
        for grip in [true, false, true, false] {
            let model = Model {
                callouts: vec![Callout {
                    target,
                    key: CalloutKey::Control(
                        Control::Parameter(Parameter::StartingCompression),
                        grip,
                    ),
                    at: Vec2::new(800.0, 450.0),
                    text: if grip {
                        String::new()
                    } else {
                        "Initial 0.0 mm".into()
                    },
                    direction: Vec2::Y,
                    pixels_per_step: 2.0,
                    invalid: false,
                    width: if grip { 26.0 } else { 260.0 },
                }],
                lines: Vec::new(),
            };
            overlay.handles.suspension.set(model.clone());
            overlay.settle();
            let bounds = overlay.handles.suspension_layout.borrow();
            let rect = bounds.get(&model.callouts[0].key).unwrap().rect;
            assert!((rect.size.width - model.callouts[0].width).abs() < 0.5);
            assert!((rect.center().x - 800.0).abs() < 0.5);
        }
    }
}
