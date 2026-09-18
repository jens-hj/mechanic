//! Reticle-operated suspension edits. Drafts never enter the simulation graph.
use crate::ConstructionGraph;
use crate::editor::build_actions::PlacedBearing;
use crate::editor::state::EditorState;
use bevy::prelude::*;
use mechanic_core::{
    BearingKind, BumpStopSpec, ShockBodyEnd, ShockSpec, SpringSpec, SuspensionSpec,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum Parameter {
    SpringLength,
    SpringOd,
    SpringId,
    Coils,
    Preload,
    ShockLength,
    ShockOd,
    StartingCompression,
    Compression,
    Rebound,
    Reverse,
    StopLength,
    StopOd,
}
impl Parameter {
    pub(crate) const ALL: [Self; 13] = [
        Self::SpringLength,
        Self::SpringOd,
        Self::SpringId,
        Self::Coils,
        Self::Preload,
        Self::ShockLength,
        Self::ShockOd,
        Self::StartingCompression,
        Self::Compression,
        Self::Rebound,
        Self::Reverse,
        Self::StopLength,
        Self::StopOd,
    ];
    pub(crate) const fn component(self) -> usize {
        match self {
            Self::SpringLength | Self::SpringOd | Self::SpringId | Self::Coils | Self::Preload => 0,
            Self::StopLength | Self::StopOd => 2,
            _ => 1,
        }
    }
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::SpringLength | Self::ShockLength => "Extended length",
            Self::SpringOd | Self::ShockOd | Self::StopOd => "Outer diameter",
            Self::SpringId => "Inner diameter",
            Self::Coils => "Active coils",
            Self::Preload => "Preload +",
            Self::StartingCompression => "Initial compression",
            Self::Compression => "Compression ×",
            Self::Rebound => "Rebound ×",
            Self::Reverse => "Reverse body end",
            Self::StopLength => "Free length",
        }
    }
    pub(crate) const fn step(self) -> f32 {
        match self {
            Self::Coils | Self::Reverse => 1.0,
            Self::Compression | Self::Rebound => 0.1,
            _ => 0.0025,
        }
    }
    pub(crate) const fn axial(self) -> bool {
        matches!(
            self,
            Self::SpringLength
                | Self::ShockLength
                | Self::Preload
                | Self::StartingCompression
                | Self::StopLength
        )
    }
    pub(crate) const fn dimension(self) -> bool {
        !matches!(
            self,
            Self::Coils | Self::Compression | Self::Rebound | Self::Reverse
        )
    }
    pub(crate) fn value(self, spec: SuspensionSpec) -> f32 {
        let s = spec.spring().unwrap_or_default();
        let h = spec.shock().unwrap_or_default();
        let b = spec.bump_stop().unwrap_or_default();
        match self {
            Self::SpringLength => s.length(),
            Self::SpringOd => s.od(),
            Self::SpringId => s.id(),
            Self::Coils => f32::from(s.coils()),
            Self::Preload => s.preload(),
            Self::ShockLength => h.length(),
            Self::ShockOd => h.od(),
            Self::StartingCompression => h.starting_compression(),
            Self::Compression => multiplier(h, true),
            Self::Rebound => multiplier(h, false),
            Self::Reverse => f32::from(h.body_end() == ShockBodyEnd::Opposite),
            Self::StopLength => b.length(),
            Self::StopOd => b.od(),
        }
    }
    pub(crate) fn readout(self, value: f32) -> String {
        if self == Self::Reverse {
            return if value == 0.0 {
                "Body: source → opposite"
            } else {
                "Body: opposite → source"
            }
            .into();
        }
        format!(
            "{} {:.1}{}",
            self.label(),
            if self.dimension() {
                value * 1000.0
            } else {
                value
            },
            if self.dimension() { " mm" } else { "" }
        )
    }
    pub(crate) fn edit(
        self,
        original: SuspensionSpec,
        value: f32,
        attached: bool,
    ) -> Result<SuspensionSpec, String> {
        let v = |p: Self| if p == self { value } else { p.value(original) };
        let mut spring = original.spring();
        let mut shock = original.shock();
        let mut stop = original.bump_stop();
        match self.component() {
            0 => {
                if spring.is_none() {
                    return Err("Spring was removed".into());
                }
                let coils = (3..=16)
                    .find(|&n| (f32::from(n) - v(Self::Coils)).abs() < 0.001)
                    .ok_or("Use 3–16 whole active coils")?;
                spring = Some(
                    SpringSpec::new(
                        v(Self::SpringLength),
                        v(Self::SpringOd),
                        v(Self::SpringId),
                        coils,
                        v(Self::Preload),
                    )
                    .map_err(|e| e.to_string())?,
                );
            }
            1 => {
                if shock.is_none() {
                    return Err("Shock was removed".into());
                }
                shock = Some(
                    ShockSpec::new(
                        v(Self::ShockLength),
                        v(Self::ShockOd),
                        if v(Self::Reverse) == 0.0 {
                            ShockBodyEnd::Source
                        } else {
                            ShockBodyEnd::Opposite
                        },
                        v(Self::StartingCompression),
                        v(Self::Compression),
                        v(Self::Rebound),
                    )
                    .map_err(|e| e.to_string())?,
                );
            }
            _ => {
                if stop.is_none() {
                    return Err("Bump stop was removed".into());
                }
                stop = Some(
                    BumpStopSpec::new(v(Self::StopLength), v(Self::StopOd))
                        .map_err(|e| e.to_string())?,
                );
            }
        }
        original
            .with_components(spring, shock, stop, attached)
            .map_err(|e| match e {
                mechanic_core::SuspensionError::AttachedSpacing => {
                    "Release the opposite attachment to change mount spacing".into()
                }
                _ => e.to_string(),
            })
    }
}
fn multiplier(shock: ShockSpec, compression: bool) -> f32 {
    shock.damping(compression)
        / ShockSpec::new(shock.length(), shock.od(), shock.body_end(), 0.0, 1.0, 1.0)
            .expect("validated shock")
            .damping(compression)
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Target(pub(crate) PlacedBearing);
impl PartialEq for Target {
    fn eq(&self, other: &Self) -> bool {
        self.0.source == other.0.source
            && self.0.anchor.distance_squared(other.0.anchor) < 1e-10
            && self.0.axis.distance_squared(other.0.axis) < 1e-10
    }
}
impl Target {
    pub(crate) fn resolve(self, state: &EditorState) -> Option<usize> {
        state.placed_bearings.iter().position(|s| {
            s.source == self.0.source
                && s.anchor.distance_squared(self.0.anchor) < 1e-10
                && s.axis.distance_squared(self.0.axis) < 1e-10
                && matches!(s.kind, BearingKind::Suspension(_))
        })
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum Control {
    Component(usize),
    Parameter(Parameter),
}
#[derive(Clone, Copy, Debug)]
pub(crate) struct Aim {
    pub(crate) control: Control,
    pub(crate) direction: Vec2,
    pub(crate) pixels_per_step: f32,
}
#[derive(Clone, Debug)]
pub(crate) struct Gesture {
    pub(crate) target: Target,
    pub(crate) original: SuspensionSpec,
    pub(crate) draft: SuspensionSpec,
    pub(crate) parameter: Parameter,
    pub(crate) value: f32,
    pub(crate) error: Option<String>,
    direction: Vec2,
    pixels_per_step: f32,
    pixels: f32,
}
impl Gesture {
    fn start(target: Target, original: SuspensionSpec, parameter: Parameter, aim: Aim) -> Self {
        let value = parameter.value(original);
        Self {
            target,
            original,
            draft: original,
            parameter,
            value: if parameter == Parameter::Reverse {
                1.0 - value
            } else {
                value
            },
            error: None,
            direction: aim.direction,
            pixels_per_step: aim.pixels_per_step.max(0.01),
            pixels: 0.0,
        }
    }
}
#[derive(Clone, Debug, Default)]
pub(crate) struct Controls {
    pub(crate) selected: Option<Target>,
    pub(crate) component: usize,
    pub(crate) aim: Option<Aim>,
    pub(crate) gesture: Option<Gesture>,
    pub(crate) feedback: Option<(Parameter, f32, String)>,
    pub(crate) dismissed: Option<Target>,
    pub(crate) consume_until_release: bool,
}
impl Controls {
    pub(crate) fn dismiss(&mut self) {
        self.dismissed = self.selected.take();
        self.gesture = None;
        self.feedback = None;
        self.aim = None;
        self.consume_until_release = true;
    }
    pub(crate) fn attempted(
        &self,
        parameter: Parameter,
        spec: SuspensionSpec,
    ) -> (f32, Option<&str>) {
        if let Some(g) = self.gesture.as_ref().filter(|g| g.parameter == parameter) {
            return (g.value, g.error.as_deref());
        }
        if let Some((_, value, message)) =
            self.feedback.as_ref().filter(|(p, _, _)| *p == parameter)
        {
            return (*value, Some(message));
        }
        (parameter.value(spec), None)
    }
    pub(crate) fn select(&mut self, socket: PlacedBearing, component: usize) {
        let target = Target(socket);
        if self.selected != Some(target) {
            self.selected = Some(target);
            self.component = component;
            self.aim = None;
            self.feedback = None;
        }
    }
}

pub(crate) fn validate_draft(
    graph: &ConstructionGraph,
    state: &EditorState,
    gesture: &mut Gesture,
) {
    let result = (|| {
        let index = gesture
            .target
            .resolve(state)
            .ok_or("Suspension target disappeared")?;
        let socket = state.placed_bearings[index];
        if socket.kind != BearingKind::Suspension(gesture.original) {
            return Err("Suspension changed; adjustment cancelled".into());
        }
        let spec = gesture.parameter.edit(
            gesture.original,
            gesture.value,
            !crate::editor::build_actions::bearing_socket_targets(graph, socket).is_empty(),
        )?;
        let half = spec.plates().diameter / 2.0;
        let center = socket.anchor + socket.axis * spec.extended_length() / 2.0;
        let extent = socket.axis.abs() * spec.extended_length() / 2.0
            + (Vec3::ONE - socket.axis.abs()) * half;
        crate::builder::validate_world_bounds(
            center - extent,
            center + extent,
            state.placement_bounds,
        )
        .map_err(|e| e.to_string())?;
        Ok(spec)
    })();
    match result {
        Ok(spec) => {
            gesture.draft = spec;
            gesture.error = None;
        }
        Err(e) => gesture.error = Some(e),
    }
}

/// Runs before connector wiring and before entering the live edit coordinate frame.
pub(crate) fn actions(
    graph: &mut crate::editor::state::EditorGraph,
    state: &mut EditorState,
    history: &mut crate::editor::history::EditorHistory,
    actions: &ButtonInput<crate::GameAction>,
    delta: Vec2,
    tool: Option<crate::Tool>,
    blocked: bool,
) -> bool {
    use crate::GameAction::{Primary, Secondary};
    if state.suspension.controls.consume_until_release {
        if !actions.pressed(Primary) && !actions.pressed(Secondary) {
            state.suspension.controls.consume_until_release = false;
        }
        return true;
    }
    if tool != Some(crate::Tool::Connector) || blocked {
        if state.suspension.controls.selected.is_some() {
            state.suspension.controls.dismiss();
        }
        return false;
    }
    if let Some(mut gesture) = state.suspension.controls.gesture.take() {
        if actions.just_pressed(Secondary) {
            state.suspension.controls.consume_until_release = true;
            return true;
        }
        if !gesture.target.resolve(state).is_some_and(|i| {
            state.placed_bearings[i].kind == BearingKind::Suspension(gesture.original)
        }) {
            state.suspension.controls.dismiss();
            state.feedback = Some("Suspension changed; adjustment cancelled".into());
            return true;
        }
        gesture.pixels += delta.dot(gesture.direction);
        let steps = (gesture.pixels / gesture.pixels_per_step).round();
        if gesture.parameter != Parameter::Reverse {
            gesture.value =
                gesture.parameter.value(gesture.original) + steps * gesture.parameter.step();
            if matches!(
                gesture.parameter,
                Parameter::Compression | Parameter::Rebound
            ) {
                gesture.value = gesture.value.clamp(0.0, 100.0);
            }
        }
        validate_draft(&graph.0, state, &mut gesture);
        if actions.just_released(Primary) {
            if let Some(error) = gesture.error {
                state.suspension.controls.feedback =
                    Some((gesture.parameter, gesture.value, error.clone()));
                state.feedback = Some(error);
            } else if gesture.draft != gesture.original {
                let index = gesture.target.resolve(state).expect("checked target");
                let mut view = crate::live_edit::EditorView::new(graph, state);
                let (graph, state) = view.parts();
                crate::suspension_editor::apply_settings(
                    &mut graph.0,
                    state,
                    history,
                    index,
                    gesture.draft,
                );
            }
        } else {
            state.suspension.controls.gesture = Some(gesture);
        }
        return true;
    }
    let Some(aim) = state.suspension.controls.aim else {
        return false;
    };
    if actions.just_pressed(Primary) {
        state.suspension.controls.feedback = None;
        match aim.control {
            Control::Component(component) => {
                state.suspension.controls.component = component;
                state.suspension.controls.consume_until_release = true;
            }
            Control::Parameter(parameter) => {
                let Some(target) = state.suspension.controls.selected else {
                    return true;
                };
                let Some(index) = target.resolve(state) else {
                    return true;
                };
                let BearingKind::Suspension(original) = state.placed_bearings[index].kind else {
                    return true;
                };
                let target = Target(state.placed_bearings[index]);
                state.suspension.controls.selected = Some(target);
                let mut gesture = Gesture::start(target, original, parameter, aim);
                validate_draft(&graph.0, state, &mut gesture);
                state.suspension.controls.gesture = Some(gesture);
            }
        }
    }
    true
}

#[cfg(test)]
pub(crate) mod tests;
