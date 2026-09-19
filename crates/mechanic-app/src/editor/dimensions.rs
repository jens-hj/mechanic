//! Bearing and cylinder dimension settings with their adjustment shortcuts.

use crate::camera::{MaterialWheelState, PlayerState};
use crate::controls::GameAction;
use crate::creation_menu::CreationMenuState;
use crate::editor::state::{EditorGraph, EditorState};
use crate::hotbar::{SelectedMaterial, SelectedTool, Tool};
use crate::{ui, world};
use bevy::prelude::{ButtonInput, Res, ResMut, Resource, State, format};
use mechanic_core::{
    BearingDimensions, CYLINDER_SWEEP_STEP_DEGREES, ConstructionMaterial, CylinderDimensions,
    FaceOwner, MAX_BEARING_OUTER_DIAMETER, MAX_CYLINDER_OUTER_DIAMETER, MAX_CYLINDER_SWEEP_DEGREES,
    MIN_BEARING_DIAMETER_GAP, MIN_BEARING_OUTER_DIAMETER, MIN_CYLINDER_DIAMETER_GAP,
    MIN_CYLINDER_OUTER_DIAMETER, MIN_CYLINDER_SWEEP_DEGREES, PipeBendDimensions,
};

pub(crate) const BEARING_DIAMETER_STEP: f32 = 0.05;

pub(crate) const CYLINDER_DIAMETER_STEP: f32 = 0.05;

pub(crate) const CYLINDER_LENGTH_STEP: f32 = 0.25;

#[derive(Resource, Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct BearingToolSettings {
    pub(crate) dimensions: BearingDimensions,
}

#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub(crate) struct CylinderToolSettings {
    pub(crate) dimensions: CylinderDimensions,
    pub(crate) bend_span: u8,
}

impl Default for CylinderToolSettings {
    fn default() -> Self {
        Self {
            dimensions: CylinderDimensions::default(),
            bend_span: PipeBendDimensions::DEFAULT_SPAN,
        }
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "bevy system resources are explicit parameters"
)]
pub(crate) fn handle_dimension_link_interaction(
    actions: Res<ButtonInput<GameAction>>,
    graph: Res<EditorGraph>,
    mut state: ResMut<EditorState>,
    mut runtime: ResMut<world::WorldRuntime>,
    space: Res<State<world::AppSpace>>,
    player: Res<PlayerState>,
    overlay: Res<ui::UiInput>,
    wheel: Res<MaterialWheelState>,
) {
    if !actions.just_pressed(GameAction::Interact)
        || !player.world_input_active()
        || overlay.blocks_keyboard()
        || wheel.open
    {
        return;
    }
    let aimed = state
        .hovered
        .and_then(|hit| match hit.face.owner {
            FaceOwner::Part(part) => Some((part, hit.distance)),
            FaceOwner::Ground => None,
        })
        .or_else(|| state.hovered_simulation.map(|hit| (hit.part, hit.distance)));
    let Some((part, distance)) = aimed else {
        return;
    };
    let Some(id) = graph.0.dimension_link_id(part) else {
        return;
    };
    if distance > 3.0 {
        state.feedback = Some("Dimension Link is out of interaction range (3 m)".to_owned());
        return;
    }
    state.feedback = Some(
        match runtime.toggle_dimension_link(*space.get(), &graph.0, part) {
            Ok(Some(_)) => format!("Activated Dimension Link {}", id.0),
            Ok(None) => format!("Deactivated Dimension Link {}", id.0),
            Err(error) => error,
        },
    );
    state.construction_mesh_dirty = true;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BearingDimensionTarget {
    Outer,
    Inner,
}

pub(crate) fn requested_bearing_dimension_adjustment(
    actions: &ButtonInput<GameAction>,
    tool: Option<Tool>,
    menu_blocks_input: bool,
) -> Option<(BearingDimensionTarget, i8)> {
    if tool != Some(Tool::Bearing) || menu_blocks_input {
        return None;
    }
    [
        (
            GameAction::BearingInnerDecrease,
            BearingDimensionTarget::Inner,
            -1,
        ),
        (
            GameAction::BearingInnerIncrease,
            BearingDimensionTarget::Inner,
            1,
        ),
        (
            GameAction::BearingOuterDecrease,
            BearingDimensionTarget::Outer,
            -1,
        ),
        (
            GameAction::BearingOuterIncrease,
            BearingDimensionTarget::Outer,
            1,
        ),
    ]
    .into_iter()
    .find_map(|(action, target, direction)| {
        actions.just_pressed(action).then_some((target, direction))
    })
}

pub(crate) fn adjusted_bearing_dimensions(
    dimensions: BearingDimensions,
    target: BearingDimensionTarget,
    direction: i8,
) -> BearingDimensions {
    let step = f32::from(direction) * BEARING_DIAMETER_STEP;
    let stepped =
        |diameter: f32| ((diameter + step) / BEARING_DIAMETER_STEP).round() * BEARING_DIAMETER_STEP;
    let (outer, inner) = match target {
        BearingDimensionTarget::Outer => {
            let outer = stepped(dimensions.outer_diameter())
                .clamp(MIN_BEARING_OUTER_DIAMETER, MAX_BEARING_OUTER_DIAMETER);
            let inner = dimensions
                .inner_diameter()
                .min(outer - MIN_BEARING_DIAMETER_GAP);
            (outer, inner)
        }
        BearingDimensionTarget::Inner => {
            let inner = stepped(dimensions.inner_diameter())
                .clamp(0.0, dimensions.outer_diameter() - MIN_BEARING_DIAMETER_GAP);
            (dimensions.outer_diameter(), inner)
        }
    };
    BearingDimensions::new(outer, inner)
        .expect("clamped bearing tool settings satisfy core dimensions")
}

pub(crate) fn handle_bearing_dimension_shortcuts(
    actions: Res<ButtonInput<GameAction>>,
    selection: Res<SelectedTool>,
    menu: Res<CreationMenuState>,
    mut settings: ResMut<BearingToolSettings>,
    mut state: ResMut<EditorState>,
) {
    let Some((target, direction)) = requested_bearing_dimension_adjustment(
        &actions,
        selection.active_editor_tool(),
        menu.blocks_keyboard(),
    ) else {
        return;
    };
    settings.dimensions = adjusted_bearing_dimensions(settings.dimensions, target, direction);
    state.feedback = Some(format!(
        "Bearing outer {:.2} m, inner {:.2} m",
        settings.dimensions.outer_diameter(),
        settings.dimensions.inner_diameter()
    ));
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CylinderDimensionTarget {
    Outer,
    Inner,
    Length,
    Sweep,
}

pub(crate) fn requested_cylinder_dimension_adjustment(
    actions: &ButtonInput<GameAction>,
    tool: Option<Tool>,
    menu_blocks_input: bool,
) -> Option<(CylinderDimensionTarget, i8)> {
    if tool != Some(Tool::Cylinder) || menu_blocks_input {
        return None;
    }
    [
        (
            GameAction::CylinderSweepDecrease,
            CylinderDimensionTarget::Sweep,
            -1,
        ),
        (
            GameAction::CylinderSweepIncrease,
            CylinderDimensionTarget::Sweep,
            1,
        ),
        (
            GameAction::CylinderLengthDecrease,
            CylinderDimensionTarget::Length,
            -1,
        ),
        (
            GameAction::CylinderLengthIncrease,
            CylinderDimensionTarget::Length,
            1,
        ),
        (
            GameAction::CylinderInnerDecrease,
            CylinderDimensionTarget::Inner,
            -1,
        ),
        (
            GameAction::CylinderInnerIncrease,
            CylinderDimensionTarget::Inner,
            1,
        ),
        (
            GameAction::CylinderOuterDecrease,
            CylinderDimensionTarget::Outer,
            -1,
        ),
        (
            GameAction::CylinderOuterIncrease,
            CylinderDimensionTarget::Outer,
            1,
        ),
    ]
    .into_iter()
    .find_map(|(action, target, direction)| {
        actions.just_pressed(action).then_some((target, direction))
    })
}

pub(crate) fn adjusted_cylinder_dimensions(
    dimensions: CylinderDimensions,
    target: CylinderDimensionTarget,
    direction: i8,
) -> CylinderDimensions {
    if target == CylinderDimensionTarget::Sweep {
        let sweep = (i32::from(dimensions.sweep_angle_degrees())
            + i32::from(direction) * i32::from(CYLINDER_SWEEP_STEP_DEGREES))
        .clamp(
            i32::from(MIN_CYLINDER_SWEEP_DEGREES),
            i32::from(MAX_CYLINDER_SWEEP_DEGREES),
        );
        return dimensions
            .with_sweep_angle_degrees(u16::try_from(sweep).expect("clamped sweep fits u16"))
            .expect("clamped cylinder sweep is valid");
    }
    let step_diameter = f32::from(direction) * CYLINDER_DIAMETER_STEP;
    let stepped_diameter = |value: f32| {
        ((value + step_diameter) / CYLINDER_DIAMETER_STEP).round() * CYLINDER_DIAMETER_STEP
    };
    let (outer, inner, length) = match target {
        CylinderDimensionTarget::Outer => {
            let outer = stepped_diameter(dimensions.outer_diameter())
                .clamp(MIN_CYLINDER_OUTER_DIAMETER, MAX_CYLINDER_OUTER_DIAMETER);
            (
                outer,
                dimensions
                    .inner_diameter()
                    .min(outer - MIN_CYLINDER_DIAMETER_GAP),
                dimensions.axial_length(),
            )
        }
        CylinderDimensionTarget::Inner => (
            dimensions.outer_diameter(),
            stepped_diameter(dimensions.inner_diameter())
                .clamp(0.0, dimensions.outer_diameter() - MIN_CYLINDER_DIAMETER_GAP),
            dimensions.axial_length(),
        ),
        CylinderDimensionTarget::Length => (
            dimensions.outer_diameter(),
            dimensions.inner_diameter(),
            (dimensions.axial_length() + f32::from(direction) * CYLINDER_LENGTH_STEP)
                .clamp(0.25, 8.0),
        ),
        CylinderDimensionTarget::Sweep => unreachable!("sweep adjustment returned above"),
    };
    CylinderDimensions::new(outer, inner, length)
        .expect("clamped cylinder tool settings satisfy core dimensions")
        .with_sweep_angle_degrees(dimensions.sweep_angle_degrees())
        .expect("existing cylinder sweep remains valid")
}

pub(crate) fn handle_cylinder_dimension_shortcuts(
    actions: Res<ButtonInput<GameAction>>,
    selection: Res<SelectedTool>,
    menu: Res<CreationMenuState>,
    material: Option<Res<SelectedMaterial>>,
    mut settings: ResMut<CylinderToolSettings>,
    mut state: ResMut<EditorState>,
) {
    if state.pipe_drag.is_some()
        || state.suspension.drag.is_some()
        || state.suspension.controls.gesture.is_some()
    {
        return;
    }
    let Some((target, direction)) = requested_cylinder_dimension_adjustment(
        &actions,
        selection.active_editor_tool(),
        menu.blocks_keyboard(),
    ) else {
        return;
    };
    if material.is_some_and(|material| material.0 == ConstructionMaterial::Rubber)
        && state.hovered_bearing.and_then(|index| state.placed_bearings.get(index))
            .is_some_and(|s| matches!(s.kind, mechanic_core::JointKind::Suspension(spec) if spec.shock().is_some())) {
        let stop = state.suspension.stop;
        let increment = f32::from(direction) * 0.0025;
        let (length, od) = match target {
            CylinderDimensionTarget::Length => ((stop.length() + increment).max(0.01), stop.od()),
            CylinderDimensionTarget::Outer => (stop.length(), (stop.od() + increment).max(0.01)),
            CylinderDimensionTarget::Inner | CylinderDimensionTarget::Sweep => {
                state.feedback = Some("Bump-stop bore and orientation follow the shock shaft".into());
                return;
            }
        };
        match mechanic_core::BumpStopSpec::new(length, od) {
            Ok(stop) => state.suspension.stop = stop,
            Err(e) => state.feedback = Some(e.to_string()),
        }
        return;
    }
    settings.dimensions = adjusted_cylinder_dimensions(settings.dimensions, target, direction);
    state.feedback = Some(format!(
        "Cylinder outer {:.2} m, inner {:.2} m, length {:.2} m, sweep {}°",
        settings.dimensions.outer_diameter(),
        settings.dimensions.inner_diameter(),
        settings.dimensions.axial_length(),
        settings.dimensions.sweep_angle_degrees(),
    ));
}
