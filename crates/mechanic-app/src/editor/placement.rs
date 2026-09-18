//! Placement grids, smart snapping, and free placement.

use crate::*;

#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub(crate) struct SmartSnapSettings {
    pub(crate) enabled: bool,
    pub(crate) range: f32,
    pub(crate) scrolled_during_hold: bool,
    pub(crate) range_adjusted_this_frame: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct FreePlacementSettings {
    pub(crate) range: f32,
    pub(crate) range_adjusted_this_frame: bool,
}

impl Default for FreePlacementSettings {
    fn default() -> Self {
        Self {
            range: 5.0,
            range_adjusted_this_frame: false,
        }
    }
}

impl FreePlacementSettings {
    pub(crate) fn update(
        &mut self,
        range_steps: f32,
        applicable: bool,
        object_snap_adjusted: bool,
    ) {
        self.range_adjusted_this_frame = false;
        if applicable && !object_snap_adjusted && range_steps != 0.0 {
            self.range = (self.range + range_steps * 0.25).clamp(0.25, 30.0);
            self.range_adjusted_this_frame = true;
        }
    }
}

impl Default for SmartSnapSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            range: 1.0,
            scrolled_during_hold: false,
            range_adjusted_this_frame: false,
        }
    }
}

impl SmartSnapSettings {
    pub(crate) fn update(
        &mut self,
        toggle_released: bool,
        range_steps: f32,
        used_during_hold: bool,
    ) {
        self.range_adjusted_this_frame = false;
        if range_steps != 0.0 {
            self.range = (self.range + range_steps * 0.25).clamp(0.25, 5.0);
            self.scrolled_during_hold = true;
            self.range_adjusted_this_frame = true;
        }
        self.scrolled_during_hold |= used_during_hold;
        if toggle_released {
            if !self.scrolled_during_hold {
                self.enabled = !self.enabled;
            }
            self.scrolled_during_hold = false;
        }
    }
}

pub(crate) fn update_smart_snap_settings(
    actions: Res<ButtonInput<GameAction>>,
    mouse: Res<ButtonInput<MouseButton>>,
    mut state: ResMut<EditorState>,
) {
    let adjustment = f32::from(actions.just_pressed(GameAction::ObjectSnapRangeIncrease))
        - f32::from(actions.just_pressed(GameAction::ObjectSnapRangeDecrease));
    let used_during_hold = actions.pressed(GameAction::ToggleObjectSnap)
        && mouse.any_pressed([MouseButton::Left, MouseButton::Right, MouseButton::Middle]);
    state.smart_snap.update(
        actions.just_released(GameAction::ToggleObjectSnap),
        adjustment,
        used_during_hold,
    );
}

pub(crate) fn update_free_placement_settings(
    actions: Res<ButtonInput<GameAction>>,
    selection: Res<SelectedTool>,
    space: Res<State<world::AppSpace>>,
    mut state: ResMut<EditorState>,
) {
    let adjustment = f32::from(actions.just_pressed(GameAction::FreePlacementRangeIncrease))
        - f32::from(actions.just_pressed(GameAction::FreePlacementRangeDecrease));
    let applicable = *space.get() == world::AppSpace::Garage
        && selection
            .active_editor_tool()
            .is_some_and(tool_supports_free_placement)
        && state.block_drag.is_none()
        && state.pipe_drag.is_none();
    let object_snap_adjusted = actions.just_pressed(GameAction::ObjectSnapRangeIncrease)
        || actions.just_pressed(GameAction::ObjectSnapRangeDecrease);
    state
        .free_placement
        .update(adjustment, applicable, object_snap_adjusted);
}

pub(crate) fn rebuild_placement_snap_index(
    graph: Res<EditorGraph>,
    mut state: ResMut<EditorState>,
) {
    if graph.is_changed() {
        let view = state
            .edit_context
            .and_then(|context| graph.0.in_edit_frame(context.frame).ok());
        state.snap_index.rebuild(view.as_ref().unwrap_or(&graph.0));
    }
}

pub(crate) fn active_placement_grid(actions: &ButtonInput<GameAction>) -> PlacementGrid {
    PlacementGrid::from_modifiers(
        actions.pressed(GameAction::FinePlacement),
        actions.pressed(GameAction::PrecisionPlacement),
    )
}

pub(crate) const fn tool_supports_free_placement(tool: Tool) -> bool {
    matches!(
        tool,
        Tool::Block
            | Tool::Cylinder
            | Tool::Controller
            | Tool::GasEngine
            | Tool::ElectricEngine
            | Tool::Servo
            | Tool::Seat
            | Tool::Input
            | Tool::DimensionLink
    )
}

pub(crate) fn free_placement_point_on_miss(
    tool: Tool,
    bounds: PlacementBounds,
    origin: Vec3,
    direction: Vec3,
    range: f32,
    secondary_pressed: bool,
) -> Option<Vec3> {
    (matches!(
        bounds,
        PlacementBounds::GarageBuild | PlacementBounds::GarageBuildFrame { .. }
    ) && tool_supports_free_placement(tool)
        && !secondary_pressed)
        .then_some(origin + direction * range)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PlacementLatticeKey {
    pub(crate) grid: PlacementGrid,
    pub(crate) low_ticks: IVec3,
    pub(crate) high_ticks: IVec3,
    pub(crate) plane: Option<PlacementPlane>,
}

#[derive(Component, Default)]
pub(crate) struct PlacementLatticeVisual {
    pub(crate) key: Option<PlacementLatticeKey>,
}

#[derive(Component, Default)]
pub(crate) struct SmartGuideVisual {
    pub(crate) guides: Vec<SmartGuide>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SmartSnapRangeKey {
    pub(crate) low_ticks: IVec3,
    pub(crate) high_ticks: IVec3,
    pub(crate) range_ticks: i32,
    pub(crate) plane: Option<PlacementPlane>,
}

#[derive(Component, Default)]
pub(crate) struct SmartSnapRangeVisual {
    pub(crate) key: Option<SmartSnapRangeKey>,
}
