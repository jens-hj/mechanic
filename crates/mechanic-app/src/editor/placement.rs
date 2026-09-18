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

#[cfg(test)]
mod tests {
    use crate::editor::overlay::coordinate_inside;
    use crate::editor::overlay::lattice_coordinates;
    use crate::editor::overlay::lattice_thickness;
    use crate::editor::overlay::placement_lattice_geometry;
    use crate::editor::overlay::smart_snap_range_geometry;
    use crate::editor::placement::FreePlacementSettings;
    use crate::editor::placement::SmartSnapSettings;
    use crate::render::mesh::construction::CUBE_POSITIONS;
    use bevy::math::DVec2;

    use crate::*;

    #[test]
    fn modifier_precedence_selects_precision_only_with_shift_and_control() {
        assert_eq!(
            PlacementGrid::from_modifiers(false, false),
            PlacementGrid::Centimetres25
        );
        assert_eq!(
            PlacementGrid::from_modifiers(false, true),
            PlacementGrid::Centimetres25
        );
        assert_eq!(
            PlacementGrid::from_modifiers(true, false),
            PlacementGrid::Centimetres5
        );
        assert_eq!(
            PlacementGrid::from_modifiers(true, true),
            PlacementGrid::Centimetres1
        );
    }

    #[test]
    fn alt_tap_toggles_but_range_adjustment_does_not() {
        let mut settings = SmartSnapSettings::default();
        settings.update(true, 0.0, false);
        assert!(!settings.enabled);

        settings.update(false, 1.0, false);
        assert!((settings.range - 1.25).abs() <= f32::EPSILON);
        assert!(settings.range_adjusted_this_frame);
        settings.update(true, 0.0, false);
        assert!(!settings.enabled);

        settings.update(false, 100.0, false);
        assert!((settings.range - 5.0).abs() <= f32::EPSILON);
        settings.update(false, -100.0, false);
        assert!((settings.range - 0.25).abs() <= f32::EPSILON);
    }

    #[test]
    fn free_range_adjustment_is_contextual_clamped_and_yields_to_object_snap() {
        let mut settings = FreePlacementSettings::default();
        settings.update(1.0, false, false);
        assert!((settings.range - 5.0).abs() <= f32::EPSILON);
        assert!(!settings.range_adjusted_this_frame);

        settings.update(1.0, true, false);
        assert!((settings.range - 5.25).abs() <= f32::EPSILON);
        assert!(settings.range_adjusted_this_frame);

        settings.update(1.0, true, true);
        assert!((settings.range - 5.25).abs() <= f32::EPSILON);
        assert!(!settings.range_adjusted_this_frame);

        settings.update(1000.0, true, false);
        assert!((settings.range - 30.0).abs() <= f32::EPSILON);
        settings.update(-1000.0, true, false);
        assert!((settings.range - 0.25).abs() <= f32::EPSILON);
    }

    #[test]
    fn free_fallback_is_garage_only_and_requires_an_eligible_tool() {
        let origin = Vec3::new(1.0, 6.0, 2.0);
        let direction = Vec3::NEG_Z;
        assert_eq!(
            free_placement_point_on_miss(
                Tool::Block,
                PlacementBounds::GarageBuild,
                origin,
                direction,
                5.0,
                false,
            ),
            Some(Vec3::new(1.0, 6.0, -3.0))
        );
        assert!(
            free_placement_point_on_miss(
                Tool::Bearing,
                PlacementBounds::GarageBuild,
                origin,
                direction,
                5.0,
                false,
            )
            .is_none()
        );
        assert!(
            free_placement_point_on_miss(
                Tool::Block,
                PlacementBounds::World {
                    origin: DVec2::ZERO,
                },
                origin,
                direction,
                5.0,
                false,
            )
            .is_none()
        );
        assert!(
            free_placement_point_on_miss(
                Tool::Block,
                PlacementBounds::GarageBuild,
                origin,
                direction,
                5.0,
                true,
            )
            .is_none()
        );
    }

    #[test]
    fn lattice_coordinates_keep_global_phase_and_emphasis_hierarchy() {
        let coordinates = lattice_coordinates(-0.25, 0.25, 0, PlacementGrid::Centimetres1);
        assert_eq!(coordinates.len(), 50);
        let mut expected = -0.245;
        for coordinate in coordinates {
            assert!((coordinate - expected).abs() < 1.0e-5);
            expected += 0.01;
        }
        assert!(lattice_thickness(0, 50) > lattice_thickness(0, 10));
        assert!(lattice_thickness(0, 10) > lattice_thickness(0, 2));
    }

    #[test]
    fn lattice_wraps_only_one_cell_beyond_the_preview() {
        let selection_low = Vec3::ZERO;
        let selection_high = Vec3::splat(0.25);
        let geometry = placement_lattice_geometry(
            selection_low,
            selection_high,
            Vec3::ZERO,
            PlacementGrid::Centimetres5,
            None,
        );

        let centers = lattice_line_centers(&geometry);
        assert!(!centers.is_empty());
        for center in centers {
            assert!(
                center.cmpge(Vec3::splat(-0.050_01)).all()
                    && center.cmple(Vec3::splat(0.300_01)).all(),
                "line centre {center:?} escaped the one-cell envelope"
            );
            assert!(
                !(0..3).all(|axis| {
                    coordinate_inside(center[axis], selection_low[axis], selection_high[axis])
                }),
                "line centre {center:?} crossed the preview interior"
            );
        }
    }

    #[test]
    fn dragging_shows_only_a_one_cell_border_on_the_whole_active_plane() {
        let selection_low = Vec3::ZERO;
        let selection_high = Vec3::new(0.75, 0.25, 0.5);
        let geometry = placement_lattice_geometry(
            selection_low,
            selection_high,
            Vec3::ZERO,
            PlacementGrid::Centimetres5,
            Some(PlacementPlane::Xz),
        );

        let centers = lattice_line_centers(&geometry);
        assert!(!centers.is_empty());
        assert!(centers.iter().all(|center| {
            (center.y - 0.125).abs() < 1.0e-6
                && center.x >= -0.05
                && center.x <= 0.80
                && center.z >= -0.05
                && center.z <= 0.55
                && !(coordinate_inside(center.x, selection_low.x, selection_high.x)
                    && coordinate_inside(center.z, selection_low.z, selection_high.z))
        }));

        for vertices in geometry.positions.chunks_exact(CUBE_POSITIONS.len()) {
            let low = vertices
                .iter()
                .map(|position| Vec3::from_array(*position))
                .fold(Vec3::splat(f32::INFINITY), Vec3::min);
            let high = vertices
                .iter()
                .map(|position| Vec3::from_array(*position))
                .fold(Vec3::splat(f32::NEG_INFINITY), Vec3::max);
            let extent = high - low;
            assert!(extent.y <= 0.004_1, "no line may run normal to XZ");
            assert!(extent.x > 0.01 || extent.z > 0.01);
        }
    }

    #[test]
    fn snap_range_wraps_the_whole_selection_on_the_active_plane() {
        let geometry = smart_snap_range_geometry(
            Vec3::ZERO,
            Vec3::new(0.75, 0.25, 0.5),
            1.0,
            Some(PlacementPlane::Xz),
        );

        let centers = lattice_line_centers(&geometry);
        assert_eq!(centers.len(), 36);
        assert!(
            centers
                .iter()
                .all(|center| (center.y - 0.125).abs() < 1.0e-6)
        );
        assert!(centers.iter().any(|center| center.x < -0.99));
        assert!(centers.iter().any(|center| center.x > 1.74));
        assert!(centers.iter().any(|center| center.z < -0.99));
        assert!(centers.iter().any(|center| center.z > 1.49));
    }

    #[test]
    fn free_preview_snap_range_uses_three_orthogonal_outlines() {
        let geometry = smart_snap_range_geometry(Vec3::ZERO, Vec3::splat(0.25), 0.5, None);

        let centers = lattice_line_centers(&geometry);
        assert_eq!(centers.len(), 108);
        for axis in 0..3 {
            assert!(
                centers
                    .iter()
                    .filter(|center| (center[axis] - 0.125).abs() < 1.0e-6)
                    .count()
                    >= 36
            );
        }
    }

    fn lattice_line_centers(geometry: &OverlayGeometry) -> Vec<Vec3> {
        geometry
            .positions
            .chunks_exact(CUBE_POSITIONS.len())
            .map(|vertices| {
                let low = vertices
                    .iter()
                    .map(|position| Vec3::from_array(*position))
                    .fold(Vec3::splat(f32::INFINITY), Vec3::min);
                let high = vertices
                    .iter()
                    .map(|position| Vec3::from_array(*position))
                    .fold(Vec3::splat(f32::NEG_INFINITY), Vec3::max);
                (low + high) * 0.5
            })
            .collect()
    }
}
